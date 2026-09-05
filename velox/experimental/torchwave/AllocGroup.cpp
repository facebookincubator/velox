/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/torchwave/AllocGroup.h"

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <tuple>
#include <utility>

#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"

namespace torch::wave {

int64_t alignAllocSize(int64_t bytes) {
  return (bytes + kAllocGroupAlign - 1) / kAllocGroupAlign * kAllocGroupAlign;
}

namespace {

int64_t requestBytes(const AllocRequest& request) {
  int64_t numel = 1;
  for (auto dim : request.dims) {
    TORCH_CHECK(dim >= 0, "Negative dim in alloc group request: ", dim);
    numel *= dim;
  }
  return numel * static_cast<int64_t>(c10::elementSize(request.dtype));
}

} // namespace

AllocGroupBuffer allocateAllocGroup(const std::vector<AllocRequest>& requests) {
  AllocGroupBuffer result;
  result.offsets.reserve(requests.size());
  result.slots.reserve(requests.size());

  int64_t cursor = 0;
  for (const auto& request : requests) {
    result.offsets.push_back(cursor);
    cursor += alignAllocSize(requestBytes(request));
  }
  result.totalBytes = cursor;

  // One allocation for the group. Untyped bytes, because the slots carved out
  // of it have their own element types; the alignment above is what lets each
  // slot's offset divide evenly by its element width.
  {
    ScopedAllocCall timed;
    result.base = at::empty(
        {result.totalBytes},
        at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
  }

  std::vector<int64_t> strides;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& request = requests[i];
    const auto elementSize =
        static_cast<int64_t>(c10::elementSize(request.dtype));
    const auto offset = result.offsets[i];
    TORCH_CHECK(
        offset % elementSize == 0,
        "Alloc group slot offset ",
        offset,
        " is not a multiple of the ",
        elementSize,
        " byte element size");
    // Contiguous, which is what set_(storage, offset, sizes) used to imply.
    strides.resize(request.dims.size());
    int64_t stride = 1;
    for (size_t d = request.dims.size(); d-- > 0;) {
      strides[d] = stride;
      stride *= request.dims[d];
    }
    // Built straight from the TensorImpl. This runs once per slot per group per
    // execution -- hundreds of times a step -- and at::empty plus set_ is two
    // dispatcher round-trips each, which measured larger than the allocation
    // the group exists to avoid. The slot still shares the base's storage, so
    // the group is freed exactly when its last slot dies.
    result.slots.push_back(aliasTensor(
        result.base,
        request.dims,
        strides,
        offset / elementSize,
        request.dtype));
  }
  return result;
}

std::vector<AllocGroup> buildAllocGroups(
    const std::vector<AllocLifetime>& lifetimes,
    std::vector<nativert::ValueId>* ungrouped) {
  // Keyed so that iteration is deterministic and, within one allocation point,
  // the groups the host can lay out without waiting come first. needsSync sits
  // ahead of the release point rather than after it so that ordering holds
  // across release points too: everything allocatable at a point is emitted
  // before anything at that point that first has to block on the device.
  struct Key {
    int32_t allocNode;
    int32_t allocStep;
    bool needsSync;
    int32_t freeNode;
    int32_t freeStep;
    bool operator<(const Key& other) const {
      return std::tie(allocNode, allocStep, needsSync, freeNode, freeStep) <
          std::tie(
                 other.allocNode,
                 other.allocStep,
                 other.needsSync,
                 other.freeNode,
                 other.freeStep);
    }
  };

  std::map<Key, AllocGroup> byKey;
  for (const auto& lifetime : lifetimes) {
    if (lifetime.freeStep < 0) {
      // Never released. Grouping it would hold every other slot in the group
      // alive for as long as it lives.
      if (ungrouped != nullptr) {
        ungrouped->push_back(lifetime.actualId);
      }
      continue;
    }
    TORCH_CHECK(
        std::tie(lifetime.freeNode, lifetime.freeStep) >=
            std::tie(lifetime.allocNode, lifetime.allocStep),
        "Alloc lifetime frees at node ",
        lifetime.freeNode,
        " step ",
        lifetime.freeStep,
        " before it allocates at node ",
        lifetime.allocNode,
        " step ",
        lifetime.allocStep);
    Key key{
        lifetime.allocNode,
        lifetime.allocStep,
        lifetime.needsSync,
        lifetime.freeNode,
        lifetime.freeStep};
    auto& group = byKey[key];
    group.allocNode = lifetime.allocNode;
    group.allocStep = lifetime.allocStep;
    group.freeNode = lifetime.freeNode;
    group.freeStep = lifetime.freeStep;
    group.needsSync = lifetime.needsSync;
    group.actualIds.push_back(lifetime.actualId);
    group.dtypes.push_back(lifetime.dtype);
  }

  std::vector<AllocGroup> groups;
  groups.reserve(byKey.size());
  for (auto& [key, group] : byKey) {
    groups.push_back(std::move(group));
  }
  return groups;
}

namespace {

// A point in the graph's execution: which node, and which step of it.
using ExecPoint = std::pair<int32_t, int32_t>;

// Where a value is written, as the lifetime scan and the concat scan both need
// it.
struct Produced {
  ExecPoint at{-1, -1};
  c10::ScalarType dtype{c10::ScalarType::Float};
  bool needsSync{false};
  int32_t numWrites{0};
};

using ProducedMap = folly::F14FastMap<nativert::ValueId, Produced>;

/// Where a value that no launch allocates first reaches the frame.
using BoundMap = folly::F14FastMap<nativert::ValueId, ExecPoint>;

// The earliest point each of LaunchFootprint::binds is put in the frame. A
// value bound by launches at several points is readable from the first of them
// on, which is the one that decides how early anything may measure it.
BoundMap boundPoints(const std::vector<NodeFootprint>& nodes) {
  BoundMap bound;
  for (const auto& node : nodes) {
    for (const auto& launch : node.launches) {
      const ExecPoint at{node.node, launch.step};
      for (auto boundId : launch.binds) {
        auto [it, inserted] = bound.try_emplace(boundId, at);
        if (!inserted && at < it->second) {
          it->second = at;
        }
      }
    }
  }
  return bound;
}

ProducedMap producedPoints(const std::vector<NodeFootprint>& nodes) {
  ProducedMap produced;
  for (const auto& node : nodes) {
    for (const auto& launch : node.launches) {
      const ExecPoint at{node.node, launch.step};
      for (size_t i = 0; i < launch.writes.size(); ++i) {
        auto& entry = produced[launch.writes[i]];
        // The first write is the one that allocates; a later one finds the
        // buffer already there. numWrites is what disqualifies the value.
        if (entry.numWrites == 0) {
          entry.at = at;
          entry.dtype = i < launch.writeDtypes.size() ? launch.writeDtypes[i]
                                                      : c10::ScalarType::Float;
          entry.needsSync =
              i < launch.writeNeedsSync.size() && launch.writeNeedsSync[i];
        }
        ++entry.numWrites;
      }
    }
  }
  return produced;
}

} // namespace

std::vector<AllocLifetime> graphAllocLifetimes(
    const std::vector<NodeFootprint>& nodes,
    AllocGroupStats* stats,
    const folly::F14FastSet<nativert::ValueId>* claimed) {
  const auto produced = producedPoints(nodes);
  folly::F14FastMap<nativert::ValueId, ExecPoint> released;
  folly::F14FastMap<nativert::ValueId, ExecPoint> lastRead;
  // Only the node matters, and a borrow cannot precede the write, so the first
  // borrow's node is the one that answers "does the allocating node take it".
  folly::F14FastMap<nativert::ValueId, int32_t> firstBorrowNode;

  for (const auto& node : nodes) {
    for (auto borrowedId : node.borrowedIds) {
      auto [it, inserted] = firstBorrowNode.try_emplace(borrowedId, node.node);
      if (!inserted && node.node < it->second) {
        it->second = node.node;
      }
    }
    for (const auto& launch : node.launches) {
      const ExecPoint at{node.node, launch.step};
      for (auto readId : launch.reads) {
        auto [it, inserted] = lastRead.try_emplace(readId, at);
        if (!inserted && it->second < at) {
          it->second = at;
        }
      }
    }
    for (size_t i = 0; i < node.releasedIds.size(); ++i) {
      const ExecPoint at{
          node.node, i < node.releaseSteps.size() ? node.releaseSteps[i] : 0};
      auto [it, inserted] = released.try_emplace(node.releasedIds[i], at);
      // A value named by two nodes' last-use sets is released by the later of
      // them; the earlier one would free a buffer the later still reads.
      if (!inserted && it->second < at) {
        it->second = at;
      }
    }
  }

  std::vector<AllocLifetime> lifetimes;
  lifetimes.reserve(produced.size());
  for (const auto& [actualId, entry] : produced) {
    AllocLifetime lifetime;
    lifetime.actualId = actualId;
    lifetime.dtype = entry.dtype;
    lifetime.allocNode = entry.at.first;
    lifetime.allocStep = entry.at.second;
    lifetime.needsSync = entry.needsSync;
    const auto releaseIt = released.find(actualId);
    const auto readIt = lastRead.find(actualId);
    if (stats != nullptr) {
      ++stats->numAllocated;
    }
    auto reject = [&](int32_t AllocGroupStats::* counter) {
      if (stats != nullptr) {
        ++(stats->*counter);
      }
    };
    const auto firstBorrowIt = firstBorrowNode.find(actualId);
    if (claimed != nullptr && claimed->count(actualId) != 0) {
      reject(&AllocGroupStats::numInConcatGroup);
    } else if (entry.numWrites > 1) {
      reject(&AllocGroupStats::numMultiWrite);
    } else if (releaseIt == released.end()) {
      reject(&AllocGroupStats::numEscaping);
    } else if (
        firstBorrowIt != firstBorrowNode.end() &&
        firstBorrowIt->second == entry.at.first) {
      reject(&AllocGroupStats::numBorrowedInOwnNode);
    } else if (releaseIt->second < entry.at) {
      reject(&AllocGroupStats::numBackwardFree);
    } else if (readIt != lastRead.end() && releaseIt->second < readIt->second) {
      // The last-use analysis puts the release before a launch that reads the
      // value. One of the two is wrong; carving a buffer on that reading would
      // free it under a live reader, so the value is left alone.
      reject(&AllocGroupStats::numReadAfterFree);
    } else {
      lifetime.freeNode = releaseIt->second.first;
      lifetime.freeStep = releaseIt->second.second;
      if (stats != nullptr) {
        ++stats->numGrouped;
      }
    }
    lifetimes.push_back(std::move(lifetime));
  }
  // The map iteration order is not stable across runs; the groups built from
  // this must be, so that a slot lands at the same offset every execution.
  std::sort(
      lifetimes.begin(),
      lifetimes.end(),
      [](const AllocLifetime& left, const AllocLifetime& right) {
        return std::tie(left.allocNode, left.allocStep, left.actualId) <
            std::tie(right.allocNode, right.allocStep, right.actualId);
      });
  return lifetimes;
}

namespace {

// Every frame value a size expression reads, including through its arguments.
void sizeExprValues(const SizeExpr& expr, std::vector<nativert::ValueId>& out) {
  out.insert(out.end(), expr.values.begin(), expr.values.end());
  for (const auto& arg : expr.args) {
    sizeExprValues(arg, out);
  }
}

// Whether the host can measure 'operand' at 'point' -- after every launch up to
// and including it has been sized.
//
// An operand an earlier launch produces is simply read out of the frame, which
// is what its size expression does. One the concat's own kernel produces is not
// there yet, so its extent has to come from a size expression over values that
// are: a reserve function is rejected rather than run early, since what it
// reads is its own business and need not exist yet.
// Why an operand's extent cannot be computed at 'point', or nullptr when it
// can. The layout needs every operand measurable at one common point, so a
// single operand answering non-null here refuses the whole concat -- and the
// first three reasons hold at every point, not just this one.
const char* operandUnmeasurableReason(
    const ConcatInputInfo& operand,
    const ProducedMap& produced,
    const BoundMap& bound,
    ExecPoint point,
    ExecPoint concatPoint) {
  // Where a value can first be read out of the frame.
  //
  // A value the scan saw is placed exactly: written by a launch, or -- for a
  // view, a delegated output or an in-place alias, none of which allocates --
  // put in the frame by the sizing pass of the launch that owns it.
  //
  // One it did not see gets the concat's own point, the last one the layout may
  // use. Barely half of this graph is covered by wave kernels; a value no
  // launch writes is as likely to be an eager op's output as a graph input, and
  // the scan cannot tell them apart. Reading "not seen" as "there from the
  // start" laid concat results out from operands whose frame slot was still
  // empty, which arrives as a zero extent rather than as an error -- an empty
  // operand is legal -- and gave a result of the wrong size. What does hold of
  // any of them is that the concat's own launch reads it, so it is there by
  // then.
  auto readableFrom = [&](nativert::ValueId valueId) -> ExecPoint {
    const auto producedIt = produced.find(valueId);
    if (producedIt != produced.end()) {
      return producedIt->second.at;
    }
    const auto boundIt = bound.find(valueId);
    if (boundIt != bound.end()) {
      return boundIt->second;
    }
    return concatPoint;
  };
  auto notReadableYet = [&](nativert::ValueId valueId) {
    return point < readableFrom(valueId);
  };
  // An extent the device settles is unknown until the step that writes it has
  // run and the host has read it back -- but known from there on, like any
  // other value an earlier step produced. It used to be refused at every point,
  // which refused the whole concat: one such operand and no point measured all
  // of them. What it really does is put a floor under the placement. A concat
  // wide enough to need host shapes has already had that producer pushed into
  // an earlier step, so the floor is the step before the concat, and the layout
  // is computed there from an extent that is by then an ordinary frame
  // tensor's.
  if (operand.hasShapeOnDevice) {
    // Strictly after: the extent is written by the step, so it is not back on
    // the host until that step has finished. An ordinary operand differs --
    // its shape is the host's from the start, so its own point will do.
    const auto it = produced.find(operand.valueId);
    return (it == produced.end() || it->second.at < point)
        ? nullptr
        : "extent settled on device, not read back until after this point";
  }
  if (operand.isSubgraphInput) {
    auto from = readableFrom(operand.valueId);
    // A copy of this operand's own reads it and writes the band, both at the
    // copy's launch -- so the operand is in the frame by then whatever the scan
    // can prove about it, and the group has to be carved no later or the copy
    // writes a band nothing has bound. The copy's destination is a delegated
    // output, which puts the copy's point in the bound map.
    if (operand.copyDestId >= 0) {
      const auto copyIt = bound.find(operand.copyDestId);
      if (copyIt != bound.end() && copyIt->second < from) {
        from = copyIt->second;
      }
    }
    return point < from ? "not in the frame at this point" : nullptr;
  }
  if (operand.hasReserveInChain) {
    return "behind a reserveShape in its chain";
  }
  if (operand.sizeExpr.op == SizeShortcut::kNone) {
    return "no size shortcut for its extent";
  }
  std::vector<nativert::ValueId> reads;
  sizeExprValues(operand.sizeExpr, reads);
  for (auto readId : reads) {
    if (notReadableYet(readId)) {
      return "its size expression reads a value not in the frame at this point";
    }
    // Measured twice -- once to lay the result out and again by the concat's
    // own reserve -- so the expression has to give the same answer both times.
    // A value written at more than one point can change in between, and the two
    // would disagree about where every operand after it starts.
    const auto it = produced.find(readId);
    if (it != produced.end() && it->second.numWrites != 1) {
      return "its size expression reads a value written more than once";
    }
  }
  return nullptr;
}

// A view is measurable even though it can never be carved: its extent is the
// host's to compute like any other operand's, and it is only its lack of
// storage of its own that keeps it out of the group's members. Refusing it
// here would instead sink the whole concat, since one view operand would make
// the layout look uncomputable at every point.
bool operandMeasurableAt(
    const ConcatInputInfo& operand,
    const ProducedMap& produced,
    const BoundMap& bound,
    ExecPoint point,
    ExecPoint concatPoint) {
  return operandUnmeasurableReason(
             operand, produced, bound, point, concatPoint) == nullptr;
}

// The concat 'layout' describes, in the frame values of the invocation that
// binds it. Both the operand ids and the size expressions that measure them are
// translated, so nothing downstream needs the invocation again.
ConcatFootprint concatFootprint(
    const ConcatLayout& layout,
    nativert::ValueId resultId,
    const OpInvocation& op,
    const IdToValueMap& idToValue) {
  const auto [spec, dtype] = layout.resolve(op.nodeMap());
  const auto& bindings = op.bindings();
  ConcatFootprint footprint;
  footprint.resultId = resultId;
  footprint.dtype = dtype;
  footprint.isStack = spec.isStack;
  footprint.dim = spec.dim;
  footprint.outRank = spec.outRank;
  footprint.layoutNode = layout.layoutNode;
  footprint.layoutStep = layout.layoutStep;
  footprint.operands.reserve(layout.inputs.size());
  for (const auto& input : layout.inputs) {
    auto operand = input;
    auto it = bindings.find(input.valueId);
    operand.valueId = it != bindings.end() ? it->second : input.valueId;
    // The value a launch writes, when the operand only names it through list
    // plumbing. Translated the same way, and separately: the two are different
    // frame slots and only this one is filled by a kernel.
    if (input.writerId >= 0) {
      auto writerIt = bindings.find(input.writerId);
      operand.writerId =
          writerIt != bindings.end() ? writerIt->second : input.writerId;
    }
    operand.sizeExpr = input.sizeExpr.toActual(bindings, idToValue);
    // An operand an earlier kernel wrote is a formal in this one's subgraph,
    // with no producer to ask, so buildConcatLayout left the flag at its
    // default. The actual has one. Answering only from the subgraph refused a
    // pitched band to every operand from the far side of a kernel boundary --
    // which is precisely the set a concat group exists to carve.
    if (!operand.mayWriteStrided) {
      const auto actual = idToValue.find(operand.valueId);
      operand.mayWriteStrided =
          actual != idToValue.end() && producerMayWriteStrided(actual->second);
    }
    // The copy destination is a value of this invocation's own, duplicated
    // from the formal one like any other intermediate, so it needs translating
    // too. Left formal it would name the first invocation's band, and every
    // later use of a deduplicated project op would fill that one instead.
    if (input.copyDestId >= 0) {
      auto destIt = bindings.find(input.copyDestId);
      operand.copyDestId =
          destIt != bindings.end() ? destIt->second : input.copyDestId;
    }
    // The reserve reads the frame through the invocation's bindings, which the
    // group no longer has by the time it measures the operand. Bound here, by
    // value: a reference to either map would dangle once the plan is the only
    // thing left holding the footprint.
    if (input.reserveShape != nullptr) {
      operand.boundReserve =
          [reserve = input.reserveShape, bindings, nodeMap = op.nodeMap()](
              nativert::ExecutionFrame& frame) {
            auto shapes = reserve(nullptr, frame, bindings, nullptr, nodeMap);
            TORCH_CHECK(
                !shapes.empty(),
                "A concat operand's reserve function returned no shape");
            return std::move(shapes[0]);
          };
    }
    footprint.operands.push_back(std::move(operand));
  }
  return footprint;
}

} // namespace

std::vector<AllocGroup> graphConcatGroups(
    const std::vector<NodeFootprint>& nodes,
    folly::F14FastSet<nativert::ValueId>& claimed,
    AllocGroupStats* stats) {
  const auto produced = producedPoints(nodes);
  const auto bound = boundPoints(nodes);
  std::vector<AllocGroup> groups;

  // Every (node, step) in execution order. A concat group is placed at the
  // earliest of these at which its layout can be computed, so the points have
  // to be walked in order rather than derived from the operands' producers.
  std::vector<ExecPoint> executionPoints;
  for (const auto& node : nodes) {
    for (const auto& launch : node.launches) {
      executionPoints.push_back({node.node, launch.step});
    }
  }
  std::sort(executionPoints.begin(), executionPoints.end());
  executionPoints.erase(
      std::unique(executionPoints.begin(), executionPoints.end()),
      executionPoints.end());

  auto count = [&](int32_t AllocGroupStats::* counter) {
    if (stats != nullptr) {
      ++(stats->*counter);
    }
  };

  // The per-reason counters say how many concats were refused but not which,
  // and a wide one left out is the difference between a parallel fill and a
  // chain of __concatCopy walked by a single block. Name them.
  //
  // Refusing a concat whose operands placement gave copies of their own is not
  // a missed optimization but a wrong answer: the copy's destination is a band
  // of the result and only this group binds it, so a refused concat leaves
  // every one of those copies writing through an empty frame slot while its
  // own kernel, told the operand is copied, fills nothing. Placement and this
  // pass have to reach the same answer; say so here rather than at the illegal
  // access it becomes on the device.
  auto checkNoOrphanCopies = [](const ConcatFootprint& concat,
                                const char* reason) {
    for (const auto& operand : concat.operands) {
      TORCH_CHECK(
          operand.copyDestId < 0,
          "Concat %",
          concat.resultId,
          " has an operand copied by an op of its own (%",
          operand.valueId,
          " into %",
          operand.copyDestId,
          ") but gets no allocation group to bind the band: ",
          reason);
    }
  };

  auto refuse = [&](int32_t AllocGroupStats::* counter,
                    const ConcatFootprint& concat,
                    const char* reason) {
    count(counter);
    checkNoOrphanCopies(concat, reason);
    if (WaveConfig::get().trace & WaveConfig::kTiming) {
      std::cout << "  concat %" << concat.resultId << " of "
                << concat.operands.size() << " operands not placed: " << reason
                << std::endl;
    }
  };

  for (const auto& node : nodes) {
    for (const auto& launch : node.launches) {
      const ExecPoint concatPoint{node.node, launch.step};
      for (const auto& concat : launch.concats) {
        if (concat.operands.size() <= 2) {
          count(&AllocGroupStats::numConcatTooFew);
          checkNoOrphanCopies(concat, "two operands or fewer");
          continue;
        }
        // The result has to be this launch's to place: one written at two
        // points, or already spoken for by another concat, is not.
        const auto resultIt = produced.find(concat.resultId);
        if (resultIt == produced.end() || resultIt->second.numWrites != 1 ||
            claimed.count(concat.resultId) != 0) {
          refuse(
              &AllocGroupStats::numConcatUnplaceableOperand,
              concat,
              "the result is not this launch's to place");
          continue;
        }

        // An operand whose extent the device settles no longer refuses the
        // concat. It puts a floor under where the layout can be computed --
        // after the step that writes it -- and the host reads the extent back
        // before carving, which is what 'needsSync' below arranges.

        // A reserve function reads whatever it likes, and nothing says what
        // it needs is in the frame when the result would be laid out.
        // uncarvedOperandShape still refuses to run one, so a group formed
        // over such an operand aborts when it measures it -- the refusal and
        // that check are a pair, and both go when the reserve becomes a
        // legitimate source of dims.
        const ConcatInputInfo* behindReserve = nullptr;
        for (const auto& operand : concat.operands) {
          if (operand.hasReserveInChain) {
            behindReserve = &operand;
            break;
          }
        }
        if (behindReserve != nullptr) {
          refuse(
              &AllocGroupStats::numConcatUnplaceableOperand,
              concat,
              "an operand's extent is known only to its own reserve function");
          continue;
        }

        // Taken from placement, not searched for again. Placement settled it
        // from the step each operand's producer landed on, and decided every
        // carve against it; a group created at any other step installs its
        // collector where the members are not sized, so nothing is captured
        // and every member silently falls back to a copy.
        ExecPoint at{concat.layoutNode, concat.layoutStep};
        if (at.first < 0) {
          // Placement took no decision for this concat -- the mode was off
          // when it was compiled. Nothing is carved either, so the group only
          // has to own the result, which its own launch can do.
          at = concatPoint;
        }
        TORCH_CHECK(
            !(concatPoint < at),
            "Concat %",
            concat.resultId,
            " is laid out at node ",
            at.first,
            " step ",
            at.second,
            ", after its own launch at node ",
            concatPoint.first,
            " step ",
            concatPoint.second);
        if (at.first < 0) {
          std::string why = "no execution point measures every operand";
          if (WaveConfig::get().trace & WaveConfig::kTiming) {
            // Which operands are in the way, tallied at the concat's own point
            // -- the latest one the layout may use, so anything unmeasurable
            // here is unmeasurable everywhere.
            std::map<std::string, std::vector<nativert::ValueId>> blockers;
            for (const auto& operand : concat.operands) {
              if (const auto* reason = operandUnmeasurableReason(
                      operand, produced, bound, concatPoint, concatPoint)) {
                blockers[reason].push_back(operand.valueId);
              }
            }
            for (const auto& [reason, ids] : blockers) {
              why += fmt::format(" [{} x {}:", ids.size(), reason);
              for (size_t i = 0; i < ids.size() && i < 6; ++i) {
                why += fmt::format(" %{}", ids[i]);
              }
              why += "]";
            }
          }
          refuse(
              &AllocGroupStats::numConcatUnplaceableOperand,
              concat,
              why.c_str());
          continue;
        }

        // A copy of an operand's own writes the band this group binds, so the
        // group has to be carved no later than the launch that holds the copy.
        // The copy's destination is a delegated output, which puts it in the
        // bound map at the copy's own point.
        ExecPoint firstCopy{-1, -1};
        nativert::ValueId firstCopyOperand = -1;
        nativert::ValueId firstCopyDest = -1;
        for (const auto& operand : concat.operands) {
          if (operand.copyDestId < 0) {
            continue;
          }
          const auto destIt = bound.find(operand.copyDestId);
          if (destIt == bound.end()) {
            continue;
          }
          if (firstCopy.first < 0 || destIt->second < firstCopy) {
            firstCopy = destIt->second;
            firstCopyOperand = operand.valueId;
            firstCopyDest = operand.copyDestId;
          }
        }
        if (firstCopy.first >= 0 && firstCopy < at) {
          // Name what held the layout back past the copy. Every operand is
          // measurable at 'at' by construction, so asking again at the copy's
          // point is what separates the operands that could have been measured
          // there from the one or two that actually moved the placement.
          // Where each blocker DOES become readable, and by which map, so the
          // answer says whether the operand's producer merely landed in a later
          // step or is invisible to the scan entirely (in which case it falls
          // back to the concat's own point).
          std::string why;
          for (const auto& operand : concat.operands) {
            const auto* reason = operandUnmeasurableReason(
                operand, produced, bound, firstCopy, concatPoint);
            if (reason == nullptr) {
              continue;
            }
            std::string from = "the concat's own point (seen by neither map)";
            if (const auto it = produced.find(operand.valueId);
                it != produced.end()) {
              from = fmt::format(
                  "written at node {} step {}",
                  it->second.at.first,
                  it->second.at.second);
            } else if (const auto boundIt = bound.find(operand.valueId);
                       boundIt != bound.end()) {
              from = fmt::format(
                  "bound at node {} step {}",
                  boundIt->second.first,
                  boundIt->second.second);
            }
            why += fmt::format(
                " [%{}: {}, readable from {}]", operand.valueId, reason, from);
          }
          TORCH_CHECK(
              false,
              "Concat %",
              concat.resultId,
              " of ",
              concat.operands.size(),
              " operands is placed at node ",
              at.first,
              " step ",
              at.second,
              ", after the copy of operand %",
              firstCopyOperand,
              " into %",
              firstCopyDest,
              " at node ",
              firstCopy.first,
              " step ",
              firstCopy.second,
              ", which would write a band nothing has bound yet."
              " Not measurable at the copy's point:",
              why);
        }

        // A concat may join the same value at more than one position --
        // cat([x, y, x]) -- and one buffer cannot be two regions of the result.
        // Neither occurrence is carved; the value keeps its own buffer and the
        // concat copies it into both.
        folly::F14FastMap<nativert::ValueId, int32_t> occurrences;
        for (const auto& operand : concat.operands) {
          ++occurrences[operand.valueId];
        }

        auto layout = std::make_shared<ConcatGroupLayout>();
        layout->resultId = concat.resultId;
        layout->dtype = concat.dtype;
        layout->isStack = concat.isStack;
        layout->dim = concat.dim;
        layout->outRank = concat.outRank;
        layout->operands = concat.operands;
        layout->memberOfOperand.assign(concat.operands.size(), -1);

        AllocGroup group;
        group.allocNode = at.first;
        group.allocStep = at.second;
        // The layout reads every operand's extent, carved or not, so the host
        // has to have them all before it lays the result out. An operand the
        // device sizes is only back after a wait, and one whose producer was
        // itself flagged carries that on -- so the whole group waits, not just
        // the members.
        //
        // Asked of the value a launch WRITES, which is what 'produced' is keyed
        // by. An operand reached through a prim.ListUnpack names the same
        // tensor as the value that was packed, but it is the packed value the
        // producing launch writes -- so looking the operand's own id up misses
        // the entry and the producer's wait is dropped. Silently: the group
        // then lays out ahead of a transfer it was supposed to wait for.
        for (const auto& operand : concat.operands) {
          if (operand.hasShapeOnDevice) {
            group.needsSync = true;
          }
          const auto producedIt = produced.find(operand.writtenValueId());
          if (producedIt != produced.end() && producedIt->second.needsSync) {
            group.needsSync = true;
          }
        }
        // The buffer is the concat result, so it goes when the result does --
        // not at a point of the group's own. Left unset rather than guessed:
        // nothing in the concat path keys on it.
        group.freeNode = -1;
        group.freeStep = -1;
        // Every operand gets a view of the region it occupies, whichever step
        // produces it: one written from 'at' onwards writes its band directly,
        // and one already materialized before 'at' is copied into it. What
        // still cannot take a view is an operand that is not a value of its
        // own to redirect.
        // Why each operand the group could not carve was left out, so a wide
        // concat that still copies says which rule cost it. Tallied only under
        // the trace bit; the loop is per operand of every concat.
        // A member and the step its own launch is sized at. Collected rather
        // than pushed straight into one group, because the collector is built
        // per step (CompiledOp.cpp, AllocGroupCollector(stepGroups)) and can
        // only intercept a member sized in the step its group belongs to.
        struct AcceptedMember {
          size_t operandIndex;
          nativert::ValueId written;
          ExecPoint sizedAt;
          bool needsSync;
        };
        std::vector<AcceptedMember> accepted;
        std::map<std::string, std::vector<nativert::ValueId>> uncarved;
        const bool tallyUncarved =
            (WaveConfig::get().trace & WaveConfig::kTiming) != 0;
        auto leaveOut = [&](const char* reason, nativert::ValueId id) {
          if (tallyUncarved) {
            uncarved[reason].push_back(id);
          }
        };
        for (size_t i = 0; i < concat.operands.size(); ++i) {
          const auto& operand = concat.operands[i];
          // Applied, not decided. Placement took this decision while the
          // concat was placed, from the same facts plus the one it alone knows
          // -- the step each operand's producer landed on -- and gave every
          // operand it did not carve a copy of its own. Deciding again here is
          // what used to let the two disagree: this pass would carve a band a
          // copy was already filling, or decline one whose copy then wrote
          // through a frame slot nothing had bound.
          if (!operand.carve) {
            leaveOut(
                operand.carveReason.empty() ? "not carved by placement"
                                            : operand.carveReason.c_str(),
                operand.valueId);
            continue;
          }
          const auto written = operand.writtenValueId();
          const auto it = produced.find(written);
          TORCH_CHECK(
              it != produced.end(),
              "Concat %",
              concat.resultId,
              " operand %",
              operand.valueId,
              " (written as %",
              written,
              ") was carved by placement but no launch of the compiled grid "
              "writes it");
          if (occurrences[written] != 1) {
            // cat([x, y, x]): one buffer cannot be two regions of the result.
            // Placement counts occurrences too, so reaching this means the
            // operand lists differ between the two.
            leaveOut("joined at more than one position", operand.valueId);
            continue;
          }
          if (claimed.count(written) != 0) {
            leaveOut("already claimed by another group", operand.valueId);
            continue;
          }
          // The band is bound to the value a launch writes. For an operand
          // reached through a prim.ListUnpack that is the packed value, not the
          // operand: the unpack aliases it into the operand's slot afterwards,
          // so binding the operand's slot would be overwritten.
          accepted.push_back({i, written, it->second.at, it->second.needsSync});
        }
        // One group per step the members are sized at, rather than one group
        // for the concat. They share the single result: materializeConcatGroup
        // keeps an existing same-shape buffer instead of allocating, so the
        // first group to run allocates it and the rest carve into that one.
        //
        // Every group measures the WHOLE result, carved members from the sizes
        // the collector captured and the rest through their own reserve or size
        // expression -- which reads the expression's leaves, not the operand's
        // tensor, so an operand not yet written still measures correctly. The
        // earliest group is therefore only legal if every operand is measurable
        // at its step; when one is not, the split is abandoned and the concat
        // keeps the single group at 'at', which is today's behaviour.
        std::map<ExecPoint, std::vector<const AcceptedMember*>> byStep;
        for (const auto& member : accepted) {
          byStep[member.sizedAt].push_back(&member);
        }
        bool splitLegal = byStep.size() > 1;
        if (splitLegal) {
          const auto earliest = byStep.begin()->first;
          for (const auto& operand : concat.operands) {
            if (!operandMeasurableAt(
                    operand, produced, bound, earliest, concatPoint)) {
              splitLegal = false;
              break;
            }
          }
        }
        if (!splitLegal) {
          byStep.clear();
          for (const auto& member : accepted) {
            byStep[at].push_back(&member);
          }
          if (byStep.empty()) {
            byStep[at] = {};
          }
        }

        int32_t totalMembers = 0;
        bool firstGroup = true;
        for (const auto& [stepPoint, members] : byStep) {
          AllocGroup stepGroup;
          stepGroup.allocNode = stepPoint.first;
          stepGroup.allocStep = stepPoint.second;
          stepGroup.needsSync = group.needsSync;
          stepGroup.freeNode = -1;
          stepGroup.freeStep = -1;
          // Each group carries its own view of which operands it carves; the
          // others read their extent from the frame, including any an earlier
          // group already carved -- that one holds a view of its band, which is
          // the right shape.
          auto stepLayout = std::make_shared<ConcatGroupLayout>(*layout);
          stepLayout->memberOfOperand.assign(concat.operands.size(), -1);
          for (const auto* member : members) {
            stepLayout->memberOfOperand[member->operandIndex] =
                static_cast<int32_t>(stepGroup.actualIds.size());
            stepGroup.actualIds.push_back(member->written);
            stepGroup.dtypes.push_back(concat.dtype);
            stepGroup.needsSync = stepGroup.needsSync || member->needsSync;
          }
          totalMembers += static_cast<int32_t>(stepGroup.actualIds.size());
          // A group with nothing to carve is still worth forming: it owns the
          // result's backing store and lays every operand's region out on the
          // host, which is what the operands are filled through. Only the first
          // needs to, though -- a later empty one would allocate nothing and
          // carve nothing.
          if (stepGroup.actualIds.empty()) {
            if (!firstGroup) {
              continue;
            }
            count(&AllocGroupStats::numConcatNoMembers);
          }
          for (auto memberId : stepGroup.actualIds) {
            claimed.insert(memberId);
          }
          if (stats != nullptr) {
            ++stats->numConcatGroups;
            stats->numConcatMembers +=
                static_cast<int32_t>(stepGroup.actualIds.size());
          }
          stepGroup.concat = std::move(stepLayout);
          groups.push_back(std::move(stepGroup));
          firstGroup = false;
        }
        claimed.insert(concat.resultId);

        if (tallyUncarved) {
          std::cout << "  concat %" << concat.resultId << " of "
                    << concat.operands.size() << " operands: " << totalMembers
                    << " carved over " << byStep.size() << " group(s)";
          if (!uncarved.empty()) {
            std::cout << ", left out";
            for (const auto& [reason, ids] : uncarved) {
              std::cout << " [" << ids.size() << " x " << reason << ":";
              for (size_t k = 0; k < ids.size() && k < 8; ++k) {
                std::cout << " %" << ids[k];
              }
              std::cout << "]";
            }
          }
          std::cout << std::endl;
        }
      }
    }
  }
  return groups;
}

std::vector<LaunchFootprint> collectLaunchFootprints(
    std::vector<OpInvocation>& ops,
    const IdToValueMap& idToValue,
    const ValueTypes& types,
    std::vector<nativert::ValueId>* borrowedIds) {
  std::vector<nativert::ValueId> borrowedSink;
  std::vector<nativert::ValueId>& borrowed =
      borrowedIds != nullptr ? *borrowedIds : borrowedSink;
  std::vector<LaunchFootprint> footprints;
  for (auto& op : ops) {
    // The grid the mode will actually run this op on, cooperative or not.
    const auto& grid = allocGroupGrid(op);
    for (size_t stepIdx = 0; stepIdx < grid.size(); ++stepIdx) {
      for (const auto& launch : grid[stepIdx]) {
        LaunchData data(launch, op, idToValue);
        LaunchFootprint footprint;
        footprint.step = static_cast<int32_t>(stepIdx);
        footprint.reads = data.actualInputs;
        // The kernel's own inputs are not everything the host touches while it
        // sizes this launch. A reserve function or a size expression reads the
        // operands of the subgraph nodes, and a host-side view output is built
        // by running its view node against them. All of those read the frame,
        // so all of them have to count as reads -- a value read by any of them
        // cannot be one whose tensor does not exist until the step's groups are
        // carved.
        const auto& bindings = op.bindings();
        auto addBorrow = [&](nativert::ValueId actualId) {
          footprint.reads.push_back(actualId);
          borrowed.push_back(actualId);
        };
        // Outputs of an in-place op. Its reserve function aliases the output to
        // the argument it mutates, so the value's buffer is deliberately
        // another value's and carving it one of its own would drop the
        // mutation. Nothing in the descriptor says so -- the aliasing is done
        // by the reserve at execution time -- but the registration does. The
        // arguments of such an op are borrowed for the same reason: the reserve
        // takes self's tensor and stores it as the output.
        folly::F14FastSet<nativert::ValueId> inPlaceOutputs;
        if (launch.op != nullptr) {
          for (const auto* node : launch.op->allNodes()) {
            const auto* meta = Registry::metadata(node->target());
            const bool mutates =
                meta != nullptr && meta->mutatesArg.has_value();
            for (const auto& input : node->inputs()) {
              if (input.value == nullptr) {
                continue;
              }
              auto it = bindings.find(input.value->id());
              const auto actualId =
                  it != bindings.end() ? it->second : input.value->id();
              if (mutates) {
                addBorrow(actualId);
              } else {
                footprint.reads.push_back(actualId);
              }
            }
            if (!mutates) {
              continue;
            }
            for (const auto* output : node->outputs()) {
              auto it = bindings.find(output->id());
              inPlaceOutputs.insert(
                  it != bindings.end() ? it->second : output->id());
            }
          }
        }
        for (const auto& desc : data.actualOutputDescs) {
          if (desc.viewNode != nullptr) {
            // Already an actual node, so its operands need no translation.
            for (const auto& input : desc.viewNode->inputs()) {
              if (input.value != nullptr) {
                addBorrow(input.value->id());
              }
            }
          }
          if (desc.aliasSelfId.has_value()) {
            addBorrow(*desc.aliasSelfId);
          }
        }
        auto addWrite = [&](nativert::ValueId actualId, bool needsSync) {
          if (inPlaceOutputs.count(actualId) != 0 ||
              static_cast<size_t>(actualId) >= types.types.size()) {
            return;
          }
          const auto* meta = types.types[actualId];
          if (meta == nullptr) {
            return;
          }
          footprint.writes.push_back(actualId);
          footprint.writeDtypes.push_back(meta->dtype());
          footprint.writeNeedsSync.push_back(needsSync);
        };
        for (size_t i = 0; i < data.actualOutputs.size(); ++i) {
          if (i >= data.actualOutputDescs.size()) {
            continue;
          }
          const auto& desc = data.actualOutputDescs[i];
          const bool needsSync =
              desc.reserveShape != nullptr || desc.shapeSetOnDevice;
          const auto kind = i < data.actualOutputTypes.size()
              ? data.actualOutputTypes[i]
              : nativert::Type::Kind::Tensor;

          // A fused concat's result. Recorded whether or not the concat scan
          // ends up placing it, so the report can say how many it saw.
          if (desc.concatLayout != nullptr &&
              kind == nativert::Type::Kind::Tensor) {
            footprint.concats.push_back(concatFootprint(
                *desc.concatLayout, data.actualOutputs[i], op, idToValue));
          }

          // A TensorList output is a header, not a buffer, but each of its
          // elements is reserved and allocated on its own and so is a candidate
          // like any other output. The element shapes come from the reserve at
          // execution time; only the identities are needed here.
          if (kind == nativert::Type::Kind::TensorList) {
            auto valueIt = idToValue.find(data.actualOutputs[i]);
            if (desc.reserveShape == nullptr || valueIt == idToValue.end()) {
              continue;
            }
            for (const auto* element : valueIt->second->getListElements()) {
              auto elementIt = bindings.find(element->id());
              addWrite(
                  elementIt != bindings.end() ? elementIt->second
                                              : element->id(),
                  needsSync);
            }
            continue;
          }
          if (kind != nativert::Type::Kind::Tensor) {
            continue;
          }
          // None of these allocates. A view and an in-place alias borrow
          // another value's storage and a delegated output is reserved by the
          // output it is coordinated with. A shape-only output is not this op's
          // buffer at all -- it is a fake tensor carrying the shape of a value
          // something else produces, typically a standalone; carving it a slot
          // replaces that value with an empty one.
          if (desc.viewNode != nullptr || desc.delegated || desc.shapeOnly ||
              desc.aliasSelfId.has_value()) {
            // No buffer of its own, but the frame does not hold it before this
            // launch is sized either, so it still fixes how early the concat
            // pass may measure it.
            footprint.binds.push_back(data.actualOutputs[i]);
            continue;
          }
          addWrite(data.actualOutputs[i], needsSync);
        }
        footprints.push_back(std::move(footprint));
      }
    }
  }
  return footprints;
}

namespace {

// Steps the compiled grid of 'op' runs, in the variant the allocation-group
// path fixes before execution.
int32_t gridSteps(OpInvocation& op) {
  return static_cast<int32_t>(allocGroupGrid(op).size());
}

} // namespace

std::vector<NodeFootprint> collectNodeFootprints(
    const std::vector<std::unique_ptr<CompiledNode>>& nodes,
    const IdToValueMap& idToValue,
    const ValueTypes& types) {
  std::vector<NodeFootprint> footprints;
  footprints.reserve(nodes.size());
  for (size_t n = 0; n < nodes.size(); ++n) {
    NodeFootprint footprint;
    footprint.node = static_cast<int32_t>(n);
    auto* invocation = nodes[n]->kernels();
    if (invocation == nullptr) {
      footprints.push_back(std::move(footprint));
      continue;
    }
    auto& ops = invocation->ops();
    footprint.launches =
        collectLaunchFootprints(ops, idToValue, types, &footprint.borrowedIds);
    for (auto& op : ops) {
      footprint.lastStep = std::max(footprint.lastStep, gridSteps(op) - 1);
    }

    // Where the node puts each of its last-use values. Under step-level release
    // a value goes as soon as every op that reads it has run out of grid steps,
    // which is what releaseLastUseAtStep does; otherwise, and for a value whose
    // readers are unknown, it goes at the node's last step.
    const bool byStep = invocation->stepLevelRelease();
    const auto& lastUseIds = invocation->lastUseIds();
    const auto& readerOps = invocation->lastUseReaderOps();
    footprint.releasedIds = lastUseIds;
    footprint.releaseSteps.reserve(lastUseIds.size());
    for (size_t i = 0; i < lastUseIds.size(); ++i) {
      int32_t step = footprint.lastStep;
      if (byStep && i < readerOps.size() && !readerOps[i].empty()) {
        step = 0;
        for (auto reader : readerOps[i]) {
          step = std::max(step, gridSteps(ops.at(reader)) - 1);
        }
      }
      footprint.releaseSteps.push_back(step);
    }
    footprints.push_back(std::move(footprint));
  }
  return footprints;
}

GraphAllocGroupPlan buildGraphAllocGroupPlan(
    const std::vector<NodeFootprint>& nodes) {
  GraphAllocGroupPlan plan;
  plan.perNode.resize(nodes.size());

  // Concats first: a concat group allocates its result and the operands it
  // carves as one tensor, so those values are no longer the lifetime pass's to
  // place. Grouping them again would carve a second buffer over the first.
  folly::F14FastSet<nativert::ValueId> claimed;
  auto concatGroups = concatAllocGroupEnabled()
      ? graphConcatGroups(nodes, claimed, &plan.stats)
      : std::vector<AllocGroup>{};

  plan.concatPlaced = claimed;

  auto lifetimes = graphAllocLifetimes(nodes, &plan.stats, &claimed);
  std::vector<nativert::ValueId> ungrouped;
  // With the lifetime grouping off every lifetime becomes an allocation of its
  // own, which is what the ordinary path does with the values it is handed.
  std::vector<AllocGroup> groups;
  if (WaveConfig::get().enableLifetimeAllocGroup) {
    groups = buildAllocGroups(lifetimes, &ungrouped);
  } else {
    for (const auto& lifetime : lifetimes) {
      ungrouped.push_back(lifetime.actualId);
    }
  }
  // Ahead of the lifetime groups at the same step: a concat group's members are
  // the outputs of that step's launches, and the ordinary path must find their
  // slots already there.
  groups.insert(
      groups.begin(),
      std::make_move_iterator(concatGroups.begin()),
      std::make_move_iterator(concatGroups.end()));

  auto nodePlan = [&](int32_t node) -> AllocGroupPlan* {
    return node >= 0 && static_cast<size_t>(node) < plan.perNode.size()
        ? &plan.perNode[node]
        : nullptr;
  };

  // Ungrouped values are reported by the node that allocates them, which is the
  // one that would have carved them; buildAllocGroups only knows their ids.
  folly::F14FastSet<nativert::ValueId> ungroupedSet(
      ungrouped.begin(), ungrouped.end());
  for (const auto& lifetime : lifetimes) {
    if (ungroupedSet.count(lifetime.actualId) == 0) {
      continue;
    }
    if (auto* target = nodePlan(lifetime.allocNode)) {
      target->ungrouped.push_back(lifetime.actualId);
    }
  }

  // The same grouping with the release step dropped from the key, to price what
  // insisting on the exact step costs.
  std::set<std::tuple<int32_t, int32_t, bool, int32_t>> byReleaseNode;
  // Groups arrive ordered by (allocation point, needsSync, release point), so
  // taking them in order keeps each node's sync-free groups ahead of its
  // waiting ones -- the point of the ordering is that the host lays out
  // everything it can before it blocks.
  for (const auto& group : groups) {
    auto* target = nodePlan(group.allocNode);
    if (target == nullptr) {
      continue;
    }
    const auto index = target->groups.size();
    if (static_cast<size_t>(group.allocStep) >= target->groupsByStep.size()) {
      target->groupsByStep.resize(static_cast<size_t>(group.allocStep) + 1);
    }
    target->groupsByStep[static_cast<size_t>(group.allocStep)].push_back(index);
    for (auto actualId : group.actualIds) {
      target->groupOfValue[actualId] = index;
    }
    target->groups.push_back(group);

    // The concat groups are counted on their own, in numConcatGroups: they fold
    // no lifetimes together, so mixing them into these would overstate what the
    // lifetime grouping found.
    if (group.concat != nullptr) {
      continue;
    }
    ++plan.stats.numGroups;
    if (group.freeNode > group.allocNode) {
      ++plan.stats.numCrossNodeGroups;
    }
    plan.stats.largestGroup = std::max(
        plan.stats.largestGroup, static_cast<int32_t>(group.actualIds.size()));
    byReleaseNode.emplace(
        group.allocNode, group.allocStep, group.needsSync, group.freeNode);
  }
  plan.stats.numGroupsByNode = static_cast<int32_t>(byReleaseNode.size());
  return plan;
}

std::string allocGroupPlanReport(const GraphAllocGroupPlan& plan) {
  const auto& stats = plan.stats;
  std::string out = fmt::format(
      "Alloc group plan: {} groups over {} of {} allocated values, "
      "{} fewer allocator calls per execution\n"
      "  not grouped: {} never released, {} borrowed by their own node, "
      "{} written at several points, {} released before allocated, "
      "{} read after released, {} placed by a concat group\n"
      "  {} groups outlive the node that allocates them, largest group {} "
      "values; ignoring the release step would give {} groups\n",
      stats.numGroups,
      stats.numGrouped,
      stats.numAllocated,
      stats.numGrouped - stats.numGroups,
      stats.numEscaping,
      stats.numBorrowedInOwnNode,
      stats.numMultiWrite,
      stats.numBackwardFree,
      stats.numReadAfterFree,
      stats.numInConcatGroup,
      stats.numCrossNodeGroups,
      stats.largestGroup,
      stats.numGroupsByNode);

  const auto concatsSeen = stats.numConcatGroups + stats.numConcatTooFew +
      stats.numConcatOnDevice + stats.numConcatUnplaceableOperand +
      stats.numConcatNoMembers;
  if (concatsSeen > 0) {
    out += fmt::format(
        "  concat groups: {} of {} fused concats place their result ahead of "
        "{} operands, each one allocation and one copy fewer\n"
        "    not placed: {} of two operands or fewer, {} with an operand the "
        "device sizes, {} with an operand the host cannot measure early, {} "
        "with no operand left to redirect\n",
        stats.numConcatGroups,
        concatsSeen,
        stats.numConcatMembers,
        stats.numConcatTooFew,
        stats.numConcatOnDevice,
        stats.numConcatUnplaceableOperand,
        stats.numConcatNoMembers);
  }

  // Group sizes, so the shape of the consolidation is visible rather than just
  // its total: a thousand pairs and ten groups of a hundred save the same
  // number of calls and are not the same result.
  std::map<size_t, int32_t> sizeHistogram;
  for (const auto& nodePlan : plan.perNode) {
    for (const auto& group : nodePlan.groups) {
      ++sizeHistogram[group.actualIds.size()];
    }
  }
  if (!sizeHistogram.empty()) {
    out += "  group sizes:";
    for (const auto& [size, count] : sizeHistogram) {
      out += fmt::format(" {}x{}", count, size);
    }
    out += '\n';
  }

  for (size_t n = 0; n < plan.perNode.size(); ++n) {
    const auto& nodePlan = plan.perNode[n];
    if (nodePlan.groups.empty() && nodePlan.ungrouped.empty()) {
      continue;
    }
    out += fmt::format(
        "  node {}: {} groups over {} values, {} ungrouped\n",
        n,
        nodePlan.groups.size(),
        nodePlan.groupOfValue.size(),
        nodePlan.ungrouped.size());
    for (const auto& group : nodePlan.groups) {
      if (group.concat != nullptr) {
        // How every operand actually reaches its band, which is the only
        // question that matters for a wide concat: 'chained' are the ones left
        // to the concat's own kernel, each waiting on the offset the one before
        // it advanced, and that is the number the standing rule caps at two.
        //
        // Not to be confused with what this line used to report, which was
        // 'crossing minus carved' -- a count of what the GROUP declined, saying
        // nothing about whether placement then gave any of them a copy.
        int32_t carved = 0;
        int32_t copied = 0;
        int32_t inPlace = 0;
        int32_t chained = 0;
        for (size_t i = 0; i < group.concat->operands.size(); ++i) {
          const auto& operand = group.concat->operands[i];
          if (operand.copyDestId >= 0) {
            ++copied;
          } else if (group.concat->memberOfOperand[i] >= 0) {
            ++carved;
          } else if (!operand.isSubgraphInput && !operand.isView) {
            ++inPlace;
          } else {
            ++chained;
          }
        }
        out += fmt::format(
            "    step {} -> concat %{} of {} operands{}: {} carved, {} copied, "
            "{} written in place, {} chained{}",
            group.allocStep,
            group.concat->resultId,
            group.concat->operands.size(),
            group.needsSync ? " (waits)" : "",
            carved,
            copied,
            inPlace,
            chained,
            chained > 2 ? " <-- OVER THE LIMIT:" : ":");
      } else {
        out += fmt::format(
            "    step {} -> node {} step {}{}:",
            group.allocStep,
            group.freeNode,
            group.freeStep,
            group.needsSync ? " (waits)" : "");
      }
      for (auto actualId : group.actualIds) {
        out += fmt::format(" %{}", actualId);
      }
      out += '\n';
    }
  }
  return out;
}

void installGraphAllocGroupPlans(WaveGraph& graph, const ValueTypes& types) {
  const auto& nodes = graph.nodes();
  auto plan = buildGraphAllocGroupPlan(
      collectNodeFootprints(nodes, graph.idToValue(), types));
  // Rendered now, printed later. The plan is settled while the graph compiles,
  // but the trace bits are typically set after that -- a caller that wants the
  // report turns tracing on around a run, by which time this has long returned.
  // Building the text unconditionally costs a few kilobytes once per graph.
  auto report = allocGroupPlanReport(plan);
  if (!WaveConfig::get().freeIntermediates) {
    // A group's buffer goes when the last of its slots leaves the frame, and
    // nothing leaves the frame with the freeing off. The grouping is still
    // correct, it just cannot pay for itself.
    report +=
        "  free_intermediates is off: no slot is ever released, so "
        "every group buffer lives to the end of the graph\n";
  }
  graph.setAllocGroupReport(std::move(report));
  for (auto id : plan.concatPlaced) {
    graph.addConcatPlaced(id);
  }
  for (size_t n = 0; n < nodes.size(); ++n) {
    if (auto* invocation = nodes[n]->kernels()) {
      invocation->setAllocGroupPlan(
          std::make_unique<AllocGroupPlan>(std::move(plan.perNode[n])));
    }
  }
}

namespace {

// One per thread: an invocation executes on one thread, and the collector is
// live only across the sizing of a single step of a single invocation. It
// cannot point to const -- recording a shape mutates the collector, which is
// the whole reason it is reachable from the sizing path.
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
thread_local AllocGroupCollector* tCollector = nullptr;

} // namespace

AllocGroupCollector* currentAllocCollector() {
  return tCollector;
}

AllocGroupCollector::AllocGroupCollector(
    const std::vector<const AllocGroup*>& groups) {
  TORCH_CHECK(
      tCollector == nullptr,
      "An allocation-group collector is already installed on this thread");
  groups_.reserve(groups.size());
  for (const auto* group : groups) {
    TORCH_CHECK(group != nullptr, "Null alloc group");
    TORCH_CHECK(
        group->actualIds.size() == group->dtypes.size(),
        "Alloc group has ",
        group->actualIds.size(),
        " values but ",
        group->dtypes.size(),
        " dtypes");
    Collected collected;
    collected.group = group;
    collected.requests.reserve(group->actualIds.size());
    for (size_t i = 0; i < group->actualIds.size(); ++i) {
      AllocRequest request;
      request.actualId = group->actualIds[i];
      request.dtype = group->dtypes[i];
      // A value produced by two groups would have one of them silently win
      // whichever way this is resolved, so it is rejected instead.
      TORCH_CHECK(
          slotOf_.count(request.actualId) == 0,
          "Value ",
          request.actualId,
          " belongs to more than one allocation group in the same step");
      slotOf_[request.actualId] =
          Slot{.group = groups_.size(), .index = collected.requests.size()};
      collected.requests.push_back(std::move(request));
    }
    collected.sized.assign(collected.requests.size(), false);
    groups_.push_back(std::move(collected));
  }
  tCollector = this;
}

AllocGroupCollector::AllocGroupCollector(const AllocGroup& group)
    : AllocGroupCollector(std::vector<const AllocGroup*>{&group}) {}

AllocGroupCollector::~AllocGroupCollector() {
  tCollector = nullptr;
}

bool AllocGroupCollector::capture(
    nativert::ValueId actualId,
    c10::IntArrayRef dims) {
  auto it = slotOf_.find(actualId);
  if (it == slotOf_.end()) {
    return false;
  }
  auto& collected = groups_[it->second.group];
  auto& request = collected.requests[it->second.index];
  // A member cannot be re-sized once its buffer exists: the slot has already
  // been handed to the frame at the old shape, and the ops that read it are
  // filled from that. This is the guard that would catch a group whose members
  // are not all sized in the same pass as each other.
  TORCH_CHECK(
      !collected.materialized,
      "Value ",
      actualId,
      " was sized again after its allocation group was materialized");
  request.dims.assign(dims.begin(), dims.end());
  collected.sized[it->second.index] = true;
  return true;
}

bool AllocGroupCollector::complete(size_t g) const {
  const auto& sized = groups_[g].sized;
  return std::all_of(
      sized.begin(), sized.end(), [](bool value) { return value; });
}

bool AllocGroupCollector::needsSync(size_t g) const {
  return groups_[g].group->needsSync;
}

std::vector<nativert::ValueId> AllocGroupCollector::missing(size_t g) const {
  std::vector<nativert::ValueId> result;
  const auto& collected = groups_[g];
  for (size_t i = 0; i < collected.sized.size(); ++i) {
    if (!collected.sized[i]) {
      result.push_back(collected.requests[i].actualId);
    }
  }
  return result;
}

std::vector<AllocRequest> AllocGroupCollector::sizedRequests(size_t g) const {
  const auto& collected = groups_[g];
  std::vector<AllocRequest> result;
  result.reserve(collected.requests.size());
  for (size_t i = 0; i < collected.requests.size(); ++i) {
    if (collected.sized[i]) {
      result.push_back(collected.requests[i]);
    }
  }
  return result;
}

AllocGroupBuffer materializeAllocGroup(
    const std::vector<AllocRequest>& requests,
    nativert::ExecutionFrame& frame) {
  auto group = allocateAllocGroup(requests);
  for (size_t i = 0; i < requests.size(); ++i) {
    frame.setIValue(requests[i].actualId, group.slots[i]);
  }
  return group;
}

const ConcatGroupLayout* AllocGroupCollector::concatLayout(size_t g) const {
  return groups_[g].group->concat.get();
}

namespace {

// The extent of an operand the group does not carve, measured against the
// frame. Every such operand was either produced by an earlier launch -- in
// which case its size expression is a plain "read this value's extent" -- or is
// produced by the concat's own kernel, in which case the expression is over
// values that already exist. graphConcatGroups refuses the concat otherwise.
std::vector<Dim> uncarvedOperandShape(
    const ConcatInputInfo& operand,
    int8_t elementRank,
    nativert::ExecutionFrame& frame) {
  // A reserve function is how this operand's extent is computed everywhere
  // else, so it is how the group computes it too. Bound to the invocation when
  // the footprint was built; without that the call would read the frame through
  // the wrong values, so its absence is an error rather than a fallback.
  if (operand.reserveShape != nullptr) {
    TORCH_CHECK(
        operand.boundReserve != nullptr,
        "Concat operand %",
        operand.valueId,
        " has a reserve function that was never bound to an invocation");
    return operand.boundReserve(frame);
  }
  auto shape = operand.sizeExpr.op == SizeShortcut::kNone
      ? std::vector<Dim>{}
      : operand.sizeExpr.dims(&frame);
  if (shape.empty()) {
    // As in the concat's own reserve: a zero-length operand arrives as an
    // undefined tensor and contributes nothing.
    return std::vector<Dim>(elementRank, 0);
  }
  if (static_cast<int8_t>(shape.size()) < elementRank) {
    // A broadcast size expression drops leading 1-dims; restore them.
    shape.insert(shape.begin(), elementRank - shape.size(), 1);
  }
  return shape;
}

} // namespace

AllocGroupBuffer materializeConcatGroup(
    const ConcatGroupLayout& layout,
    const std::vector<AllocRequest>& requests,
    const std::vector<bool>& sized,
    nativert::ExecutionFrame& frame) {
  const ConcatSpec spec{
      .isStack = layout.isStack, .dim = layout.dim, .outRank = layout.outRank};
  const auto elementRank = spec.elementRank();

  std::vector<std::vector<Dim>> operandShapes;
  operandShapes.reserve(layout.operands.size());
  std::vector<bool> carve(layout.operands.size(), false);
  for (size_t i = 0; i < layout.operands.size(); ++i) {
    const auto member = layout.memberOfOperand[i];
    if (member >= 0 && sized.at(member)) {
      const auto& dims = requests.at(member).dims;
      operandShapes.emplace_back(dims.begin(), dims.end());
      carve[i] = true;
    } else {
      operandShapes.push_back(
          uncarvedOperandShape(layout.operands[i], elementRank, frame));
    }
    TORCH_CHECK(
        static_cast<int8_t>(operandShapes.back().size()) == elementRank,
        "Concat operand %",
        layout.operands[i].valueId,
        " measured as rank ",
        operandShapes.back().size(),
        ", expected rank ",
        static_cast<int>(elementRank));
  }

  const auto outShape = concatResultShape(spec, operandShapes);
  const std::vector<int64_t> outSizes(outShape.begin(), outShape.end());
  if (WaveConfig::get().trace & WaveConfig::kTiming) {
    // What the group actually laid out, against what it was told to. An
    // operand measured as empty here that is not empty when the concat runs is
    // how a group placed too early corrupts the result: its band is a
    // zero-extent slice, and whatever fills it writes past the end.
    // How much of the result its operands write in place, against how much a
    // copy has to move. The operand COUNT does not answer this -- the carved
    // and copied sets are wildly different sizes on the ROO graph -- so tally
    // the elements each side actually accounts for.
    // Three ways an operand's bytes reach the result, and they have to be
    // counted apart: the group carves it a band, the concat's own kernel writes
    // it in place, or a copy moves it. Only the last is traffic. Counting
    // in-place as copied -- which lumping the two non-carved cases together
    // does -- reports a graph that copies nothing as copying everything.
    int64_t carvedElements = 0;
    int64_t inPlaceElements = 0;
    int64_t copiedElements = 0;
    for (size_t i = 0; i < layout.operands.size(); ++i) {
      int64_t elements = 1;
      for (const auto dim : operandShapes[i]) {
        elements *= static_cast<int64_t>(dim);
      }
      if (carve[i]) {
        carvedElements += elements;
      } else if (layout.operands[i].copyDestId >= 0) {
        copiedElements += elements;
      } else {
        inPlaceElements += elements;
      }
    }
    const auto bytesPer = static_cast<int64_t>(c10::elementSize(layout.dtype));
    std::cout << "  concat %" << layout.resultId << " landed: carved "
              << carvedElements * bytesPer << " B, inplace "
              << inPlaceElements * bytesPer << " B, copied "
              << copiedElements * bytesPer << " B" << std::endl;
    std::vector<nativert::ValueId> measuredEmpty;
    for (size_t i = 0; i < layout.operands.size(); ++i) {
      const auto& shape = operandShapes[i];
      if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
        measuredEmpty.push_back(layout.operands[i].valueId);
      }
    }
    std::cout << "  concat group %" << layout.resultId << " laid out as "
              << fmt::format("{}", fmt::join(outSizes, "x")) << " from "
              << layout.operands.size() << " operands, " << measuredEmpty.size()
              << " measured empty";
    for (size_t i = 0; i < measuredEmpty.size() && i < 8; ++i) {
      std::cout << " %" << measuredEmpty[i];
    }
    std::cout << std::endl;
  }
  // A group whose layout is measurable no earlier than the concat's own launch
  // is carved after that launch has been sized, which is after the concat's own
  // reserve has allocated the result and handed a band of it to every operand
  // its kernel computes. Allocating a second result there would leave those
  // bands pointing into a buffer nothing reads. Keeping the one that is already
  // the right shape makes the two agree on which buffer the bands are in --
  // this one and the reserve carve the same regions of it, so whichever runs
  // second finds its own work already done.
  auto& existing = frame.getIValue(layout.resultId);
  at::Tensor result;
  if (existing.isTensor() && existing.toTensor().is_cuda() &&
      existing.toTensor().sizes() == c10::IntArrayRef(outSizes)) {
    result = existing.toTensor();
  } else {
    result = at::empty(
        outSizes, at::TensorOptions().dtype(layout.dtype).device(at::kCUDA));
  }

  AllocGroupBuffer buffer;
  buffer.base = result;
  buffer.totalBytes = result.numel() * result.element_size();
  buffer.slots.assign(requests.size(), at::Tensor());
  buffer.offsets.assign(requests.size(), 0);

  const auto joinStride = result.strides()[spec.dim];
  const auto elementSize = static_cast<int64_t>(c10::elementSize(layout.dtype));
  int64_t offset = 0;
  for (size_t i = 0; i < layout.operands.size(); ++i) {
    const int64_t extent =
        spec.isStack ? 1 : static_cast<int64_t>(operandShapes[i].at(spec.dim));
    // A cat operand spans 'extent' positions along the join axis from where the
    // ones before it ended; a stack operand occupies the single position 'i'.
    const int64_t start = spec.isStack ? static_cast<int64_t>(i) : offset;
    offset += extent;
    // Carved and copied are the two ways an operand reaches its band, and they
    // are exclusive: a carved operand's producer writes the band itself, so a
    // copy of it would have no destination to be given and would write to
    // whatever the frame slot happened to hold. Placement and the group decide
    // this separately and have to reach the same answer; when they do not, say
    // so here rather than at the illegal access it becomes on the device.
    TORCH_CHECK(
        !(carve[i] && layout.operands[i].copyDestId >= 0),
        "Concat operand %",
        layout.operands[i].valueId,
        " of result %",
        layout.resultId,
        " is both carved into the result and copied into it");
    if (!carve[i]) {
      // An operand an op of its own copies gets the band as that copy's
      // destination. It has to be bound HERE rather than at the concat's own
      // reserve: the group is materialized ahead of the operands, and the copy
      // runs before the concat, so binding it later would hand the copy a
      // buffer of its own and then drop what it wrote when the band replaced
      // it. The operand itself keeps its buffer -- it is what the copy reads.
      if (layout.operands[i].copyDestId >= 0) {
        frame.setIValue(
            layout.operands[i].copyDestId,
            concatOperandView(result, spec, start, extent));
      }
      continue;
    }
    const auto member = layout.memberOfOperand[i];
    auto view = concatOperandView(result, spec, start, extent);
    buffer.offsets[member] = start * joinStride * elementSize;
    frame.setIValue(layout.operands[i].valueId, view);
    buffer.slots[member] = std::move(view);
  }

  // Written last: the concat's own reserve keeps a result that is already there
  // at the right shape, and reaching that state before every operand has been
  // placed would let a failure leave a result with nothing behind it.
  frame.setIValue(layout.resultId, std::move(result));
  return buffer;
}

bool concatAllocGroupEnabled() {
  return allocGroupEnabled() && WaveConfig::get().enableConcatAllocGroup;
}

bool allocGroupEnabled() {
  const auto& config = WaveConfig::get();
  // The plan is expressed in step indices of a grid that does not change once
  // execution starts. Both compiled grids satisfy that -- compilation settles
  // the multi-kernel one exactly as it settles the cooperative one -- so what
  // is required is that the choice between them is made, not which way. Left at
  // auto, the op could run either and a step index names nothing.
  //
  // The dynamic single-block switch (chooseGridVariant) is a remaining seam,
  // not one this mode introduced: an op small enough to want the single-block
  // variant switches to a grid with a different number of steps, and the plan's
  // indices are not rebuilt. The cooperative path has always run that way. It
  // is left alone here rather than pinned, because pinning the grid costs more
  // on the graphs that switch than the grouping saves.
  //
  // The freeing is deliberately not asked for. With it off a group's buffer is
  // never released, so it stays in the frame and the next run finds it there,
  // sizes it, resizes it and carves the views out of it exactly as it would
  // from a base allocated fresh. Folding lifetimes is worth less that way, but
  // nothing about the mode is unsound without it -- and requiring it made the
  // mode unusable at the one moment several of its decisions have to be taken:
  // the freeing is set after the graph is loaded, so a gate that reads it is
  // false while the graph compiles.
  return config.enableAllocGroup && config.isCg.has_value();
}

LaunchGrid& allocGroupGrid(OpInvocation& op) {
  const auto& config = WaveConfig::get();
  auto* projectOp = op.projectOp();
  // Forced single block is a whole-run choice, so the plan is built on that
  // grid and chooseGridVariant has nothing left to switch.
  if (config.useSingleBlock.has_value() && *config.useSingleBlock &&
      !projectOp->singleBlockGrid().empty()) {
    return projectOp->singleBlockGrid();
  }
  // Only where the op actually has a cooperative variant. Reading cgGrid()
  // unconditionally described nothing for an op without one, so no value got a
  // lifetime and the mode grouped nothing at all.
  if (config.isCg.has_value() && *config.isCg && !projectOp->cgGrid().empty()) {
    return projectOp->cgGrid();
  }
  return projectOp->grid();
}

} // namespace torch::wave
