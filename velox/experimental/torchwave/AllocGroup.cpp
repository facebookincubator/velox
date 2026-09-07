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
bool operandMeasurableAt(
    const ConcatInputInfo& operand,
    const ProducedMap& produced,
    ExecPoint point) {
  // A view is measurable even though it can never be carved: its extent is the
  // host's to compute like any other operand's, and it is only its lack of
  // storage of its own that keeps it out of the group's members. Refusing it
  // here would instead sink the whole concat, since one view operand would make
  // the layout look uncomputable at every point.
  if (operand.hasShapeOnDevice) {
    return false;
  }
  if (operand.isSubgraphInput) {
    const auto it = produced.find(operand.valueId);
    return it == produced.end() || !(point < it->second.at);
  }
  if (operand.reserveShape != nullptr || operand.hasReserveInChain ||
      operand.sizeExpr.op == SizeShortcut::kNone) {
    return false;
  }
  std::vector<nativert::ValueId> reads;
  sizeExprValues(operand.sizeExpr, reads);
  for (auto readId : reads) {
    const auto it = produced.find(readId);
    if (it == produced.end()) {
      continue;
    }
    // Measured twice -- once to lay the result out and again by the concat's
    // own reserve -- so the expression has to give the same answer both times.
    // A value written at more than one point can change in between, and the two
    // would disagree about where every operand after it starts.
    if (point < it->second.at || it->second.numWrites != 1) {
      return false;
    }
  }
  return true;
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
  footprint.operands.reserve(layout.inputs.size());
  for (const auto& input : layout.inputs) {
    auto operand = input;
    auto it = bindings.find(input.valueId);
    operand.valueId = it != bindings.end() ? it->second : input.valueId;
    operand.sizeExpr = input.sizeExpr.toActual(bindings, idToValue);
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

  for (const auto& node : nodes) {
    for (const auto& launch : node.launches) {
      const ExecPoint concatPoint{node.node, launch.step};
      for (const auto& concat : launch.concats) {
        if (concat.operands.size() <= 2) {
          count(&AllocGroupStats::numConcatTooFew);
          continue;
        }
        // The result has to be this launch's to place: one written at two
        // points, or already spoken for by another concat, is not.
        const auto resultIt = produced.find(concat.resultId);
        if (resultIt == produced.end() || resultIt->second.numWrites != 1 ||
            claimed.count(concat.resultId) != 0) {
          count(&AllocGroupStats::numConcatUnplaceableOperand);
          continue;
        }

        bool onDevice = false;
        for (const auto& operand : concat.operands) {
          if (operand.hasShapeOnDevice) {
            onDevice = true;
          }
        }
        if (onDevice) {
          count(&AllocGroupStats::numConcatOnDevice);
          continue;
        }

        // The group is made as early as the layout can be computed: the first
        // point at which every operand's extent is known. Everything produced
        // from there on writes its band directly; only what was already
        // materialized before it has to be copied in. Placing it any later
        // would turn operands that could have written in place into copies.
        ExecPoint at{-1, -1};
        for (const auto& point : executionPoints) {
          if (concatPoint < point) {
            break;
          }
          bool allMeasurable = true;
          for (const auto& operand : concat.operands) {
            if (!operandMeasurableAt(operand, produced, point)) {
              allMeasurable = false;
              break;
            }
          }
          if (allMeasurable) {
            at = point;
            break;
          }
        }
        if (at.first < 0) {
          count(&AllocGroupStats::numConcatUnplaceableOperand);
          continue;
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
        for (size_t i = 0; i < concat.operands.size(); ++i) {
          const auto& operand = concat.operands[i];
          // A view aliases somebody else's buffer, so there is no write of its
          // own to redirect into the region; it is copied in instead.
          if (operand.isView) {
            continue;
          }
          // Joining anywhere but the outermost axis makes the operand's region
          // a pitched band. Only a producer that maps its writes through the
          // output's strides can fill one; the rest keep a dense buffer of
          // their own and the concat copies it in.
          if (concat.dim != 0 && !operand.mayWriteStrided) {
            count(&AllocGroupStats::numConcatStridedBand);
            continue;
          }
          // cat([x, y, x]): one buffer cannot be two regions of the result.
          if (occurrences[operand.valueId] != 1) {
            continue;
          }
          const auto it = produced.find(operand.valueId);
          if (it == produced.end() || it->second.numWrites != 1) {
            continue;
          }
          // A view cannot value-convert, so an operand the concat would promote
          // has to keep its own buffer and be copied in.
          if (it->second.dtype != concat.dtype) {
            continue;
          }
          if (claimed.count(operand.valueId) != 0) {
            continue;
          }
          layout->memberOfOperand[i] =
              static_cast<int32_t>(group.actualIds.size());
          group.actualIds.push_back(operand.valueId);
          group.dtypes.push_back(concat.dtype);
          group.needsSync = group.needsSync || it->second.needsSync;
        }
        // A group with nothing to carve is still worth forming: it owns the
        // result's backing store and lays every operand's region out on the
        // host, which is what the operands are filled through. There is no
        // serial path to fall back to for a concat this wide, so refusing here
        // would leave it with nothing at all.
        if (group.actualIds.empty()) {
          count(&AllocGroupStats::numConcatNoMembers);
        }

        claimed.insert(concat.resultId);
        for (auto memberId : group.actualIds) {
          claimed.insert(memberId);
        }
        if (stats != nullptr) {
          ++stats->numConcatGroups;
          stats->numConcatMembers +=
              static_cast<int32_t>(group.actualIds.size());
        }
        group.concat = std::move(layout);
        groups.push_back(std::move(group));
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
    // The grid the mode will actually run, which is the cooperative one only
    // where the op has it. Walking cgGrid() unconditionally described nothing
    // for an op without a cooperative variant, so no value got a lifetime and
    // the mode grouped nothing at all.
    auto& cg = op.projectOp()->cgGrid();
    auto& grid = cg.empty() ? op.projectOp()->grid() : cg;
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
  auto& cg = op.projectOp()->cgGrid();
  return static_cast<int32_t>(
      cg.empty() ? op.projectOp()->grid().size() : cg.size());
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
      stats.numConcatNoMembers + stats.numConcatStridedBand;
  if (concatsSeen > 0) {
    out += fmt::format(
        "  concat groups: {} of {} fused concats place their result ahead of "
        "{} operands, each one allocation and one copy fewer\n"
        "    not placed: {} of two operands or fewer, {} joined off the "
        "outermost axis, {} with an operand the device sizes, {} with an "
        "operand the host cannot measure early, {} with no operand left to "
        "redirect\n",
        stats.numConcatGroups,
        concatsSeen,
        stats.numConcatMembers,
        stats.numConcatTooFew,
        stats.numConcatStridedBand,
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
        // What the placement actually removed. An operand the concat's own
        // kernel computes was already writing into its region of the result
        // (reserveConcatOutput hands it a view), so it never cost a copy; only
        // one that crosses the kernel boundary does, and of those only the
        // carved ones stop being copied. The remainder is what a wider
        // placement would still have to go after.
        int32_t crossing = 0;
        for (const auto& operand : group.concat->operands) {
          crossing += operand.isSubgraphInput ? 1 : 0;
        }
        const auto placed = static_cast<int32_t>(group.actualIds.size());
        out += fmt::format(
            "    step {} -> concat %{} of {} operands{} ({} fused in place, {} "
            "crossing: {} placed, {} still copied):",
            group.allocStep,
            group.concat->resultId,
            group.concat->operands.size(),
            group.needsSync ? " (waits)" : "",
            static_cast<int32_t>(group.concat->operands.size()) - crossing,
            crossing,
            placed,
            crossing - placed);
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
  if (WaveConfig::get().trace & WaveConfig::kTiming) {
    std::cout << allocGroupPlanReport(plan);
    if (!WaveConfig::get().freeIntermediates) {
      // A group's buffer goes when the last of its slots leaves the frame, and
      // nothing leaves the frame with the freeing off. The grouping is still
      // correct, it just cannot pay for itself.
      std::cout << "  free_intermediates is off: no slot is ever released, so "
                   "every group buffer lives to the end of the graph\n";
    }
  }
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
  TORCH_CHECK(
      operand.reserveShape == nullptr,
      "Concat operand %",
      operand.valueId,
      " is measured by a reserve function, which the concat allocation group "
      "cannot run ahead of the launch that owns it");
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
  auto result = at::empty(
      outSizes, at::TensorOptions().dtype(layout.dtype).device(at::kCUDA));

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
    if (!carve[i]) {
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
  // execution starts, which is only true of the cooperative grid.
  //
  // And a group's buffer goes when the last of its slots leaves the frame, so
  // with the freeing off nothing ever leaves and every group lives to the end
  // of the graph. The mode would still fold the allocator calls together, but
  // folding lifetimes into one buffer is the point, and there are no lifetimes
  // to fold.
  return config.enableAllocGroup && config.isCg.has_value() && *config.isCg &&
      config.freeIntermediates;
}

} // namespace torch::wave
