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

#include "velox/experimental/torchwave/GraphPrep.h"

#include <glog/logging.h>
#include <string>
#include <vector>

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/Device.h>
#include <torch/nativert/graph/Graph.h>

#include "velox/experimental/torchwave/Registry.h"

namespace torch::wave {

void stripDataAsserts(nativert::Graph& graph) {
  std::vector<nativert::Node*> toDrop;
  for (auto& node : graph.nodes()) {
    const auto& target = node.target();
    if (target == "torch.ops.aten._assert_async.msg" ||
        target == "torch.ops.aten._assert_scalar.default") {
      toDrop.push_back(&node);
    }
  }
  if (toDrop.empty()) {
    return;
  }
  for (auto* node : toDrop) {
    node->destroy();
  }
  // Removing the asserts orphans their symbolic feeder chains (e.g.
  // _operator.or_, _operator.eq). Generic nativert has no kernels for those
  // pure _operator.* / sym_* ops, so DCE any now-userless one. Iterative, since
  // removing a node can orphan its inputs. Graph outputs stay safe: the graph's
  // output node counts as a user, so a live output never looks dead.
  size_t deadRemoved = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<nativert::Node*> dead;
    for (auto& node : graph.nodes()) {
      const auto target = node.target();
      // Only DCE the pure logical/bitwise _operator ops that generic nativert
      // has no SymIntOpKernel for; these only feed the stripped torch._check
      // asserts. Leave every other _operator.* (add/eq/...) in place even when
      // it becomes dead: nativert can build those, and its SymIntOpKernels
      // resolve inputs by symbolic name through a map, so removing a
      // symbol-defining node (that looks edge-userless) makes a surviving
      // kernel's name lookup throw std::out_of_range.
      if (target != "_operator.or_" && target != "_operator.and_" &&
          target != "_operator.xor_") {
        continue;
      }
      bool anyUser = false;
      for (const auto* out : node.outputs()) {
        if (!out->users().empty()) {
          anyUser = true;
          break;
        }
      }
      if (!anyUser) {
        dead.push_back(&node);
      }
    }
    for (auto* node : dead) {
      node->destroy();
      ++deadRemoved;
      changed = true;
    }
  }
  graph.renumberValues();
  graph.finalize();
  graph.lint();
  LOG(INFO) << "stripDataAsserts: removed " << toDrop.size()
            << " assert node(s) and " << deadRemoved << " dead sym feeder(s)";
}

void normalizeSymOpArgNames(nativert::Graph& graph) {
  size_t renamed = 0;
  for (auto& node : graph.nodes()) {
    const auto target = node.target();
    // Only the getSymInputs-based kernels expect "a"/"b": every _operator.*
    // scalar/sym op plus torch.sym_max/min/float. NOT sym_size / sym_numel,
    // which use self/dim.
    const bool needsAB = target.rfind("_operator.", 0) == 0 ||
        target == "torch.sym_max" || target == "torch.sym_min" ||
        target == "torch.sym_float";
    if (!needsAB) {
      continue;
    }
    // A constant operand is carried as an attribute named "a" or "b"; only the
    // symbolic operands are inputs. Assign inputs to the a/b slots an attribute
    // does not already occupy, in order. Blindly renaming input[0] -> "a" would
    // corrupt an op whose constant is operand a, e.g. _operator.sub(%x, a=64)
    // (64 - %x): its input is operand b, and overwriting it with "a" collides
    // with the constant and leaves the op with no "b".
    auto& ins = node.inputs();
    const bool slotTaken[2] = {
        node.tryGetAttribute("a") != nullptr,
        node.tryGetAttribute("b") != nullptr};
    size_t slot = 0;
    for (size_t i = 0; i < ins.size() && slot < 2; ++i) {
      while (slot < 2 && slotTaken[slot]) {
        ++slot;
      }
      if (slot >= 2) {
        break;
      }
      const char* want = (slot == 0) ? "a" : "b";
      if (ins.at(i).name != want) {
        ins.at(i).name = want;
        ++renamed;
      }
      ++slot;
    }
  }
  if (renamed > 0) {
    LOG(INFO) << "normalizeSymOpArgNames: renamed " << renamed
              << " sym-op input name(s) to a/b";
  }
}

int32_t rewriteGpuIncompatibleOps(nativert::Graph& graph) {
  // fb::simple_1d_concat has no CUDA implementation -- its CUDA registration is
  // a throwing Dummy
  // (caffe2/torch/fb/sparsenn/cpu_operators/simple_concat.cpp). It is a plain
  // 1-D concat, identical to aten.cat(tensors, dim=0), which does have a CUDA
  // kernel. Mirror wave's rewrite (MoreBuiltins.cpp) so the generic nativert
  // executor can run it on GPU. (The other fb ops in this model -- sigrid_hash,
  // grouped_masked_select_jagged_1d, batch_flip_and_truncate_sparse,
  // group_length_guard_sparse, fused_datafm_merge_and_dedup_by_reference --
  // have real CUDA kernels in sparsenn_operators_gpu and self-register.)
  int32_t rewritten = 0;
  for (auto& node : graph.nodes()) {
    if (node.target() == "torch.ops.fb.simple_1d_concat.default") {
      node.setTarget("torch.ops.aten.cat.default");
      // nativert matches node inputs to schema args by name; rename the
      // TensorList input "inputs" -> "tensors" to match aten::cat's schema.
      if (!node.inputs().empty()) {
        node.inputs()[0].name = "tensors";
      }
      node.addAttribute({"dim", static_cast<int64_t>(0)});
      ++rewritten;
    }
  }
  LOG(INFO) << "rewriteGpuIncompatibleOps: rewrote " << rewritten
            << " fb.simple_1d_concat -> aten.cat.default";
  return rewritten;
}

int32_t insertCpuOnlyCopies(nativert::Graph& graph) {
  // Collect insertion sites first; don't mutate the node list while iterating.
  struct Site {
    nativert::Node* consumer;
    size_t inputIdx;
    nativert::Value* deviceValue;
  };
  std::vector<Site> sites;
  for (auto& node : graph.nodes()) {
    const auto* meta = Registry::metadata(node.target());
    if (!meta) {
      continue;
    }
    auto& inputs = node.inputs();
    for (size_t i = 0; i < inputs.size() && i < meta->argumentMeta.size();
         ++i) {
      // cpuOnly is only set on tensor args (e.g. tensor_split indices), so a
      // non-null value here is the tensor we must keep on host.
      if (meta->argumentMeta.at(i).cpuOnly && inputs.at(i).value) {
        sites.push_back({&node, i, inputs.at(i).value});
      }
    }
  }

  // For each cpuOnly tensor arg, insert aten._to_copy(self, device=cpu) right
  // before the consumer and repoint only that edge. tensor_split (the only
  // cpuOnly op) reads its indices on the host and returns views of self, so
  // self and the outputs stay on GPU -- no move-back is needed. Lets the
  // generic nativert executor run the graph on GPU (wave handles cpuOnly args
  // itself at runtime; see Launch in CompiledOp.cpp).
  for (const auto& site : sites) {
    auto* copyNode = graph.createNode(
        "torch.ops.aten._to_copy.default", {{"self", site.deviceValue}});
    copyNode->addAttribute({"dtype", torch::nativert::None{}});
    copyNode->addAttribute({"layout", torch::nativert::None{}});
    copyNode->addAttribute({"device", c10::Device(c10::kCPU)});
    copyNode->addAttribute({"pin_memory", torch::nativert::None{}});
    copyNode->addAttribute({"non_blocking", false});
    copyNode->addAttribute({"memory_format", torch::nativert::None{}});
    auto* cpuValue = copyNode->addOutput(
        graph.getUniqueValueName(), site.deviceValue->type());
    graph.insertBefore(copyNode, site.consumer);
    site.consumer->inputs()[site.inputIdx].value = cpuValue;
    site.deviceValue->eraseUser(site.consumer);
    cpuValue->addUser(site.consumer);
  }
  LOG(INFO) << "insertCpuOnlyCopies: inserted " << sites.size()
            << " _to_copy(device=cpu) node(s)";
  return static_cast<int32_t>(sites.size());
}

int32_t rewriteMergeAndDedupToTw(nativert::Graph& graph) {
  // (base node target, fused _tw node target, _tw dispatcher op name). The _tw
  // ops are the fused TorchWave CUDA kernels registered by
  // registerTorchWaveMergeAndDedup; only retarget a node when its _tw op is
  // actually in the dispatcher, so this is a no-op for the base engine (where
  // the sparsenn op runs as a standalone) and only fuses in a build that linked
  // and registered the kernels.
  struct Entry {
    const char* from;
    const char* to;
    const char* op;
  };
  static const Entry kEntries[] = {
      {"torch.ops.fb.fused_datafm_merge_and_dedup_by_reference.default",
       "torch.ops.fb.fused_datafm_merge_and_dedup_by_reference_tw.default",
       "fb::fused_datafm_merge_and_dedup_by_reference_tw"},
      {"torch.ops.fb.fused_datafm_merge_and_dedup_by_reference_optimized.default",
       "torch.ops.fb.fused_datafm_merge_and_dedup_by_reference_optimized_tw.default",
       "fb::fused_datafm_merge_and_dedup_by_reference_optimized_tw"},
  };
  int32_t rewritten = 0;
  for (auto& node : graph.nodes()) {
    for (const auto& e : kEntries) {
      if (node.target() == e.from &&
          c10::Dispatcher::singleton()
              .findOp(c10::OperatorName(e.op, ""))
              .has_value()) {
        node.setTarget(e.to);
        ++rewritten;
      }
    }
  }
  LOG(INFO) << "rewriteMergeAndDedupToTw: rewrote " << rewritten
            << " merge-and-dedup node(s) to fused _tw ops";
  return rewritten;
}

} // namespace torch::wave
