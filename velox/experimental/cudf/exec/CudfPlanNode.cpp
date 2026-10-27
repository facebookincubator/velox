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
#include "velox/experimental/cudf/exec/CudfPlanNode.h"
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/exec/CudfScanUtils.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <algorithm>
#include <functional>
#include <mutex>
#include <typeinfo>
#include <unordered_set>

namespace facebook::velox::cudf_velox {
namespace {
using Descriptor = std::shared_ptr<const CudfPlanNodeDescriptor>;
std::mutex descriptorMutex;
std::vector<std::pair<std::string, Descriptor>> descriptors;

Descriptor findDescriptor(const core::PlanNode& node) {
  std::vector<std::pair<std::string, Descriptor>> snapshot;
  {
    std::lock_guard lock(descriptorMutex);
    snapshot = descriptors;
  }
  for (const auto& [name, descriptor] : snapshot) {
    if (descriptor->canHandle(node)) {
      return descriptor;
    }
  }
  return nullptr;
}

CudfPlanNodeTraits deviceTraits(size_t sources) {
  return {
      std::vector<InputRepresentation>(sources, InputRepresentation::kCudf),
      VectorRepresentation::kCudf,
      {}};
}
} // namespace

void registerCudfPlanNodeDescriptor(std::string name, Descriptor descriptor) {
  VELOX_USER_CHECK_NOT_NULL(descriptor);
  std::lock_guard lock(descriptorMutex);
  VELOX_USER_CHECK(
      std::none_of(
          descriptors.begin(),
          descriptors.end(),
          [&](const auto& entry) { return entry.first == name; }),
      "cuDF plan descriptor already registered: {}",
      name);
  descriptors.emplace_back(std::move(name), std::move(descriptor));
}

void unregisterCudfPlanNodeDescriptor(const std::string& name) {
  std::lock_guard lock(descriptorMutex);
  std::erase_if(
      descriptors, [&](const auto& entry) { return entry.first == name; });
}

namespace detail {
core::PlanNodePtr rebuildCudfLocalPartition(
    const core::LocalPartitionNode& node,
    std::vector<core::PlanNodePtr> sources) {
  VELOX_USER_CHECK(
      !node.scaleWriter(), "GPU local exchange cannot scale writers");
  auto rebuilt = core::LocalPartitionNode::Builder(node)
                     .sources(std::move(sources))
                     .build();
  return std::shared_ptr<const CudfLocalPartitionNode>(
      new CudfLocalPartitionNode(*rebuilt));
}

std::optional<CudfPlanNodeTraits> customCudfPlanNodeTraits(
    const core::PlanNodePtr& node,
    const core::QueryCtx& queryCtx) {
  if (auto* physical = dynamic_cast<const CudfPlanNode*>(node.get())) {
    return physical->cudfTraits();
  }
  if (dynamic_cast<const CudfLocalPartitionNode*>(node.get()) ||
      dynamic_cast<const CudfLocalMergeNode*>(node.get())) {
    return deviceTraits(node->sources().size());
  }
  if (auto descriptor = findDescriptor(*node)) {
    return descriptor->traits(*node, queryCtx);
  }
  return std::nullopt;
}

CudfPlanNodeTraits cudfPlanNodeTraits(
    const core::PlanNodePtr& node,
    const core::QueryCtx& queryCtx) {
  if (auto traits = customCudfPlanNodeTraits(node, queryCtx)) {
    return *traits;
  }
  if (auto* scan = dynamic_cast<const core::TableScanNode*>(node.get());
      scan && isGpuTableScan(*scan)) {
    return deviceTraits(0);
  }
  if (auto* exchange = dynamic_cast<const core::ExchangeNode*>(node.get());
      exchange && exchange->transportKind() == core::TransportKind::kUcx) {
    VELOX_USER_CHECK(
        !node->is<core::MergeExchangeNode>(),
        "UCX merge exchange requires an explicit Exchange and CudfOrderBy");
    return deviceTraits(0);
  }
  if (auto* output =
          dynamic_cast<const core::PartitionedOutputNode*>(node.get())) {
    return {
        {output->transportKind() == core::TransportKind::kUcx
             ? InputRepresentation::kCudf
             : InputRepresentation::kVelox},
        {},
        {}};
  }
  return {
      std::vector<InputRepresentation>(
          node->sources().size(), InputRepresentation::kVelox),
      VectorRepresentation::kVelox,
      {}};
}

core::PlanNodePtr rebuildCustomCudfPlanNode(
    const core::PlanNodePtr& node,
    std::vector<core::PlanNodePtr> sources) {
  if (sources == node->sources()) {
    return node;
  }
  core::PlanNodePtr rebuilt;
  if (auto* physical = dynamic_cast<const CudfPlanNode*>(node.get())) {
    rebuilt = physical->withSources(sources);
  } else if (
      auto* partition =
          dynamic_cast<const CudfLocalPartitionNode*>(node.get())) {
    rebuilt = partition->withSources(sources);
  } else if (
      auto* merge = dynamic_cast<const CudfLocalMergeNode*>(node.get())) {
    rebuilt = merge->withSources(sources);
  } else if (auto descriptor = findDescriptor(*node)) {
    rebuilt = descriptor->withSources(node, sources);
  }
  VELOX_USER_CHECK_NOT_NULL(
      rebuilt,
      "No rebuilding support for plan node {} ({})",
      node->name(),
      node->id());
  VELOX_USER_CHECK(
      rebuilt->id() == node->id() && typeid(*rebuilt) == typeid(*node),
      "Rebuilding {} ({}) must preserve its ID and dynamic type",
      node->name(),
      node->id());
  VELOX_USER_CHECK(
      rebuilt->sources() == sources,
      "Rebuilding {} ({}) did not install the rewritten sources",
      node->name(),
      node->id());
  VELOX_USER_CHECK(
      *rebuilt->outputType() == *node->outputType(),
      "Rebuilding {} ({}) must preserve its output schema",
      node->name(),
      node->id());
  return rebuilt;
}
} // namespace detail

void validateCudfPlan(
    const core::PlanNodePtr& plan,
    const core::QueryCtx& queryCtx,
    VectorRepresentation taskOutput) {
  VELOX_USER_CHECK_NOT_NULL(plan);
  using Output = std::optional<VectorRepresentation>;
  std::unordered_set<core::PlanNodeId> ids;
  std::function<Output(const core::PlanNodePtr&)> visit =
      [&](const core::PlanNodePtr& node) -> Output {
    VELOX_USER_CHECK(
        ids.insert(node->id()).second,
        "Duplicate plan node ID: {}",
        node->id());
    const auto traits = detail::cudfPlanNodeTraits(node, queryCtx);
    VELOX_USER_CHECK_EQ(
        traits.inputs.size(),
        node->sources().size(),
        "Incorrect input contract for {} ({})",
        node->name(),
        node->id());
    VELOX_USER_CHECK(
        !(traits.output && traits.outputSource),
        "Output contract must be fixed or source-preserving, not both");
    std::vector<Output> outputs;
    for (size_t i = 0; i < traits.inputs.size(); ++i) {
      auto output = visit(node->sources()[i]);
      VELOX_USER_CHECK(
          output.has_value(), "A terminal sink cannot feed another node");
      const auto need = traits.inputs[i];
      VELOX_USER_CHECK(
          need == InputRepresentation::kEither ||
              (need == InputRepresentation::kCudf) ==
                  (*output == VectorRepresentation::kCudf),
          "Vector representation mismatch at source {} of {} ({})",
          i,
          node->name(),
          node->id());
      outputs.push_back(output);
    }
    if (dynamic_cast<const CudfFromVeloxNode*>(node.get()) ||
        dynamic_cast<const CudfToVeloxNode*>(node.get())) {
      VELOX_USER_CHECK(
          isTypeSupportedByCudf(node->sources()[0]->outputType()),
          "Unsupported cuDF conversion type at {}: {}",
          node->id(),
          node->sources()[0]->outputType()->toString());
    }
    if (traits.outputSource) {
      VELOX_USER_CHECK_LT(*traits.outputSource, outputs.size());
      return outputs[*traits.outputSource];
    }
    return traits.output;
  };
  const auto output = visit(plan);
  VELOX_USER_CHECK(
      !output || *output == taskOutput,
      "Plan output does not match the requested task representation");
}
} // namespace facebook::velox::cudf_velox
