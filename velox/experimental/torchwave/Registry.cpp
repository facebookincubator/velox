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

#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/Utils.h"

#include <c10/util/StringUtil.h>

namespace torch::wave {

std::unordered_map<std::string, Metadata>& Registry::registry() {
  static std::unordered_map<std::string, Metadata> map;
  return map;
}

bool Metadata::isStandalone(NodeCP node, const ValueTypes& types) const {
  if (isStandalone_) {
    return true;
  }
  if (only1d) {
    for (const auto& input : node->inputs()) {
      auto* value = input.value;
      if (value->type().kind() != nativert::Type::Kind::Tensor) {
        continue;
      }
      auto r = types.rank(value);
      if (r < 0 || r > 1) {
        return true;
      }
    }
  }
  if (isStandaloneFunc) {
    return isStandaloneFunc(node, types);
  }
  return false;
}

void Metadata::defaultInputMeta() {
  TORCH_CHECK(functionSchema, "defaultInputMeta requires functionSchema");
  if (argumentMeta.empty()) {
    argumentMeta.resize(functionSchema->arguments().size());
  }
}

void Metadata::defaultOutputMeta() {
  TORCH_CHECK(functionSchema, "defaultOutputMeta requires functionSchema");
  if (returnMeta.empty()) {
    returnMeta.resize(functionSchema->returns().size());
  }
}

void Registry::registerMetadata(std::string_view op, Metadata metadata) {
  if (metadata.functionSchema) {
    auto numArgs = metadata.functionSchema->arguments().size();
    auto numReturns = metadata.functionSchema->returns().size();
    TORCH_CHECK(
        metadata.argumentMeta.size() == numArgs,
        op,
        ": argumentMeta size ",
        metadata.argumentMeta.size(),
        " != schema argument count ",
        numArgs);
    TORCH_CHECK(
        metadata.returnMeta.size() == numReturns,
        op,
        ": returnMeta size ",
        metadata.returnMeta.size(),
        " != schema return count ",
        numReturns);
    for (auto idx : metadata.sizeArgs.ordinal) {
      TORCH_CHECK(
          idx >= 0 && idx < static_cast<int32_t>(numArgs),
          op,
          ": sizeArgs index ",
          idx,
          " out of range [0, ",
          numArgs,
          ")");
    }
    for (auto idx : metadata.typeTemplateParams) {
      TORCH_CHECK(
          idx >= 0 && idx < static_cast<int32_t>(numArgs),
          op,
          ": typeTemplateParams index ",
          idx,
          " out of range [0, ",
          numArgs,
          ")");
    }
    if (metadata.inputFromPreviousKernel.has_value()) {
      auto idx = metadata.inputFromPreviousKernel.value();
      TORCH_CHECK(
          idx >= 0 && idx < static_cast<int32_t>(numArgs),
          op,
          ": inputFromPreviousKernel index ",
          idx,
          " out of range [0, ",
          numArgs,
          ")");
    }
  }
  registry()[std::string(op)] = std::move(metadata);
}

const Metadata* Registry::metadata(std::string_view op) {
  auto& map = registry();
  auto it = map.find(std::string(op));
  if (it == map.end()) {
    return nullptr;
  }
  return &it->second;
}

Metadata Registry::unregister(std::string_view name) {
  auto& map = registry();
  auto it = map.find(std::string(name));
  TORCH_CHECK(it != map.end(), "Registry entry not found: ", name);
  auto metadata = std::move(it->second);
  map.erase(it);
  return metadata;
}

void Registry::restoreRegistry(std::string_view name, Metadata metadata) {
  registry()[std::string(name)] = std::move(metadata);
}

void Registry::registerElementwise(
    std::string_view qualifiedName,
    std::vector<std::string> attributeArgs) {
  auto atoms = c10::split(qualifiedName, '.');
  TORCH_CHECK(atoms.size() >= 3, "Invalid qualified op name: ", qualifiedName);
  auto opName = atoms[atoms.size() - 2];

  const auto* schema = findFunctionSchema(qualifiedName);
  TORCH_CHECK(schema, "FunctionSchema not found for: ", qualifiedName);

  Metadata md;

  md.functionSchema = schema;
  md.sizeArgs.ordinal = {0};
  md.inPlaceIfLastUse = true;
  md.argumentMeta.resize(
      schema->arguments().size(), ArgumentMeta{.isRegister = true});
  md.returnMeta = {ArgumentMeta{.isRegister = true}};
  md.elementwise = std::make_unique<ElementwiseOp>();
  md.elementwise->functionName = fmt::format("--{}", opName);
  md.elementwise->attributeArgs = std::move(attributeArgs);

  registerMetadata(qualifiedName, std::move(md));
}

void Registry::registerElementwiseOp(
    std::string_view qualifiedName,
    std::string_view elementwiseFuncName,
    bool isStandalone,
    std::vector<std::string> attributeArgs) {
  const auto* schema = findFunctionSchema(qualifiedName);
  TORCH_CHECK(schema, "FunctionSchema not found for: ", qualifiedName);

  Metadata md;

  md.functionSchema = schema;
  md.sizeArgs.ordinal = {0};
  md.inPlaceIfLastUse = true;
  md.isStandalone_ = isStandalone;
  md.argumentMeta.resize(
      schema->arguments().size(), ArgumentMeta{.isRegister = true});
  md.returnMeta = {ArgumentMeta{.isRegister = true}};
  md.elementwise = std::make_unique<ElementwiseOp>();
  md.elementwise->functionName = fmt::format("--{}", elementwiseFuncName);
  md.elementwise->attributeArgs = std::move(attributeArgs);

  registerMetadata(qualifiedName, std::move(md));
}

std::vector<std::unique_ptr<c10::FunctionSchema>>& Registry::schemaStorage() {
  static std::vector<std::unique_ptr<c10::FunctionSchema>> storage;
  return storage;
}

const c10::FunctionSchema* Registry::ownSchema(
    std::unique_ptr<c10::FunctionSchema> schema) {
  auto* ptr = schema.get();
  schemaStorage().push_back(std::move(schema));
  return ptr;
}

} // namespace torch::wave
