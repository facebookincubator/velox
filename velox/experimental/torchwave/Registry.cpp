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

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/StringUtil.h>

namespace torch::wave {

std::unordered_map<std::string, Metadata>& Registry::registry() {
  static std::unordered_map<std::string, Metadata> map;
  return map;
}

void Registry::registerMetadata(std::string_view op, Metadata metadata) {
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

void Registry::registerElementwise(
    std::string_view qualifiedName,
    std::vector<std::string> attributeArgs) {
  auto atoms = c10::split(qualifiedName, '.');
  TORCH_CHECK(atoms.size() >= 3, "Invalid qualified op name: ", qualifiedName);

  auto numAtoms = atoms.size();
  auto ns = atoms[numAtoms - 3];
  auto opName = atoms[numAtoms - 2];
  auto overloadName = atoms[numAtoms - 1];

  auto operatorName = fmt::format("{}::{}", ns, opName);
  std::string normalizedOverload =
      (overloadName == "default") ? "" : std::string(overloadName);

  auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(
      operatorName.c_str(), normalizedOverload.c_str());
  const auto& schema = handle.schema();

  Metadata md;
  md.kind = Metadata::kMetadata;
  md.functionSchema = &schema;
  md.sizeArgs = {0};
  md.inPlaceIfLastUse = true;
  md.argumentMeta.resize(
      schema.arguments().size(), ArgumentMeta{.isRegister = true});
  md.returnMeta = {ArgumentMeta{.isRegister = true}};
  md.elementWise = std::make_unique<ElementwiseOp>();
  md.elementWise->functionName = fmt::format("--{}", opName);
  md.elementWise->attributeArgs = std::move(attributeArgs);

  registerMetadata(qualifiedName, std::move(md));
}

} // namespace torch::wave
