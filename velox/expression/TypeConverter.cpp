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

#include "velox/expression/TypeConverter.h"
#include <boost/algorithm/string.hpp>

namespace facebook::velox::exec {
TypeSignature toTypeSignature(const TypePtr& type) {
  std::vector<TypeSignature> children;
  if (type->size()) {
    children.reserve(type->size());
    for (auto i = 0; i < type->size(); i++) {
      children.emplace_back(toTypeSignature(type->childAt(i)));
    }
  }
  const std::string& typeName = type->name();
  return TypeSignature(
      boost::algorithm::to_lower_copy(typeName), std::move(children));
}

} // namespace facebook::velox::exec
