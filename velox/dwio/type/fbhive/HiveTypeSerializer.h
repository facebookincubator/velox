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

#pragma once

#include <memory>
#include <string>

#include "velox/type/Type.h"

namespace facebook {
namespace velox {
namespace dwio {
namespace type {
namespace fbhive {

class HiveTypeSerializer {
 public:
  static std::string serialize(const std::shared_ptr<const velox::Type>& type);

 private:
  std::string visit(const velox::Type& type) const;

 private:
  HiveTypeSerializer() = default;

  ~HiveTypeSerializer() = default;

  std::string visitChildren(const velox::RowType& t) const;

  std::string visitChildren(const velox::Type& t) const;
};

} // namespace fbhive
} // namespace type
} // namespace dwio
} // namespace velox
} // namespace facebook
