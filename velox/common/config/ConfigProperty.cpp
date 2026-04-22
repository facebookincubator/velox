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
#include "velox/common/config/ConfigProperty.h"

#include "velox/common/EnumDefine.h"

namespace facebook::velox::config {

namespace {

const auto& configPropertyTypeNames() {
  static const folly::F14FastMap<ConfigPropertyType, std::string_view> kNames =
      {
          {ConfigPropertyType::kBoolean, "BOOLEAN"},
          {ConfigPropertyType::kInteger, "INTEGER"},
          {ConfigPropertyType::kDouble, "DOUBLE"},
          {ConfigPropertyType::kString, "STRING"},
      };
  return kNames;
}

} // namespace

VELOX_DEFINE_ENUM_NAME(ConfigPropertyType, configPropertyTypeNames);

} // namespace facebook::velox::config
