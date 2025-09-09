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

#include "velox/connectors/hive/iceberg/PartitionSpec.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

const auto& transformTypeNames() {
  static const folly::F14FastMap<TransformType, std::string_view>
      kTransformNames = {
          {TransformType::kIdentity, "identity"},
          {TransformType::kHour, "hour"},
          {TransformType::kDay, "day"},
          {TransformType::kMonth, "month"},
          {TransformType::kYear, "year"},
          {TransformType::kBucket, "bucket"},
          {TransformType::kTruncate, "trunc"}};
  return kTransformNames;
}

} // namespace

VELOX_DEFINE_ENUM_NAME(TransformType, transformTypeNames);

} // namespace facebook::velox::connector::hive::iceberg
