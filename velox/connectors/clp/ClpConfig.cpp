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

#include <boost/algorithm/string.hpp>

#include "velox/connectors/clp/ClpConfig.h"

namespace facebook::velox::connector::clp {

namespace {

ClpConfig::StorageType stringToStorageType(const std::string& strValue) {
  auto upperValue = boost::algorithm::to_upper_copy(strValue);
  if (upperValue == "FS") {
    return ClpConfig::StorageType::kFs;
  }
  if (upperValue == "S3") {
    return ClpConfig::StorageType::kS3;
  }
  VELOX_UNSUPPORTED("Unsupported storage type: {}.", strValue);
}

} // namespace

ClpConfig::StorageType ClpConfig::storageType() const {
  return stringToStorageType(config_->get<std::string>(kStorageType, "FS"));
}

} // namespace facebook::velox::connector::clp
