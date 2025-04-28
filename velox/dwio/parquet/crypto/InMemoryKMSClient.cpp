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
#include "velox/dwio/parquet/crypto/InMemoryKMSClient.h"
#include "velox/dwio/parquet/crypto/Exception.h"

namespace facebook::velox::parquet {

std::string InMemoryKMSClient::getKey(
    const std::string& keyMetadata,
    const std::string& doAs) {
  auto it = keyMap_.find(keyMetadata);
  if (it != keyMap_.end()) {
    return it->second;
  }
  throw CryptoException("[CLAC] http status code 403");
}

void InMemoryKMSClient::putKey(
    const std::string& keyMetadata,
    const std::string& key) {
  keyMap_[keyMetadata] = key;
}

} // namespace facebook::velox::parquet
