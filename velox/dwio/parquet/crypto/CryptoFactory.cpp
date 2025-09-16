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
#include "velox/dwio/parquet/crypto/CryptoFactory.h"

namespace facebook::velox::parquet {

CryptoFactory& CryptoFactory::getInstance() {
  return getInstance(nullptr, false);
}

CryptoFactory& CryptoFactory::getInstance(
    std::shared_ptr<DecryptionKeyRetriever> kmsClient,
    bool clacEnabled) {
  static const auto instance = std::unique_ptr<CryptoFactory>(
      new CryptoFactory(kmsClient, clacEnabled));
  return *instance;
}

}
