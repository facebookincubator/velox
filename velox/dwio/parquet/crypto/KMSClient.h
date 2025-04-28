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
#include "velox/dwio/parquet/crypto/Cache.h"
#include "velox/dwio/parquet/crypto/EncryptionKey.h"
#include "velox/dwio/parquet/crypto/KeyRetriever.h"

namespace facebook::velox::parquet {

constexpr int kCacheCleanupInternalSeconds = 60;

// This KMS client is a reference implementation. You may inherit it and
// implement the decryptKey function to call your KMS to decrypt the key or
// create a new KMS Client by implementation the interface
// DecryptionKeyRetriever directly.
class KMSClient : public DecryptionKeyRetriever {
 public:
  std::string getKey(const std::string& keyMetadata, const std::string& doAs)
      override;

 protected:
  virtual std::shared_ptr<EncryptedKeyVersion> parseKeyMetadata(
      const std::string& keyMetadata);
  virtual KeyVersion decryptKey(
      std::shared_ptr<EncryptedKeyVersion>& encryptedKeyVersion,
      const std::string& doAs) = 0;

 private:
  Cache cache_{kCacheCleanupInternalSeconds};
  Cache exceptionCache_{kCacheCleanupInternalSeconds};
};

} // namespace facebook::velox::parquet
