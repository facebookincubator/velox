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
#include "FileDecryptionProperties.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::parquet {

FileDecryptionProperties::Builder*
FileDecryptionProperties::Builder::keyRetriever(
    const std::shared_ptr<DecryptionKeyRetriever>& keyRetriever) {
  if (keyRetriever == nullptr)
    return this;

  keyRetriever_ = keyRetriever;
  return this;
}

FileDecryptionProperties::Builder* FileDecryptionProperties::Builder::aadPrefix(
    const std::string& aadPrefix) {
  if (aadPrefix.empty()) {
    return this;
  }
  aadPrefix_ = aadPrefix;
  return this;
}

FileDecryptionProperties::Builder*
FileDecryptionProperties::Builder::aadPrefixVerifier(
    std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier) {
  if (aadPrefixVerifier == nullptr)
    return this;

  aadPrefixVerifier_ = std::move(aadPrefixVerifier);
  return this;
}

FileDecryptionProperties::FileDecryptionProperties(
    std::shared_ptr<DecryptionKeyRetriever> keyRetriever,
    bool checkPlaintextFooterIntegrity,
    const std::string& aadPrefix,
    std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier,
    bool plaintextFilesAllowed) {
  DCHECK(nullptr != keyRetriever);
  aadPrefixVerifier_ = std::move(aadPrefixVerifier);
  checkPlaintextFooterIntegrity_ = checkPlaintextFooterIntegrity;
  keyRetriever_ = std::move(keyRetriever);
  aadPrefix_ = aadPrefix;
  plaintextFilesAllowed_ = plaintextFilesAllowed;
}

} // namespace facebook::velox::parquet
