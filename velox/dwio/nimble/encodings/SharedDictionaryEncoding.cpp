/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"

namespace facebook::nimble {

SharedDictionaryAlphabet::SharedDictionaryAlphabet(
    std::string_view encoded,
    const Encoding::Options& options,
    std::shared_ptr<const void> encodedAlphabetOwner,
    velox::memory::MemoryPool* pool)
    : encodedAlphabetOwner_{std::move(encodedAlphabetOwner)},
      dataType_{EncodingPrefix::dataType(encoded)},
      encodingType_{EncodingPrefix::encodingType(encoded)},
      entryPayload_{*velox::checkedNotNull(pool)},
      entries_{nullptr},
      entryView_{
          supportsEncodingView(encodingType_)
              ? createEncodingView(encoded, pool, options)
              : nullptr} {
  if (entryView_ != nullptr) {
    entryCount_ = entryView_->rowCount();
    return;
  }

  // No view for this encoding, so decode every entry once and serve lookups
  // from the decoded buffer instead.
  auto encoding = EncodingFactory{options}.create(
      *pool, encoded, [this](uint32_t size) -> void* {
        return entryPayload_.reserve(size);
      });
  NIMBLE_CHECK_EQ(
      encoding->dataType(),
      dataType_,
      "Shared dictionary alphabet has an inconsistent data type.");
  entryCount_ = encoding->rowCount();
  if (entryCount_ == 0) {
    return;
  }
  decodeEntries(*encoding, pool);
}

std::shared_ptr<const SharedDictionaryAlphabet>
SharedDictionaryAlphabet::create(
    std::string_view encodedAlphabet,
    std::shared_ptr<const void> encodedAlphabetOwner,
    velox::memory::MemoryPool* pool) {
  NIMBLE_CHECK_NOT_NULL(pool);
  NIMBLE_CHECK_NOT_NULL(encodedAlphabetOwner);
  NIMBLE_CHECK_FILE(
      !encodedAlphabet.empty(), "Shared dictionary alphabet is empty.");
  auto alphabet =
      std::shared_ptr<SharedDictionaryAlphabet>{new SharedDictionaryAlphabet{
          encodedAlphabet,
          Encoding::Options{},
          std::move(encodedAlphabetOwner),
          pool}};
  NIMBLE_CHECK_GT(
      alphabet->entryCount(), 0, "Shared dictionary alphabet has no entries.");
  return alphabet;
}

} // namespace facebook::nimble
