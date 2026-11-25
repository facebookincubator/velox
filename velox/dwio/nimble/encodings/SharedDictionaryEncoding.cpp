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

#include "folly/container/F14Map.h"
#include "velox/common/EnumDefine.h"

namespace facebook::nimble {

namespace {
const auto& sharedDictionaryScopeNames() {
  static const folly::F14FastMap<SharedDictionaryScope, std::string_view>
      kNames = {
          {SharedDictionaryScope::Stripe, "Stripe"},
          {SharedDictionaryScope::File, "File"},
          {SharedDictionaryScope::External, "External"},
      };
  return kNames;
}

// Shares the ownership of an encoded alphabet with a buffer view over it, so
// that strings pointing into the alphabet stay valid while the view lives.
class EncodedAlphabetReleaser {
 public:
  explicit EncodedAlphabetReleaser(std::shared_ptr<const void> owner)
      : owner_{std::move(owner)} {}

  void addRef() const {}

  void release() const {}

 private:
  // Keeps the encoded alphabet alive.
  const std::shared_ptr<const void> owner_;
};

} // namespace

VELOX_DEFINE_ENUM_NAME(SharedDictionaryScope, sharedDictionaryScopeNames)

SharedDictionaryAlphabet::SharedDictionaryAlphabet(
    std::string_view encoded,
    const Encoding::Options& options,
    velox::BufferPtr encodedAlphabetBuffer,
    std::shared_ptr<const void> encodedAlphabetOwner,
    velox::memory::MemoryPool* pool)
    : encodedAlphabetOwner_{std::move(encodedAlphabetOwner)},
      encodedAlphabetBuffer_{std::move(encodedAlphabetBuffer)},
      encodedAlphabet_{encoded},
      dataType_{EncodingPrefix::dataType(encoded)},
      encodingType_{EncodingPrefix::encodingType(encoded)},
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
      *pool, encoded, [this, pool](uint32_t size) -> void* {
        auto& entryBuffer = entryBuffers_.emplace_back(
            velox::AlignedBuffer::allocate<char>(size, pool));
        return entryBuffer->asMutable<void>();
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

  // The alphabet serves every read of the streams bound to it for as long as
  // the reader keeps it, so its view builds lookup structures once, such as
  // the offsets of string entries, instead of for each lookup. The input
  // string buffer lets the view pass the encoded bytes to the reads whose
  // strings point into them, and keeps those bytes alive after the alphabet is
  // gone. The alphabet keeps the buffer, which the options of its view point
  // to.
  auto inputStringBuffer = velox::BufferView<EncodedAlphabetReleaser>::create(
      reinterpret_cast<const uint8_t*>(encodedAlphabet.data()),
      encodedAlphabet.size(),
      EncodedAlphabetReleaser{encodedAlphabetOwner});
  const Encoding::Options options{
      .cacheEncodingView = true, .inputStringBuffer = inputStringBuffer.get()};
  auto alphabet =
      std::shared_ptr<SharedDictionaryAlphabet>{new SharedDictionaryAlphabet{
          encodedAlphabet,
          options,
          std::move(inputStringBuffer),
          std::move(encodedAlphabetOwner),
          pool}};
  NIMBLE_CHECK_GT(
      alphabet->entryCount(), 0, "Shared dictionary alphabet has no entries.");
  return alphabet;
}

void SharedDictionaryAlphabet::readStrings(
    std::span<const uint32_t> indices,
    const std::function<void(velox::BufferPtr)>& addStringBuffer,
    std::string_view* output) const {
  NIMBLE_CHECK_NOT_NULL(addStringBuffer);
  checkEntryType<std::string_view>();
  if (indices.empty()) {
    return;
  }
  if (entryView_ != nullptr) {
    entryView_->read(
        indices, /*setNull=*/[](uint32_t) {}, addStringBuffer, output);
    return;
  }
  materialize<std::string_view>(indices, output);
  for (const auto& entryBuffer : entryBuffers_) {
    // A zero-size request of the decoder leaves an empty buffer, which holds
    // no string bytes and must not be passed.
    if (entryBuffer->size() > 0) {
      addStringBuffer(entryBuffer);
    }
  }
}

} // namespace facebook::nimble
