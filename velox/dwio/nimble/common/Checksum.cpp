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
#include "velox/dwio/nimble/common/Checksum.h"
#include "velox/dwio/nimble/common/Exceptions.h"

#define XXH_INLINE_ALL
#include <xxhash.h>

namespace facebook::nimble {

namespace {
class Xxh3_64Checksum : public Checksum {
 public:
  Xxh3_64Checksum() : state_{XXH3_createState()} {
    NIMBLE_DCHECK_NOT_NULL(state_, "Failed to initialize Xxh3_64Checksum.");
    reset();
  }

  ~Xxh3_64Checksum() override {
    XXH3_freeState(state_);
  }

  void reset() override {
    const auto result = XXH3_64bits_reset(state_);
    NIMBLE_CHECK(result != XXH_ERROR, "XXH3_64bits_reset error.");
  }

  void update(std::string_view data) override {
    const auto result = XXH3_64bits_update(state_, data.data(), data.size());
    NIMBLE_CHECK(result != XXH_ERROR, "XXH3_64bits_update error.");
  }

  uint64_t getChecksum64(bool reset) override {
    auto ret = static_cast<uint64_t>(XXH3_64bits_digest(state_));
    if (UNLIKELY(reset)) {
      this->reset();
    }
    return ret;
  }

  uint32_t getChecksum32(bool reset) override {
    return narrow(getChecksum64(reset));
  }

  uint64_t computeChecksum64(std::string_view data) override {
    // The one-shot entry point skips the streaming state machine entirely and
    // reaches XXH3's specialized small-input paths. Reset so accumulated state
    // cannot leak into a later getChecksum64().
    reset();
    return static_cast<uint64_t>(XXH3_64bits(data.data(), data.size()));
  }

  uint32_t computeChecksum32(std::string_view data) override {
    return narrow(computeChecksum64(data));
  }

  ChecksumType getType() const override {
    return ChecksumType::XXH3_64;
  }

 private:
  // Folds the high half into the low half, so the result depends on all 64
  // bits even for a digest whose low half is the weaker one.
  //
  // This value is persisted in stripe-group metadata. Changing it invalidates
  // the checksums in every file already written.
  static uint32_t narrow(uint64_t checksum) {
    return static_cast<uint32_t>(checksum ^ (checksum >> 32));
  }

  XXH3_state_t* const state_;
};
} // namespace

std::unique_ptr<Checksum> ChecksumFactory::create(ChecksumType type) {
  switch (type) {
    case ChecksumType::XXH3_64:
      return std::make_unique<Xxh3_64Checksum>();
    default:
      NIMBLE_UNSUPPORTED("Unsupported checksum type: {}", toString(type));
  }
}

} // namespace facebook::nimble
