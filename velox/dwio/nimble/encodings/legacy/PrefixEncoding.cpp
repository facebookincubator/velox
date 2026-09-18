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
#include "velox/dwio/nimble/encodings/legacy/PrefixEncoding.h"

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

namespace facebook::nimble::legacy {

PrefixEncoding::PrefixEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> /* stringBufferFactory */)
    : TypedEncoding<std::string_view, std::string_view>{pool, data},
      restartInterval_{readRestartInterval(data, dataOffset())},
      numRestarts_{computeNumRestarts(rowCount_, restartInterval_)},
      restartOffsets_{restartOffsets(data, dataOffset())},
      dataStart_{dataStart(data, dataOffset(), numRestarts_)},
      decodedValue_{pool_},
      materializedValues_{pool_} {
  reset();
}

// static
uint32_t PrefixEncoding::readRestartInterval(
    std::string_view data,
    uint32_t startOffset) {
  const auto* pos = data.data() + startOffset;
  return encoding::readUint32(pos);
}

// static
uint32_t PrefixEncoding::computeNumRestarts(
    uint32_t rowCount,
    uint32_t restartInterval) {
  return velox::bits::divRoundUp(rowCount, restartInterval);
}

// static
const char* PrefixEncoding::restartOffsets(
    std::string_view data,
    uint32_t startOffset) {
  return data.data() + startOffset + 4;
}

// static
const char* PrefixEncoding::dataStart(
    std::string_view data,
    uint32_t startOffset,
    uint32_t numRestarts) {
  return data.data() + startOffset + 4 + (numRestarts * sizeof(uint32_t));
}

void PrefixEncoding::reset() {
  currentPos_ = dataStart_;
  currentRow_ = 0;
  decodedValue_.clear();
}

void PrefixEncoding::skip(uint32_t rowCount) {
  NIMBLE_CHECK_LE(currentRow_ + rowCount, rowCount_, "Invalid skip row count");

  if (rowCount == 0) {
    return;
  }

  const uint32_t targetRow = currentRow_ + rowCount;

  const uint32_t currentRestartIndex = currentRow_ / restartInterval_;
  const uint32_t targetRestartIndex = targetRow / restartInterval_;
  NIMBLE_CHECK_LE(currentRestartIndex, targetRestartIndex);

  if (targetRestartIndex > currentRestartIndex &&
      targetRestartIndex < numRestarts_) {
    seekToRestartPoint(targetRestartIndex);
  }

  while (currentRow_ < targetRow) {
    decodeEntry();
  }
}

std::string_view PrefixEncoding::decodeEntry() {
  const uint32_t sharedPrefixLen = encoding::readUint32(currentPos_);
  const uint32_t suffixLen = encoding::readUint32(currentPos_);

  const uint32_t fullLen = sharedPrefixLen + suffixLen;
  NIMBLE_CHECK_LE(
      sharedPrefixLen, decodedValue_.size(), "Invalid shared prefix length");
  decodedValue_.resize(fullLen);

  if (suffixLen > 0) {
    std::memcpy(decodedValue_.data() + sharedPrefixLen, currentPos_, suffixLen);
    currentPos_ += suffixLen;
  }
  ++currentRow_;
  return std::string_view(decodedValue_.data(), fullLen);
}

void PrefixEncoding::materialize(uint32_t rowCount, void* buffer) {
  NIMBLE_CHECK_LE(currentRow_ + rowCount, rowCount_, "Invalid row count");

  materializedValues_.clear();

  Vector<std::pair<uint32_t, uint32_t>> valueOffsets{pool_};
  valueOffsets.reserve(rowCount);

  for (uint32_t i = 0; i < rowCount; ++i) {
    const std::string_view decodedValue = decodeEntry();
    const uint32_t valueOffset = materializedValues_.size();
    const uint32_t valueLength = decodedValue.size();
    materializedValues_.insert(
        materializedValues_.end(),
        decodedValue.data(),
        decodedValue.data() + valueLength);
    valueOffsets.push_back({valueOffset, valueLength});
  }

  std::string_view* valueOutput = static_cast<std::string_view*>(buffer);
  for (uint32_t i = 0; i < rowCount; ++i) {
    const auto [offset, len] = valueOffsets[i];
    valueOutput[i] = std::string_view(materializedValues_.data() + offset, len);
  }
}

uint32_t PrefixEncoding::restartOffset(uint32_t restartIndex) const {
  NIMBLE_CHECK_LT(restartIndex, numRestarts_, "Restart index out of bounds");
  const char* offsetPos = restartOffsets_ + (restartIndex * sizeof(uint32_t));
  return encoding::readUint32(offsetPos);
}

void PrefixEncoding::seekToRestartPoint(uint32_t restartIndex) {
  NIMBLE_CHECK_LT(restartIndex, numRestarts_, "Restart index out of bounds");

  currentPos_ = dataStart_ + restartOffset(restartIndex);
  currentRow_ = restartIndex * restartInterval_;
  decodedValue_.clear();
}

std::string PrefixEncoding::debugString(int offset) const {
  std::string log = Encoding::debugString(offset);
  log += fmt::format(
      "\n{}restart_interval={}, num_restarts={}",
      std::string(offset, ' '),
      restartInterval_,
      numRestarts_);
  return log;
}

} // namespace facebook::nimble::legacy
