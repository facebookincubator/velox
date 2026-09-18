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

#pragma once

#include <algorithm>

#include "velox/dwio/nimble/velox/StreamData.h"

namespace facebook::nimble {
namespace detail {
inline bool shouldOmitDataStream(
    uint64_t dataVectorSize,
    uint64_t minChunkSize,
    bool allNulls,
    bool isFirstChunk) {
  if (dataVectorSize > minChunkSize) {
    return false;
  }
  // When all values are null, the values stream is omitted.
  return isFirstChunk || allNulls;
}

inline bool shouldOmitNullStream(
    const NullsStreamData& streamData,
    uint64_t minChunkSize,
    bool isFirstChunk) {
  // All-non-null rows can be omitted only if the whole stream is omitted.
  if (!streamData.hasNullValues()) {
    return isFirstChunk;
  }
  if (streamData.nonNulls().size() > minChunkSize) {
    return false;
  }
  return isFirstChunk || streamData.empty();
}
} // namespace detail

/**
 * Options for configuring StreamChunker behavior.
 */
struct StreamChunkerOptions {
  uint64_t minChunkSize{};
  uint64_t maxChunkSize{};
  bool ensureFullChunks{};
  bool isFirstChunk{};
};

/**
 * Breaks streams into manageable chunks based on size thresholds.
 * The `compact()` method should be called after use to reclaim memory from
 * processed StreamData values.
 */
class StreamChunker {
 public:
  StreamChunker() = default;

  /// Returns the next chunk of stream data.
  virtual std::optional<StreamDataView> next() = 0;

  /// Erases processed data to reclaim memory.
  virtual void compact() = 0;

  virtual ~StreamChunker() = default;

 protected:
  // Helper struct to hold result of nextChunkSize helper methods.
  struct ChunkSize {
    size_t dataElementCount{0};
    size_t nullElementCount{0};
    uint64_t rollingChunkSize{0};
    uint64_t extraMemory{0};
  };
};

std::unique_ptr<StreamChunker> getStreamChunker(
    StreamData& streamData,
    const StreamChunkerOptions& options);

template <typename T, typename StreamDataT = ContentStreamData<T>>
class ContentStreamChunker final : public StreamChunker {
 public:
  explicit ContentStreamChunker(
      StreamDataT& streamData,
      const StreamChunkerOptions& options)
      : streamData_{&streamData},
        minChunkSize_{options.minChunkSize},
        maxChunkSize_{options.maxChunkSize},
        ensureFullChunks_{options.ensureFullChunks},
        extraMemory_{streamData_->extraMemory()} {
    static_assert(
        std::is_same_v<StreamDataT, ContentStreamData<T>> ||
            std::is_same_v<StreamDataT, NullableContentStreamData<T>>,
        "StreamDataT must be either ContentStreamData<T> or NullableContentStreamData<T>");
    NIMBLE_DCHECK(
        !streamData_->hasNulls(),
        "Streams with nulls should be handled by NullableContentStreamData");
    NIMBLE_DCHECK_GE(
        maxChunkSize_,
        sizeof(T),
        "MaxChunkSize must be at least the size of a single data element.");
  }

  std::optional<StreamDataView> next() override {
    const auto& chunkSize = nextChunkSize();
    if (chunkSize.rollingChunkSize == 0) {
      return std::nullopt;
    }

    const std::string_view dataChunk = {
        reinterpret_cast<const char*>(
            streamData_->mutableData().data() + dataElementOffset_),
        chunkSize.dataElementCount * sizeof(T)};
    dataElementOffset_ += chunkSize.dataElementCount;
    extraMemory_ -= chunkSize.extraMemory;
    return StreamDataView{
        streamData_->descriptor(),
        dataChunk,
        static_cast<uint32_t>(chunkSize.dataElementCount)};
  }

 private:
  ChunkSize nextChunkSize() {
    const size_t maxChunkValuesCount = maxChunkSize_ / sizeof(T);
    const size_t remainingValuesCount =
        streamData_->mutableData().size() - dataElementOffset_;
    if ((ensureFullChunks_ && remainingValuesCount < maxChunkValuesCount) ||
        (remainingValuesCount < minChunkSize_ / sizeof(T))) {
      return ChunkSize{};
    }
    const size_t chunkValuesCount =
        std::min(maxChunkValuesCount, remainingValuesCount);
    return ChunkSize{
        .dataElementCount = chunkValuesCount,
        .rollingChunkSize = chunkValuesCount * sizeof(T)};
  }

  ChunkSize nextStringChunkSize() {
    const auto& data = streamData_->mutableData();
    size_t stringCount{0};
    uint64_t rollingChunkSize{0};
    uint64_t rollingExtraMemory{0};
    bool fullChunk{false};
    for (size_t i = dataElementOffset_; i < data.size(); ++i) {
      const auto& str = data[i];
      const uint64_t strSize = str.size() + sizeof(std::string_view);

      if (rollingChunkSize == 0 && strSize > maxChunkSize_) {
        // Allow a single oversized string as its own chunk.
        fullChunk = true;
        rollingExtraMemory += str.size();
        rollingChunkSize += strSize;
        ++stringCount;
        break;
      }

      if (rollingChunkSize + strSize > maxChunkSize_) {
        fullChunk = true;
        break;
      }

      rollingExtraMemory += str.size();
      rollingChunkSize += strSize;
      ++stringCount;
    }

    fullChunk = fullChunk || (rollingChunkSize == maxChunkSize_);
    if ((ensureFullChunks_ && !fullChunk) ||
        (rollingChunkSize < minChunkSize_)) {
      return ChunkSize{};
    }

    return ChunkSize{
        .dataElementCount = stringCount,
        .rollingChunkSize = rollingChunkSize,
        .extraMemory = rollingExtraMemory};
  }

  void compact() override {
    // No chunks consumed from stream data, we should not compact.
    if (dataElementOffset_ == 0) {
      return;
    }

    auto& currentData = streamData_->mutableData();
    const uint64_t remainingDataCount = currentData.size() - dataElementOffset_;

    // Move and clear existing buffer
    auto tempData = std::move(currentData);
    streamData_->reset();
    NIMBLE_DCHECK(
        streamData_->empty(), "StreamData should be empty after reset");

    auto& mutableData = streamData_->mutableData();
    mutableData.reserve(remainingDataCount);
    NIMBLE_DCHECK_GE(
        mutableData.capacity(),
        remainingDataCount,
        "Data buffer capacity should be at least new capacity");

    mutableData.resize(remainingDataCount);
    NIMBLE_DCHECK_EQ(
        mutableData.size(),
        remainingDataCount,
        "Data buffer size should be equal to remaining data count");

    std::copy_n(
        tempData.begin() + dataElementOffset_,
        remainingDataCount,
        mutableData.begin());
    dataElementOffset_ = 0;
    streamData_->extraMemory() = extraMemory_;

    // Ensure nulls capacity for nullable content streams without nulls.
    if (auto* nullableContentStreamData =
            dynamic_cast<NullableContentStreamData<T>*>(streamData_)) {
      nullableContentStreamData->ensureAdditionalNullsCapacity(
          /*mayHaveNulls=*/false, static_cast<uint32_t>(remainingDataCount));
    }
  }

  StreamDataT* const streamData_;
  const uint64_t minChunkSize_;
  const uint64_t maxChunkSize_;
  const bool ensureFullChunks_;

  size_t dataElementOffset_{0};
  size_t extraMemory_{0};
};

template <>
inline StreamChunker::ChunkSize ContentStreamChunker<
    std::string_view,
    ContentStreamData<std::string_view>>::nextChunkSize() {
  return nextStringChunkSize();
}

template <>
inline StreamChunker::ChunkSize ContentStreamChunker<
    std::string_view,
    NullableContentStreamData<std::string_view>>::nextChunkSize() {
  return nextStringChunkSize();
}

class NullsStreamChunker final : public StreamChunker {
 public:
  NullsStreamChunker(
      NullsStreamData& streamData,
      const StreamChunkerOptions& options)
      : streamData_{&streamData},
        minChunkSize_{options.minChunkSize},
        maxChunkSize_{options.maxChunkSize},
        omitStream_{detail::shouldOmitNullStream(
            streamData,
            options.minChunkSize,
            options.isFirstChunk)},
        ensureFullChunks_{options.ensureFullChunks} {
    static_assert(sizeof(bool) == 1);
    NIMBLE_DCHECK_GE(
        maxChunkSize_,
        1,
        "MaxChunkSize must be at least the size of a single element.");
  }

  std::optional<StreamDataView> next() override {
    if (omitStream_) {
      return std::nullopt;
    }

    const size_t remainingNonNulls = streamData_->rowCount() - nonNullsOffset_;
    const size_t nonNullsInChunk = std::min(maxChunkSize_, remainingNonNulls);
    if (nonNullsInChunk == 0 || nonNullsInChunk < minChunkSize_ ||
        (ensureFullChunks_ && nonNullsInChunk < maxChunkSize_)) {
      return std::nullopt;
    }

    // Null stream values are converted to boolean data for encoding.
    streamData_->materialize();
    auto& nonNulls = streamData_->mutableNonNulls();
    const auto* chunkBegin = nonNulls.data() + nonNullsOffset_;
    // Validity booleans are the payload (no bitmap): nulls are the false bits.
    const auto nullCount = static_cast<uint32_t>(
        std::count(chunkBegin, chunkBegin + nonNullsInChunk, false));
    std::string_view dataChunk = {
        reinterpret_cast<const char*>(chunkBegin), nonNullsInChunk};
    nonNullsOffset_ += nonNullsInChunk;
    return StreamDataView{
        streamData_->descriptor(),
        dataChunk,
        static_cast<uint32_t>(nonNullsInChunk),
        /*nonNulls=*/std::nullopt,
        nullCount};
  }

 private:
  void compact() override {
    // No chunks consumed from stream data, we should not compact.
    if (nonNullsOffset_ == 0) {
      return;
    }
    auto& nonNulls = streamData_->mutableNonNulls();
    const auto remainingNonNullsCount = nonNulls.size() - nonNullsOffset_;

    // Move and clear existing buffer
    auto tempNonNulls = std::move(nonNulls);
    const bool hasNulls = streamData_->hasNulls();
    streamData_->reset();
    NIMBLE_CHECK(
        streamData_->empty(), "StreamData should be empty after reset");

    auto& mutableNonNulls = streamData_->mutableNonNulls();
    mutableNonNulls.reserve(remainingNonNullsCount);
    NIMBLE_DCHECK_GE(
        mutableNonNulls.capacity(),
        remainingNonNullsCount,
        "NonNulls buffer capacity should be at least new capacity");

    streamData_->ensureAdditionalNullsCapacity(
        hasNulls, static_cast<uint32_t>(remainingNonNullsCount));
    if (hasNulls) {
      mutableNonNulls.resize(remainingNonNullsCount);
      NIMBLE_CHECK_EQ(
          mutableNonNulls.size(),
          remainingNonNullsCount,
          "NonNulls buffer size should be equal to remaining non-nulls count");

      std::copy_n(
          tempNonNulls.begin() + nonNullsOffset_,
          remainingNonNullsCount,
          mutableNonNulls.begin());
    }
    nonNullsOffset_ = 0;
  }

  NullsStreamData* const streamData_;
  const uint64_t minChunkSize_;
  const uint64_t maxChunkSize_;
  const bool omitStream_;
  const bool ensureFullChunks_;

  size_t nonNullsOffset_{0};
};

template <typename T>
class NullableContentStreamChunker final : public StreamChunker {
 public:
  explicit NullableContentStreamChunker(
      NullableContentStreamData<T>& streamData,
      const StreamChunkerOptions& options)
      : streamData_{&streamData},
        minChunkSize_{options.minChunkSize},
        maxChunkSize_{options.maxChunkSize},
        omitStream_{detail::shouldOmitDataStream(
            streamData.data().size(),
            options.minChunkSize,
            streamData.nonNulls().empty(),
            options.isFirstChunk)},
        ensureFullChunks_{options.ensureFullChunks},
        extraMemory_{streamData_->extraMemory()} {
    static_assert(sizeof(bool) == 1);
    NIMBLE_CHECK(
        streamData.hasNulls(),
        "ContentStreamChunker should be used when no nulls are present in stream.");
    NIMBLE_CHECK_GE(
        maxChunkSize_,
        sizeof(T) + 1,
        "MaxChunkSize must be at least the size of a single element.");

    streamData.materialize();
  }

  std::optional<StreamDataView> next() override {
    if (omitStream_) {
      return std::nullopt;
    }
    const auto& chunkSize = nextChunkSize();
    if (chunkSize.rollingChunkSize == 0) {
      return std::nullopt;
    }

    // Chunk content
    std::string_view dataChunk = {
        reinterpret_cast<const char*>(
            streamData_->mutableData().data() + dataElementOffset_),
        chunkSize.dataElementCount * sizeof(T)};

    // Chunk nulls
    std::span<const bool> nonNullsChunk(
        streamData_->mutableNonNulls().data() + nonNullsOffset_,
        chunkSize.nullElementCount);

    dataElementOffset_ += chunkSize.dataElementCount;
    nonNullsOffset_ += chunkSize.nullElementCount;
    extraMemory_ -= chunkSize.extraMemory;

    if (chunkSize.nullElementCount > chunkSize.dataElementCount) {
      return StreamDataView{
          streamData_->descriptor(),
          dataChunk,
          static_cast<uint32_t>(chunkSize.nullElementCount),
          nonNullsChunk,
          static_cast<uint32_t>(
              chunkSize.nullElementCount - chunkSize.dataElementCount)};
    }
    NIMBLE_CHECK_EQ(chunkSize.dataElementCount, chunkSize.nullElementCount);
    return StreamDataView{
        streamData_->descriptor(),
        dataChunk,
        static_cast<uint32_t>(chunkSize.dataElementCount)};
  }

 private:
  ChunkSize nextChunkSize() {
    const auto& nonNulls = streamData_->mutableNonNulls();
    const auto bufferedCount = nonNulls.size();

    ChunkSize chunkSize;
    bool fullChunk{false};
    // Calculate how many entries we can fit in the chunk
    for (size_t nonNullIdx = nonNullsOffset_; nonNullIdx < bufferedCount;
         ++nonNullIdx) {
      uint32_t elementSize = sizeof(bool); // Always account for null indicator
      uint32_t numDataElements{0};
      if (nonNulls[nonNullIdx]) {
        elementSize += sizeof(T); // Add data size if non-null
        ++numDataElements;
      }

      if (chunkSize.rollingChunkSize + elementSize > maxChunkSize_) {
        fullChunk = true;
        break;
      }

      chunkSize.dataElementCount += numDataElements;
      ++chunkSize.nullElementCount;
      chunkSize.rollingChunkSize += elementSize;
    }

    fullChunk = fullChunk || (chunkSize.rollingChunkSize == maxChunkSize_);
    if ((ensureFullChunks_ && !fullChunk) ||
        (chunkSize.rollingChunkSize < minChunkSize_)) {
      return ChunkSize{};
    }
    return chunkSize;
  }

  void compact() override {
    // No chunks consumed from stream data, we should not compact.
    if (nonNullsOffset_ == 0) {
      return;
    }
    auto& currentNonNulls = streamData_->mutableNonNulls();
    auto& currentData = streamData_->mutableData();
    const auto remainingNonNullsCount =
        currentNonNulls.size() - nonNullsOffset_;
    const auto remainingDataCount = currentData.size() - dataElementOffset_;

    // Move and clear existing buffers
    auto tempNonNulls = std::move(currentNonNulls);
    auto tempData = std::move(currentData);
    const bool hasNulls = streamData_->hasNulls();
    streamData_->reset();
    NIMBLE_CHECK(
        streamData_->empty(), "StreamData should be empty after reset");

    auto& mutableData = streamData_->mutableData();
    mutableData.reserve(remainingDataCount);
    NIMBLE_DCHECK_GE(
        mutableData.capacity(),
        remainingDataCount,
        "Data buffer capacity should be at least new capacity");

    mutableData.resize(remainingDataCount);
    NIMBLE_CHECK_EQ(
        mutableData.size(),
        remainingDataCount,
        "Data buffer size should be equal to remaining data count");

    std::copy_n(
        tempData.begin() + dataElementOffset_,
        remainingDataCount,
        mutableData.begin());

    auto& mutableNonNulls = streamData_->mutableNonNulls();
    mutableNonNulls.reserve(remainingNonNullsCount);
    NIMBLE_DCHECK_GE(
        mutableNonNulls.capacity(),
        remainingNonNullsCount,
        "NonNulls buffer capacity should be at least new capacity");

    streamData_->ensureAdditionalNullsCapacity(
        hasNulls, static_cast<uint32_t>(remainingNonNullsCount));
    if (hasNulls) {
      mutableNonNulls.resize(remainingNonNullsCount);
      NIMBLE_CHECK_EQ(
          mutableNonNulls.size(),
          remainingNonNullsCount,
          "NonNulls buffer size should be equal to remaining non-nulls count");

      std::copy_n(
          tempNonNulls.begin() + nonNullsOffset_,
          remainingNonNullsCount,
          mutableNonNulls.begin());
    }
    streamData_->extraMemory() = extraMemory_;
    dataElementOffset_ = 0;
    nonNullsOffset_ = 0;
  }

  NullableContentStreamData<T>* const streamData_;
  const uint64_t minChunkSize_;
  const uint64_t maxChunkSize_;
  const bool omitStream_;
  const bool ensureFullChunks_;

  size_t dataElementOffset_{0};
  size_t nonNullsOffset_{0};
  size_t extraMemory_{0};
};

template <>
inline StreamChunker::ChunkSize
NullableContentStreamChunker<std::string_view>::nextChunkSize() {
  const auto& data = streamData_->mutableData();
  const auto& nonNulls = streamData_->mutableNonNulls();
  const auto bufferedCount = nonNulls.size();
  ChunkSize chunkSize;
  bool fullChunk{false};
  // Calculate how many entries we can fit in the chunk
  for (size_t nonNullsIndex = nonNullsOffset_; nonNullsIndex < bufferedCount;
       ++nonNullsIndex) {
    uint64_t currentElementTotalSize{sizeof(bool)};
    size_t currentElementExtraMemory{0};
    uint32_t currentElementDataCount{0};
    if (nonNulls[nonNullsIndex]) {
      const auto& currentString =
          data[dataElementOffset_ + chunkSize.dataElementCount];
      currentElementTotalSize +=
          currentString.size() + sizeof(std::string_view);
      currentElementExtraMemory = currentString.size();
      ++currentElementDataCount;
    }

    if (chunkSize.rollingChunkSize == 0 &&
        currentElementTotalSize > maxChunkSize_) {
      // Allow a single oversized string as its own chunk.
      fullChunk = true;
      chunkSize.dataElementCount += currentElementDataCount;
      ++chunkSize.nullElementCount;
      chunkSize.rollingChunkSize += currentElementTotalSize;
      chunkSize.extraMemory += currentElementExtraMemory;
      break;
    }

    if (chunkSize.rollingChunkSize + currentElementTotalSize > maxChunkSize_) {
      fullChunk = true;
      break;
    }

    chunkSize.dataElementCount += currentElementDataCount;
    ++chunkSize.nullElementCount;
    chunkSize.rollingChunkSize += currentElementTotalSize;
    chunkSize.extraMemory += currentElementExtraMemory;
  }

  fullChunk = fullChunk || (chunkSize.rollingChunkSize == maxChunkSize_);
  if ((ensureFullChunks_ && !fullChunk) ||
      (chunkSize.rollingChunkSize < minChunkSize_)) {
    chunkSize = ChunkSize{};
  }
  return chunkSize;
}

class NullableContentStringStreamChunker final : public StreamChunker {
 public:
  explicit NullableContentStringStreamChunker(
      NullableContentStringStreamData& streamData,
      const StreamChunkerOptions& options)
      : streamData_{&streamData},
        minChunkSize_{options.minChunkSize},
        maxChunkSize_{options.maxChunkSize},
        omitStream_{detail::shouldOmitDataStream(
            streamData.bufferSize(),
            options.minChunkSize,
            streamData.nonNulls().empty(),
            options.isFirstChunk)},
        ensureFullChunks_{options.ensureFullChunks} {
    static_assert(sizeof(bool) == 1);
    streamData_->materializeNulls();
  }

  std::optional<StreamDataView> next() override {
    if (omitStream_) {
      return std::nullopt;
    }
    const auto& chunkSize = nextChunkSize();
    if (chunkSize.rollingChunkSize == 0) {
      return std::nullopt;
    }

    // Content
    auto& output = streamData_->mutableStringViews();
    output.resize(0);
    output.reserve(chunkSize.dataElementCount);
    auto mutableData = streamData_->mutableData();
    auto& mutableLengths = mutableData.lengths;
    auto& mutableBuffer = mutableData.buffer;
    auto currentBufferOffset = bufferOffset_;
    for (size_t i = 0; i < chunkSize.dataElementCount; ++i) {
      const auto currentLength = mutableLengths[lengthsOffset_ + i];
      output.emplace_back(
          std::string_view{
              mutableBuffer.data() + currentBufferOffset, currentLength});
      currentBufferOffset += currentLength;
    }
    std::string_view dataChunk = {
        reinterpret_cast<const char*>(output.data()),
        chunkSize.dataElementCount * sizeof(std::string_view)};

    // Nulls
    std::span<const bool> nonNullsChunk(
        streamData_->mutableNonNulls().data() + nonNullsOffset_,
        chunkSize.nullElementCount);

    lengthsOffset_ += chunkSize.dataElementCount;
    nonNullsOffset_ += chunkSize.nullElementCount;
    bufferOffset_ += chunkSize.extraMemory;

    if (chunkSize.nullElementCount > chunkSize.dataElementCount) {
      return StreamDataView{
          streamData_->descriptor(),
          dataChunk,
          static_cast<uint32_t>(chunkSize.nullElementCount),
          nonNullsChunk,
          static_cast<uint32_t>(
              chunkSize.nullElementCount - chunkSize.dataElementCount)};
    }
    NIMBLE_DCHECK_EQ(chunkSize.dataElementCount, chunkSize.nullElementCount);
    return StreamDataView{
        streamData_->descriptor(),
        dataChunk,
        static_cast<uint32_t>(chunkSize.dataElementCount)};
  }

 private:
  ChunkSize nextChunkSize() {
    const auto& stringLengths = streamData_->mutableData().lengths;
    const auto& nonNulls = streamData_->mutableNonNulls();
    ChunkSize chunkSize;
    bool fullChunk{false};
    // Calculate how many entries we can fit in the chunk
    for (size_t idx = nonNullsOffset_; idx < nonNulls.size(); ++idx) {
      uint64_t currentTotalSize{sizeof(bool)};
      uint32_t currentDataCount{0};
      size_t currentExtraMemory{0};
      if (nonNulls[idx]) {
        currentExtraMemory =
            stringLengths[lengthsOffset_ + chunkSize.dataElementCount];
        // NOTE: This is not a bug, we use sizeof(std::string_view) instead of
        // sizeof(uint64_t) to prevent reader regressions with the current
        // default writer settings.
        currentTotalSize += currentExtraMemory + sizeof(std::string_view);
        ++currentDataCount;
      }

      if (chunkSize.rollingChunkSize == 0 && currentTotalSize > maxChunkSize_) {
        // Allow a single oversized string as its own chunk.
        fullChunk = true;
        chunkSize.extraMemory += currentExtraMemory;
        chunkSize.dataElementCount += currentDataCount;
        chunkSize.rollingChunkSize += currentTotalSize;
        ++chunkSize.nullElementCount;
        break;
      }

      if (chunkSize.rollingChunkSize + currentTotalSize > maxChunkSize_) {
        fullChunk = true;
        break;
      }

      chunkSize.extraMemory += currentExtraMemory;
      chunkSize.dataElementCount += currentDataCount;
      chunkSize.rollingChunkSize += currentTotalSize;
      ++chunkSize.nullElementCount;
    }

    fullChunk = fullChunk || (chunkSize.rollingChunkSize == maxChunkSize_);
    if ((ensureFullChunks_ && !fullChunk) ||
        (chunkSize.rollingChunkSize < minChunkSize_)) {
      chunkSize = ChunkSize{};
    }
    return chunkSize;
  }

  void compact() override {
    // Clear existing outputvector before beginning compaction.
    streamData_->mutableStringViews().clear();

    // No chunks consumed from stream data, we should not compact.
    if (nonNullsOffset_ == 0) {
      return;
    }

    const bool hasNulls = streamData_->hasNulls();
    // Move and clear existing buffers
    auto tempBuffer = std::move(streamData_->mutableData().buffer);
    auto tempLengths = std::move(streamData_->mutableData().lengths);
    auto tempNonNulls = std::move(streamData_->mutableNonNulls());
    streamData_->reset();
    NIMBLE_DCHECK(
        streamData_->empty(), "StreamData should be empty after reset");

    {
      const auto remainingDataCount = tempLengths.size() - lengthsOffset_;
      auto& mutableDataLength = streamData_->mutableData().lengths;
      mutableDataLength.resize(remainingDataCount);
      NIMBLE_DCHECK_EQ(
          mutableDataLength.size(),
          remainingDataCount,
          "Data length size should be equal to remaining data count");

      std::copy_n(
          tempLengths.begin() + lengthsOffset_,
          remainingDataCount,
          mutableDataLength.begin());
    }

    {
      const auto remainingDataBytes = tempBuffer.size() - bufferOffset_;
      auto& mutableDataBuffer = streamData_->mutableData().buffer;
      mutableDataBuffer.resize(remainingDataBytes);
      NIMBLE_DCHECK_EQ(
          mutableDataBuffer.size(),
          remainingDataBytes,
          "Data buffer size should be equal to remaining data bytes");

      std::copy_n(
          tempBuffer.begin() + bufferOffset_,
          remainingDataBytes,
          mutableDataBuffer.begin());
    }

    {
      auto& mutableNonNulls = streamData_->mutableNonNulls();
      const auto remainingNonNullsCount = tempNonNulls.size() - nonNullsOffset_;
      streamData_->ensureAdditionalNullsCapacity(
          hasNulls, static_cast<uint32_t>(remainingNonNullsCount));
      if (hasNulls) {
        mutableNonNulls.resize(remainingNonNullsCount);
        NIMBLE_DCHECK_EQ(
            mutableNonNulls.size(),
            remainingNonNullsCount,
            "NonNulls buffer size should be equal to remaining non-nulls count");

        std::copy_n(
            tempNonNulls.begin() + nonNullsOffset_,
            remainingNonNullsCount,
            mutableNonNulls.begin());
      }
    }
    lengthsOffset_ = 0;
    nonNullsOffset_ = 0;
    bufferOffset_ = 0;
  }

  NullableContentStringStreamData* const streamData_;
  const uint64_t minChunkSize_;
  const uint64_t maxChunkSize_;
  const bool omitStream_;
  const bool ensureFullChunks_;

  size_t lengthsOffset_{0};
  size_t nonNullsOffset_{0};
  uint64_t bufferOffset_{0};
};
} // namespace facebook::nimble
