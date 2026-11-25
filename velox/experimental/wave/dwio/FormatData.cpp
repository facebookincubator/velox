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

#include "velox/experimental/wave/dwio/FormatData.h"
#include "velox/experimental/wave/dwio/ColumnReader.h"
#include "velox/experimental/wave/exec/Wave.h"

DECLARE_int32(wave_reader_rows_per_tb);

DEFINE_int32(
    staging_bytes_per_thread,
    300000,
    "Make a parallel memcpy shard per this many bytes");

namespace facebook::velox::wave {

BufferId SplitStaging::add(Staging& staging) {
  VELOX_CHECK_NULL(deviceBuffer_);
  staging_.push_back(staging);
  offsets_.push_back(fill_);
  fill_ += bits::roundUp(staging.size, 8);
  return offsets_.size() - 1;
}

void SplitStaging::registerPointerInternal(
    BufferId id,
    void** ptr,
    bool clear) {
  VELOX_CHECK_NOT_NULL(ptr);
  VELOX_CHECK_NULL(deviceBuffer_);
  if (clear) {
    *ptr = nullptr;
  }
#if 0 // ndef NDEBUG
  for (auto& pair : patch_) {
    VELOX_CHECK(pair.second != ptr, "Must not register the same pointer twice");
  }
#endif
  patch_.push_back(std::make_pair(id, ptr));
}

void SplitStaging::copyColumns(
    int32_t begin,
    int32_t end,
    char* destination,
    bool release) {
  for (auto i = begin; i < end; ++i) {
    memcpy(destination + offsets_[i], staging_[i].hostData, staging_[i].size);
  }
  if (release) {
    sem_.release();
  }
}

// Shared pool of 1-2GB of pinned host memory for staging. May
// transiently exceed 2GB but settles to 2GB after the peak.
GpuArena& getTransferArena() {
  static std::unique_ptr<GpuArena> arena = std::make_unique<GpuArena>(
      1UL << 30, getHostAllocator(nullptr), 2UL << 30);
  return *arena;
}

// Starts the transfers registered with add(). 'stream' is set to a stream
// where operations depending on the transfer may be queued.
void SplitStaging::transfer(
    WaveStream& waveStream,
    Stream& stream,
    bool recordEvent,
    std::function<void(WaveStream&, Stream&)> asyncTail) {
  if (fill_ == 0 || deviceBuffer_ != nullptr) {
    if (recordEvent && !event_) {
      event_ = std::make_unique<Event>();
      event_->record(stream);
    }
    if (asyncTail) {
      asyncTail(waveStream, stream);
    }
    return;
  }
  WaveTime startTime = WaveTime::now();
  deviceBuffer_ = waveStream.deviceArena().allocate<char>(fill_);
  hostBuffer_ = getTransferArena().allocate<char>(fill_);
  auto transferBuffer = hostBuffer_->as<char>();
  int firstToCopy = 0;
  int64_t copySize = 0;
  auto targetCopySize = FLAGS_staging_bytes_per_thread;
  int32_t numThreads = 0;
  if (fill_ > 2000000) {
    for (auto i = 0; i < staging_.size(); ++i) {
      auto columnSize = staging_[i].size;
      copySize += columnSize;
      if (copySize >= targetCopySize && i < staging_.size() - 1) {
        ++numThreads;
        WaveStream::copyExecutor()->add(
            [i, firstToCopy, transferBuffer, this]() {
              copyColumns(firstToCopy, i + 1, transferBuffer, true);
            });
        transferBuffer += copySize;
        copySize = 0;
        firstToCopy = i + 1;
      }
    }
  }
  auto deviceData = deviceBuffer_->as<char>();
  for (auto& pair : patch_) {
    *reinterpret_cast<int64_t*>(pair.second) +=
        reinterpret_cast<int64_t>(deviceData) + offsets_[pair.first];
  }
  if (asyncTail) {
    WaveStream::syncExecutor()->add([firstToCopy,
                                     numThreads,
                                     transferBuffer,
                                     asyncTail,
                                     &waveStream,
                                     &stream,
                                     recordEvent,
                                     startTime,
                                     this]() {
      copyColumns(firstToCopy, staging_.size(), transferBuffer, false);
      for (auto i = 0; i < numThreads; ++i) {
        sem_.acquire();
      }
      stream.hostToDeviceAsync(
          deviceBuffer_->as<char>(), hostBuffer_->as<char>(), fill_);
      waveStream.stats().stagingTime += WaveTime::now() - startTime;
      if (recordEvent) {
        event_ = std::make_unique<Event>();
        event_->record(stream);
      }
      fill_ = 0;
      patch_.clear();
      offsets_.clear();
      asyncTail(waveStream, stream);
    });
  } else {
    copyColumns(firstToCopy, staging_.size(), transferBuffer, false);
    for (auto i = 0; i < numThreads; ++i) {
      sem_.acquire();
    }
    stream.hostToDeviceAsync(
        deviceBuffer_->as<char>(), hostBuffer_->as<char>(), fill_);
    waveStream.stats().stagingTime += WaveTime::now() - startTime;
    if (recordEvent) {
      event_ = std::make_unique<Event>();
      event_->record(stream);
    }
    fill_ = 0;
    patch_.clear();
    offsets_.clear();
  }
}

namespace {
void setFilter(GpuDecode* step, ColumnReader* reader, Stream* stream) {
  auto* veloxFilter = reader->scanSpec().filter();
  if (!veloxFilter) {
    step->filterKind = WaveFilterKind::kAlwaysTrue;
    return;
  }
  switch (veloxFilter->kind()) {
    case common::FilterKind::kBigintRange: {
      step->filterKind = WaveFilterKind::kBigintRange;
      step->nullsAllowed = veloxFilter->testNull();
      step->filter._.int64Range[0] =
          reinterpret_cast<const common::BigintRange*>(veloxFilter)->lower();
      step->filter._.int64Range[1] =
          reinterpret_cast<const common::BigintRange*>(veloxFilter)->upper();
      break;
    }

    default:
      VELOX_UNSUPPORTED(
          "Unsupported filter kind", static_cast<int32_t>(veloxFilter->kind()));
  }
}
} // namespace

std::unique_ptr<GpuDecode> FormatData::makeStep(
    ColumnOp& op,
    const ColumnOp* previousFilter,
    ResultStaging& deviceStaging,
    SplitStaging& splitStaging,
    ReadStream& stream,
    WaveTypeKind columnKind,
    int32_t blockIdx) {
  auto rowsPerBlock = FLAGS_wave_reader_rows_per_tb;
  auto maxRowsPerThread = (rowsPerBlock / kBlockSize);
  int32_t numBlocks =
      bits::roundUp(op.rows.size(), rowsPerBlock) / rowsPerBlock;

  auto rowsInBlock = std::min<int32_t>(
      rowsPerBlock, op.rows.size() - (blockIdx * rowsPerBlock));

  auto step = std::make_unique<GpuDecode>();
  bool hasNulls = false;
  if (nullsBufferId_ != kNoBufferId) {
    hasNulls = true;
    if (grid_.nulls) {
      step->nonNullBases = grid_.numNonNull;
      step->nulls = grid_.nulls;
    } else {
      // The nulls transfer is staged but no pointer yet. Happens when the nulls
      // decoding is in the same kernel as decode, i.e. single TB per column.
      splitStaging.registerPointer(nullsBufferId_, &step->nulls, true);
      step->nonNullBases = nullptr;
    }
  }
  step->numRowsPerThread = bits::roundUp(rowsInBlock, kBlockSize) / kBlockSize;
  step->gridNumRowsPerThread = maxRowsPerThread;
  setFilter(step.get(), op.reader, nullptr);
  bool dense = previousFilter == nullptr &&
      simd::isDense(op.rows.data(), op.rows.size());
  step->nullMode = hasNulls
      ? (dense ? NullMode::kDenseNullable : NullMode::kSparseNullable)
      : (dense ? NullMode::kDenseNonNull : NullMode::kSparseNonNull);
  step->nthBlock = blockIdx;
  if (step->filterKind != WaveFilterKind::kAlwaysTrue && op.waveVector) {
    /// Filtres get to record an extra copy of their passing rows if they make
    /// values.
    if (blockIdx == 0) {
      op.extraRowCountId = deviceStaging.reserve(
          numBlocks * step->numRowsPerThread * sizeof(int32_t));
      deviceStaging.registerPointer(
          op.extraRowCountId, &step->filterRowCount, true);
      deviceStaging.registerPointer(
          op.extraRowCountId, &op.extraRowCount, true);
    } else {
      step->filterRowCount = reinterpret_cast<int32_t*>(
          blockIdx * sizeof(int32_t) * maxRowsPerThread);
      deviceStaging.registerPointer(
          op.extraRowCountId, &step->filterRowCount, false);
    }
  }
  step->dataType = columnKind;
  auto kindSize = waveTypeKindSize(columnKind);
  step->step =
      kindSize == 4 ? DecodeStep::kSelective32 : DecodeStep::kSelective64;
  if (previousFilter) {
    step->maxRow = GpuDecode::kFilterHits;
  } else {
    step->maxRow = currentRow_ + (blockIdx * rowsPerBlock) + rowsInBlock;
  }
  step->baseRow = currentRow_ + rowsPerBlock * blockIdx;

  if (op.waveVector) {
    if (blockIdx == 0) {
      VELOX_CHECK_GE(op.waveVector->size(), op.rows.size());
    }
    step->result =
        op.waveVector->values<char>() + kindSize * blockIdx * rowsPerBlock;
    step->resultNulls = op.waveVector->nulls()
        ? op.waveVector->nulls() + blockIdx * rowsPerBlock
        : nullptr;
  }
  // A filter that produces no values still needs the rows passing the
  // previous filter and records the rows passing itself.
  if (previousFilter) {
    if (previousFilter->deviceResult) {
      // This is when the previous filter is in the previous kernel and its
      // device side result is allocated.
      step->rows = previousFilter->deviceResult + blockIdx * rowsPerBlock;
    } else {
      // NOLINTBEGIN(performance-no-int-to-ptr)
      step->rows =
          reinterpret_cast<int32_t*>(blockIdx * rowsPerBlock * sizeof(int32_t));
      // NOLINTEND(performance-no-int-to-ptr)
      deviceStaging.registerPointer(
          previousFilter->deviceResultId, &step->rows, false);
    }
  }
  if (op.reader->scanSpec().filter()) {
    if (blockIdx == 0) {
      op.deviceResultId = deviceStaging.reserve(
          static_cast<int32_t>(op.rows.size() * sizeof(int32_t)));
      deviceStaging.registerPointer(op.deviceResultId, &step->resultRows, true);
      deviceStaging.registerPointer(op.deviceResultId, &op.deviceResult, true);
    } else {
      // NOLINTBEGIN(performance-no-int-to-ptr)
      step->resultRows =
          reinterpret_cast<int32_t*>(blockIdx * rowsPerBlock * sizeof(int32_t));
      // NOLINTEND(performance-no-int-to-ptr)
      deviceStaging.registerPointer(
          op.deviceResultId, &step->resultRows, false);
    }
  }
  return step;
}

bool StagedBuffers::stage(SplitStaging& staging) {
  if (buffers_.empty()) {
    return false;
  }
  if (staged_) {
    pending_ = false;
    staging.addDependency(stagingId_);
    return false;
  }
  staged_ = true;
  pending_ = true;
  for (auto& buffer : buffers_) {
    buffer.id = staging.add(buffer.staging);
    staging.registerPointer(buffer.id, &buffer.device, true);
  }
  stagingId_ = staging.id();
  return true;
}

int32_t chunkIndex(const std::vector<ChunkDesc>& chunks, int32_t row) {
  auto it = std::upper_bound(
      chunks.begin(), chunks.end(), row, [](int32_t row, const auto& chunk) {
        return row < chunk.firstRow;
      });
  VELOX_CHECK(it != chunks.begin());
  return static_cast<int32_t>(it - chunks.begin() - 1);
}

FormatData::FormatData(ColumnRole role, std::vector<const FormatData*> levels)
    : role_(role), levels_(std::move(levels)) {
  VELOX_CHECK_LE(
      levels_.size(), kMaxLevels, "Too many nested nullable structs");
}

void FormatData::startOp(
    ColumnOp& op,
    const ColumnOp* previousFilter,
    ResultStaging& deviceStaging,
    ResultStaging& /*resultStaging*/,
    SplitStaging& splitStaging,
    DecodePrograms& program,
    ReadStream& stream) {
  stage(deviceStaging, splitStaging);
  const auto rowsPerBlock = FLAGS_wave_reader_rows_per_tb;
  const auto numRows = static_cast<int32_t>(op.rows.size());
  const int32_t numBlocks = bits::divRoundUp(numRows, rowsPerBlock);
  VELOX_CHECK_LT(numBlocks, 256 * 256, "Overflow 16 bit block number");
  if (numBlocks > 1) {
    VELOX_CHECK(griddized_);
  }
  for (auto blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
    auto step = makeStep(
        op,
        previousFilter,
        deviceStaging,
        splitStaging,
        stream,
        stepKind(),
        blockIdx);
    setDecode(*step, op, blockIdx, splitStaging);
    if (role_ == ColumnRole::kLengths) {
      // The element starts and sizes of the top level rows of the TB.
      if (blockIdx == 0) {
        op.totals = stream.arena().allocate<int32_t>(numBlocks);
      }
      setNested(*step);
      auto& column = step->column();
      column.offsets = op.waveVector->offsets() + blockIdx * rowsPerBlock;
      column.sizes = op.waveVector->sizes() + blockIdx * rowsPerBlock;
      column.totals = op.totals->as<int32_t>();
      column.numResults =
          bits::roundUp(numRows, kBlockSize) - blockIdx * rowsPerBlock;
    }
    op.isFinal = true;
    std::vector<std::unique_ptr<GpuDecode>>* steps;
    // Programs are parallel after filters.
    if (stream.filtersDone() || !previousFilter) {
      program.programs.emplace_back();
      steps = &program.programs.back();
    } else {
      VELOX_CHECK_LT(blockIdx, static_cast<int32_t>(program.programs.size()));
      steps = &program.programs[blockIdx];
    }
    steps->push_back(std::move(step));
  }
}

void FormatData::startNestedOp(
    NestedOp& op,
    ResultStaging& deviceStaging,
    SplitStaging& staging,
    DecodePrograms& program,
    ReadStream& /*stream*/) {
  if (role_ == ColumnRole::kStructNulls && !op.vector->nulls()) {
    // The struct and the structs above it up to the array have no nulls.
    return;
  }
  stage(deviceStaging, staging);
  const int32_t rowsPerBlock = FLAGS_wave_reader_rows_per_tb;
  const int32_t numBlocks = bits::divRoundUp(op.numRows, rowsPerBlock);
  const auto kind = stepKind();
  const auto kindSize =
      role_ == ColumnRole::kValues ? waveTypeKindSize(kind) : sizeof(int32_t);
  for (auto blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
    const int32_t firstRow = blockIdx * rowsPerBlock;
    const int32_t numRows = std::min(rowsPerBlock, op.numRows - firstRow);
    auto step = std::make_unique<GpuDecode>();
    step->numRowsPerThread = bits::divRoundUp(numRows, kBlockSize);
    step->gridNumRowsPerThread = rowsPerBlock / kBlockSize;
    step->nthBlock = blockIdx;
    step->nullMode = NullMode::kDenseNonNull;
    step->baseRow = firstRow;
    step->maxRow = firstRow + numRows;
    step->dataType = kind;
    step->resultNulls =
        op.vector->nulls() ? op.vector->nulls() + firstRow : nullptr;
    setNestedDecode(*step, staging);
    auto& column = step->column();
    setLevels(column);
    column.elementRows = op.elementRows;
    column.numResults = op.numRows - firstRow;
    if (role_ == ColumnRole::kValues) {
      step->step =
          kindSize == 4 ? DecodeStep::kSelective32 : DecodeStep::kSelective64;
      step->result = op.vector->values<char>() + kindSize * firstRow;
    } else {
      setNested(*step);
      if (role_ == ColumnRole::kLengths) {
        column.offsets = op.vector->offsets() + firstRow;
        column.sizes = op.vector->sizes() + firstRow;
        column.totals = op.totals->as<int32_t>();
      }
    }
    program.programs.emplace_back();
    program.programs.back().push_back(std::move(step));
  }
}

void FormatData::setLevels(ColumnAccess& column) const {
  column.numLevels = static_cast<int32_t>(levels_.size());
  for (auto i = 0; i < levels_.size(); ++i) {
    column.levelAux[i] = levels_[i]->deviceAux();
    VELOX_CHECK_NOT_NULL(column.levelAux[i]);
    column.levelRows[i] = levels_[i]->numEncodedRows();
  }
}

void FormatData::setNested(GpuDecode& step) const {
  auto& column = step.column();
  step.step = DecodeStep::kNested;
  column.isStruct = role_ == ColumnRole::kStructNulls;
  column.streamRows = numEncodedRows();
  column.aux = deviceAux_;
  if (role_ == ColumnRole::kLengths) {
    column.aux += lengthsAux_;
  }
}

void FormatData::allocateAux(ReadStream& stream, int32_t bytes) {
  auxBuffer_ = stream.deviceArena().allocate<char>(bytes);
  deviceAux_ = auxBuffer_->as<char>();
}

std::pair<int32_t, int32_t> FormatData::tbChunks(
    const std::vector<ChunkDesc>& chunks,
    int32_t firstRow,
    int32_t lastRow) const {
  if (!levels_.empty()) {
    return {0, static_cast<int32_t>(chunks.size())};
  }
  const auto first = chunkIndex(chunks, firstRow);
  return {first, chunkIndex(chunks, lastRow) - first + 1};
}

std::unique_ptr<GpuDecode> FormatData::makePrepareStep(
    ColumnOp& op,
    bool withFilter,
    ResultStaging& deviceStaging,
    SplitStaging& splitStaging,
    ReadStream& stream) {
  if (withFilter) {
    return makeAlphabetStep(
        op, deviceStaging, splitStaging, stream, WaveTypeKind::BIGINT, 0, 1);
  }
  auto step = std::make_unique<GpuDecode>();
  step->numRowsPerThread = 1;
  step->gridNumRowsPerThread = FLAGS_wave_reader_rows_per_tb / kBlockSize;
  step->nullMode = NullMode::kDenseNonNull;
  step->dataType = WaveTypeKind::BIGINT;
  return step;
}

std::unique_ptr<GpuDecode> FormatData::makeAlphabetStep(
    ColumnOp& op,
    ResultStaging& deviceStaging,
    SplitStaging& splitStaging,
    ReadStream& stream,
    WaveTypeKind columnKind,
    int32_t blockIdx,
    int32_t numRows) {
  auto rowsPerBlock = FLAGS_wave_reader_rows_per_tb;
  auto maxRowsPerThread = (rowsPerBlock / kBlockSize);

  auto rowsInBlock =
      std::min<int32_t>(rowsPerBlock, numRows - (blockIdx * rowsPerBlock));

  auto step = std::make_unique<GpuDecode>();
  step->numRowsPerThread = bits::roundUp(rowsInBlock, kBlockSize) / kBlockSize;
  step->gridNumRowsPerThread = maxRowsPerThread;
  setFilter(step.get(), op.reader, nullptr);
  step->nullMode = NullMode::kDenseNonNull;
  step->nthBlock = blockIdx;
  step->dataType = columnKind;
  auto kindSize = waveTypeKindSize(columnKind);
  step->step =
      kindSize == 4 ? DecodeStep::kSelective32 : DecodeStep::kSelective64;
  step->maxRow = (blockIdx * rowsPerBlock) + rowsInBlock;

  step->baseRow = rowsPerBlock * blockIdx;

  return step;
}

} // namespace facebook::velox::wave
