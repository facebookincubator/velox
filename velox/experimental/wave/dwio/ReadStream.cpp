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

#include "velox/common/process/TraceContext.h"
#include "velox/experimental/wave/dwio/ColumnReader.h"
#include "velox/experimental/wave/dwio/StructColumnReader.h"

DEFINE_int32(
    wave_reader_rows_per_tb,
    1024,
    "Number of items per thread block in Wave reader");

DECLARE_int32(wave_max_reader_batch_rows);

DEFINE_int32(
    wave_max_nested_elements,
    64 << 20,
    "Maximum number of elements of the arrays and maps of one level of nesting "
    "in a batch of a Wave reader");

DEFINE_int32(
    wave_max_nesting_depth,
    64,
    "Maximum depth of arrays and maps in the Wave reader");

namespace facebook::velox::wave {

void allOperands(
    const ColumnReader* reader,
    OperandSet& operands,
    std::vector<AbstractOperand*>* abstractOperands) {
  auto op = reader->operand();
  if (op != nullptr) {
    operands.add(op->id);
    if (abstractOperands) {
      abstractOperands->push_back(op);
    }
  }

  for (auto& child : reader->children()) {
    allOperands(child, operands, abstractOperands);
  }
}

namespace {
// Appends the readers below 'reader' to 'all' and the ones that read top
// level rows to 'leaves', in depth first order. These are the readers without
// children and the arrays and maps that are not below an array or map.
void collectReaders(
    const ColumnReader* reader,
    std::vector<ColumnReader*>& all,
    std::vector<ColumnReader*>& leaves,
    bool isNested = false) {
  for (auto* child : reader->children()) {
    all.push_back(child);
    if (child->isRepeated()) {
      if (!isNested && child->operand()) {
        leaves.push_back(child);
      }
      collectReaders(child, all, leaves, true);
    } else if (child->children().empty()) {
      // A reader without a result or filter, e.g. a map without requested
      // keys, has nothing to read.
      if (!isNested && (child->operand() || child->scanSpec().filter())) {
        leaves.push_back(child);
      }
    } else {
      collectReaders(child, all, leaves, isNested);
    }
  }
}
} // namespace

ReadStream::ReadStream(
    StructColumnReader* columnReader,
    WaveStream& _waveStream,
    io::IoStatistics* ioStatistics,
    FileInfo& fileInfo,
    const OperandSet* firstColumns)
    : Executable(), ioStats_(ioStatistics), fileInfo_(fileInfo) {
  VELOX_CHECK_EQ(
      0,
      FLAGS_wave_reader_rows_per_tb & 1023,
      "wave_reader_rows_per_tb must be a multiple of 1K");
  waveStream = &_waveStream;
  allOperands(columnReader, outputOperands, &abstractOperands_);
  output.resize(outputOperands.size());
  reader_ = columnReader;
  collectReaders(reader_, allReaders_, leaves_);
  reader_->splitStaging().push_back(
      std::make_unique<SplitStaging>(fileInfo_, 0));
  currentStaging_ = reader_->splitStaging().back().get();
}

void ReadStream::setBlockStatusAndTemp(Stream* stream) {
  auto* status = control_->deviceData->as<BlockStatus>();
  prefetchStatus(stream);
  auto maxRowsPerThread = FLAGS_wave_reader_rows_per_tb / kBlockSize;
  auto tempSize = programs_.programs[0][0]->tempSize();
  auto size = programs_.programs.size() * tempSize;
  auto id = deviceStaging_.reserve(size);
  for (auto blockIdx = 0; blockIdx < programs_.programs.size(); ++blockIdx) {
    auto& program = programs_.programs[blockIdx];
    for (auto& op : program) {
      op->temp = reinterpret_cast<int32_t*>(blockIdx * tempSize);
      deviceStaging_.registerPointer(id, &op->temp, false);
      op->blockStatus = status + maxRowsPerThread * op->nthBlock;
    }
  }
}

void ReadStream::prefetchStatus(Stream* stream) {
  if (!stream) {
    return;
  }
  char* data = control_->deviceData->as<char>();
  auto size = control_->deviceData->size() - statusBytes_;
  stream->prefetch(getDevice(), data + statusBytes_, size);
}

namespace {
void maybeRecordTransferTime(Stream& stream, WaveStream& waveStream) {
  if (stream.getAndClearIsTransfer() && FLAGS_wave_transfer_timing) {
    WaveTimer t(waveStream.mutableStats().transferWaitTime);
    stream.wait();
  }
}

void recordDecodeLaunch(WaveStream& waveStream, const DecodeLaunchInfo& info) {
  WaveStats& stats = waveStream.stats();
  stats.numThreadBlocks += info.numBlocks;
  ++stats.numReaderKernels;
  stats.numNewReaderKernels += info.isNewKernel;
  stats.readerLaunchMicros += info.launchMicros;
}
} // namespace

bool ReadStream::makeGrid(Stream* stream) {
  programs_.clear();
  auto total = reader_->formatData()->totalRows();
  auto blockSize = FLAGS_wave_reader_rows_per_tb;
  auto numBlocks = bits::roundUp(total, blockSize) / blockSize;
  int32_t nthFilter = 0;
  int32_t nthNonFilter = 0;
  int32_t nthLeaf = 0;
  for (auto* child : allReaders_) {
    // A struct or a column below an array has no op. Its griddize prepares
    // the data that does not depend on the rows that are read.
    ColumnOp localOp;
    localOp.reader = child;
    auto* op = &localOp;
    if (nthLeaf < leaves_.size() && leaves_[nthLeaf] == child) {
      ++nthLeaf;
      op = child->scanSpec().filter() ? &filters_[nthFilter++]
                                      : &ops_[nthNonFilter++];
    }
    child->formatData()->griddize(
        *op,
        blockSize,
        numBlocks,
        deviceStaging_,
        resultStaging_,
        *currentStaging_,
        programs_,
        *this);
  }
  if (!programs_.programs.empty()) {
    WaveStats& stats = waveStream->stats();
    auto bytes = currentStaging_->bytesToDevice();
    ioStats_->incRawBytesRead(bytes);
    stats.bytesToDevice += bytes;
    ++stats.numKernels;
    stats.numPrograms += programs_.programs.size();
    stats.numThreads +=
        programs_.programs.size() * std::min<int32_t>(rows_.size(), kBlockSize);
    setBlockStatusAndTemp();
    deviceStaging_.makeDeviceBuffer(waveStream->arena());
    currentStaging_->transfer(*waveStream, *stream, true);
    LaunchParams params(waveStream->deviceArena());
    WaveBufferPtr extra;
    {
      maybeRecordTransferTime(*stream, *waveStream);
      PrintTime l("grid");
      recordDecodeLaunch(*waveStream, launchDecode(programs_, params, stream));
    }
    reader_->recordGriddize(*stream);
    if (params.device) {
      commands_.push_back(std::move(params));
    }
    auto nth = reader_->splitStaging().size();
    reader_->splitStaging().push_back(
        std::make_unique<SplitStaging>(fileInfo_, nth));
    currentStaging_ = reader_->splitStaging().back().get();
    return true;
  }
  return false;
}

void ReadStream::makeCompact(bool isSerial) {
  auto rowsPerBlock = FLAGS_wave_reader_rows_per_tb;
  auto maxRowsPerThread = FLAGS_wave_reader_rows_per_tb / kBlockSize;
  for (int32_t i = 0; i < static_cast<int32_t>(filters_.size()) - 1; ++i) {
    if (filters_[i].waveVector) {
      int32_t numTBs =
          bits::roundUp(numBlocks_, maxRowsPerThread) / maxRowsPerThread;
      for (auto blockIdx = 0; blockIdx < numTBs; ++blockIdx) {
        auto step = std::make_unique<GpuDecode>();
        step->step = DecodeStep::kCompact64;
        step->nthBlock = blockIdx;
        step->numRowsPerThread = blockIdx == numTBs - 1
            ? numBlocks_ - (numTBs - 1) * maxRowsPerThread
            : maxRowsPerThread;
        step->gridNumRowsPerThread = maxRowsPerThread;
        if (filters_.back().deviceResult) {
          step->data.compact.finalRows =
              filters_.back().deviceResult + blockIdx * rowsPerBlock;
          step->data.compact.sourceNumRows =
              filters_[i].extraRowCount + blockIdx * maxRowsPerThread;
        } else {
          step->data.compact.finalRows = reinterpret_cast<int32_t*>(
              blockIdx * rowsPerBlock * sizeof(int32_t));
          deviceStaging_.registerPointer(
              filters_.back().deviceResultId,
              &step->data.compact.finalRows,
              false);
          step->data.compact.sourceNumRows = reinterpret_cast<int32_t*>(
              blockIdx * maxRowsPerThread * sizeof(int32_t));
          deviceStaging_.registerPointer(
              filters_[i].extraRowCountId,
              &step->data.compact.sourceNumRows,
              false);
        }
        if (filters_[i].deviceResult) {
          step->data.compact.sourceRows =
              filters_[i].deviceResult + blockIdx * rowsPerBlock;
        } else {
          step->data.compact.sourceRows =
              reinterpret_cast<int32_t*>(blockIdx * rowsPerBlock);
          deviceStaging_.registerPointer(
              filters_[i].deviceResultId,
              &step->data.compact.sourceRows,
              false);
        }
        auto& vector = filters_[i].waveVector;
        step->dataType = static_cast<WaveTypeKind>(vector->type()->kind());
        step->data.compact.source = vector->values<char>() +
            waveTypeKindSize(step->dataType) * blockIdx * rowsPerBlock;
        if (vector->nulls()) {
          step->data.compact.sourceNull =
              vector->nulls() + blockIdx * rowsPerBlock;
        }
        if (isSerial) {
          programs_.programs[blockIdx].push_back(std::move(step));
        } else {
          programs_.programs.emplace_back();
          programs_.programs.back().push_back(std::move(step));
        }
      }
    }
  }
}

void ReadStream::makeOps() {
  for (auto* child : leaves_) {
    if (child->scanSpec().filter()) {
      hasFilters_ = true;
      filters_.emplace_back();
      bool filterOnly = !child->scanSpec().keepValues();
      child->makeOp(
          this,
          filterOnly ? ColumnAction::kFilter : ColumnAction::kValues,
          filters_.back());
    }
  }
  for (auto* child : leaves_) {
    if (child->scanSpec().filter()) {
      continue;
    }
    ops_.emplace_back();
    auto& op = ops_.back();
    child->makeOp(this, ColumnAction::kValues, op);
  }
}

bool ReadStream::decodenonFiltersInFiltersKernel() {
  return ops_.size() == 1;
}

void ReadStream::prepareRead() {
  filtersDone_ = false;
  filtersCompacted_ = false;
  for (auto& op : filters_) {
    op.reader->formatData()->newBatch(row_);
    op.isFinal = false;
    op.rows = rows_;
  }
  for (auto& op : ops_) {
    op.reader->formatData()->newBatch(row_);
    op.isFinal = false;
    op.rows = rows_;
  }
}

bool ReadStream::makePrograms(bool& needSync) {
  bool allDone = true;
  needSync = false;
  programs_.clear();
  ColumnOp* previousFilter = nullptr;

  auto filterOpsFinal = [&]() {
    return std::all_of(filters_.begin(), filters_.end(), [](auto& filter) {
      return filter.isFinal;
    });
  };

  if (!filtersDone_ && !filters_.empty()) {
    // Filters are done consecutively, each TB does all the filters for its
    // range.
    for (auto& filter : filters_) {
      filter.reader->formatData()->startOp(
          filter,
          previousFilter,
          deviceStaging_,
          resultStaging_,
          *currentStaging_,
          programs_,
          *this);
      previousFilter = &filter;
    }
    filtersDone_ = filterOpsFinal();

    // the decode is already done here if the filters are dispatched and there
    // is only one filter kernel (hence makeCompact is not needed) and there are
    // no non-filter columns to decode.
    return filtersDone_ && filters_.size() == 1 && ops_.empty();
    // TODO(bowenwu): revisit this optimization of
    // decodenonFiltersInFiltersKernel().
  }
  if (!filtersCompacted_) {
    // makeCompact is not idempotent can only be called once per batch.
    makeCompact(!filtersDone_);
    filtersCompacted_ = true;
  }
  previousFilter = filters_.empty() ? nullptr : &filters_.back();
  for (auto i = 0; i < ops_.size(); ++i) {
    auto& op = ops_[i];
    if (op.isFinal) {
      continue;
    }
    if (op.prerequisite == ColumnOp::kNoPrerequisite ||
        ops_[op.prerequisite].isFinal) {
      op.reader->formatData()->startOp(
          op,
          previousFilter,
          deviceStaging_,
          resultStaging_,
          *currentStaging_,
          programs_,
          *this);
      if (!op.isFinal) {
        allDone = false;
      }
      if (op.needsResult) {
        needSync = true;
      }
    } else {
      allDone = false;
    }
  }
  filtersDone_ = true;
  if ((filters_.empty() || gridStatusBytes_ > 0) && allDone) {
    auto setCount = std::make_unique<GpuDecode>();
    setCount->step = DecodeStep::kRowCountNoFilter;
    setCount->data.rowCountNoFilter.numRows = rows_.size();
    setCount->data.rowCountNoFilter.status =
        control_->deviceData->as<BlockStatus>();
    setCount->data.rowCountNoFilter.gridStatusSize = gridStatusBytes_;
    setCount->data.rowCountNoFilter.gridOnly = !filters_.empty();

    programs_.programs.emplace_back();
    programs_.programs.back().push_back(std::move(setCount));
  }
  ++nthWave_;
  resultStaging_.setReturnBuffer(waveStream->arena(), programs_.result);
  return allDone;
}

void ReadStream::syncStaging(Stream& stream) {
  auto& set = currentStaging_->dependsOn();
  if (set.empty()) {
    return;
  }
  set.forEach([&](int32_t id) {
    auto dep = reader_->splitStaging()[id].get();
    auto event = dep->event();
    VELOX_CHECK(event);
    event->wait(stream);
  });
}

void ReadStream::initializeResultNulls(Stream& stream) {
  for (auto i = 0; i < filters_.size(); ++i) {
    if (filters_[i].reader->formatData()->hasNulls()) {
      auto waveVector = filters_[i].waveVector;
      if (waveVector && waveVector->nulls()) {
        stream.memset(waveVector->nulls(), 1, waveVector->size());
      }
    }
  }
  for (auto i = 0; i < ops_.size(); ++i) {
    if (ops_[i].reader->formatData()->hasNulls()) {
      auto waveVector = ops_[i].waveVector;
      VELOX_CHECK(waveVector != nullptr && waveVector->nulls() != nullptr);
      stream.memset(waveVector->nulls(), 1, waveVector->size());
    }
  }
}

void ReadStream::launch(
    std::unique_ptr<ReadStream> readStream,
    int32_t row,
    RowSet rows) {
  using UniqueExe = std::unique_ptr<Executable>;
  readStream->row_ = row;
  readStream->rows_ = rows;

  // The function of control here is to have a status and row count for each
  // kBlockSize top level rows of output and to have Operand structs for the
  // produced column.
  readStream->makeControl();
  auto waveStream = readStream->waveStream;
  WaveStats& stats = waveStream->stats();
  bool firstLaunch = true;
  waveStream->installExecutables(
      folly::Range<UniqueExe*>(reinterpret_cast<UniqueExe*>(&readStream), 1),
      [&](Stream* stream, folly::Range<Executable**> exes) {
        auto* readStream = reinterpret_cast<ReadStream*>(exes[0]);
        bool needSync = false;
        bool griddizedHere = false;
        if (!readStream->inited_) {
          readStream->makeOps();
          // A ReadStream for a later batch of the same stripe makes no
          // prepare ops. Those of the first batch may still run on another
          // stream.
          griddizedHere = readStream->makeGrid(stream);
          readStream->initializeResultNulls(*stream);
          readStream->inited_ = true;
        }
        // The decodes read the aux data made by the prepare ops.
        auto waitGriddize = [&]() {
          if (!griddizedHere && firstLaunch) {
            if (auto* event = readStream->reader_->griddizeEvent()) {
              event->wait(*stream);
            }
          }
          firstLaunch = false;
        };
        readStream->prepareRead();
        for (;;) {
          bool done = readStream->makePrograms(needSync);
          auto bytes = readStream->currentStaging_->bytesToDevice();
          readStream->ioStats_->incRawBytesRead(bytes);
          stats.bytesToDevice += bytes;
          ++stats.numKernels;
          stats.numPrograms += readStream->programs_.programs.size();
          stats.numThreads += readStream->programs_.programs.size() *
              std::min<int32_t>(readStream->rows_.size(), kBlockSize);
          readStream->currentStaging_->transfer(*waveStream, *stream, true);
          if (done) {
            break;
          }
          readStream->setBlockStatusAndTemp(stream);
          readStream->deviceStaging_.makeDeviceBuffer(waveStream->arena());
          WaveBufferPtr extra;
          waitGriddize();
          readStream->syncStaging(*stream);
          LaunchParams params(waveStream->deviceArena());
          {
            maybeRecordTransferTime(*stream, *readStream->waveStream);
            PrintTime l("decode");
            recordDecodeLaunch(
                *readStream->waveStream,
                launchDecode(readStream->programs(), params, stream));
          }
          if (params.device) {
            readStream->commands_.push_back(std::move(params));
          }
          auto nth = readStream->reader_->splitStaging().size();
          readStream->reader_->splitStaging().push_back(
              std::make_unique<SplitStaging>(readStream->fileInfo_, nth));
          readStream->currentStaging_ =
              readStream->reader_->splitStaging().back().get();
          if (needSync) {
            waveStream->setState(WaveStream::State::kWait);
            stream->wait();
            readStream->waveStream->setState(WaveStream::State::kHost);
          } else {
            readStream->waveStream->setState(WaveStream::State::kParallel);
          }
        }

        readStream->setBlockStatusAndTemp(stream);
        readStream->deviceStaging_.makeDeviceBuffer(waveStream->arena());
        LaunchParams params(readStream->waveStream->deviceArena());
        waitGriddize();
        readStream->syncStaging(*stream);
        {
          maybeRecordTransferTime(*stream, *readStream->waveStream);
          PrintTime l("decode-f");
          recordDecodeLaunch(
              *readStream->waveStream,
              launchDecode(readStream->programs(), params, stream));
        }
        if (params.device) {
          readStream->commands_.push_back(std::move(params));
        }
        readStream->readNested(stream);
        readStream->waveStream->setState(WaveStream::State::kParallel);
        readStream->waveStream->markLaunch(*stream, *readStream);
      });
}

std::unique_ptr<WaveVector> ReadStream::makeNestedVector(
    const ColumnReader& reader,
    int32_t numRows) {
  auto vector =
      std::make_unique<WaveVector>(reader.requestedType(), waveStream->arena());
  vector->resize(numRows, reader.formatData()->hasNulls());
  if (!reader.isRepeated() && !reader.children().empty()) {
    std::vector<std::unique_ptr<WaveVector>> children;
    for (auto* child : reader.children()) {
      children.push_back(makeNestedVector(*child, numRows));
    }
    vector->setChildren(std::move(children));
  }
  return vector;
}

void ReadStream::addNestedOps(NestedOp op, std::vector<NestedOp>& arrays) {
  auto* reader = op.reader;
  if (reader->isRepeated()) {
    const int32_t numBlocks =
        bits::divRoundUp(op.numRows, FLAGS_wave_reader_rows_per_tb);
    op.totals =
        waveStream->arena().allocate<int32_t>(std::max<int32_t>(numBlocks, 1));
  }
  reader->formatData()->startNestedOp(
      op, deviceStaging_, *currentStaging_, programs_, *this);
  if (reader->isRepeated()) {
    arrays.push_back(std::move(op));
    return;
  }
  for (auto i = 0; i < reader->children().size(); ++i) {
    NestedOp child = op;
    child.reader = reader->children()[i];
    child.vector = &op.vector->childAt(i);
    addNestedOps(std::move(child), arrays);
  }
}

void ReadStream::launchNested(Stream* stream) {
  WaveStats& stats = waveStream->stats();
  auto bytes = currentStaging_->bytesToDevice();
  ioStats_->incRawBytesRead(bytes);
  stats.bytesToDevice += bytes;
  ++stats.numKernels;
  stats.numPrograms += static_cast<int64_t>(programs_.programs.size());
  currentStaging_->transfer(*waveStream, *stream, true);
  setBlockStatusAndTemp(stream);
  deviceStaging_.makeDeviceBuffer(waveStream->arena());
  syncStaging(*stream);
  LaunchParams params(waveStream->deviceArena());
  {
    PrintTime l("decode-nested");
    recordDecodeLaunch(*waveStream, launchDecode(programs_, params, stream));
  }
  if (params.device) {
    commands_.push_back(std::move(params));
  }
  auto nth = reader_->splitStaging().size();
  reader_->splitStaging().push_back(
      std::make_unique<SplitStaging>(fileInfo_, nth));
  currentStaging_ = reader_->splitStaging().back().get();
}

void ReadStream::readNested(Stream* stream) {
  std::vector<NestedOp> arrays;
  for (auto& op : ops_) {
    if (op.reader->isRepeated()) {
      NestedOp array;
      array.reader = op.reader;
      array.numRows =
          static_cast<int32_t>(bits::roundUp(rows_.size(), kBlockSize));
      array.vector = op.waveVector;
      array.totals = op.totals;
      arrays.push_back(std::move(array));
    }
  }
  if (arrays.empty()) {
    return;
  }
  const int32_t rowsPerBlock = FLAGS_wave_reader_rows_per_tb;
  auto nth = reader_->splitStaging().size();
  reader_->splitStaging().push_back(
      std::make_unique<SplitStaging>(fileInfo_, nth));
  currentStaging_ = reader_->splitStaging().back().get();
  // Each level of nesting syncs to get the number of elements of the arrays
  // and maps of the level. Then a kernel turns the sizes into offsets and
  // element rows and a kernel reads the elements, including the sizes of the
  // next level.
  int32_t depth = 0;
  while (!arrays.empty()) {
    VELOX_CHECK_LT(
        depth++, FLAGS_wave_max_nesting_depth, "Arrays nested too deep");
    waveStream->setState(WaveStream::State::kWait);
    stream->wait();
    waveStream->setState(WaveStream::State::kHost);
    programs_.clear();
    std::vector<NestedOp> elements;
    for (auto& array : arrays) {
      const int32_t numBlocks = bits::divRoundUp(array.numRows, rowsPerBlock);
      const auto* totals = array.totals->as<int32_t>();
      int32_t numElements = 0;
      std::vector<GpuDecode*> steps;
      for (auto block = 0; block < numBlocks; ++block) {
        auto step = std::make_unique<GpuDecode>();
        step->step = DecodeStep::kExpandLengths;
        step->nthBlock = block;
        auto& expand = step->data.expandLengths;
        expand.offsets = array.vector->offsets() + block * rowsPerBlock;
        expand.sizes = array.vector->sizes() + block * rowsPerBlock;
        expand.base = numElements;
        expand.numRows =
            std::min(rowsPerBlock, array.numRows - block * rowsPerBlock);
        // Element counts come from device. Bad counts must not make huge
        // allocations.
        VELOX_CHECK_GE(totals[block], 0, "Negative array element count");
        VELOX_CHECK_LE(
            static_cast<int64_t>(numElements) + totals[block],
            FLAGS_wave_max_nested_elements,
            "Too many array elements in a batch");
        numElements += totals[block];
        steps.push_back(step.get());
        programs_.programs.emplace_back();
        programs_.programs.back().push_back(std::move(step));
      }
      VLOG(1) << "Nested level " << depth << ": " << numBlocks << " TBs, "
              << numElements << " elements";
      auto rows = waveStream->arena().allocate<int32_t>(
          std::max<int32_t>(numElements, 1));
      for (auto* step : steps) {
        step->data.expandLengths.rows = rows->as<int32_t>();
      }
      std::vector<std::unique_ptr<WaveVector>> children;
      for (auto* child : array.reader->children()) {
        children.push_back(makeNestedVector(*child, numElements));
      }
      array.vector->setChildren(std::move(children));
      for (auto i = 0; i < array.reader->children().size(); ++i) {
        NestedOp element;
        element.reader = array.reader->children()[i];
        element.numRows = numElements;
        element.elementRows = rows->as<int32_t>();
        element.vector = &array.vector->childAt(i);
        elements.push_back(std::move(element));
      }
      nestedBuffers_.push_back(std::move(rows));
    }
    if (!programs_.programs.empty()) {
      launchNested(stream);
    }
    programs_.clear();
    arrays.clear();
    for (auto& element : elements) {
      addNestedOps(std::move(element), arrays);
    }
    if (!programs_.programs.empty()) {
      launchNested(stream);
    }
  }
  programs_.clear();
}

void ReadStream::makeControl() {
  auto numRows = rows_.size();
  numBlocks_ = bits::roundUp(numRows, kBlockSize) / kBlockSize;
  waveStream->setNumRows(numRows);
  WaveStream::ExeLaunchInfo info;
  waveStream->exeLaunchInfo(*this, numBlocks_, info);
  auto instructionStatus = waveStream->instructionStatus();
  int32_t instructionBytes =
      instructionStatusSize(instructionStatus, numBlocks_);
  statusBytes_ = bits::roundUp(sizeof(BlockStatus) * numBlocks_, 8);
  auto deviceBytes = statusBytes_ + instructionBytes + info.totalBytes;
  auto control = std::make_unique<LaunchControl>(0, numRows);
  control->deviceData = waveStream->arena().allocate<char>(deviceBytes);
  // The operand section must be cleared before written on host. The statuses
  // are cleared on device.
  memset(
      control->deviceData->as<char>() + statusBytes_,
      0,
      instructionBytes + info.totalBytes);
  control->params.status = control->deviceData->as<BlockStatus>();
  // The WaveStream may have read another stripe where the nulls differ.
  for (auto* reader : leaves_) {
    if (auto* operand = reader->operand()) {
      waveStream->operandNullable()[operand->id] =
          reader->formatData()->hasNulls() && !reader->hasNonNullFilter();
    }
  }
  operands = waveStream->fillOperands(
      *this,
      control->deviceData->as<char>() + statusBytes_ + instructionBytes,
      info)[0];
  control_ = control.get();
  waveStream->setLaunchControl(0, 0, std::move(control));
}

} // namespace facebook::velox::wave
