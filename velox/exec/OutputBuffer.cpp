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
#include "velox/exec/OutputBuffer.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

using core::PartitionedOutputNode;

void ArbitraryBuffer::noMoreData() {
  // Drop duplicate end markers.
  if (!pages_.empty() && pages_.back() == nullptr) {
    return;
  }
  pages_.push_back(nullptr);
}

void ArbitraryBuffer::enqueue(std::unique_ptr<SerializedPage> page) {
  VELOX_CHECK_NOT_NULL(page, "Unexpected null page");
  VELOX_CHECK(!hasNoMoreData(), "Arbitrary buffer has set no more data marker");
  pages_.push_back(std::shared_ptr<SerializedPage>(page.release()));
}

void ArbitraryBuffer::getAvailablePageSizes(std::vector<int64_t>& out) const {
  out.reserve(out.size() + pages_.size());
  for (const auto& page : pages_) {
    if (page != nullptr) {
      out.push_back(page->size());
    }
  }
}

std::vector<std::shared_ptr<SerializedPage>> ArbitraryBuffer::getPages(
    uint64_t maxBytes) {
  if (maxBytes == 0 && !pages_.empty() && pages_.front() == nullptr) {
    // Always give out an end marker when this buffer is finished and fully
    // consumed.  When multiple `DestinationBuffer' polling the same
    // `ArbitraryBuffer', we can simplify the code in
    // `DestinationBuffer::getData' since we will always get a null marker and
    // not going through the callback path, eliminate the chance of getting
    // stuck.
    VELOX_CHECK_EQ(pages_.size(), 1);
    return {nullptr};
  }
  std::vector<std::shared_ptr<SerializedPage>> pages;
  uint64_t bytesRemoved{0};
  while (bytesRemoved < maxBytes && !pages_.empty()) {
    if (pages_.front() == nullptr) {
      // NOTE: keep the end marker in arbitrary buffer to signal all the
      // destination buffers after the buffers have all been consumed.
      VELOX_CHECK_EQ(pages_.size(), 1);
      pages.push_back(nullptr);
      break;
    }
    bytesRemoved += pages_.front()->size();
    pages.push_back(std::move(pages_.front()));
    pages_.pop_front();
  }
  return pages;
}

std::string ArbitraryBuffer::toString() const {
  return fmt::format(
      "[ARBITRARY_BUFFER PAGES[{}] NO MORE DATA[{}]]",
      pages_.size() - !!hasNoMoreData(),
      hasNoMoreData());
}

uint64_t DestinationBuffer::BufferedPages::PageSpiller::size() const {
  return pageSizes_.size();
}

uint64_t DestinationBuffer::BufferedPages::PageSpiller::totalBytes() const {
  return totalBytes_;
}

std::shared_ptr<SerializedPage>
DestinationBuffer::BufferedPages::PageSpiller::at(uint64_t index) {
  VELOX_CHECK_LT(index, pageSizes_.size());
  const auto numBufferedPages = bufferedPages_.size();
  if (index < numBufferedPages) {
    return bufferedPages_[index];
  }

  const auto pagesToUnspill = index + 1 - numBufferedPages;
  bufferedPages_.reserve(numBufferedPages + pagesToUnspill);
  for (uint32_t i = 0; i < pagesToUnspill; ++i) {
    bufferedPages_.push_back(unspillNextPage());
  }
  return bufferedPages_[index];
}

bool DestinationBuffer::BufferedPages::PageSpiller::isNullAt(
    uint64_t index) const {
  VELOX_CHECK_LT(index, pageSizes_.size());
  return !pageSizes_[index].has_value();
}

uint64_t DestinationBuffer::BufferedPages::PageSpiller::sizeAt(
    uint64_t index) const {
  VELOX_CHECK_LT(index, pageSizes_.size());
  const auto& pageSize = pageSizes_[index];
  VELOX_CHECK(pageSize.has_value());
  return pageSize.value();
}

void DestinationBuffer::BufferedPages::PageSpiller::deleteFront(
    uint64_t numPages) {
  VELOX_CHECK_LE(numPages, pageSizes_.size());
  for (uint32_t i = 0; i < numPages; ++i) {
    totalBytes_ -= pageSizes_[i].has_value() ? pageSizes_[i].value() : 0;
  }
  pageSizes_.erase(pageSizes_.begin(), pageSizes_.begin() + numPages);

  const auto numBuffered = std::min(numPages, (uint64_t)bufferedPages_.size());
  bufferedPages_.erase(
      bufferedPages_.begin(), bufferedPages_.begin() + numBuffered);
  numPages -= numBuffered;

  for (; numPages > 0; --numPages) {
    unspillNextPage();
  }
}

std::vector<std::shared_ptr<SerializedPage>>
DestinationBuffer::BufferedPages::PageSpiller::deleteAll() {
  while (curFileStream_ != nullptr || !spillFilePaths_.empty()) {
    bufferedPages_.push_back(unspillNextPage());
  }
  VELOX_CHECK(spillFilePaths_.empty());
  VELOX_CHECK_NULL(curFileStream_);
  auto deletedPages = std::move(bufferedPages_);
  bufferedPages_.clear();
  pageSizes_.clear();
  totalBytes_ = 0;
  return deletedPages;
}

std::tuple<std::string, std::unique_ptr<WriteFile>>
DestinationBuffer::BufferedPages::PageSpiller::nextSpillWriteFile() {
  std::string path = fmt::format("{}-{}", filePrefix_, nextFileId_++);
  auto fs = filesystems::getFileSystem(path, nullptr);
  return {
      path,
      fs->openFileForWrite(
          path,
          filesystems::FileOptions{
              {{filesystems::FileOptions::kFileCreateConfig.toString(),
                fileCreateConfig_}},
              nullptr,
              std::nullopt})};
}

namespace {
// A wrapper around a write file that provides buffer capability.
class BufferedWriteFile {
 public:
  BufferedWriteFile(
      std::unique_ptr<WriteFile> writeFile,
      uint64_t flushThresholdBytes,
      memory::MemoryPool* pool)
      : flushThresholdBytes_(flushThresholdBytes),
        pool_(pool),
        writeFile_(std::move(writeFile)),
        bufferStream_(std::make_unique<IOBufOutputStream>(*pool_)) {}

  ~BufferedWriteFile() {
    close();
  }

  void append(char* payload, int64_t bytes) {
    VELOX_CHECK_NOT_NULL(bufferStream_);
    if (flushThresholdBytes_ == 0) {
      // Bypass the copy if no buffer is intended.
      writeFile_->append(std::string_view(payload, bytes));
      return;
    }
    bufferStream_->write(payload, bytes);
    bufferBytes_ += bytes;
    checkFlush();
  }

  void append(std::unique_ptr<folly::IOBuf>&& iobuf) {
    VELOX_CHECK_NOT_NULL(bufferStream_);
    if (flushThresholdBytes_ == 0) {
      // Bypass the copy if no buffer is intended.
      writeFile_->append(std::move(iobuf));
      return;
    }
    for (auto range = iobuf->begin(); range != iobuf->end(); range++) {
      bufferStream_->write(
          reinterpret_cast<const char*>(range->data()), range->size());
    }
    bufferBytes_ += iobuf->computeChainDataLength();
    checkFlush();
  }

  void close() {
    auto iobuf = bufferStream_->getIOBuf();
    if (iobuf->computeChainDataLength() != 0) {
      writeFile_->append(std::move(iobuf));
    }
    writeFile_->close();
    bufferStream_.reset();
  }

 private:
  void checkFlush() {
    if (bufferBytes_ < flushThresholdBytes_) {
      return;
    }
    writeFile_->append(bufferStream_->getIOBuf());
    bufferStream_ = std::make_unique<IOBufOutputStream>(*pool_);
    bufferBytes_ = 0;
  }

  const uint64_t flushThresholdBytes_;
  memory::MemoryPool* const pool_;

  std::unique_ptr<WriteFile> writeFile_;
  std::unique_ptr<IOBufOutputStream> bufferStream_;
  uint64_t bufferBytes_{0};
};

} // namespace

void DestinationBuffer::BufferedPages::PageSpiller::spill() {
  if (pages_->empty()) {
    return;
  }

  auto [path, writeFile] = nextSpillWriteFile();
  BufferedWriteFile bufferedWriteFile(
      std::move(writeFile), writeBufferSize_, pool_);
  spillFilePaths_.push_back(path);

  // Spill file layout:
  // --- Payload 0 ---
  // (1B) is null page at 0
  // [
  //  (8B) payload size at 0
  //  (1B) has num rows at 0
  //  [(8B) num rows at 0]
  //  (xB) payload at 0
  // ]
  // --- Payload 1 ---
  //      ...
  // --- Payload n ---

  uint64_t spilledBytes{0};
  uint64_t spillWriteTimeNanos{0};
  const auto totalBytesBeforeSpill = totalBytes_;
  pageSizes_.reserve(pageSizes_.size() + pages_->size());
  {
    NanosecondTimer timer(&spillWriteTimeNanos);
    for (auto& page : *pages_) {
      // Fill spilled page metadata to keep in memory.
      if (page == nullptr) {
        pageSizes_.push_back(std::nullopt);
      } else {
        const auto pageSize = page->size();
        pageSizes_.push_back(pageSize);
        totalBytes_ += pageSize;
      }

      // Spill payload headers.
      uint8_t isNull = (page == nullptr) ? 1 : 0;
      bufferedWriteFile.append(
          reinterpret_cast<char*>(&isNull), sizeof(uint8_t));
      if (page == nullptr) {
        continue;
      }

      int64_t pageBytes = 0;
      if (page != nullptr) {
        pageBytes = page->size();
      }
      bufferedWriteFile.append(
          reinterpret_cast<char*>(&pageBytes), sizeof(int64_t));

      auto numRowsOpt = page->numRows();
      uint8_t hasNumRows = numRowsOpt.has_value() ? 1 : 0;
      bufferedWriteFile.append(reinterpret_cast<char*>(&hasNumRows), 1);
      if (numRowsOpt.has_value()) {
        int64_t numRows = numRowsOpt.value();
        bufferedWriteFile.append(
            reinterpret_cast<char*>(&numRows), sizeof(int64_t));
      }

      // Spill payload.
      bufferedWriteFile.append(page->getIOBuf());
      VELOX_CHECK_GE(totalBytes_, totalBytesBeforeSpill);
      spilledBytes += totalBytes_ - totalBytesBeforeSpill;
    }
  }
  auto spillStatsLocked = spillStats_->wlock();
  spillStatsLocked->spilledBytes += spilledBytes;
  spillStatsLocked->spilledFiles++;
  spillStatsLocked->spillWriteTimeNanos += spillWriteTimeNanos;
}

bool DestinationBuffer::BufferedPages::PageSpiller::empty() const {
  if (!bufferedPages_.empty()) {
    return false;
  }
  return curFileStream_ == nullptr && spillFilePaths_.empty();
}

void DestinationBuffer::BufferedPages::PageSpiller::ensureFileStream() {
  if (curFileStream_ != nullptr) {
    return;
  }
  VELOX_CHECK(!spillFilePaths_.empty());
  auto filePath = spillFilePaths_.front();
  auto fs = filesystems::getFileSystem(filePath, nullptr);
  auto file = fs->openFileForRead(filePath);
  curFileStream_ = std::make_unique<common::FileInputStream>(
      std::move(file), readBufferSize_, pool_);
  spillFilePaths_.pop_front();
}

namespace {
struct FreeData {
  std::shared_ptr<memory::MemoryPool> pool;
  int64_t bytesToFree;
};

void freeFunc(void* data, void* userData) {
  auto* freeData = reinterpret_cast<FreeData*>(userData);
  freeData->pool->free(data, freeData->bytesToFree);
  delete freeData;
}
} // namespace

std::shared_ptr<SerializedPage>
DestinationBuffer::BufferedPages::PageSpiller::unspillNextPage() {
  VELOX_CHECK(!empty());
  ensureFileStream();

  // Read payload headers
  auto isNull = !!(curFileStream_->read<uint8_t>());
  if (isNull) {
    if (curFileStream_->atEnd()) {
      curFileStream_.reset();
    }
    return nullptr;
  }
  auto iobufBytes = curFileStream_->read<int64_t>();
  auto hasNumRows = curFileStream_->read<uint8_t>() == 0 ? false : true;
  int64_t numRows{0};
  if (hasNumRows) {
    numRows = curFileStream_->read<int64_t>();
  }

  // Read payload
  VELOX_CHECK_GE(curFileStream_->remainingSize(), iobufBytes);
  void* rawBuf = pool_->allocate(iobufBytes);
  curFileStream_->readBytes(reinterpret_cast<uint8_t*>(rawBuf), iobufBytes);
  if (curFileStream_->atEnd()) {
    curFileStream_.reset();
  }

  auto* userData = new FreeData();
  userData->pool = pool_->shared_from_this();
  userData->bytesToFree = iobufBytes;
  auto iobuf =
      folly::IOBuf::takeOwnership(rawBuf, iobufBytes, freeFunc, userData, true);

  return std::make_shared<SerializedPage>(
      std::move(iobuf),
      nullptr,
      hasNumRows ? std::optional(numRows) : std::nullopt);
}

void DestinationBuffer::setupSpiller(
    memory::MemoryPool* pool,
    const common::SpillConfig* spillConfig,
    folly::Synchronized<common::SpillStats>* spillStats) {
  data_.setupSpiller(pool, spillConfig, destinationIdx_, spillStats);
}

void DestinationBuffer::BufferedPages::setupSpiller(
    memory::MemoryPool* pool,
    const common::SpillConfig* spillConfig,
    int32_t destinationIdx,
    folly::Synchronized<common::SpillStats>* spillStats) {
  auto spillDir = spillConfig->getSpillDirPathCb();
  VELOX_CHECK(!spillDir.empty(), "Spill directory does not exist");

  spiller_ = std::make_unique<DestinationBuffer::BufferedPages::PageSpiller>(
      &pages_,
      fmt::format(
          "{}/{}-dest-{}-spill",
          spillDir,
          spillConfig->fileNamePrefix,
          destinationIdx),
      spillConfig->fileCreateConfig,
      spillConfig->readBufferSize,
      spillConfig->writeBufferSize,
      pool,
      spillStats);
}

uint64_t DestinationBuffer::BufferedPages::size() const {
  return (spiller_ == nullptr ? 0 : spiller_->size()) + pages_.size();
}

std::shared_ptr<SerializedPage> DestinationBuffer::BufferedPages::at(
    uint64_t index) {
  VELOX_CHECK_LT(index, size());
  if (spiller_ == nullptr) {
    return pages_[index];
  }
  const auto numSpilledPages = spiller_->size();
  if (index >= numSpilledPages) {
    return pages_[index - numSpilledPages];
  }
  return spiller_->at(index);
}

bool DestinationBuffer::BufferedPages::isNullAt(uint64_t index) const {
  VELOX_CHECK_LT(index, size());
  if (spiller_ == nullptr) {
    return pages_[index] == nullptr;
  }
  const auto numSpilledPages = spiller_->size();
  if (index >= numSpilledPages) {
    return pages_[index - numSpilledPages] == nullptr;
  }
  return spiller_->isNullAt(index);
}

uint64_t DestinationBuffer::BufferedPages::sizeAt(uint64_t index) const {
  VELOX_CHECK_LT(index, size());
  if (spiller_ == nullptr) {
    VELOX_CHECK_NOT_NULL(pages_[index]);
    return pages_[index]->size();
  }
  const auto numSpilledPages = spiller_->size();
  if (index >= numSpilledPages) {
    VELOX_CHECK_NOT_NULL(pages_[index - numSpilledPages]);
    return pages_[index - numSpilledPages]->size();
  }
  return spiller_->sizeAt(index);
}

bool DestinationBuffer::BufferedPages::empty() const {
  return (spiller_ == nullptr || spiller_->empty()) && pages_.empty();
}

void DestinationBuffer::spill() {
  data_.spill();
  stats_.bytesSpilled = data_.spilledBytes();
}

void DestinationBuffer::BufferedPages::spill() {
  VELOX_CHECK_NOT_NULL(spiller_);
  spiller_->spill();
  pages_.clear();
}

uint64_t DestinationBuffer::BufferedPages::spilledBytes() const {
  return spiller_ == nullptr ? 0 : spiller_->totalBytes();
}

void DestinationBuffer::BufferedPages::append(
    std::shared_ptr<SerializedPage> page) {
  pages_.push_back(std::move(page));
}

void DestinationBuffer::BufferedPages::deleteFront(uint64_t numPages) {
  VELOX_CHECK_LE(numPages, size());
  if (spiller_ != nullptr) {
    const auto numSpillerPages = std::min(spiller_->size(), numPages);
    spiller_->deleteFront(numSpillerPages);
    numPages -= numSpillerPages;
    if (numPages == 0) {
      return;
    }
  }
  pages_.erase(pages_.begin(), pages_.begin() + numPages);
}

void DestinationBuffer::Stats::recordEnqueue(const SerializedPage& data) {
  const auto numRows = data.numRows();
  VELOX_CHECK(numRows.has_value(), "SerializedPage's numRows must be valid");
  bytesBuffered += data.size();
  rowsBuffered += numRows.value();
  ++pagesBuffered;
}

void DestinationBuffer::Stats::recordAcknowledge(const SerializedPage& data) {
  const auto numRows = data.numRows();
  VELOX_CHECK(numRows.has_value(), "SerializedPage's numRows must be valid");
  const int64_t size = data.size();
  bytesBuffered -= size;
  VELOX_DCHECK_GE(bytesBuffered, 0, "bytesBuffered must be non-negative");
  rowsBuffered -= numRows.value();
  VELOX_DCHECK_GE(rowsBuffered, 0, "rowsBuffered must be non-negative");
  --pagesBuffered;
  VELOX_DCHECK_GE(pagesBuffered, 0, "pagesBuffered must be non-negative");
  bytesSent += size;
  rowsSent += numRows.value();
  ++pagesSent;
}

void DestinationBuffer::Stats::recordDelete(const SerializedPage& data) {
  recordAcknowledge(data);
}

DestinationBuffer::Data DestinationBuffer::getData(
    uint64_t maxBytes,
    int64_t sequence,
    DataAvailableCallback notify,
    DataConsumerActiveCheckCallback activeCheck,
    ArbitraryBuffer* arbitraryBuffer) {
  VELOX_CHECK_GE(
      sequence, sequence_, "Get received for an already acknowledged item");
  if (arbitraryBuffer != nullptr) {
    loadData(arbitraryBuffer, maxBytes);
  }

  const auto totalPages = data_.size();
  if (sequence - sequence_ >= totalPages) {
    if (sequence - sequence_ > totalPages) {
      VLOG(1) << this << " Out of order get: " << sequence << " over "
              << sequence_ << " Setting second notify " << notifySequence_
              << " / " << sequence;
    }
    if (maxBytes == 0) {
      std::vector<int64_t> remainingBytes;
      if (arbitraryBuffer != nullptr) {
        arbitraryBuffer->getAvailablePageSizes(remainingBytes);
      }
      if (!remainingBytes.empty()) {
        return {{}, std::move(remainingBytes), true};
      }
    }
    notify_ = std::move(notify);
    aliveCheck_ = std::move(activeCheck);
    if (sequence - sequence_ > totalPages) {
      notifySequence_ = std::min(notifySequence_, sequence);
    } else {
      notifySequence_ = sequence;
    }
    notifyMaxBytes_ = maxBytes;
    return {};
  }

  std::vector<std::unique_ptr<folly::IOBuf>> data;
  uint64_t resultBytes = 0;
  auto i = sequence - sequence_;
  if (maxBytes > 0) {
    for (; i < totalPages; ++i) {
      // nullptr is used as end marker
      auto page = data_.at(i);
      if (page == nullptr) {
        VELOX_CHECK_EQ(i, totalPages - 1, "null marker found in the middle");
        data.push_back(nullptr);
        break;
      }
      data.push_back(page->getIOBuf());
      resultBytes += page->size();
      if (resultBytes >= maxBytes) {
        ++i;
        break;
      }
    }
  }
  bool atEnd = false;
  std::vector<int64_t> remainingBytes;
  remainingBytes.reserve(totalPages - i);
  for (; i < totalPages; ++i) {
    const auto page = data_.at(i);
    if (data_.isNullAt(i)) {
      VELOX_CHECK_EQ(i, totalPages - 1, "null marker found in the middle");
      atEnd = true;
      break;
    }
    remainingBytes.push_back(data_.sizeAt(i));
  }
  if (!atEnd && arbitraryBuffer != nullptr) {
    arbitraryBuffer->getAvailablePageSizes(remainingBytes);
  }
  if (data.empty() && remainingBytes.empty() && atEnd) {
    data.push_back(nullptr);
  }
  return {std::move(data), std::move(remainingBytes), true};
}

DestinationBuffer::DestinationBuffer(int32_t destinationIdx)
    : destinationIdx_(destinationIdx) {}

void DestinationBuffer::enqueue(std::shared_ptr<SerializedPage> data) {
  // Drop duplicate end markers.
  if (data == nullptr && !data_.empty() && data_.isNullAt(data_.size() - 1)) {
    return;
  }

  if (data != nullptr) {
    stats_.recordEnqueue(*data);
  }
  data_.append(std::move(data));
}

DataAvailable DestinationBuffer::getAndClearNotify() {
  if (notify_ == nullptr) {
    VELOX_CHECK_NULL(aliveCheck_);
    return DataAvailable();
  }
  DataAvailable result;
  result.callback = notify_;
  result.sequence = notifySequence_;
  auto data = getData(notifyMaxBytes_, notifySequence_, nullptr, nullptr);
  result.data = std::move(data.data);
  result.remainingBytes = std::move(data.remainingBytes);
  clearNotify();
  return result;
}

void DestinationBuffer::clearNotify() {
  notify_ = nullptr;
  aliveCheck_ = nullptr;
  notifySequence_ = 0;
  notifyMaxBytes_ = 0;
}

void DestinationBuffer::finish() {
  VELOX_CHECK_NULL(notify_, "notify must be cleared before finish");
  VELOX_CHECK(data_.empty(), "data must be fetched before finish");
  stats_.finished = true;
}

void DestinationBuffer::maybeLoadData(ArbitraryBuffer* buffer) {
  VELOX_CHECK(!buffer->empty() || buffer->hasNoMoreData());
  if (notify_ == nullptr) {
    return;
  }
  if (aliveCheck_ != nullptr && !aliveCheck_()) {
    // Skip load data to an inactive destination buffer.
    clearNotify();
    return;
  }
  loadData(buffer, notifyMaxBytes_);
}

void DestinationBuffer::loadData(ArbitraryBuffer* buffer, uint64_t maxBytes) {
  auto pages = buffer->getPages(maxBytes);
  for (auto& page : pages) {
    enqueue(std::move(page));
  }
}

std::vector<std::shared_ptr<SerializedPage>> DestinationBuffer::acknowledge(
    int64_t sequence,
    bool fromGetData) {
  const int64_t numDeleted = sequence - sequence_;
  if (numDeleted == 0 && fromGetData) {
    // If called from getData, it is expected that there will be
    // nothing to delete because a previous acknowledgement has been
    // received before the getData. This is not guaranteed though
    // because the messages may arrive out of order. Note that getData
    // implicitly acknowledges all messages with a lower sequence
    // number than the one in getData.
    return {};
  }
  if (numDeleted <= 0) {
    // Acknowledges come out of order, e.g. ack of 10 and 9 have
    // swapped places in flight.
    VLOG(1) << this << " Out of order ack: " << sequence << " over "
            << sequence_;
    return {};
  }

  const auto totalPages = data_.size();
  VELOX_CHECK_LE(
      numDeleted, totalPages, "Ack received for a not yet produced item");
  std::vector<std::shared_ptr<SerializedPage>> freed;
  for (auto i = 0; i < numDeleted; ++i) {
    const auto page = data_.at(i);
    if (page == nullptr) {
      VELOX_CHECK_EQ(i, totalPages - 1, "null marker found in the middle");
      break;
    }
    stats_.recordAcknowledge(*page);
    freed.push_back(std::move(page));
  }
  data_.deleteFront(numDeleted);
  stats_.bytesSpilled = data_.spilledBytes();
  sequence_ += numDeleted;
  return freed;
}

std::vector<std::shared_ptr<SerializedPage>>
DestinationBuffer::BufferedPages::deleteAll() {
  std::vector<std::shared_ptr<SerializedPage>> freed;
  if (spiller_ != nullptr) {
    spiller_->deleteAll();
  }
  for (auto i = 0; i < pages_.size(); ++i) {
    if (pages_[i] == nullptr) {
      VELOX_CHECK_EQ(i, pages_.size() - 1, "null marker found in the middle");
      break;
    }
    freed.push_back(std::move(pages_[i]));
  }
  pages_.clear();
  return freed;
}

std::vector<std::shared_ptr<SerializedPage>>
DestinationBuffer::deleteResults() {
  std::vector<std::shared_ptr<SerializedPage>> freed = data_.deleteAll();
  stats_.bytesSpilled = data_.spilledBytes();
  for (const auto& page : freed) {
    stats_.recordDelete(*page);
  }
  return freed;
}

DestinationBuffer::Stats DestinationBuffer::stats() const {
  return stats_;
}

std::string DestinationBuffer::toString() {
  std::stringstream out;
  out << "[available: " << data_.size() << ", " << "sequence: " << sequence_
      << ", " << (notify_ ? "notify registered, " : "") << this << "]";
  return out.str();
}

namespace {
// Frees 'freed' and realizes 'promises'. Used after
// updateAfterAcknowledgeLocked. This runs outside of the mutex, so
// that we do the expensive free outside and only then continue the
// producers which will allocate more memory.
void releaseAfterAcknowledge(
    std::vector<std::shared_ptr<SerializedPage>>& freed,
    std::vector<ContinuePromise>& promises) {
  freed.clear();
  for (auto& promise : promises) {
    promise.setValue();
  }
}

bool isPartitionedOutputPool(const memory::MemoryPool& pool) {
  return folly::StringPiece(pool.name()).endsWith("PartitionedOutput");
}
} // namespace

PartitionedOutputNodeReclaimer::PartitionedOutputNodeReclaimer(
    core::PartitionedOutputNode::Kind kind,
    int32_t priority)
    : MemoryReclaimer(priority), kind_(kind) {}

bool PartitionedOutputNodeReclaimer::reclaimableBytes(
    const memory::MemoryPool& pool,
    uint64_t& reclaimableBytes) const {
  reclaimableBytes = 0;
  if (kind_ != core::PartitionedOutputNode::Kind::kPartitioned) {
    return false;
  }
  reclaimableBytes = pool.reservedBytes();
  return true;
}

uint64_t PartitionedOutputNodeReclaimer::reclaim(
    memory::MemoryPool* pool,
    uint64_t targetBytes,
    uint64_t maxWaitMs,
    memory::MemoryReclaimer::Stats& stats) {
  const auto prevNodeReservedMemory = pool->reservedBytes();
  pool->visitChildren([&](memory::MemoryPool* child) {
    VELOX_CHECK_EQ(child->kind(), memory::MemoryPool::Kind::kLeaf);
    if (isPartitionedOutputPool(*child)) {
      child->reclaim(targetBytes, maxWaitMs, stats);
      return false;
    }
    return true;
  });
  return prevNodeReservedMemory - pool->reservedBytes();
}

OutputBuffer::OutputBuffer(
    std::shared_ptr<Task> task,
    PartitionedOutputNode::Kind kind,
    int numDestinations,
    uint32_t numDrivers,
    memory::MemoryPool* pool)
    : task_(std::move(task)),
      kind_(kind),
      maxSize_(task_->queryCtx()->queryConfig().maxOutputBufferSize()),
      continueSize_((maxSize_ * kContinuePct) / 100),
      arbitraryBuffer_(
          isArbitrary() ? std::make_unique<ArbitraryBuffer>() : nullptr),
      pool_(pool),
      numDrivers_(numDrivers) {
  buffers_.reserve(numDestinations);
  for (int i = 0; i < numDestinations; i++) {
    buffers_.push_back(createDestinationBuffer(i));
  }
  finishedBufferStats_.resize(numDestinations);
}

void OutputBuffer::updateOutputBuffers(int numBuffers, bool noMoreBuffers) {
  if (isPartitioned()) {
    VELOX_CHECK_EQ(buffers_.size(), numBuffers);
    VELOX_CHECK(noMoreBuffers);
    noMoreBuffers_ = true;
    return;
  }

  std::vector<ContinuePromise> promises;
  bool isFinished;
  {
    std::lock_guard<std::mutex> l(mutex_);

    if (numBuffers > buffers_.size()) {
      addOutputBuffersLocked(numBuffers);
    }

    if (!noMoreBuffers) {
      return;
    }

    noMoreBuffers_ = true;
    isFinished = isFinishedLocked();
    updateAfterAcknowledgeLocked(dataToBroadcast_, promises);
  }

  releaseAfterAcknowledge(dataToBroadcast_, promises);
  if (isFinished) {
    task_->setAllOutputConsumed();
  }
}

void OutputBuffer::updateNumDrivers(uint32_t newNumDrivers) {
  bool isNoMoreDrivers{false};
  {
    std::lock_guard<std::mutex> l(mutex_);
    numDrivers_ = newNumDrivers;
    // If we finished all drivers, ensure we register that we are 'done'.
    if (numDrivers_ == numFinished_) {
      isNoMoreDrivers = true;
    }
  }
  if (isNoMoreDrivers) {
    noMoreDrivers();
  }
}

std::unique_ptr<DestinationBuffer> OutputBuffer::createDestinationBuffer(
    int32_t destinationIdx) const {
  return std::make_unique<DestinationBuffer>(destinationIdx);
}

void OutputBuffer::addOutputBuffersLocked(int numBuffers) {
  VELOX_CHECK(!noMoreBuffers_);
  VELOX_CHECK(!isPartitioned());
  buffers_.reserve(numBuffers);
  for (int32_t i = buffers_.size(); i < numBuffers; ++i) {
    auto buffer = createDestinationBuffer(i);
    if (isBroadcast()) {
      for (const auto& data : dataToBroadcast_) {
        buffer->enqueue(data);
      }
      if (atEnd_) {
        buffer->enqueue(nullptr);
      }
    }
    buffers_.emplace_back(std::move(buffer));
  }
  finishedBufferStats_.resize(numBuffers);
}

void OutputBuffer::updateStatsWithEnqueuedPageLocked(
    int64_t pageBytes,
    int64_t pageRows) {
  updateTotalBufferedBytesMsLocked();

  bufferedBytes_ += pageBytes;
  ++bufferedPages_;

  ++numOutputPages_;
  numOutputRows_ += pageRows;
  numOutputBytes_ += pageBytes;
}

void OutputBuffer::updateStatsWithFreedPagesLocked(
    int numPages,
    int64_t pageBytes) {
  updateTotalBufferedBytesMsLocked();

  bufferedBytes_ -= pageBytes;
  VELOX_CHECK_GE(bufferedBytes_, 0);
  bufferedPages_ -= numPages;
  VELOX_CHECK_GE(bufferedPages_, 0);
}

void OutputBuffer::updateTotalBufferedBytesMsLocked() {
  const auto nowMs = getCurrentTimeMs();
  if (bufferedBytes_ > 0) {
    const auto deltaMs = nowMs - bufferStartMs_;
    totalBufferedBytesMs_ += bufferedBytes_ * deltaMs;
  }

  bufferStartMs_ = nowMs;
}

bool OutputBuffer::enqueue(
    int destination,
    std::unique_ptr<SerializedPage> data,
    ContinueFuture* future) {
  VELOX_CHECK_NOT_NULL(data);
  VELOX_CHECK(
      task_->isRunning(), "Task is terminated, cannot add data to output.");
  std::vector<DataAvailable> dataAvailableCallbacks;
  bool blocked = false;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_LT(destination, buffers_.size());

    updateStatsWithEnqueuedPageLocked(data->size(), data->numRows().value());

    switch (kind_) {
      case PartitionedOutputNode::Kind::kBroadcast:
        VELOX_CHECK_EQ(destination, 0, "Bad destination {}", destination);
        enqueueBroadcastOutputLocked(std::move(data), dataAvailableCallbacks);
        break;
      case PartitionedOutputNode::Kind::kArbitrary:
        VELOX_CHECK_EQ(destination, 0, "Bad destination {}", destination);
        enqueueArbitraryOutputLocked(std::move(data), dataAvailableCallbacks);
        break;
      case PartitionedOutputNode::Kind::kPartitioned:
        enqueuePartitionedOutputLocked(
            destination, std::move(data), dataAvailableCallbacks);
        break;
      default:
        VELOX_UNREACHABLE(PartitionedOutputNode::kindString(kind_));
    }

    if (spilled_ && bufferedBytes_ >= maxSize_) {
      reclaimLocked();
      // Skip notifying data availability below if this output buffer is in
      // spilled state.
      return false;
    }

    if (bufferedBytes_ >= maxSize_ && future) {
      promises_.emplace_back("OutputBuffer::enqueue");
      *future = promises_.back().getSemiFuture();
      blocked = true;
    }
  }

  // Outside mutex_.
  for (auto& callback : dataAvailableCallbacks) {
    callback.notify();
  }

  return blocked;
}

void OutputBuffer::enqueueBroadcastOutputLocked(
    std::unique_ptr<SerializedPage> data,
    std::vector<DataAvailable>& dataAvailableCbs) {
  VELOX_DCHECK(isBroadcast());
  VELOX_CHECK_NULL(arbitraryBuffer_);
  VELOX_DCHECK(dataAvailableCbs.empty());

  std::shared_ptr<SerializedPage> sharedData(data.release());
  for (auto& buffer : buffers_) {
    if (buffer != nullptr) {
      buffer->enqueue(sharedData);
      dataAvailableCbs.emplace_back(buffer->getAndClearNotify());
    }
  }

  // NOTE: we don't need to add new buffer to 'dataToBroadcast_' if there is no
  // more output buffers.
  if (!noMoreBuffers_) {
    dataToBroadcast_.emplace_back(sharedData);
  }
}

void OutputBuffer::enqueueArbitraryOutputLocked(
    std::unique_ptr<SerializedPage> data,
    std::vector<DataAvailable>& dataAvailableCbs) {
  VELOX_DCHECK(isArbitrary());
  VELOX_DCHECK_NOT_NULL(arbitraryBuffer_);
  VELOX_DCHECK(dataAvailableCbs.empty());
  VELOX_CHECK(!arbitraryBuffer_->hasNoMoreData());

  arbitraryBuffer_->enqueue(std::move(data));
  VELOX_CHECK_LT(nextArbitraryLoadBufferIndex_, buffers_.size());
  int32_t bufferId = nextArbitraryLoadBufferIndex_;
  for (int32_t i = 0; i < buffers_.size();
       ++i, bufferId = (bufferId + 1) % buffers_.size()) {
    if (arbitraryBuffer_->empty()) {
      nextArbitraryLoadBufferIndex_ = bufferId;
      break;
    }
    auto* buffer = buffers_[bufferId].get();
    if (buffer == nullptr) {
      continue;
    }
    buffer->maybeLoadData(arbitraryBuffer_.get());
    dataAvailableCbs.emplace_back(buffer->getAndClearNotify());
  }
}

void OutputBuffer::enqueuePartitionedOutputLocked(
    int destination,
    std::unique_ptr<SerializedPage> data,
    std::vector<DataAvailable>& dataAvailableCbs) {
  VELOX_DCHECK(isPartitioned());
  VELOX_CHECK_NULL(arbitraryBuffer_);
  VELOX_DCHECK(dataAvailableCbs.empty());

  VELOX_CHECK_LT(destination, buffers_.size());
  auto* buffer = buffers_[destination].get();
  if (buffer != nullptr) {
    buffer->enqueue(std::move(data));
    dataAvailableCbs.emplace_back(buffer->getAndClearNotify());
  } else {
    // Some downstream tasks may finish early and delete the corresponding
    // buffers. Further data for these buffers is dropped.
    updateStatsWithFreedPagesLocked(1, data->size());
  }
}

void OutputBuffer::noMoreData() {
  // Increment number of finished drivers.
  checkIfDone(true);
}

void OutputBuffer::noMoreDrivers() {
  // Do not increment number of finished drivers.
  checkIfDone(false);
}

void OutputBuffer::checkIfDone(bool oneDriverFinished) {
  std::vector<DataAvailable> finished;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (oneDriverFinished) {
      ++numFinished_;
    }
    VELOX_CHECK_LE(
        numFinished_,
        numDrivers_,
        "Each driver should call noMoreData exactly once");
    atEnd_ = numFinished_ == numDrivers_;
    if (!atEnd_) {
      return;
    }
    if (isArbitrary()) {
      arbitraryBuffer_->noMoreData();
      for (auto& buffer : buffers_) {
        if (buffer != nullptr) {
          buffer->maybeLoadData(arbitraryBuffer_.get());
          finished.push_back(buffer->getAndClearNotify());
        }
      }
    } else {
      for (auto& buffer : buffers_) {
        if (buffer != nullptr) {
          buffer->enqueue(nullptr);
          finished.push_back(buffer->getAndClearNotify());
        }
      }
      if (spilled_) {
        reclaimLocked();
      }
    }
  }

  // Notify outside of mutex.
  for (auto& notification : finished) {
    notification.notify();
  }
}

bool OutputBuffer::isFinished() {
  std::lock_guard<std::mutex> l(mutex_);
  return isFinishedLocked();
}

bool OutputBuffer::isFinishedLocked() {
  // NOTE: for broadcast output buffer, we can only mark it as finished after
  // receiving the no more (destination) buffers signal.
  if (isBroadcast() && !noMoreBuffers_) {
    return false;
  }
  for (auto& buffer : buffers_) {
    if (buffer != nullptr) {
      return false;
    }
  }
  return true;
}

void OutputBuffer::acknowledge(int destination, int64_t sequence) {
  std::vector<std::shared_ptr<SerializedPage>> freed;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_LT(destination, buffers_.size());
    auto* buffer = buffers_[destination].get();
    if (!buffer) {
      VLOG(1) << "Ack received after final ack for destination " << destination
              << " and sequence " << sequence;
      return;
    }
    freed = buffer->acknowledge(sequence, false);
    updateAfterAcknowledgeLocked(freed, promises);
  }
  releaseAfterAcknowledge(freed, promises);
}

void OutputBuffer::updateAfterAcknowledgeLocked(
    const std::vector<std::shared_ptr<SerializedPage>>& freed,
    std::vector<ContinuePromise>& promises) {
  uint64_t freedBytes{0};
  int freedPages{0};
  for (const auto& free : freed) {
    if (free.use_count() == 1) {
      ++freedPages;
      freedBytes += free->size();
    }
  }
  if (freedPages == 0) {
    VELOX_CHECK_EQ(freedBytes, 0);
    return;
  }
  VELOX_CHECK_GT(freedBytes, 0);

  updateStatsWithFreedPagesLocked(freedPages, freedBytes);

  if (bufferedBytes_ < continueSize_) {
    promises = std::move(promises_);
  }
}

bool OutputBuffer::deleteResults(int destination) {
  std::vector<std::shared_ptr<SerializedPage>> freed;
  std::vector<ContinuePromise> promises;
  bool isFinished;
  DataAvailable dataAvailable;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_LT(destination, buffers_.size());
    auto* buffer = buffers_[destination].get();
    if (buffer == nullptr) {
      VLOG(1) << "Extra delete received for destination " << destination;
      return false;
    }
    freed = buffer->deleteResults();
    dataAvailable = buffer->getAndClearNotify();
    buffer->finish();
    VELOX_CHECK_LT(destination, finishedBufferStats_.size());
    finishedBufferStats_[destination] = buffers_[destination]->stats();
    buffers_[destination] = nullptr;
    ++numFinalAcknowledges_;
    isFinished = isFinishedLocked();
    updateAfterAcknowledgeLocked(freed, promises);
  }

  // Outside of mutex.
  dataAvailable.notify();

  if (!promises.empty()) {
    VLOG(1) << "Delete of results unblocks producers. Can happen in early end "
            << "due to error or limit";
  }
  releaseAfterAcknowledge(freed, promises);
  if (isFinished) {
    task_->setAllOutputConsumed();
  }
  return isFinished;
}

void OutputBuffer::getData(
    int destination,
    uint64_t maxBytes,
    int64_t sequence,
    DataAvailableCallback notify,
    DataConsumerActiveCheckCallback activeCheck) {
  DestinationBuffer::Data data;
  std::vector<std::shared_ptr<SerializedPage>> freed;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);

    if (spilled_ && !atEnd_) {
      // If spilled, only start to respond with data when all output is produced
      // and spilled. Otherwise, let the request timeout.-
      return;
    }

    if (!isPartitioned() && destination >= buffers_.size()) {
      addOutputBuffersLocked(destination + 1);
    }

    VELOX_CHECK_LT(destination, buffers_.size());
    auto* buffer = buffers_[destination].get();
    if (buffer) {
      freed = buffer->acknowledge(sequence, true);
      updateAfterAcknowledgeLocked(freed, promises);
      data = buffer->getData(
          maxBytes, sequence, notify, activeCheck, arbitraryBuffer_.get());
    } else {
      data.data.emplace_back(nullptr);
      data.immediate = true;
      VLOG(1) << "getData received after deleteResults for destination "
              << destination << " and sequence " << sequence;
    }
  }
  releaseAfterAcknowledge(freed, promises);
  if (data.immediate) {
    notify(std::move(data.data), sequence, std::move(data.remainingBytes));
  }
}

void OutputBuffer::setupSpiller(
    const common::SpillConfig* spillConfig,
    folly::Synchronized<common::SpillStats>* spillStats) {
  if (spillConfig_.has_value()) {
    // Only need to be set once.
    return;
  }
  // A copy needs to be stored in output buffer because it out-lives
  // corresponding drivers, and hence the producing partitioned output
  // operators.
  spillConfig_ = *spillConfig;
  std::lock_guard<std::mutex> l(mutex_);
  for (auto& buffer : buffers_) {
    if (buffer != nullptr) {
      buffer->setupSpiller(pool_, spillConfig, spillStats);
    }
  }
}

bool OutputBuffer::canReclaim() const {
  // We only enable spilling for partitioned mode.
  return isPartitioned() && spillConfig_.has_value();
}

void OutputBuffer::reclaim() {
  std::lock_guard<std::mutex> l(mutex_);
  reclaimLocked();
}

void OutputBuffer::reclaimLocked() {
  VELOX_CHECK(canReclaim());
  VELOX_CHECK(isPartitioned());
  VELOX_CHECK(spillConfig_.has_value());
  spilled_ = true;

  struct Candidate {
    uint32_t destinationIdx;
    int64_t reclaimableBytes;
  };

  // Make reclaim order based on buffers' in-memory size, from high to low.
  std::vector<Candidate> candidates;
  candidates.reserve(buffers_.size());
  for (uint32_t i = 0; i < buffers_.size(); ++i) {
    if (buffers_[i] == nullptr) {
      continue;
    }
    const auto bufferStats = buffers_[i]->stats();
    const auto spillableBytes =
        bufferStats.bytesBuffered - bufferStats.bytesSpilled;
    VELOX_CHECK_GE(spillableBytes, 0);
    if (spillableBytes == 0) {
      continue;
    }
    candidates.push_back({i, spillableBytes});
  }
  if (candidates.empty()) {
    return;
  }

  std::sort(
      candidates.begin(),
      candidates.end(),
      [&](auto& lhsCandidate, auto& rhsCandidate) {
        return lhsCandidate.reclaimableBytes > rhsCandidate.reclaimableBytes;
      });

  struct SpillResult {
    const std::exception_ptr error{nullptr};

    explicit SpillResult(std::exception_ptr _error)
        : error(std::move(_error)) {}
  };

  auto* spillExecutor = spillConfig_->executor;
  std::vector<std::shared_ptr<AsyncSource<SpillResult>>> spillTasks;
  spillTasks.reserve(candidates.size());
  uint64_t spillBytes{0};
  for (const auto candidate : candidates) {
    if (candidate.reclaimableBytes == 0) {
      break;
    }
    spillTasks.push_back(memory::createAsyncMemoryReclaimTask<SpillResult>(
        [destinationIdx = candidate.destinationIdx,
         buffer = buffers_[candidate.destinationIdx].get()]() {
          try {
            buffer->spill();
            return std::make_unique<SpillResult>(nullptr);
          } catch (const std::exception& e) {
            LOG(ERROR) << "Reclaim from DestinationBuffer " << destinationIdx
                       << " failed: " << e.what();
            return std::make_unique<SpillResult>(std::current_exception());
          }
        }));
    if ((spillTasks.size() > 1) && (spillExecutor != nullptr)) {
      auto priority = spillExecutor->getNumPriorities();
      spillExecutor->add([source = spillTasks.back()]() { source->prepare(); });
    }
    spillBytes += candidate.reclaimableBytes;
  }

  SCOPE_EXIT {
    for (auto& spillTask : spillTasks) {
      // We consume the result for the pending tasks. This is a cleanup in the
      // guard and must not throw. The first error is already captured before
      // this runs.
      try {
        spillTask->move();
      } catch (const std::exception&) {
      }
    }
  };

  for (auto& spillTask : spillTasks) {
    const auto result = spillTask->move();
    if (result->error) {
      std::rethrow_exception(result->error);
    }
  }
}

void OutputBuffer::terminate() {
  VELOX_CHECK(!task_->isRunning());

  std::vector<ContinuePromise> outstandingPromises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    outstandingPromises.swap(promises_);
  }
  for (auto& promise : outstandingPromises) {
    promise.setValue();
  }
}

std::string OutputBuffer::toString() {
  std::lock_guard<std::mutex> l(mutex_);
  return toStringLocked();
}

std::string OutputBuffer::toStringLocked() const {
  std::stringstream out;
  out << "[OutputBuffer[" << kind_ << "] bufferedBytes_=" << bufferedBytes_
      << "b, num producers blocked=" << promises_.size()
      << ", completed=" << numFinished_ << "/" << numDrivers_ << ", "
      << (atEnd_ ? "at end, " : "") << "destinations: " << std::endl;
  for (auto i = 0; i < buffers_.size(); ++i) {
    auto buffer = buffers_[i].get();
    out << i << ": " << (buffer ? buffer->toString() : "none") << std::endl;
  }
  if (isArbitrary()) {
    out << arbitraryBuffer_->toString();
  }
  out << "]" << std::endl;
  return out.str();
}

double OutputBuffer::getUtilization() const {
  return bufferedBytes_ / static_cast<double>(maxSize_);
}

bool OutputBuffer::isOverUtilized() const {
  return (bufferedBytes_ > (0.5 * maxSize_)) || atEnd_;
}

int64_t OutputBuffer::getAverageBufferTimeMsLocked() const {
  if (numOutputBytes_ > 0) {
    return totalBufferedBytesMs_ / numOutputBytes_;
  }

  return 0;
}

namespace {

// Find out how many buffers hold 80% of the data. Useful to identify skew.
int32_t countTopBuffers(
    const std::vector<DestinationBuffer::Stats>& bufferStats,
    int64_t totalBytes) {
  std::vector<int64_t> bufferSizes;
  bufferSizes.reserve(bufferStats.size());
  for (auto i = 0; i < bufferStats.size(); ++i) {
    const auto& stats = bufferStats[i];
    bufferSizes.push_back(stats.bytesBuffered + stats.bytesSent);
  }

  // Sort descending.
  std::sort(bufferSizes.begin(), bufferSizes.end(), std::greater<int64_t>());

  const auto limit = totalBytes * 0.8;
  int32_t numBuffers = 0;
  int32_t runningTotal = 0;
  for (auto size : bufferSizes) {
    runningTotal += size;
    numBuffers++;

    if (runningTotal >= limit) {
      break;
    }
  }

  return numBuffers;
}

} // namespace

OutputBuffer::Stats OutputBuffer::stats() {
  std::lock_guard<std::mutex> l(mutex_);
  std::vector<DestinationBuffer::Stats> bufferStats;
  VELOX_CHECK_EQ(buffers_.size(), finishedBufferStats_.size());
  bufferStats.resize(buffers_.size());
  for (auto i = 0; i < buffers_.size(); ++i) {
    auto buffer = buffers_[i].get();
    if (buffer != nullptr) {
      bufferStats[i] = buffer->stats();
    } else {
      bufferStats[i] = finishedBufferStats_[i];
    }
  }

  updateTotalBufferedBytesMsLocked();

  return OutputBuffer::Stats(
      kind_,
      noMoreBuffers_,
      atEnd_,
      isFinishedLocked(),
      bufferedBytes_,
      bufferedPages_,
      numOutputBytes_,
      numOutputRows_,
      numOutputPages_,
      getAverageBufferTimeMsLocked(),
      countTopBuffers(bufferStats, numOutputBytes_),
      bufferStats);
}

} // namespace facebook::velox::exec
