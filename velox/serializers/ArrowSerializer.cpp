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
#include "velox/serializers/ArrowSerializer.h"

#include <cstdint>
#include <limits>

#include <arrow/buffer.h>
#include <arrow/c/bridge.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <folly/Range.h>
#include <folly/ScopeGuard.h>

#include "velox/vector/ComplexVector.h"
#include "velox/vector/arrow/Bridge.h"

namespace facebook::velox::serializer::arrow {
namespace {

// Adapts a Velox OutputStream to an arrow::io::OutputStream so that Arrow IPC
// writes flow directly into the Velox output without an intermediate buffer.
class VeloxArrowOutputStream : public ::arrow::io::OutputStream {
 public:
  explicit VeloxArrowOutputStream(velox::OutputStream* sink) : sink_(sink) {}

  ::arrow::Status Write(const void* data, int64_t nbytes) override {
    sink_->write(reinterpret_cast<const char*>(data), nbytes);
    position_ += nbytes;
    return ::arrow::Status::OK();
  }

  ::arrow::Result<int64_t> Tell() const override {
    return position_;
  }

  ::arrow::Status Close() override {
    closed_ = true;
    return ::arrow::Status::OK();
  }

  bool closed() const override {
    return closed_;
  }

 private:
  velox::OutputStream* sink_;
  int64_t position_{0};
  bool closed_{false};
};

// Extracts the specified row ranges from a RowVector into a new contiguous
// RowVector. Returns the input directly when ranges cover the full vector.
RowVectorPtr extractRanges(
    const RowVectorPtr& vector,
    const folly::Range<const IndexRange*>& ranges,
    memory::MemoryPool* pool) {
  if (ranges.size() == 1 && ranges[0].begin == 0 &&
      ranges[0].size == vector->size()) {
    return vector;
  }

  vector_size_t totalRows = 0;
  for (const auto& range : ranges) {
    totalRows += range.size;
  }

  auto rowType = asRowType(vector->type());
  const auto numChildren = vector->childrenSize();
  std::vector<VectorPtr> children;
  children.reserve(numChildren);
  for (int32_t i = 0; i < numChildren; ++i) {
    auto child = vector->childAt(i);
    auto newChild = BaseVector::create(child->type(), totalRows, pool);
    vector_size_t outIdx = 0;
    for (const auto& range : ranges) {
      newChild->copy(child.get(), outIdx, range.begin, range.size);
      outIdx += range.size;
    }
    children.push_back(std::move(newChild));
  }

  BufferPtr nulls;
  if (vector->mayHaveNulls()) {
    nulls = AlignedBuffer::allocate<bool>(totalRows, pool);
    auto rawNulls = nulls->asMutable<uint64_t>();
    vector_size_t outIdx = 0;
    for (const auto& range : ranges) {
      for (vector_size_t j = 0; j < range.size; ++j) {
        bits::setBit(rawNulls, outIdx++, !vector->isNullAt(range.begin + j));
      }
    }
  }

  return std::make_shared<RowVector>(
      pool, rowType, nulls, totalRows, std::move(children));
}

void serializeToArrowIpc(
    const RowVectorPtr& vector,
    const folly::Range<const IndexRange*>& ranges,
    memory::MemoryPool* pool,
    OutputStream* stream) {
  auto toSerialize = extractRanges(vector, ranges, pool);

  // Export Velox vector to Arrow C Data Interface (zero-copy for fixed-width).
  ArrowArray arrowArray{};
  ArrowSchema arrowSchema{};
  auto releaseArrowArray = folly::makeGuard([&]() {
    if (arrowArray.release) {
      arrowArray.release(&arrowArray);
    }
  });
  auto releaseArrowSchema = folly::makeGuard([&]() {
    if (arrowSchema.release) {
      arrowSchema.release(&arrowSchema);
    }
  });
  exportToArrow(toSerialize, arrowArray, pool);
  exportToArrow(toSerialize, arrowSchema);

  // Import into Arrow C++ RecordBatch (zero-copy wrapper).
  // ImportRecordBatch takes ownership on success.
  auto batch = ::arrow::ImportRecordBatch(&arrowArray, &arrowSchema);
  if (!batch.ok()) {
    VELOX_FAIL("Arrow ImportRecordBatch failed: {}", batch.status().ToString());
  }
  releaseArrowArray.dismiss();
  releaseArrowSchema.dismiss();

  // Write IPC stream directly to the Velox output stream.
  auto sink = std::make_shared<VeloxArrowOutputStream>(stream);
  auto writer = ::arrow::ipc::MakeStreamWriter(sink, (*batch)->schema());
  VELOX_CHECK(
      writer.ok(),
      "Arrow MakeStreamWriter failed: {}",
      writer.status().ToString());
  auto writeStatus = (*writer)->WriteRecordBatch(**batch);
  VELOX_CHECK(
      writeStatus.ok(),
      "Arrow WriteRecordBatch failed: {}",
      writeStatus.ToString());
  auto closeStatus = (*writer)->Close();
  VELOX_CHECK(
      closeStatus.ok(),
      "Arrow writer Close failed: {}",
      closeStatus.ToString());
}

// Batch serializer: single-shot serialize of a RowVector subset.
class ArrowBatchVectorSerializer : public BatchVectorSerializer {
 public:
  explicit ArrowBatchVectorSerializer(memory::MemoryPool* pool) : pool_(pool) {}

  void serialize(
      const RowVectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges,
      Scratch& /*scratch*/,
      OutputStream* stream) override {
    serializeToArrowIpc(vector, ranges, pool_, stream);
  }

  void estimateSerializedSize(
      VectorPtr /*vector*/,
      const folly::Range<const IndexRange*>& ranges,
      vector_size_t** sizes,
      Scratch& /*scratch*/) override {
    for (int32_t i = 0; i < static_cast<int32_t>(ranges.size()); ++i) {
      *sizes[i] += static_cast<vector_size_t>(ranges[i].size * sizeof(int64_t));
    }
  }

 private:
  memory::MemoryPool* pool_;
};

// Iterative serializer: accumulates rows across multiple append() calls,
// materializes into a single RowVector on flush().
class ArrowIterativeVectorSerializer : public IterativeVectorSerializer {
 public:
  ArrowIterativeVectorSerializer(
      RowTypePtr type,
      int32_t /*numRows*/,
      StreamArena* streamArena)
      : type_(std::move(type)), pool_(streamArena->pool()) {}

  void append(
      const RowVectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges,
      Scratch& /*scratch*/) override {
    for (const auto& range : ranges) {
      accumulated_.emplace_back(vector, range);
      numRows_ += range.size;
    }
  }

  size_t maxSerializedSize() const override {
    // Rough upper-bound estimate.
    return numRows_ * type_->size() * sizeof(int64_t) + 1'024;
  }

  void flush(OutputStream* stream) override {
    if (numRows_ == 0) {
      return;
    }

    auto& sourceVector = accumulated_.at(0).first;
    const auto numChildren = sourceVector->childrenSize();
    std::vector<VectorPtr> children;
    children.reserve(numChildren);
    for (int32_t col = 0; col < numChildren; ++col) {
      auto child = BaseVector::create(
          sourceVector->childAt(col)->type(), numRows_, pool_);
      vector_size_t outIdx = 0;
      for (const auto& [accVector, range] : accumulated_) {
        child->copy(
            accVector->childAt(col).get(), outIdx, range.begin, range.size);
        outIdx += range.size;
      }
      children.push_back(std::move(child));
    }

    // Arrow RecordBatch does not support top-level (struct-level) nulls.
    // RowVector nulls are not propagated; this matches the remote function
    // use case where the wrapping RowVector is never null at the row level.
    auto merged = std::make_shared<RowVector>(
        pool_, type_, BufferPtr{}, numRows_, std::move(children));

    IndexRange fullRange{0, numRows_};
    serializeToArrowIpc(merged, folly::Range(&fullRange, 1), pool_, stream);

    accumulated_.clear();
    numRows_ = 0;
  }

 private:
  RowTypePtr type_;
  memory::MemoryPool* pool_;
  std::vector<std::pair<RowVectorPtr, IndexRange>> accumulated_;
  vector_size_t numRows_{0};
};

} // namespace

std::unique_ptr<IterativeVectorSerializer>
ArrowVectorSerde::createIterativeSerializer(
    RowTypePtr type,
    int32_t numRows,
    StreamArena* streamArena,
    const Options* /*options*/) {
  return std::make_unique<ArrowIterativeVectorSerializer>(
      std::move(type), numRows, streamArena);
}

std::unique_ptr<BatchVectorSerializer> ArrowVectorSerde::createBatchSerializer(
    memory::MemoryPool* pool,
    const Options* /*options*/) {
  return std::make_unique<ArrowBatchVectorSerializer>(pool);
}

namespace {

// Parses an Arrow IPC stream from an arrow::Buffer and produces a Velox
// RowVector. The buffer's data stays alive via the Arrow ownership chain:
// arrow::Buffer → RecordBatch (SliceBuffer) → ArrowArray → BufferView.
RowVectorPtr deserializeFromArrowBuffer(
    const std::shared_ptr<::arrow::Buffer>& arrowBuffer,
    velox::memory::MemoryPool* pool,
    const RowTypePtr& type) {
  auto bufferReader = std::make_shared<::arrow::io::BufferReader>(arrowBuffer);

  auto reader = ::arrow::ipc::RecordBatchStreamReader::Open(bufferReader);
  VELOX_CHECK(
      reader.ok(),
      "Arrow IPC reader open failed: {}",
      reader.status().ToString());

  std::shared_ptr<::arrow::RecordBatch> batch;
  auto readStatus = (*reader)->ReadNext(&batch);
  VELOX_CHECK(
      readStatus.ok(), "Arrow IPC ReadNext failed: {}", readStatus.ToString());
  VELOX_CHECK_NOT_NULL(batch, "Arrow IPC stream contained no record batches.");

  ArrowArray arrowArray{};
  ArrowSchema arrowSchema{};
  auto exportStatus =
      ::arrow::ExportRecordBatch(*batch, &arrowArray, &arrowSchema);
  if (!exportStatus.ok()) {
    if (arrowArray.release) {
      arrowArray.release(&arrowArray);
    }
    if (arrowSchema.release) {
      arrowSchema.release(&arrowSchema);
    }
    VELOX_FAIL("Arrow ExportRecordBatch failed: {}", exportStatus.ToString());
  }

  // importFromArrowAsOwner takes ownership of arrowArray and arrowSchema.
  auto imported = importFromArrowAsOwner(arrowSchema, arrowArray, pool);
  auto importedRow = std::dynamic_pointer_cast<RowVector>(imported);
  VELOX_CHECK_NOT_NULL(
      importedRow, "Arrow IPC deserialized data is not a RowVector.");

  return std::make_shared<RowVector>(
      pool,
      type,
      importedRow->nulls(),
      importedRow->size(),
      importedRow->children());
}

// Wraps a folly::IOBuf as an arrow::Buffer. Coalesces if fragmented (zero-copy
// if already contiguous). Captures the IOBuf to keep the memory alive.
// The exposed data_/size_ pointers are valid for the lifetime of this object.
// Thread safety: not thread-safe; callers must not access the buffer
// concurrently with destruction.
class IOBufArrowBuffer : public ::arrow::Buffer {
 public:
  explicit IOBufArrowBuffer(const folly::IOBuf& iobuf)
      : ::arrow::Buffer(nullptr, 0),
        ownedBuf_(iobuf.cloneCoalescedAsValueWithHeadroomTailroom(0, 0)) {
    data_ = ownedBuf_.data();
    size_ = static_cast<int64_t>(ownedBuf_.length());
    capacity_ = size_;
  }

 private:
  folly::IOBuf ownedBuf_;
};

std::shared_ptr<::arrow::Buffer> wrapIOBufAsArrowBuffer(
    const folly::IOBuf& iobuf) {
  return std::make_shared<IOBufArrowBuffer>(iobuf);
}

} // namespace

void ArrowVectorSerde::deserialize(
    ByteInputStream* source,
    velox::memory::MemoryPool* pool,
    RowTypePtr type,
    RowVectorPtr* result,
    const Options* /*options*/) {
  const auto totalSize = source->remainingSize();
  VELOX_CHECK_GT(totalSize, 0, "Empty input to Arrow IPC deserializer.");
  // readBytes() takes int32_t, so cap at INT32_MAX (~2 GB).
  constexpr size_t kMaxArrowIpcPayloadSize =
      static_cast<size_t>(std::numeric_limits<int32_t>::max());
  VELOX_CHECK_LE(
      totalSize,
      kMaxArrowIpcPayloadSize,
      "Arrow IPC payload too large: {} bytes.",
      totalSize);

  std::string data;
  data.resize(totalSize);
  source->readBytes(
      reinterpret_cast<uint8_t*>(data.data()), static_cast<int32_t>(totalSize));

  auto arrowBuffer = ::arrow::Buffer::FromString(std::move(data));
  *result = deserializeFromArrowBuffer(arrowBuffer, pool, type);
}

void ArrowVectorSerde::deserialize(
    const folly::IOBuf& source,
    velox::memory::MemoryPool* pool,
    RowTypePtr type,
    RowVectorPtr* result,
    const Options* /*options*/) {
  VELOX_CHECK_GT(
      source.computeChainDataLength(),
      0,
      "Empty input to Arrow IPC deserializer.");

  // Wrap IOBuf as arrow::Buffer (zero-copy for contiguous IOBufs).
  auto arrowBuffer = wrapIOBufAsArrowBuffer(source);
  *result = deserializeFromArrowBuffer(arrowBuffer, pool, type);
}

void ArrowVectorSerde::registerVectorSerde() {
  velox::registerVectorSerde(std::make_unique<ArrowVectorSerde>());
}

void ArrowVectorSerde::registerNamedVectorSerde() {
  velox::registerNamedVectorSerde(
      std::string(kSerdeName), std::make_unique<ArrowVectorSerde>());
}

void ArrowVectorSerde::tryRegisterNamedVectorSerde() {
  if (!velox::isRegisteredNamedVectorSerde(std::string(kSerdeName))) {
    registerNamedVectorSerde();
  }
}

void ArrowVectorSerde::registerVectorSerdeFactory() {
  velox::registerVectorSerdeFactory(std::string(kSerdeName), [] {
    return std::make_unique<ArrowVectorSerde>();
  });
}

} // namespace facebook::velox::serializer::arrow
