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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfPrestoOutput.h"
#include "velox/experimental/cudf/exec/PrestoPageDevice.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/compression/Compression.h"
#include "velox/common/time/Timer.h"
#include "velox/exec/OutputTransportRegistry.h"
#include "velox/exec/SerializedPage.h"
#include "velox/exec/Task.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/serializers/PrestoSerializerSerializationUtils.h"
#include "velox/vector/VectorStream.h"

#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/pinned_memory.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/aligned.hpp>

#include <cuda_runtime.h>

#include <folly/io/IOBuf.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>

namespace facebook::velox::cudf_velox {
namespace {

using serializer::presto::PrestoOutputStreamListener;
using serializer::presto::detail::kCheckSumBitMask;
using serializer::presto::detail::kHeaderSize;

// Header fields set on the host when the pages are checksummed.
constexpr int32_t kCodecMarkerOffset = sizeof(int32_t);
constexpr int32_t kChecksumOffset = kHeaderSize - sizeof(int64_t);
// A wave of pages is laid out on the device at once and copied to the host in
// batches. Both are a number of pages, bounded in bytes.
constexpr int64_t kPagesPerWave = 32;
constexpr int64_t kMaxWaveBytes = 256LL << 20;
constexpr int64_t kPagesPerHostBatch = 8;
constexpr int64_t kMaxHostBatchBytes = 64LL << 20;
constexpr size_t kPinnedAlignment = rmm::CUDA_ALLOCATION_ALIGNMENT;

/// Size in bytes of a value of 'type' in its Presto block, 0 for the
/// VARIABLE_WIDTH encoding, nullopt when the type has no flat encoding.
std::optional<int32_t> flatWidth(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
      return 1;
    case TypeKind::SMALLINT:
      return 2;
    case TypeKind::INTEGER:
    case TypeKind::REAL:
      return 4;
    case TypeKind::BIGINT:
    case TypeKind::DOUBLE:
    case TypeKind::TIMESTAMP:
      return 8;
    case TypeKind::HUGEINT:
      return 16;
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      return 0;
    default:
      return std::nullopt;
  }
}

// Scale from the cuDF timestamp unit to Presto milliseconds.
bool timestampScale(cudf::type_id id, int64_t& mul, int64_t& div) {
  mul = 1;
  div = 1;
  switch (id) {
    case cudf::type_id::TIMESTAMP_SECONDS:
      mul = 1'000;
      return true;
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return true;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      div = 1'000;
      return true;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      div = 1'000'000;
      return true;
    default:
      return false;
  }
}

int64_t pageBytesFor(const core::QueryConfig& config) {
  const auto pageBytes = config.get<int64_t>(
      CudfConfig::kCudfPrestoOutputPageBytes,
      CudfConfig::getInstance().prestoOutputPageBytes);
  VELOX_USER_CHECK_GT(
      pageBytes,
      0,
      "{} must be positive",
      CudfConfig::kCudfPrestoOutputPageBytes);
  VELOX_USER_CHECK_LE(
      pageBytes,
      std::numeric_limits<int32_t>::max() / 2,
      "{} must fit a Presto page",
      CudfConfig::kCudfPrestoOutputPageBytes);
  return pageBytes;
}

// Frees a page's pinned memory once the IOBuf is released, which happens on
// the consumer's thread long after the operator may be gone. Holds the task
// meanwhile, as the bufferReleaseFn of exec::PartitionedOutput does.
struct PageRelease {
  size_t bytes;
  rmm::host_device_async_resource_ref mr;
  std::shared_ptr<exec::Task> task;
};

void releasePage(void* data, void* userData) {
  auto* release = static_cast<PageRelease*>(userData);
  try {
    release->mr.deallocate_sync(data, release->bytes, kPinnedAlignment);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to free a pinned Presto page: " << e.what();
  }
  delete release;
}

int64_t nowNanos() {
  return static_cast<int64_t>(SteadyTime::now<std::chrono::nanoseconds>());
}

template <typename T>
void putValue(std::vector<uint8_t>& out, T value) {
  uint8_t bytes[sizeof(T)];
  std::memcpy(bytes, &value, sizeof(T));
  out.insert(out.end(), bytes, bytes + sizeof(T));
}

} // namespace

// Device layout of one output column of the table being serialized.
struct CudfPrestoOutput::Column {
  std::string_view encodingName;
  // Value size in bytes, 0 for strings.
  int32_t width{0};
  // Fixed width: values of row 0 on the device; dense valid rows when
  // 'compacted'.
  const uint8_t* values{nullptr};
  bool compacted{false};
  // Strings: chars addressed by the absolute cuDF offsets.
  const char* chars{nullptr};
  detail::StringOffsets offsets{};
  // Null handling; 'mask' is null when the column has no nulls.
  const cudf::bitmask_type* mask{nullptr};
  int64_t firstBit{0};
  std::vector<int32_t> pageNulls;
  std::vector<int64_t> validBefore;
  // Strings: absolute chars offset at each page boundary (numPages + 1).
  std::vector<int64_t> charsBounds;
  // Owned conversions: a cast or purged column, and dense value buffers.
  std::unique_ptr<cudf::column> owned;
  rmm::device_buffer dense;

  bool isString() const {
    return width == 0;
  }
};

// Serialization state of one table.
struct CudfPrestoOutput::Serializer {
  explicit Serializer(cuda::stream_ref stream_) : stream(stream_) {}

  cuda::stream_ref stream;
  // At most one of the two owns the table; neither does for zero columns.
  CudfVectorPtr input;
  std::unique_ptr<cudf::table> table;
  int32_t numRows{0};
  std::vector<Column> columns;
  // Row boundaries of the pages (numPages + 1).
  std::vector<int64_t> bounds;
  std::vector<int64_t> pageSizes;
  size_t nextPage{0};

  size_t numPages() const {
    return pageSizes.size();
  }

  bool done() const {
    return nextPage >= pageSizes.size();
  }
};

CudfPrestoOutput::CudfPrestoOutput(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
    bool eagerFlush,
    const std::shared_ptr<exec::DefaultOutputBufferManager>& bufferManager)
    : CudfOperatorBase(
          operatorId,
          ctx,
          planNode->outputType(),
          planNode->id(),
          "CudfPrestoOutput",
          nvtx3::rgb{148, 0, 211}, // Purple
          NvtxMethodFlag::kAll,
          std::nullopt,
          planNode),
      bufferManager_(bufferManager),
      inputType_(planNode->inputType()),
      outputChannels_(
          exec::calculateOutputChannels(
              planNode->inputType(),
              planNode->outputType(),
              planNode->outputType())),
      pageBytes_(pageBytesFor(ctx->queryConfig())),
      waveBytes_(std::min(kMaxWaveBytes, kPagesPerWave * pageBytes_)),
      hostBatchBytes_(
          std::min(kMaxHostBatchBytes, kPagesPerHostBatch * pageBytes_)),
      eagerFlush_(eagerFlush),
      listener_(
          bufferManager == nullptr ? nullptr : bufferManager->newListener()),
      checksumListener_(
          dynamic_cast<PrestoOutputStreamListener*>(listener_.get())),
      pinnedMr_(cudf::get_pinned_memory_resource()) {
  VELOX_CHECK_NOT_NULL(
      bufferManager, "CudfPrestoOutput requires an output buffer manager");
  VELOX_CHECK_EQ(planNode->numPartitions(), 1);
}

CudfPrestoOutput::~CudfPrestoOutput() = default;

// static
std::shared_ptr<exec::DefaultOutputBufferManager>
CudfPrestoOutput::bufferManagerFor(
    const core::PartitionedOutputNode& node,
    const core::QueryCtx& queryCtx) {
  auto entry =
      exec::OutputTransportRegistry::tryGet(queryCtx, node.transportKind());
  if (entry == nullptr) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<exec::DefaultOutputBufferManager>(
      entry->manager);
}

// static
bool CudfPrestoOutput::isEnabled(const core::QueryConfig& config) {
  return config.get<bool>(
      CudfConfig::kCudfPrestoOutputEnabled,
      CudfConfig::getInstance().prestoOutputEnabled);
}

// static
std::optional<std::string> CudfPrestoOutput::cpuFallbackReason(
    const core::PartitionedOutputNode& node,
    const core::QueryCtx& queryCtx) {
  const auto& config = queryCtx.queryConfig();
  if (!isEnabled(config)) {
    return fmt::format("disabled by {}", CudfConfig::kCudfPrestoOutputEnabled);
  }
  if (node.numPartitions() != 1 || !node.keys().empty()) {
    return fmt::format(
        "output partitioned across {} destinations", node.numPartitions());
  }
  if (node.serdeKind() != VectorSerde::kindName(VectorSerde::Kind::kPresto)) {
    return fmt::format("{} serde", node.serdeKind());
  }
  const auto compression = config.shuffleCompressionKind();
  if (common::stringToCompressionKind(compression) !=
      common::CompressionKind::CompressionKind_NONE) {
    return fmt::format("shuffle compression {}", compression);
  }
  if (bufferManagerFor(node, queryCtx) == nullptr) {
    return fmt::format(
        "output transport {} without the default output buffer manager",
        node.transportKind());
  }
  for (const auto& type : node.outputType()->children()) {
    if (!isTypeSupportedByCudf(type)) {
      return fmt::format("cuDF cannot hold {}", type->toString());
    }
    if (!flatWidth(type).has_value()) {
      return fmt::format("{} has no flat Presto encoding", type->toString());
    }
  }
  return std::nullopt;
}

void CudfPrestoOutput::doAddInput(RowVectorPtr input) {
  VELOX_CHECK_EQ(
      blockingReason_,
      exec::BlockingReason::kNotBlocked,
      "addInput while blocked on the consumer");
  if (input->size() == 0) {
    return;
  }
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput, "CudfPrestoOutput expects GPU input");
  pendingRows_ += cudfInput->size();
  pendingBytes_ += cudfInput->estimateFlatSize();
  pendingInputs_.push_back(std::move(cudfInput));
}

RowVectorPtr CudfPrestoOutput::doGetOutput() {
  drain();
  return nullptr;
}

exec::BlockingReason CudfPrestoOutput::isBlocked(ContinueFuture* future) {
  if (blockingReason_ != exec::BlockingReason::kNotBlocked) {
    *future = std::move(future_);
    blockingReason_ = exec::BlockingReason::kNotBlocked;
    return exec::BlockingReason::kWaitForConsumer;
  }
  return exec::BlockingReason::kNotBlocked;
}

void CudfPrestoOutput::startSerializer() {
  VELOX_CHECK(!pendingInputs_.empty());
  auto serializer =
      std::make_unique<Serializer>(pendingInputs_.front()->stream());
  auto mr = get_temp_mr();
  const auto numColumns = outputType_->size();

  VELOX_CHECK_LE(
      pendingRows_,
      std::numeric_limits<int32_t>::max(),
      "Presto output batch exceeds the cuDF row limit");
  serializer->numRows = static_cast<int32_t>(pendingRows_);

  cudf::table_view view;
  if (numColumns == 0) {
    // The pages carry only row counts.
  } else if (pendingInputs_.size() == 1) {
    serializer->input = std::move(pendingInputs_.front());
    view = serializer->input->getTableView();
  } else {
    serializer->table = getConcatenatedTable(
        std::move(pendingInputs_), inputType_, serializer->stream, mr);
    view = serializer->table->view();
  }
  pendingInputs_.clear();
  pendingRows_ = 0;
  pendingBytes_ = 0;
  if (numColumns > 0 && !outputChannels_.empty()) {
    view = view.select(outputChannels_.begin(), outputChannels_.end());
  }

  auto stream = serializer->stream;
  const int32_t numRows = serializer->numRows;
  VELOX_CHECK_EQ(static_cast<size_t>(view.num_columns()), numColumns);
  serializer->columns.resize(numColumns);

  // Serialized bits per row apart from string bytes: values, offsets and null
  // bits. Page boundaries follow from them and the string lengths.
  int64_t fixedBitsPerRow = 0;
  std::vector<detail::StringOffsets> stringOffsets;
  for (size_t i = 0; i < numColumns; ++i) {
    auto col = view.column(i);
    auto& column = serializer->columns[i];
    const auto& type = outputType_->childAt(i);
    const auto width = flatWidth(type);
    VELOX_CHECK(
        width.has_value(), "No flat Presto encoding for {}", type->toString());
    column.width = *width;
    column.encodingName = serializer::presto::detail::typeToEncodingName(type);
    VELOX_CHECK_EQ(col.size(), numRows);

    if (column.isString()) {
      VELOX_CHECK_EQ(
          static_cast<int32_t>(col.type().id()),
          static_cast<int32_t>(cudf::type_id::STRING),
          "Expected a cuDF STRING column for {}",
          type->toString());
      // Presto repeats the previous offset for a null string, so the bytes of
      // null rows must go.
      if (col.null_count() > 0 && cudf::may_have_nonempty_nulls(col) &&
          cudf::has_nonempty_nulls(col, stream)) {
        column.owned = cudf::purge_nonempty_nulls(col, stream, mr);
        col = column.owned->view();
      }
      cudf::strings_column_view strings(col);
      auto offsetsView = strings.offsets();
      column.offsets = {
          offsetsView.head(),
          offsetsView.type().id() == cudf::type_id::INT64,
          col.offset()};
      column.chars = strings.chars_begin(stream);
      stringOffsets.push_back(column.offsets);
      fixedBitsPerRow += 8 * sizeof(int32_t);
    } else {
      // Bring the values to the representation cuDF uses for the Velox type,
      // e.g. a compact DECIMAL32 reader column to DECIMAL64.
      if (type->kind() != TypeKind::TIMESTAMP &&
          static_cast<int32_t>(cudf::size_of(col.type())) != column.width) {
        column.owned = cudf::cast(col, veloxToCudfDataType(type), stream, mr);
        col = column.owned->view();
      }
      VELOX_CHECK_EQ(
          static_cast<int32_t>(cudf::size_of(col.type())),
          column.width,
          "Unexpected cuDF value width for {}",
          type->toString());
      const uint8_t* raw = col.head<uint8_t>() +
          static_cast<int64_t>(col.offset()) * column.width;
      int32_t numValues = numRows;
      if (col.null_count() > 0) {
        // Presto blocks hold the values of the non-null rows only. The values
        // of null rows are undefined, so nothing below may read them.
        numValues = numRows - col.null_count();
        rmm::device_buffer dense(
            static_cast<size_t>(numValues) * column.width, stream, mr);
        const auto copied = detail::compactValid(
            raw,
            column.width,
            col.null_mask(),
            col.offset(),
            numRows,
            static_cast<uint8_t*>(dense.data()),
            stream);
        VELOX_CHECK_EQ(copied, numValues);
        column.dense = std::move(dense);
        raw = static_cast<const uint8_t*>(column.dense.data());
        column.compacted = true;
      }
      // Converted values go into the compacted copy in place, or into a copy
      // of their own.
      auto converted = [&]() -> uint8_t* {
        if (!column.compacted) {
          column.dense = rmm::device_buffer(
              static_cast<size_t>(numValues) * column.width, stream, mr);
        }
        return static_cast<uint8_t*>(column.dense.data());
      };
      if (type->kind() == TypeKind::TIMESTAMP) {
        int64_t mul;
        int64_t div;
        VELOX_CHECK(
            timestampScale(col.type().id(), mul, div),
            "Expected a cuDF timestamp column for {}",
            type->toString());
        if (mul != 1 || div != 1) {
          auto* millis = reinterpret_cast<int64_t*>(converted());
          VELOX_USER_CHECK(
              detail::timestampsToMillis(
                  reinterpret_cast<const int64_t*>(raw),
                  numValues,
                  mul,
                  div,
                  millis,
                  stream,
                  mr),
              "Could not convert the TIMESTAMP values of {} to milliseconds",
              outputType_->nameOf(i));
          raw = reinterpret_cast<const uint8_t*>(millis);
        }
      } else if (type->kind() == TypeKind::HUGEINT) {
        auto* java = reinterpret_cast<__int128_t*>(converted());
        detail::decimalsToJava(
            reinterpret_cast<const __int128_t*>(raw), numValues, java, stream);
        raw = reinterpret_cast<const uint8_t*>(java);
      }
      fixedBitsPerRow += 8 * column.width;
      column.values = raw;
    }

    if (col.null_count() > 0) {
      column.mask = col.null_mask();
      column.firstBit = col.offset();
      fixedBitsPerRow += 1;
    }
  }

  serializer->bounds = detail::prestoPageBoundaries(
      numRows,
      fixedBitsPerRow,
      stringOffsets,
      pageBytes_,
      kMaxPageRows,
      stream,
      mr);
  const auto& bounds = serializer->bounds;
  const size_t numPages = bounds.size() - 1;

  for (auto& column : serializer->columns) {
    if (column.mask != nullptr) {
      std::vector<cudf::size_type> ranges;
      ranges.reserve(2 * numPages);
      for (size_t p = 0; p < numPages; ++p) {
        ranges.push_back(
            static_cast<cudf::size_type>(column.firstBit + bounds[p]));
        ranges.push_back(
            static_cast<cudf::size_type>(column.firstBit + bounds[p + 1]));
      }
      column.pageNulls =
          cudf::segmented_null_count(column.mask, ranges, stream);
      VELOX_CHECK_EQ(column.pageNulls.size(), numPages);
      if (!column.isString()) {
        column.validBefore.resize(numPages);
        int64_t valid = 0;
        for (size_t p = 0; p < numPages; ++p) {
          column.validBefore[p] = valid;
          valid += (bounds[p + 1] - bounds[p]) - column.pageNulls[p];
        }
      }
    }
    if (column.isString()) {
      column.charsBounds =
          detail::gatherStringOffsets(column.offsets, bounds, stream, mr);
    }
  }

  serializer->pageSizes.resize(numPages);
  for (size_t p = 0; p < numPages; ++p) {
    const int64_t rows = bounds[p + 1] - bounds[p];
    int64_t size = kHeaderSize + sizeof(int32_t);
    for (const auto& column : serializer->columns) {
      const int32_t nulls = column.mask != nullptr ? column.pageNulls[p] : 0;
      // Encoding name, position count and the has-nulls byte.
      size +=
          sizeof(int32_t) + column.encodingName.size() + sizeof(int32_t) + 1;
      if (nulls > 0) {
        size += (rows + 7) / 8;
      }
      if (column.isString()) {
        size += sizeof(int32_t) * rows + sizeof(int32_t) +
            (column.charsBounds[p + 1] - column.charsBounds[p]);
      } else {
        size += (rows - nulls) * column.width;
      }
    }
    VELOX_CHECK_LE(
        size,
        std::numeric_limits<int32_t>::max(),
        "A Presto page of {} rows exceeds the int32 size limit; lower {}",
        rows,
        CudfConfig::kCudfPrestoOutputPageBytes);
    serializer->pageSizes[p] = size;
  }

  serializer_ = std::move(serializer);
}

void CudfPrestoOutput::buildNextWave() {
  auto& serializer = *serializer_;
  auto stream = serializer.stream;
  auto mr = get_temp_mr();

  const size_t first = serializer.nextPage;
  size_t last = first;
  int64_t total = 0;
  while (last < serializer.numPages() &&
         (last == first || total + serializer.pageSizes[last] <= waveBytes_)) {
    total += serializer.pageSizes[last];
    ++last;
  }

  Wave wave;
  wave.data = rmm::device_buffer(static_cast<size_t>(total), stream, mr);
  auto* base = static_cast<uint8_t*>(wave.data.data());

  // Small fields are assembled on the host, copied to the device once and
  // scattered by the batched copy together with the chars and value ranges.
  struct BlobCopy {
    size_t blobOffset;
    size_t dstOffset;
    size_t size;
  };
  std::vector<uint8_t> blob;
  std::vector<BlobCopy> blobCopies;
  std::vector<detail::CopyTask> copies;
  std::vector<detail::NullBytesTask> nullTasks;
  std::vector<detail::StringOffsetsTask> offsetTasks;
  const auto numColumns = static_cast<int32_t>(serializer.columns.size());
  blob.reserve((last - first) * (32 + 32 * numColumns));

  size_t waveOffset = 0;
  for (size_t p = first; p < last; ++p) {
    const auto begin = serializer.bounds[p];
    const auto rows =
        static_cast<int32_t>(serializer.bounds[p + 1] - serializer.bounds[p]);
    const int64_t pageSize = serializer.pageSizes[p];
    uint8_t* page = base + waveOffset;
    size_t pos = 0;

    size_t blobStart = blob.size();
    putValue<int32_t>(blob, rows);
    // No compression. The codec marker and the checksum stay zero unless
    // checksumPage() sets them on the host.
    putValue<int8_t>(blob, 0);
    putValue<int32_t>(blob, static_cast<int32_t>(pageSize - kHeaderSize));
    putValue<int32_t>(blob, static_cast<int32_t>(pageSize - kHeaderSize));
    putValue<int64_t>(blob, 0);
    putValue<int32_t>(blob, numColumns);
    blobCopies.push_back({blobStart, waveOffset, blob.size() - blobStart});
    pos += blob.size() - blobStart;

    for (const auto& column : serializer.columns) {
      const auto& name = column.encodingName;
      blobStart = blob.size();
      putValue<int32_t>(blob, static_cast<int32_t>(name.size()));
      blob.insert(blob.end(), name.begin(), name.end());
      putValue<int32_t>(blob, rows);
      blobCopies.push_back(
          {blobStart, waveOffset + pos, blob.size() - blobStart});
      pos += blob.size() - blobStart;

      if (column.isString()) {
        offsetTasks.push_back({column.offsets, begin, rows, page + pos});
        pos += sizeof(int32_t) * static_cast<size_t>(rows);
      }

      const int32_t nulls = column.mask != nullptr ? column.pageNulls[p] : 0;
      blobStart = blob.size();
      putValue<int8_t>(blob, nulls > 0 ? 1 : 0);
      blobCopies.push_back({blobStart, waveOffset + pos, 1});
      pos += 1;
      if (nulls > 0) {
        nullTasks.push_back(
            {column.mask, column.firstBit + begin, rows, page + pos});
        pos += (rows + 7) / 8;
      }

      if (column.isString()) {
        const int64_t bytes = column.charsBounds[p + 1] - column.charsBounds[p];
        blobStart = blob.size();
        putValue<int32_t>(blob, static_cast<int32_t>(bytes));
        blobCopies.push_back({blobStart, waveOffset + pos, sizeof(int32_t)});
        pos += sizeof(int32_t);
        if (bytes > 0) {
          copies.push_back(
              {reinterpret_cast<const uint8_t*>(column.chars) +
                   column.charsBounds[p],
               page + pos,
               static_cast<size_t>(bytes)});
        }
        pos += bytes;
      } else {
        const int64_t bytes = static_cast<int64_t>(rows - nulls) * column.width;
        if (bytes > 0) {
          const uint8_t* src = column.compacted
              ? column.values + column.validBefore[p] * column.width
              : column.values + begin * column.width;
          copies.push_back({src, page + pos, static_cast<size_t>(bytes)});
        }
        pos += bytes;
      }
    }
    VELOX_CHECK_EQ(pos, static_cast<size_t>(pageSize));
    wave.pages.push_back({waveOffset, static_cast<size_t>(pageSize), rows});
    waveOffset += pageSize;
  }

  rmm::device_buffer deviceBlob(blob.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      deviceBlob.data(),
      blob.data(),
      blob.size(),
      cudaMemcpyHostToDevice,
      stream.get()));
  for (const auto& blobCopy : blobCopies) {
    copies.push_back(
        {static_cast<const uint8_t*>(deviceBlob.data()) + blobCopy.blobOffset,
         base + blobCopy.dstOffset,
         blobCopy.size});
  }
  detail::batchedDeviceCopy(copies, stream, mr);
  detail::fillPrestoNullBytes(nullTasks, stream, mr);
  detail::fillPrestoStringOffsets(offsetTasks, stream, mr);
  // The pageable blob was staged by cudaMemcpyAsync before it returned, and
  // 'deviceBlob' is freed in stream order after the batched copy.

  serializer.nextPage = last;
  wave_ = std::move(wave);
}

void CudfPrestoOutput::copyPagesToHost() {
  auto& wave = *wave_;
  auto stream = serializer_->stream;
  const auto* base = static_cast<const uint8_t*>(wave.data.data());
  std::vector<HostPage> batch;
  int64_t batchBytes = 0;
  try {
    while (wave.next < wave.pages.size() &&
           (batch.empty() ||
            batchBytes + static_cast<int64_t>(wave.pages[wave.next].size) <=
                hostBatchBytes_)) {
      const auto& range = wave.pages[wave.next];
      HostPage page{nullptr, range.size, range.numRows};
      // Allocated in stream order: a block recycled from another stream is
      // free of that stream's work before the copy writes it.
      page.data = pinnedMr_.allocate(stream, range.size, kPinnedAlignment);
      batch.push_back(page);
      batchBytes += range.size;
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          page.data,
          base + range.offset,
          range.size,
          cudaMemcpyDeviceToHost,
          stream.get()));
      ++wave.next;
    }
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  } catch (...) {
    cudaStreamSynchronize(stream.get());
    for (auto& page : batch) {
      releaseHostPage(page);
    }
    throw;
  }
  for (auto& page : batch) {
    if (checksumListener_ != nullptr) {
      checksumPage(page);
    }
    ready_.push_back(page);
  }
}

void CudfPrestoOutput::checksumPage(HostPage& page) {
  // The CRC32 that exec::PartitionedOutput computes through the listener:
  // over the bytes after the header, the codec marker, the row count and the
  // uncompressed size.
  auto* bytes = static_cast<char*>(page.data);
  const auto uncompressedSize = static_cast<int32_t>(page.size) - kHeaderSize;
  checksumListener_->reset();
  checksumListener_->onWrite(bytes + kHeaderSize, uncompressedSize);
  const int64_t checksum = serializer::presto::detail::computeChecksum(
      checksumListener_, kCheckSumBitMask, page.numRows, uncompressedSize);
  bytes[kCodecMarkerOffset] = kCheckSumBitMask;
  std::memcpy(bytes + kChecksumOffset, &checksum, sizeof(checksum));
}

bool CudfPrestoOutput::enqueueReadyPages(
    exec::DefaultOutputBufferManager& bufferManager) {
  while (!ready_.empty()) {
    HostPage page = ready_.front();
    ready_.pop_front();
    auto* release = new PageRelease{page.size, pinnedMr_, operatorCtx_->task()};
    auto iobuf = folly::IOBuf::takeOwnership(
        page.data, page.size, page.size, releasePage, release);
    auto serialized = std::make_unique<exec::PrestoSerializedPage>(
        std::move(iobuf), nullptr, page.numRows);
    const bool blocked =
        bufferManager.enqueue(taskId(), 0, std::move(serialized), &future_);
    {
      auto lockedStats = stats_.wlock();
      lockedStats->addOutputVector(page.size, page.numRows);
    }
    if (blocked) {
      blockingReason_ = exec::BlockingReason::kWaitForConsumer;
      return true;
    }
  }
  return false;
}

void CudfPrestoOutput::drain() {
  auto bufferManager = bufferManager_.lock();
  VELOX_CHECK_NOT_NULL(
      bufferManager, "DefaultOutputBufferManager was already destructed");
  while (blockingReason_ == exec::BlockingReason::kNotBlocked) {
    if (!ready_.empty()) {
      if (enqueueReadyPages(*bufferManager)) {
        return;
      }
      continue;
    }
    if (wave_.has_value()) {
      if (wave_->next < wave_->pages.size()) {
        const auto start = nowNanos();
        copyPagesToHost();
        copyNanos_ += nowNanos() - start;
        continue;
      }
      // Every copy of the wave has completed: the device buffer can go.
      wave_.reset();
      continue;
    }
    if (serializer_ != nullptr) {
      if (!serializer_->done()) {
        const auto start = nowNanos();
        buildNextWave();
        layoutNanos_ += nowNanos() - start;
        continue;
      }
      serializer_.reset();
      continue;
    }
    // Fill a page before serializing, unless the output is flushed eagerly or
    // complete.
    if (!pendingInputs_.empty() &&
        (noMoreInput_ || eagerFlush_ || pendingBytes_ >= pageBytes_ ||
         pendingRows_ >= kMaxPageRows)) {
      const auto start = nowNanos();
      startSerializer();
      layoutNanos_ += nowNanos() - start;
      continue;
    }
    break;
  }

  if (blockingReason_ == exec::BlockingReason::kNotBlocked && noMoreInput_ &&
      pendingInputs_.empty() && serializer_ == nullptr && !wave_.has_value() &&
      ready_.empty() && !finished_) {
    bufferManager->noMoreData(taskId());
    finished_ = true;
  }
}

void CudfPrestoOutput::releaseHostPage(HostPage& page) {
  if (page.data != nullptr) {
    pinnedMr_.deallocate_sync(page.data, page.size, kPinnedAlignment);
    page.data = nullptr;
  }
}

void CudfPrestoOutput::doClose() {
  Operator::close();
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        exec::Operator::kShuffleSerdeKind,
        RuntimeCounter(static_cast<int64_t>(VectorSerde::Kind::kPresto)));
    lockedStats->addRuntimeStat(
        exec::Operator::kShuffleCompressionKind,
        RuntimeCounter(
            static_cast<int64_t>(
                common::CompressionKind::CompressionKind_NONE)));
    lockedStats->addRuntimeStat(
        "pageLayoutNanos",
        RuntimeCounter(layoutNanos_, RuntimeCounter::Unit::kNanos));
    lockedStats->addRuntimeStat(
        "deviceToHostNanos",
        RuntimeCounter(copyNanos_, RuntimeCounter::Unit::kNanos));
  }
  if (serializer_ != nullptr) {
    // Pending device-to-host copies must finish before their buffers go away.
    cudaStreamSynchronize(serializer_->stream.get());
  }
  for (auto& page : ready_) {
    releaseHostPage(page);
  }
  ready_.clear();
  wave_.reset();
  serializer_.reset();
  pendingInputs_.clear();
}

} // namespace facebook::velox::cudf_velox
