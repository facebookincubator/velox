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

#pragma once

#include <folly/container/F14Map.h>
#include <deque>

#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/experimental/wave/dwio/ColumnReader.h"
#include "velox/experimental/wave/dwio/nimble/NimbleStream.h"
#include "velox/type/Subfield.h"

namespace facebook::velox::wave {

/// The streams of a Nimble stripe that a scan reads, in host memory.
struct NimbleStripe {
  /// Bytes after each stream. Bit unpacking reads whole words that may extend
  /// past the last value.
  static constexpr int32_t kStreamPadding = 16;

  /// Returns the bytes of stream 'streamId', nullptr if the stripe does not
  /// have the stream.
  const std::string* stream(uint32_t streamId) const;

  int32_t numRows{0};
  /// Stream bytes by stream id. Each is followed by kStreamPadding bytes.
  folly::F14FastMap<uint32_t, std::string> streams;
  /// Streams made on the host for columns without a stream in the stripe.
  /// See NimbleFormatParams::nullStream().
  std::deque<std::string> nullStreams;
};

/// FormatData of a column of a Nimble stripe. A scalar column has one stream.
class NimbleFormatData : public FormatData {
 public:
  /// 'stream' is nullptr for a column without a data stream, e.g. the root
  /// struct. 'filter' is the filter of the column or nullptr. Compressed
  /// payloads are decompressed on the host with memory from 'pool'. 'levels'
  /// are the nulls of the nullable structs between the column and the
  /// enclosing array or map or the top level. 'isNested' is true below an
  /// array or map.
  NimbleFormatData(
      int32_t totalRows,
      const std::string* stream,
      WaveTypeKind kind,
      const common::Filter* filter,
      memory::MemoryPool* pool,
      ColumnRole role,
      std::vector<const FormatData*> levels,
      bool isNested);

  /// True if the column has nulls in its stream or an enclosing struct has
  /// nulls.
  bool hasNulls() const override {
    return stream_.hasNulls || !levels_.empty();
  }

  /// A stripe may have nulls in a column that has none in another stripe.
  bool isNullabilityStable() const override {
    return false;
  }

  /// Number of rows in the stream.
  int32_t numEncodedRows() const override {
    return stream_.numRows;
  }

  /// Returns the sum of the lengths of an array or map, i.e. the number of
  /// rows of its elements. Decodes the lengths on the host.
  int64_t sumOfLengths(memory::MemoryPool& pool) const;

  int32_t totalRows() const override {
    return totalRows_;
  }

  /// Adds a prepare op for each node that needs aux data: run ends, rank
  /// bitmaps of nulls and of MainlyConstant, and the alphabet filter bitmap.
  void griddize(
      ColumnOp& op,
      int32_t blockSize,
      int32_t numBlocks,
      ResultStaging& deviceStaging,
      ResultStaging& resultStaging,
      SplitStaging& staging,
      DecodePrograms& programs,
      ReadStream& stream) override;

 protected:
  void stage(ResultStaging& /*deviceStaging*/, SplitStaging& staging) override {
    buffers_.stage(staging);
  }

  WaveTypeKind stepKind() const override {
    return role_ == ColumnRole::kLengths ? WaveTypeKind::INTEGER : kind_;
  }

  void setDecode(
      GpuDecode& step,
      ColumnOp& op,
      int32_t blockIdx,
      SplitStaging& staging) override;

  void setNestedDecode(GpuDecode& step, SplitStaging& staging) override;

 private:
  // Indices of the buffers in 'buffers_'.
  static constexpr int32_t kData = 0;
  static constexpr int32_t kDescriptors = 1;

  // Decides which part of 'filter' is evaluated on the host and which
  // dictionaries get an alphabet filter prepare op.
  void setFilterPreparation(const common::Filter& filter);

  // Returns the size of the aux data of 'node'.
  int32_t auxBytes(const NimbleNode& node) const;

  // Sets the device addresses of the stream, the 'numChunks' chunks from
  // 'firstChunk', the nodes and the aux data in 'nimble'.
  void setPointers(
      GpuDecode::Nimble& nimble,
      int32_t firstChunk,
      int32_t numChunks,
      SplitStaging& staging) const;

  const int32_t totalRows_;
  // The bytes copied to device: the stream or 'withDecompressed_'.
  const std::string* data_;
  // The stream followed by the decompressed payloads of compressed leaves.
  std::string withDecompressed_;
  const WaveTypeKind kind_;
  const bool isNested_;
  NimbleStream stream_;
  // Nodes with a prepare op.
  std::vector<int32_t> prepareNodes_;
  // Total size of the aux data.
  int32_t auxSize_{0};
  // The chunks of 'stream_' followed by the nodes, copied to device.
  std::string descriptors_;
  // Offset of the nodes in 'descriptors_'.
  int64_t nodesOffset_{0};
  int32_t accessorsId_{0};
  // True if the column has a filter.
  bool hasFilter_{false};
  // 'data_' and 'descriptors_'.
  StagedBuffers buffers_;
};

class NimbleFormatParams : public FormatParams {
 public:
  /// Value for setInMapStream() for a flat map key that is not in the file.
  static constexpr int32_t kMissingKey = -2;
  static constexpr int32_t kNoInMap = -1;

  NimbleFormatParams(
      memory::MemoryPool& pool,
      dwio::common::ColumnRuntimeStats& stats,
      NimbleStripe& stripe)
      : FormatParams(pool, stats), stripe_(stripe) {}

  /// Sets the Nimble schema node of the column for the next toFormatData().
  /// nullptr for a column that is not in the file. Its rows are null.
  void setNimbleType(const nimble::Type* type) {
    nimbleType_ = type;
  }

  /// Makes the next toFormatData() return the in-map stream 'streamId' of a
  /// flat map key as the nulls of a struct. kMissingKey makes nulls without
  /// a stream, i.e. no nulls. kNoInMap returns to schema nodes.
  void setInMapStream(int32_t streamId) {
    inMapStream_ = streamId;
  }

  std::unique_ptr<FormatData> toFormatData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type,
      const velox::common::ScanSpec& scanSpec,
      OperandId operand) override;

 private:
  // Returns a stream made on the host where all rows are null, for a column
  // without a stream in the stripe. The number of rows is rowBound(). If
  // 'isNulls' is true, the stream is the bool nulls of a struct.
  const std::string* nullStream(nimble::DataType dataType, bool isNulls);

  // Returns an upper bound for the number of rows of the stream of the next
  // column: the rows of the stripe, or below an array or map, the number of
  // elements.
  int32_t rowBound();

  NimbleStripe& stripe_;
  const nimble::Type* nimbleType_{nullptr};
  // In-map stream for the next toFormatData(), kNoInMap if none.
  int32_t inMapStream_{kNoInMap};
  // The number of elements of each array or map that encloses a column.
  folly::F14FastMap<const FormatData*, int64_t> numElements_;
};

class NimbleFormatReader {
 public:
  /// Makes the reader tree for a stripe. 'nimbleType' is the Nimble schema
  /// node that corresponds to 'fileType'. nullptr for a column that is not in
  /// the file. Its rows are null.
  static std::unique_ptr<ColumnReader> build(
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      const nimble::Type* nimbleType,
      NimbleFormatParams& params,
      common::ScanSpec& scanSpec,
      std::vector<std::unique_ptr<common::Subfield::PathElement>>& path,
      const DefinesMap& defines,
      bool isRoot = false);

  /// Appends the ids of the streams of 'type' that the scan reads to
  /// 'streamIds'. 'scanSpec' selects the struct fields and the flat map keys
  /// to read. nullptr selects all.
  static void addStreamIds(
      const nimble::Type& type,
      const common::ScanSpec* scanSpec,
      std::vector<uint32_t>& streamIds);

  /// Returns true if the scan reads key 'key' of a flat map with
  /// 'scanSpec'. A flat map read as a struct reads the keys that are fields
  /// of the struct. Otherwise, a filter on the keys selects the keys.
  static bool isKeySelected(
      const common::ScanSpec* scanSpec,
      const std::string& key);
};

} // namespace facebook::velox::wave
