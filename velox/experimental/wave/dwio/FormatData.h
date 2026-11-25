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

#include <folly/Range.h>
#include "velox/common/base/Semaphore.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/file/Region.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/experimental/wave/dwio/decode/DecodeStep.h"
#include "velox/experimental/wave/exec/OperandSet.h"
#include "velox/experimental/wave/vector/WaveVector.h"

namespace facebook::velox::wave {

class ReadStream;
class WaveStream;

/// Use generic bit set to track depemdemce pon staging.
using StagingSet = OperandSet;

// Describes how a column is staged on GPU, for example, copy from host RAM,
// direct read, already on device etc.
struct Staging {
  Staging(const void* hostData, int32_t size, const common::Region& region)
      : hostData(hostData),
        size(hostData ? size : region.length),
        fileOffset(region.offset) {}

  // Pointer to data in pageable host memory, if applicable.
  const void* hostData{nullptr};

  //  Size in bytes.
  size_t size;

  /// If 'hostData' is nullptr, this is the start offset for 'size'
  /// bytes in 'fileInfo_' of containing SplitStaging.
  int64_t fileOffset{0};
};

struct FileInfo {
  ReadFile* file{nullptr};
  StringIdLease* fileId{nullptr};
  cache::AsyncDataCache* cache{nullptr};
};

/// Describes how columns to be read together are staged on device. This is
/// anything from a set of host to device copies, GPU direct IO, or no-op if
/// data already on device.
class SplitStaging {
 public:
  /// id indicating no dependence on other staging.
  static constexpr int32_t kNoStaging = ~0;

  SplitStaging(FileInfo& fileInfo, int32_t id) : id_(id), fileInfo_(fileInfo) {}

  /// Adds a transfer described by 'staging'. Returns an id of the
  /// device side buffer. The id will be mapped to an actual buffer
  /// when the transfers are queud. At this time, pointers that
  /// are registered to the id are patched to the actual device side
  /// address.
  BufferId add(Staging& staging);

  /// Registers '*ptr' to be patched to the device side address of the transfer
  /// identified by 'id'. The *ptr is an offset into the buffer identified by
  /// id, so that the actual start of the area is added to the offset at *ptr.
  /// If 'clear' is true, *ptr is set to nullptr first. This may not be called
  /// after transfer().
  template <typename T>
  void registerPointer(BufferId id, T pointer, bool clear) {
    registerPointerInternal(
        id,
        reinterpret_cast<void**>(reinterpret_cast<uint64_t>(pointer)),
        clear);
  }

  int64_t bytesToDevice() const {
    return fill_;
  }
  // Starts the transfers registered with add( on 'stream'). Does
  // nothing after first call or if no pointers are registered. If
  // 'recordEvent' is true, records an event that is completed after
  // the transfer arrives. Use event() to access the event. If
  // 'asyncTail' is non-nullptr, it is called after the data transfer
  // is enqueued. The call is on an executor and transfer() returns as
  // soon as the work is enqueud. If asyncTail is not given,
  // transfer() returns after the transfer is enqueued on
  // 'stream'. event() is not set until the transfer is enqueued.
  void transfer(
      WaveStream& waveStream,
      Stream& stream,
      bool recordEvent = false,
      std::function<void(WaveStream&, Stream&)> asyncTail = nullptr);

  Event* event() const {
    return event_.get();
  }

  int32_t id() const {
    return id_;
  }

  void addDependency(int32_t id) {
    dependsOn_.add(id);
  }

  const StagingSet& dependsOn() {
    return dependsOn_;
  }

 private:
  void registerPointerInternal(BufferId id, void** ptr, bool clear);

  void copyColumns(int32_t begin, int32_t end, char* destination, bool release);

  const int32_t id_;

  // Pinned host memory for transfer to device. May be nullptr if using unified
  // memory.
  WaveBufferPtr hostBuffer_;

  // Device accessible memory (device or unified) with the data to read.
  WaveBufferPtr deviceBuffer_;

  std::vector<Staging> staging_;
  // Offsets into device buffer for each id returned by add().
  std::vector<int64_t> offsets_;

  // List of pointers to patch to places inside deviceBuffer once this is
  // allocated.
  std::vector<std::pair<int32_t, void**>> patch_;

  // Total device side space reserved so farr.
  int64_t fill_{0};

  // Optional event recorded after end of transfer. Use to sync dependent
  // kernels on other streams.
  std::unique_ptr<Event> event_;

  // Synchronizes arrival of multithreaded memcpy
  Semaphore sem_{0};

  FileInfo& fileInfo_;

  // Set of other SplitStaging ids of which 'this' is a
  // duplicate. These need to be complete efore dependents of 'this'
  // can run. A staging can both have transfers and dependencies.
  StagingSet dependsOn_;
};

using RowSet = folly::Range<const int32_t*>;
class ColumnReader;

/// Information that allows a column to be read in parallel independent thread
/// blocks. This represents an array of starting points inside the encoded
/// column.
struct ColumnGridInfo {
  /// Number of independently schedulable blocks.
  int32_t numBlocks;

  ///
  BlockStatus* status{nullptr};

  /// Device readable nulls as a flat bitmap. 1 is non-null. nullptr means
  /// non-null.
  char* nulls{nullptr};

  /// Device side array of non-null counts. Decoding for values for the ith
  /// block starts at index 'nonNullCount[i - 1]' in encoded values. nullptr if
  /// non nulls.
  int32_t* numNonNull{nullptr};
};

// Specifies an action on a column. A column is not indivisible. It
// has parts and another column's decode may depend on one part of
// another column but not another., e.g. a child of a nullable struct
// needs the nulls of the struct but no other parts to decode.
enum class ColumnAction { kNulls = 1, kLengths = 2, kFilter = 4, kValues = 8 };

/// A generic description of a decode step. The actual steps are
/// provided by FormatData specializations but this captures
/// dependences, e.g. filters before non-filters, nulls and lengths
/// of repeated containers before decoding the values. A dependency
/// can be device side only or may need host decision. Items that
/// depend device side can be made into consecutive decode ops in
/// one kernel launch or can be in consecutively queued
/// kernels. dependences which need host require the prerequisite
/// kernel to ship data to host, which will sync on the stream and
/// only then may schedule the dependents in another kernel.
struct ColumnOp {
  static constexpr int32_t kNoPrerequisite = -1;
  static constexpr int32_t kNoOperand = -1;

  // Is the op completed after this? If so, any dependent action can be
  // queued as soon as this is set.
  bool isFinal{false};
  // True if needs a result on the host before proceeding.
  bool needsResult{false};
  OperandId producesOperand{kNoOperand};
  // Index of another op in column ops array in ReadStream.
  int32_t prerequisite{kNoPrerequisite};
  ColumnAction action;
  // Non-owning view on rows to read.
  RowSet rows;
  ColumnReader* reader{nullptr};
  // Vector completed by arrival of this. nullptr if no vector.
  WaveVector* waveVector{nullptr};

  // Host side result size. 0 for unconditional decoding. Can be buffer size for
  // passing rows, length/offset array etc.
  int32_t resultSize{0};

  // Device side non-vector result, like set of passing rows, array of
  // lengths/starts etc.
  int32_t* deviceResult{nullptr};

  // Id of 'deviceResult' from resultStaging. A subsequent op must refer to the
  // result of the previous one before the former is allocated.
  BufferId deviceResultId{kNoBufferId};

  // Id of extra filter passing row count. Needed for aligning values from
  // non-last filtered columns to final.
  int32_t* extraRowCount{nullptr};

  BufferId extraRowCountId{kNoBufferId};

  int32_t* hostResult{nullptr};

  // Arrays and maps: the number of elements of the rows of each TB, written
  // by device and read by host after a sync.
  WaveBufferPtr totals;
};

/// Read of the rows of a column below an array or map. The rows are the
/// elements of the enclosing array or map, from 0 to 'numRows'.
struct NestedOp {
  ColumnReader* reader{nullptr};
  int32_t numRows{0};
  // Device side row of each element in the stream of the enclosing array's
  // elements.
  const int32_t* elementRows{nullptr};
  // Values and nulls of the elements. For a struct, the nulls.
  WaveVector* vector{nullptr};
  // Arrays and maps: see ColumnOp::totals.
  WaveBufferPtr totals;
};

/// Role of a FormatData in the reader tree.
enum class ColumnRole {
  /// Values of a scalar column.
  kValues,
  /// Nulls of a struct, or what acts as the nulls of a struct, e.g. the
  /// in-map flags of a key of a Nimble flat map.
  kStructNulls,
  /// Lengths of an array or map.
  kLengths,
};

/// Host buffers of a column that are copied to the device once per split.
/// The first stage() adds them to a SplitStaging. Later stage() calls add a
/// dependency on that staging. The device addresses are known after the first
/// staging is transferred.
class StagedBuffers {
 public:
  /// Adds a buffer and returns its index.
  int32_t add(const Staging& staging) {
    buffers_.push_back({staging});
    return static_cast<int32_t>(buffers_.size() - 1);
  }

  bool empty() const {
    return buffers_.empty();
  }

  /// Adds the buffers to 'staging' on the first call. Otherwise adds a
  /// dependency on the staging of the first call. Returns true on the first
  /// call.
  bool stage(SplitStaging& staging);

  /// Sets 'pointer' to the device address of buffer 'index' plus 'offset'. If
  /// the latest stage() added the buffers, registers 'pointer' to be patched
  /// when 'staging' is transferred.
  template <typename T>
  void setPointer(
      int32_t index,
      T*& pointer,
      int64_t offset,
      SplitStaging& staging) const {
    const auto& buffer = buffers_[index];
    if (pending_) {
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      pointer = reinterpret_cast<T*>(offset);
      staging.registerPointer(buffer.id, &pointer, false);
    } else {
      VELOX_CHECK_NOT_NULL(buffer.device);
      pointer = reinterpret_cast<T*>(buffer.device + offset);
    }
  }

  /// Device address of buffer 'index', nullptr before the first staging is
  /// transferred.
  char* deviceAddress(int32_t index) const {
    return buffers_[index].device;
  }

 private:
  struct Buffer {
    Staging staging;
    BufferId id{kNoBufferId};
    char* device{nullptr};
  };

  std::vector<Buffer> buffers_;
  bool staged_{false};
  // True from the stage() that adds the buffers to the next stage().
  bool pending_{false};
  int32_t stagingId_{SplitStaging::kNoStaging};
};

/// Returns the index of the chunk of 'chunks' that contains 'row'. 'chunks'
/// are in row order.
int32_t chunkIndex(const std::vector<ChunkDesc>& chunks, int32_t row);

/// Operations on the encoded data of a column. This is specialized for each
/// file format. The format independent parts of making decode steps are here:
/// the TBs of an op, the nulls of enclosing structs, reading lengths and
/// struct nulls below the top level, and data made by prepare ops that lives
/// as long as 'this'.
class FormatData {
 public:
  /// 'levels' are the FormatDatas of the nulls of the nullable structs between
  /// the column and the enclosing array or map or the top level, outermost
  /// first.
  explicit FormatData(
      ColumnRole role = ColumnRole::kValues,
      std::vector<const FormatData*> levels = {});

  virtual ~FormatData() = default;

  ColumnRole role() const {
    return role_;
  }

  virtual int32_t totalRows() const = 0;

  virtual bool hasNulls() const = 0;

  /// True if hasNulls() is the same for the column in all parts of the file
  /// that the reader tree of 'this' is not made for. If false, another
  /// reader tree of the column, e.g. for the next stripe, may have nulls when
  /// 'this' has none.
  virtual bool isNullabilityStable() const {
    return true;
  }

  /// Number of rows of the encoded data of 'this'. Differs from totalRows() in
  /// a nullable struct, where the rows are the non-null rows of the struct,
  /// and below an array or map, where the rows are elements.
  virtual int32_t numEncodedRows() const {
    return totalRows();
  }

  /// Device address of the data made by the prepare ops of 'this' in
  /// griddize(). For struct nulls, this starts with the RankBitmap of non-null
  /// rows. nullptr before griddize() or if there are no prepare ops.
  const char* deviceAux() const {
    return deviceAux_;
  }

  /// Prepares a new batch of reads. The batch starts at 'startRow', which is a
  /// row number in terms of the column of 'this'. The row number for a nested
  /// column is in terms of the column, not in terms of top level rows.
  virtual void newBatch(int32_t startRow) {
    currentRow_ = startRow;
  }

  /// Schedules operations for preparing the encoded data to be
  /// consumed in 'numBlocks' parallel blocks of 'blockSize' rows. For
  /// example, for a column of 11M nullable varints, this with 1024
  /// blocksize and 2048 blocks, this would count 2M bits and write a
  /// prefix sum every 1K bits, so that we know the corresponding
  /// position in the varints for non-nulls. Then for the varints, we
  /// write the starting offset every 1K nulls, e.g, supposing 2 bytes
  /// per varint and 800 non-nulls for every 1K bits, we get 0, 1600,
  /// 3600, ... as starts for the varints. The FormatData stores the
  /// intermediates. This is a no-op for encodings that are random
  /// access capable, e.g. non-null bit packings. this is a also a
  /// no-op if there are less than 'blockSize' rows left.
  virtual void griddize(
      ColumnOp& op,
      int32_t blockSize,
      int32_t numBlocks,
      ResultStaging& deviceStaging,
      ResultStaging& resultStaging,
      SplitStaging& staging,
      DecodePrograms& program,
      ReadStream& stream) = 0;

  /// Adds the TBs that read the rows of 'op'. If the column is a filter
  /// depending on another filter, the previous filter is given on the first
  /// call. Updates status of 'op'. Stages the encoded data with stage() and
  /// sets the format specific fields of each TB's step with setDecode(). For
  /// an array or map, the step reads the lengths into the offsets and sizes of
  /// op.waveVector and sets op.totals.
  virtual void startOp(
      ColumnOp& op,
      const ColumnOp* previousFilter,
      ResultStaging& deviceStaging,
      ResultStaging& resultStaging,
      SplitStaging& staging,
      DecodePrograms& program,
      ReadStream& stream);

  /// Adds the TBs that read the rows of 'op' below an array or map. Values go
  /// to op.vector. An array or map writes its element starts and sizes and
  /// sets op.totals. A struct writes its nulls. Sets the format specific
  /// fields of each TB's step with setNestedDecode().
  virtual void startNestedOp(
      NestedOp& op,
      ResultStaging& deviceStaging,
      SplitStaging& staging,
      DecodePrograms& program,
      ReadStream& stream);

 protected:
  /// Adds the encoded data of 'this' to 'staging'. Called at the start of
  /// startOp() and startNestedOp().
  virtual void stage(
      ResultStaging& /*deviceStaging*/,
      SplitStaging& /*staging*/) {}

  /// Type of the values that the steps of startOp() produce.
  virtual WaveTypeKind stepKind() const = 0;

  /// Sets the format specific fields of the step of TB 'blockIdx' of 'op'.
  virtual void setDecode(
      GpuDecode& step,
      ColumnOp& op,
      int32_t blockIdx,
      SplitStaging& staging) = 0;

  /// Sets the format specific fields of a TB of startNestedOp(), i.e. where
  /// the encoded data is. The other fields are set by startNestedOp().
  virtual void setNestedDecode(GpuDecode& /*step*/, SplitStaging& /*staging*/) {
    VELOX_NYI("Columns below arrays and maps are not supported");
  }

  /// Sets the fields of 'column' for the nulls of the enclosing structs.
  void setLevels(ColumnAccess& column) const;

  /// Makes 'step' a kNested step that reads the lengths or struct nulls of
  /// 'this' from the aux data.
  void setNested(GpuDecode& step) const;

  /// Allocates 'bytes' of device memory for the data made by prepare ops. It
  /// is read by the ops of all batches of the split and lives as long as
  /// 'this', not as long as the ReadStream that makes it.
  void allocateAux(ReadStream& stream, int32_t bytes);

  /// Returns the first chunk and the number of chunks of 'chunks' that a TB
  /// reading rows 'firstRow' to 'lastRow' needs. Without enclosing struct
  /// nulls, these are the chunks of the rows. Otherwise the rows depend on the
  /// nulls and all chunks are needed.
  std::pair<int32_t, int32_t> tbChunks(
      const std::vector<ChunkDesc>& chunks,
      int32_t firstRow,
      int32_t lastRow) const;

  std::unique_ptr<GpuDecode> makeStep(
      ColumnOp& op,
      const ColumnOp* previousFilter,
      ResultStaging& deviceStaging,
      SplitStaging& splitStaging,
      ReadStream& stream,
      WaveTypeKind columnKind,
      int32_t blockIdx);

  std::unique_ptr<GpuDecode> makeAlphabetStep(
      ColumnOp& op,
      ResultStaging& deviceStaging,
      SplitStaging& splitStaging,
      ReadStream& stream,
      WaveTypeKind columnKind,
      int32_t blockIdx,
      int32_t numRows);

  /// Makes a step of a prepare op that runs in one TB. If 'withFilter' is
  /// true, the step has the filter of the column, e.g. for testing the
  /// alphabet of a dictionary.
  std::unique_ptr<GpuDecode> makePrepareStep(
      ColumnOp& op,
      bool withFilter,
      ResultStaging& deviceStaging,
      SplitStaging& splitStaging,
      ReadStream& stream);

  const ColumnRole role_;

  /// The FormatDatas of the nulls of the enclosing nullable structs,
  /// outermost first.
  const std::vector<const FormatData*> levels_;

  /// Offset in the aux data of the element starts and null flags of the rows
  /// of an array or map. See nestedRows() in ColumnOps.cuh.
  int32_t lengthsAux_{0};

  /// Device address of the aux data. Set by allocateAux().
  char* deviceAux_{nullptr};

  // A format with a flat nulls bitmap stages it separately from the values.
  // Staging id for nulls.
  int32_t nullsStagingId_{SplitStaging::kNoStaging};
  // Id for nulls buffer. The nulls buffer has no address at time of scheduling
  // if nulls decode is in the same kernel as decoding.
  BufferId nullsBufferId_{kNoBufferId};

  // First unaccessed row number relative to start of 'this'.
  int32_t currentRow_{0};

  ColumnGridInfo grid_;
  bool griddized_{false};

 private:
  // Owns 'deviceAux_'.
  WaveBufferPtr auxBuffer_;
};

class FormatParams {
 public:
  explicit FormatParams(
      memory::MemoryPool& pool,
      dwio::common::ColumnRuntimeStats& stats)
      : pool_(pool), stats_(stats) {}

  virtual ~FormatParams() = default;

  /// Makes format-specific structures for the column given by  'type'.
  /// 'scanSpec' is given as extra context.
  virtual std::unique_ptr<FormatData> toFormatData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type,
      const velox::common::ScanSpec& scanSpec,
      OperandId operand) = 0;

  memory::MemoryPool& pool() {
    return pool_;
  }

  dwio::common::ColumnRuntimeStats& runtimeStatistics() {
    return stats_;
  }

  /// Adds the nulls of a struct whose columns are made next.
  void pushLevel(const FormatData* structNulls) {
    levels_.push_back(structNulls);
  }

  void popLevel() {
    levels_.pop_back();
  }

  /// The nulls of the nullable structs that enclose the next column,
  /// outermost first.
  const std::vector<const FormatData*>& levels() const {
    return levels_;
  }

  /// Starts the columns below an array or map. Their rows are elements. The
  /// nulls of structs above the array or map do not apply. 'lengths' is the
  /// FormatData of the array or map.
  void pushNested(const FormatData* lengths) {
    nestedLevels_.push_back(std::move(levels_));
    levels_.clear();
    nestedLengths_.push_back(lengths);
  }

  void popNested() {
    levels_ = std::move(nestedLevels_.back());
    nestedLevels_.pop_back();
    nestedLengths_.pop_back();
  }

  /// True if the next column is below an array or map.
  bool isNested() const {
    return !nestedLevels_.empty();
  }

  /// The FormatData of the lengths of the innermost array or map that encloses
  /// the next column, nullptr at the top level.
  const FormatData* nestedLengths() const {
    return nestedLengths_.empty() ? nullptr : nestedLengths_.back();
  }

 private:
  memory::MemoryPool& pool_;
  dwio::common::ColumnRuntimeStats& stats_;
  std::vector<const FormatData*> levels_;
  // 'levels_' of the columns above each enclosing array or map.
  std::vector<std::vector<const FormatData*>> nestedLevels_;
  // The FormatData of the lengths of each enclosing array or map.
  std::vector<const FormatData*> nestedLengths_;
};

}; // namespace facebook::velox::wave
