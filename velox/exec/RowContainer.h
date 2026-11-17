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

#include <folly/CPortability.h>
#include <span>

#include "velox/common/memory/Allocation.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/ContainerRowSerde.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/VectorTypeUtils.h"

namespace facebook::velox::exec {

class Aggregate;

class Accumulator {
 public:
  /// 'destroyFunction' frees the memory the accumulator holds outside each
  /// group row. May be null when there is nothing to free. Required when
  /// 'usesExternalMemory' is true.
  Accumulator(
      bool isFixedSize,
      int32_t fixedSize,
      bool usesExternalMemory,
      int32_t alignment,
      TypePtr spillType,
      std::function<void(folly::Range<char**> groups, VectorPtr& result)>
          spillExtractFunction,
      std::function<void(folly::Range<char**> groups)> destroyFunction);

  explicit Accumulator(Aggregate* aggregate, TypePtr spillType);

  bool isFixedSize() const;

  int32_t fixedWidthSize() const;

  bool usesExternalMemory() const;

  int32_t alignment() const;

  const TypePtr& spillType() const;

  void extractForSpill(folly::Range<char**> groups, VectorPtr& result) const;

  void destroy(folly::Range<char**> groups);

 private:
  const bool isFixedSize_;
  const int32_t fixedSize_;
  const bool usesExternalMemory_;
  const int32_t alignment_;
  const TypePtr spillType_;
  std::function<void(folly::Range<char**>, VectorPtr&)> spillExtractFunction_;
  std::function<void(folly::Range<char**> groups)> destroyFunction_;
};

using normalized_key_t = uint64_t;

/// Pointers to rows of a RowContainer.
using RowSpan = std::span<char* const>;

struct RowContainerIterator {
  int32_t allocationIndex = 0;
  int32_t rowOffset = 0;
  /// Number of unvisited entries that are prefixed by an uint64_t for
  /// normalized key. Set in listRows() on first call.
  int64_t normalizedKeysLeft = 0;
  int normalizedKeySize = 0;

  /// Ordinal position of 'currentRow' in RowContainer.
  int32_t rowNumber{0};
  char* rowBegin{nullptr};
  /// First byte after the end of the range containing 'currentRow'.
  char* endOfRun{nullptr};
  /// Cursor of the list row operation.
  int32_t listRowCursor{0};

  /// Returns the current row, skipping a possible normalized key below the
  /// first byte of row.
  inline char* currentRow() const {
    return (rowBegin && normalizedKeysLeft) ? rowBegin + normalizedKeySize
                                            : rowBegin;
  }

  void reset() {
    *this = {};
  }

  std::string toString() const;
};

/// Container with a 8-bit partition number field for each row in a
/// RowContainer. The partition number bytes correspond 1:1 to rows. Used only
/// for parallel hash join build.
class RowPartitions {
 public:
  /// Initializes this to hold up to 'numRows'.
  RowPartitions(int32_t numRows, memory::MemoryPool& pool);

  /// Appends 'partitions' to the end of 'this'. Throws if adding more than the
  /// capacity given at construction.
  void appendPartitions(std::span<const uint8_t> partitions);

  const memory::Allocation& allocation() const {
    return allocation_;
  }

  int32_t size() const {
    return size_;
  }

  void reset() {
    size_ = 0;
  }

 private:
  const int32_t capacity_;

  // Number of partition numbers added.
  int32_t size_{0};

  // Partition numbers. 1 byte each.
  memory::Allocation allocation_;
};

/// Stats of a key or dependent column of a RowContainer: the number of nulls
/// and, for a variable-width column, the sum and the maximum of the sizes of
/// the non-null values. The number of non-null values is the row count of the
/// container minus the nulls, so it is not kept here.
///
/// Removing a value keeps the null count and the sum exact. The maximum cannot
/// be recomputed, so after a removal it is a bound: no smaller than the largest
/// remaining size.
class RowColumnStats {
 public:
  /// Adds a null.
  void addNull() {
    ++nullCount_;
  }

  /// Removes a null.
  void removeNull() {
    VELOX_DCHECK_GT(nullCount_, 0);
    --nullCount_;
  }

  /// Adds a non-null value of 'bytes' of a variable-width column.
  void addValue(int32_t bytes) {
    maxBytes_ = std::max(maxBytes_, bytes);
    sumBytes_ += bytes;
  }

  /// Removes a non-null value of 'bytes' of a variable-width column.
  void removeValue(int32_t bytes) {
    VELOX_DCHECK_GE(sumBytes_, bytes);
    sumBytes_ -= bytes;
  }

  uint32_t nullCount() const {
    return nullCount_;
  }

  uint64_t sumBytes() const {
    return sumBytes_;
  }

  int32_t maxBytes() const {
    return maxBytes_;
  }

  /// Returns the average size of the non-null values of a variable-width
  /// column, 0 if there are none.
  ///
  /// @param numRows The number of rows the stats cover: the row count of the
  /// container, or the total of the containers whose stats were merged.
  uint64_t avgBytes(uint64_t numRows) const {
    VELOX_DCHECK_GE(numRows, nullCount_);
    const auto numValues = numRows - nullCount_;
    return numValues == 0 ? 0 : sumBytes_ / numValues;
  }

  /// Merges the stats of the same column in several containers.
  static RowColumnStats merge(const std::vector<RowColumnStats>& statsList);

 private:
  uint32_t nullCount_{0};
  uint64_t sumBytes_{0};
  int32_t maxBytes_{0};
};

/// Packed representation of offset, null byte offset and null mask for
/// a column inside a RowContainer.
class RowColumn {
 public:
  /// Used as null offset for a non-null column.
  static constexpr int32_t kNotNullOffset = -1;

  RowColumn(int32_t offset, int32_t nullOffset)
      : packedOffsets_(packOffsets(offset, nullOffset)) {}

  int32_t offset() const {
    return packedOffsets_ >> 32;
  }

  int32_t nullByte() const {
    return static_cast<uint32_t>(packedOffsets_) >> 8;
  }

  uint8_t nullMask() const {
    return packedOffsets_ & 0xff;
  }

  /// The null bits and the initialized bits for accumulators start at the
  /// beginning of the first byte following the null bits for the keys.  This
  /// guarantees that they always appear on the same byte for any given
  /// accumulator (since 2 evenly divides 8).
  int32_t initializedByte() const {
    return nullByte();
  }

  /// The initialized bit for an accumulator is guaranteed to appear on the same
  /// byte immediately following the null bit for that accumulator.
  int32_t initializedMask() const {
    return nullMask() << 1;
  }

 private:
  static uint64_t packOffsets(int32_t offset, int32_t nullOffset) {
    if (nullOffset == kNotNullOffset) {
      // If the column is not nullable, The low word is 0, meaning
      // that a null check will AND 0 to the 0th byte of the row,
      // which is always false and always safe to do.
      return static_cast<uint64_t>(offset) << 32;
    }
    return (1UL << (nullOffset & 7)) | ((nullOffset & ~7UL) << 5) |
        static_cast<uint64_t>(offset) << 32;
  }

  const uint64_t packedOffsets_;
};

/// Stores data by row, unlike vectors, which store it by column. Each row is a
/// fixed-width record; strings and complex type values live in separate
/// allocations the row points to. A row has these parts:
/// - Keys: the values rows are compared and hashed by.
/// - Accumulators: mutable state that changes after the row is added, such as
///   the state of an aggregate function for a group. Each has null and
///   initialized flags, and hooks to spill and free it.
/// - Dependent columns: values stored with the row and only read afterward,
///   such as the build side columns of a hash join. A container has either
///   accumulators or dependent columns, not both.
/// - Raw dependent: like a dependent column, written once with the row, but of
///   a type the caller defines and manages; the container only reserves them.
/// - Flags and fields that options ask for: the probed flag, the next row with
///   the same keys, the count, the normalized key.
///
/// Rows are allocated in pages, not one contiguous buffer, and never move, so
/// a row is addressed by a stable pointer.
///
/// Suits large data accessed by row in random order: the build side of a hash
/// join, the groups of an aggregation, the input of an order by or a window.
/// Vectors need contiguous memory for each column, which gets hard to
/// allocate as data grows, and reading one row from them touches every
/// column's buffer.
///
/// The API, in the order the class declares it:
/// - Construction, with getters for what the constructor was given and the
///   resulting row shape: fixedRowSize(), rowSizeOffset(), nextOffset(),
///   hasCount().
/// - Adding and removing rows: newRow(), eraseRows(), clear(). Erased rows
///   are reused by later newRow() calls.
/// - Storing values: store() copies a value from a vector into a row;
///   extractSerializedRows() and storeSerializedRow() move whole rows between
///   containers.
/// - Reading values: extractColumn() copies a column of chosen rows into a
///   vector; columnAt() locates a column in a row for direct access.
/// - Listing rows: listRows() walks the rows in allocation order with a
///   RowContainerIterator, optionally only those probed or not probed by a
///   hash join.
/// - Partitioning rows for a parallel hash join build: createRowPartitions()
///   and listPartitionRows().
/// - Comparing and hashing rows: compare(), compareRows() and hashKeys(), for
///   sorting and hash tables.
/// - Hash table fields: the probed flag, the count and the normalized key, each
///   present if the constructor's options ask for it.
/// - Memory and column statistics: allocated and used bytes, the size of a
///   row, row size estimates, and per-column null counts and value sizes.
class RowContainer {
 public:
  // Construction, what the constructor was given and the resulting row shape.

  /// What each row holds besides its columns. The defaults make rows of
  /// nullable keys, accumulators and dependent columns only.
  struct Options {
    /// Whether the keys have null flags.
    bool nullableKeys{true};

    /// Whether each row has a pointer to the next row with the same keys, for
    /// a hash join build side with non-unique keys.
    bool hasNext{false};

    /// Whether each row has a bit for the probed state of a full or right
    /// outer join.
    bool hasProbedFlag{false};

    /// Whether each row has a count, for counting joins.
    bool hasCountFlag{false};

    /// Whether each row has a word below it for a normalized key that
    /// collapses all keys into one word for faster comparison.
    bool hasNormalizedKey{false};

    /// Whether to keep an array of row pointers, so that listRowsFast() lists
    /// rows without scanning the allocations or checking the free and probed
    /// flags.
    bool useListRowIndex{false};

    /// The number of bytes to add to the fixed-width part of each row for the
    /// caller to manage. The container reserves them, after the dependent
    /// columns and 8-byte aligned, and otherwise leaves them alone: they are
    /// uninitialized in a new row, have no null flag, and are never compared,
    /// hashed or extracted. The caller reads and writes them through
    /// rawDependent(), for example to keep a fixed-length array of floats or a
    /// pointer to an object it owns.
    ///
    /// A container with a raw dependent cannot be spilled.
    int32_t rawDependentSize{0};
  };

  /// Makes an empty container for rows of the given keys, accumulators and
  /// dependent columns, with what 'options' adds.
  ///
  /// @param keyTypes The types of the keys.
  /// @param accumulators The accumulators of a group by, empty otherwise.
  /// @param dependentTypes The types of the non-key columns of a hash join
  /// build side or an order by. Must be empty if 'accumulators' is not.
  /// @param pool The pool all memory of the container comes from. The
  /// container makes two allocators on it and owns them: one for the
  /// fixed-width parts of rows, and stringAllocator() for variable-width data.
  RowContainer(
      const std::vector<TypePtr>& keyTypes,
      const std::vector<Accumulator>& accumulators,
      const std::vector<TypePtr>& dependentTypes,
      memory::MemoryPool* pool,
      const Options& options);

  /// Makes a container for rows of nullable keys and dependent columns. For
  /// example, the input of a top N row number: the columns rows are compared
  /// by and the other columns.
  RowContainer(
      const std::vector<TypePtr>& keyTypes,
      const std::vector<TypePtr>& dependentTypes,
      memory::MemoryPool* pool)
      : RowContainer(
            keyTypes,
            /*accumulators=*/{},
            dependentTypes,
            pool,
            Options{}) {}

  /// Makes a container for rows of nullable keys and nothing else. For
  /// example, the rows of a top N, or the inputs of an aggregation over sorted
  /// inputs, which may be compared by any of their columns.
  RowContainer(const std::vector<TypePtr>& keyTypes, memory::MemoryPool* pool)
      : RowContainer(
            keyTypes,
            /*dependentTypes=*/std::vector<TypePtr>{},
            pool) {}

  ~RowContainer();

  const std::vector<TypePtr>& keyTypes() const {
    return keyTypes_;
  }

  const std::vector<Accumulator>& accumulators() const {
    return accumulators_;
  }

  /// Returns the types of the keys followed by the types of the dependent
  /// columns. Accumulators have no type and are not included.
  const std::vector<TypePtr>& columnTypes() const {
    return layout_.types;
  }

  int32_t rawDependentSize() const {
    return layout_.rawDependentSize;
  }

  /// The row size excluding any out-of-line stored variable length values.
  int32_t fixedRowSize() const {
    return layout_.fixedRowSize;
  }

  /// Returns the offset of a uint32_t row size, or 0 if the row has no
  /// variable-width columns and no variable-width accumulators.
  int32_t rowSizeOffset() const {
    return layout_.rowSizeOffset;
  }

  /// For a hash join table with possible non-unique entries, the offset of the
  /// pointer to the next row with the same key. 0 if keys are guaranteed
  /// unique, e.g. for a group by or semijoin build.
  ///
  /// A new row's pointer is null.
  int32_t nextOffset() const {
    return layout_.nextOffset;
  }

  /// Whether rows have a count, for counting joins.
  bool hasCount() const {
    return layout_.countOffset != 0;
  }

  /// The allocator of the variable-width data of rows: strings and complex
  /// type values of columns, and the variable-width state of accumulators.
  HashStringAllocator& stringAllocator() {
    return *stringAllocator_;
  }

  // Adding and removing rows.

  /// Allocates a new row and initializes possible aggregates to null.
  ///
  /// The caller then stores each key and dependent column of the row at most
  /// once. A column not stored reads as an empty non-null value.
  char* newRow();

  /// Makes 'row' a new row in place and returns it, as newRow() does. If
  /// 'reuse' is true, 'row' is a row of this container in use: the result is
  /// the same as erasing it and making a new row, without the free list.
  char* initializeRow(char* row, bool reuse);

  /// Sets the keys, accumulators and dependent columns of 'row' to null.
  /// Non-nullable keys keep their values. Used when making a row with
  /// uninitialized keys for aggregates with no-op partial aggregation.
  void setAllNull(char* row);

  /// Adds 'rows' to the free rows list and frees any associated variable length
  /// data.
  void eraseRows(std::span<char*> rows);

  /// Writes to 'result' the elements of 'rows' that point to rows of this
  /// container, in order, and returns their number. 'result' must have space
  /// for 'rows.size()' elements.
  int32_t filterOwnedRows(RowSpan rows, std::span<char*> result) const;

  /// Resets the state to be as after construction. Frees memory for payload.
  void clear();

  /// Returns the number of used rows in 'this'. This is the number of rows a
  /// RowContainerIterator would access.
  int64_t numRows() const {
    return numRows_;
  }

  // Storing values.
  //
  // Key and dependent values are written only through store() and
  // storeSerializedRow(), once per row. The column stats and the variable-width
  // data rely on it.

  /// Stores the 'index'th value in 'decoded' into 'row' at 'columnIndex'.
  void store(
      const DecodedVector& decoded,
      vector_size_t rowIndex,
      char* row,
      column_index_t columnIndex);

  /// Stores the first 'rows.size()' values of 'decoded' into the column at
  /// 'columnIndex', value i into 'rows[i]'. 'rows' are rows of this
  /// container, usually just made with newRow(), one per input row. Faster
  /// than calling store() for each row: it dispatches on the type of the
  /// column and checks for nulls once.
  void
  store(const DecodedVector& decoded, RowSpan rows, column_index_t columnIndex);

  /// Copies the keys and dependent columns of each of 'rows' into one value of
  /// 'result', a flat VARBINARY vector. The format of the value is an
  /// implementation detail; storeSerializedRow() loads it back. Used for
  /// spilling, as it is cheaper than converting rows to columns.
  ///
  /// The container must have no accumulators and no raw dependent, which the
  /// format does not hold.
  void extractSerializedRows(RowSpan rows, const VectorPtr& result) const;

  /// Copies the value at 'index' in 'vector', made by extractSerializedRows(),
  /// into 'row'.
  ///
  /// The container must have no accumulators and no raw dependent.
  void storeSerializedRow(
      const FlatVector<StringView>& vector,
      vector_size_t index,
      char* row);

  // Reading values.

  /// Copies the values of 'column' in the first 'numRows' of 'rows' into
  /// 'result', starting at 'resultOffset'. A null entry in 'rows' sets null
  /// in 'result'.
  ///
  /// For rows of several containers with the same layout, such as the tables
  /// a parallel join build merges. For the rows of one container, use the
  /// overloads taking a column index.
  ///
  /// @param columnHasNulls Whether 'column' may have nulls in any of the
  /// containers. If false, the copy skips null checks.
  static void extractColumn(
      const char* const* rows,
      int32_t numRows,
      RowColumn column,
      bool columnHasNulls,
      const VectorPtr& result,
      vector_size_t resultOffset = 0);

  /// Copies the values of the column at 'columnIndex' in the first 'numRows'
  /// of 'rows' into 'result', starting at 'resultOffset'. A null entry in
  /// 'rows' sets null in 'result'.
  void extractColumn(
      const char* const* rows,
      int32_t numRows,
      column_index_t columnIndex,
      const VectorPtr& result,
      vector_size_t resultOffset = 0) const {
    extractColumn(
        rows,
        numRows,
        columnAt(columnIndex),
        columnHasNulls(columnIndex),
        result,
        resultOffset);
  }

  /// Like the overload above, but copies the rows at the positions in
  /// 'rowNumbers' of 'rows'.
  ///
  /// Positions may repeat and be out of order. A negative position sets null
  /// in 'result'.
  void extractColumn(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      column_index_t columnIndex,
      const VectorPtr& result,
      vector_size_t resultOffset = 0) const {
    extractColumn(
        rows,
        rowNumbers,
        columnAt(columnIndex),
        columnHasNulls(columnIndex),
        result,
        resultOffset);
  }

  /// Writes to 'result' a nulls buffer of the first 'numRows' of 'rows': row
  /// i is null if 'rows[i]' is null or its value in 'column' is null. Returns
  /// the number of null rows.
  ///
  /// 'result' must have at least bits::nbytes(numRows) bytes.
  static vector_size_t extractNulls(
      const char* const* rows,
      int32_t numRows,
      RowColumn column,
      const BufferPtr& result);

  /// Like the static overload, for the column at 'columnIndex'.
  vector_size_t extractNulls(
      const char* const* rows,
      int32_t numRows,
      column_index_t columnIndex,
      const BufferPtr& result) const {
    return extractNulls(rows, numRows, columnAt(columnIndex), result);
  }

  /// The raw dependent of 'row': 'rawDependentSize' bytes reserved at
  /// construction.
  std::span<char> rawDependent(char* row) const {
    VELOX_DCHECK_GT(layout_.rawDependentSize, 0);
    return {
        row + layout_.rawDependentOffset,
        static_cast<size_t>(layout_.rawDependentSize)};
  }

  /// Returns the location in a row of the column at 'columnIndex': a key, an
  /// accumulator or a dependent column, in that order. The raw dependent is not
  /// a column.
  ///
  /// Callers may read keys and dependent columns at this location, and read and
  /// write accumulators. Keys and dependent columns are written through
  /// store().
  RowColumn columnAt(column_index_t columnIndex) const {
    return layout_.rowColumns[columnIndex];
  }

  /// Returns the RowColumn for the i-th accumulator. Accumulators are laid
  /// out in row storage after the keys.
  RowColumn accumulatorColumnAt(column_index_t accumulatorIndex) const {
    return columnAt(keyTypes_.size() + accumulatorIndex);
  }

  /// Returns true if the value of 'rowColumn' in 'row' is null.
  static inline bool isNullAt(const char* row, const RowColumn& rowColumn) {
    return (row[rowColumn.nullByte()] & rowColumn.nullMask()) != 0;
  }

  /// Returns true if the value at rowColumn in row is NaN.
  template <typename T>
    requires std::is_floating_point_v<T>
  static inline bool isNanAt(const char* row, const RowColumn& rowColumn) {
    if (isNullAt(row, rowColumn)) {
      return false;
    }
    return std::isnan(valueAt<T>(row, rowColumn.offset()));
  }

  /// Returns the per row size of a fixed size column.
  int32_t fixedSizeAt(column_index_t columnIndex) const;

  /// Returns the size of a string or complex types value stored in the
  /// specified row and column.
  int32_t variableSizeAt(const char* row, column_index_t columnIndex) const;

  // Listing rows.

  /// Pass as 'maxBytes' to listRows() to list rows regardless of their size.
  static constexpr uint64_t kUnlimited = std::numeric_limits<uint64_t>::max();

  /// Which rows listRows() returns: all, those with the probed flag set, or
  /// those without.
  enum class ProbedFilter { kAll, kProbed, kNotProbed };

  /// Software-prefetch look-ahead for listRows() on x86, in bytes. A fixed byte
  /// distance keeps the run-ahead stable across row widths. Exposed as a
  /// template parameter so benchmarks can sweep the real scan implementation.
  static constexpr int32_t kListRowsPrefetchDistanceBytes = 2'048;

  /// Writes to 'rows', up to its size, rows that pass 'probedFilter', starting
  /// at the position of 'iter' and stopping after their total size exceeds
  /// 'maxBytes'. Returns the number of rows written, 0 at the end. No tsan
  /// because probed flags may have been set by a different thread. There is a
  /// barrier but tsan does not know this.
  template <
      ProbedFilter probedFilter,
      int32_t prefetchDistanceBytes = kListRowsPrefetchDistanceBytes>
  FOLLY_DISABLE_THREAD_SANITIZER int32_t listRows(
      RowContainerIterator* iter,
      std::span<char*> rows,
      uint64_t maxBytes) const;

  /// Lists rows from the array of row pointers kept if 'useListRowIndex' is
  /// true, without scanning the allocations or checking the free and probed
  /// flags.
  int32_t listRowsFast(RowContainerIterator* iter, std::span<char*> rows)
      const {
    size_t count = 0;
    while (count < rows.size() && iter->listRowCursor < rowPointers_.size()) {
      rows[count++] = rowPointers_[iter->listRowCursor];
      ++iter->listRowCursor;
    }
    return static_cast<int32_t>(count);
  }

  /// Writes to 'rows', up to its size, the rows from the position of 'iter'. A
  /// default constructed or reset iter starts at the beginning. Returns the
  /// number of rows written, 0 at the end. Stops after the total size of the
  /// rows written exceeds 'maxBytes'.
  int32_t listRows(
      RowContainerIterator* iter,
      std::span<char*> rows,
      uint64_t maxBytes) const {
    return listRows<ProbedFilter::kAll>(iter, rows, maxBytes);
  }

  int32_t listRows(RowContainerIterator* iter, std::span<char*> rows) const {
    if (useListRowIndex_) {
      return listRowsFast(iter, rows);
    }
    return listRows<ProbedFilter::kAll>(iter, rows, kUnlimited);
  }

  // Partitioning rows for a parallel hash join build. The build splits the
  // buckets of the table into ranges, the partitions, one per thread. It
  // records the partition of each row in a RowPartitions, then each thread
  // lists the rows of its partition and inserts them.

  /// Makes a container for a partition number per row of this container, for
  /// a parallel join build to fill. Allocates it from 'pool', which may differ
  /// from the pool of this container: a parallel join build allocates the
  /// RowPartitions of every table it merges from its own pool.
  ///
  /// Makes this container immutable, so it can be called only once.
  std::unique_ptr<RowPartitions> createRowPartitions(memory::MemoryPool& pool);

  /// Retrieves rows from 'iterator' whose partition equals 'partition'. Writes
  /// pointers to the rows to 'rows', up to its size. 'rowPartitions' contains
  /// the partition number of each row in this container. The function returns
  /// the number of rows retrieved, 0 when no more rows are found. 'iterator' is
  /// expected to be in initial state on first call.
  int32_t listPartitionRows(
      RowContainerIterator& iterator,
      uint8_t partition,
      const RowPartitions& rowPartitions,
      std::span<char*> rows) const;

  // Comparing and hashing rows.

  /// Compares the value at 'column' in 'row' with the value at 'index' in
  /// 'decoded'. Returns 0 for equal, < 0 for 'row' < 'decoded', > 0 otherwise.
  /// 'mayHaveNulls' specifies if nulls need to be checked. This is a fast path
  /// for compare().
  template <bool mayHaveNulls = true>
  int32_t compare(
      const char* row,
      RowColumn column,
      const DecodedVector& decoded,
      vector_size_t index,
      CompareFlags flags = CompareFlags()) const;

  /// Compares the value at 'columnIndex' between 'left' and 'right'. Returns
  /// 0 for equal, < 0 for left < right, > 0 otherwise.
  int32_t compare(
      const char* left,
      const char* right,
      column_index_t columnIndex,
      CompareFlags flags = CompareFlags()) const;

  /// Compares the value between 'left' at 'leftIndex' and 'right' and
  /// 'rightIndex'. Returns 0 for equal, < 0 for left < right, > 0 otherwise.
  /// Both columns should have the same type.
  int32_t compare(
      const char* left,
      const char* right,
      int leftColumnIndex,
      int rightColumnIndex,
      CompareFlags flags = CompareFlags()) const;

  /// Compares rows 'left' and 'right' of this container by their keys only,
  /// in order. Returns 0 if all keys are equal, even if other columns differ,
  /// and otherwise the result of the first key that differs: < 0 if 'left'
  /// comes first, > 0 if 'right' does.
  ///
  /// @param flags The comparison flags of each key, or empty for the
  /// defaults.
  int32_t compareRows(
      const char* left,
      const char* right,
      const std::vector<CompareFlags>& flags = {}) const;

  /// Writes to 'result' the hashes of 'rows' by all their keys. 'result' must
  /// have space for 'rows.size()' hashes.
  void hashKeys(RowSpan rows, std::span<uint64_t> result) const;

  // Hash table fields: the probed flag, the count and the normalized key, each
  // present if the constructor's options ask for it.

  /// Sets 'probed' flag for the specified rows. Used by the right and
  /// full join to mark build-side rows that matches join
  /// condition. 'rows' may contain duplicate entries for the cases
  /// where single probe row matched multiple build rows. In case of
  /// the full join, 'rows' may include null entries that correspond
  /// to probe rows with no match. No tsan because any thread can set
  /// this without synchronization. There is a barrier between setting
  /// and reading.
  FOLLY_DISABLE_THREAD_SANITIZER
  void setProbedFlag(RowSpan rows);

  /// Sets 'probed' flag for a single non-null row. Returns true if the flag
  /// transitioned from false to true.
  FOLLY_DISABLE_THREAD_SANITIZER
  bool testAndSetProbedFlag(char* row);

  /// Writes the probed flags of 'rows' into 'result',
  /// which must be a flat BOOLEAN vector. Used by right semi project joins,
  /// null-aware or not.
  ///
  /// @param setNullForNullKeysRow Whether a row with a null key gets null
  /// instead of its flag.
  /// @param setNullForNonProbedRow Whether a row that was not probed gets null
  /// instead of false.
  void extractProbedFlags(
      RowSpan rows,
      bool setNullForNullKeysRow,
      bool setNullForNonProbedRow,
      const VectorPtr& result) const;

  /// Returns the count stored at the given row. Used for counting joins.
  int32_t count(const char* row) const {
    VELOX_DCHECK_NE(layout_.countOffset, 0);
    return countRef(const_cast<char*>(row));
  }

  /// Increments the count at the given row. Used during hash table build for
  /// counting joins.
  void incrementCount(char* row) const {
    VELOX_DCHECK_NE(layout_.countOffset, 0);
    ++countRef(row);
  }

  /// Decrements the count at the given row. Used during hash table probe for
  /// counting joins.
  void decrementCount(char* row) const {
    VELOX_DCHECK_NE(layout_.countOffset, 0);
    --countRef(row);
  }

  /// Adds 'n' to the count at the given row. Used during hash table merge
  /// to combine counts from multiple build-side tables.
  void addCount(char* row, int32_t n) const {
    VELOX_DCHECK_NE(layout_.countOffset, 0);
    countRef(row) += n;
  }

  /// Allows get/set of the normalized key. If normalized keys are used, they
  /// are stored in the word immediately below the row.
  static inline normalized_key_t& normalizedKey(char* row) {
    return reinterpret_cast<normalized_key_t*>(row)[-1];
  }

  /// Stops reserving a normalized key below new rows. Rows made before keep
  /// theirs. Lasts until clear(). Used by a hash table that switches to hash
  /// mode, which does not use normalized keys.
  void disableNormalizedKeys() {
    normalizedKeySize_ = 0;
  }

  // Memory and column statistics.

  /// Returns the bytes 'row' takes: the fixed-width part, plus the
  /// variable-width data it holds in the string allocator if the container
  /// has variable-width columns or variable-width accumulators. Excludes the
  /// normalized key below the row.
  uint32_t rowSize(const char* row) const {
    return layout_.fixedRowSize +
        (layout_.rowSizeOffset
             ? *reinterpret_cast<const uint32_t*>(row + layout_.rowSizeOffset)
             : 0);
  }

  /// Returns the bytes the container holds from its memory pool: the
  /// allocations for fixed-width rows and for variable-width data. Includes
  /// space not yet used and space of erased rows and freed values kept for
  /// reuse, so it can exceed the sum of rowSize() over all rows.
  uint64_t allocatedBytes() const {
    return rows_.allocatedBytes() + stringAllocator_->retainedSize();
  }

  /// Returns the bytes of variable-width data the rows hold, with allocator
  /// overhead. Leaves out free space in the string allocator. Includes the
  /// array of row pointers kept if 'useListRowIndex' is true, which the string
  /// allocator also holds.
  int64_t usedVariableWidthBytes() const {
    return stringAllocator_->retainedSize() - stringAllocator_->freeSpace();
  }

  /// Returns the average bytes a row uses: its fixed-width part, including
  /// padding and the normalized key, plus its variable-width data with
  /// allocator overhead. Leaves out space not used yet and the space of erased
  /// rows and freed values, so it is smaller than allocatedBytes() / numRows().
  ///
  /// Returns nullopt if the container has no rows.
  std::optional<int64_t> estimateRowSize() const;

  /// Returns an upper bound on the bytes the container takes from its pool to
  /// add 'numRows' rows holding 'variableWidthBytes' of variable-width data in
  /// total. Erased rows, the rest of the current allocation run and free
  /// variable-width space are used first, and each part is rounded up to a
  /// huge page. Operators use it to reserve memory
  /// before adding input.
  int64_t maxGrowthBytes(vector_size_t numRows, int64_t variableWidthBytes)
      const;

  /// Returns the stats of the key or dependent column at 'columnIndex'.
  ///
  /// The container keeps them up to date: store() and storeSerializedRow() add
  /// a value, eraseRows() and initializeRow() with 'reuse' remove the values
  /// of a row, setAllNull() turns them into nulls, and clear() resets them. A
  /// column not stored in a row counts as an empty non-null value. After a
  /// value is removed, the maximum size is a bound; see RowColumnStats.
  const RowColumnStats& columnStats(column_index_t columnIndex) const {
    return rowColumnsStats_[columnIndex];
  }

  /// Returns the stats of the column at 'columnIndex' over all of
  /// 'containers', which have the same layout. For example, the containers of
  /// a hash table and the tables merged into it.
  static RowColumnStats columnStats(
      std::span<RowContainer* const> containers,
      column_index_t columnIndex);

  /// Returns true if the key or dependent column at 'columnIndex' has nulls.
  bool columnHasNulls(column_index_t columnIndex) const {
    return rowColumnsStats_[columnIndex].nullCount() > 0;
  }

  // Debugging and testing.

  /// Returns a summary of the container: key types, dependent types, number of
  /// accumulators and number of rows.
  std::string toString() const;

  /// Returns a string representation of the specified row in the same format as
  /// BaseVector::toString(index).
  std::string toString(const char* row) const;

  /// Checks that the list of free rows, the row count and the column stats
  /// agree with the rows. Throws if they do not.
  ///
  /// Visits every row, so it is meant for tests.
  void checkConsistency() const;

  const std::vector<char*, StlAllocator<char*>>& testingRowPointers() const {
    return rowPointers_;
  }

  /// Adds 'bytes' to the variable-width size of 'row', saturating at the
  /// maximum of uint32_t, as storing variable-width data would. Lets tests
  /// make rows of any size.
  void testingIncrementRowSize(char* row, uint64_t bytes) {
    uint32_t* size = reinterpret_cast<uint32_t*>(row + layout_.rowSizeOffset);
    *size =
        std::min<uint64_t>(*size + bytes, std::numeric_limits<uint32_t>::max());
  }

  /// Advances 'iterator' by 'numRows' rows. Its current row is then null if
  /// past the end.
  void testingSkip(RowContainerIterator& iterator, int32_t numRows) const {
    skip(iterator, numRows);
  }

  bool testingMutable() const {
    return mutable_;
  }

 private:
  // Where each part of a row is. Fixed at construction.
  //
  // A row has these parts, in order:
  // - The keys. A string or a complex type value is a StringView or a
  //   std::string_view pointing to its data in the string allocator. The keys
  //   take at least the size of a pointer, which holds the next free row on a
  //   free row.
  // - The flags, one bit each: the null flags of the keys if they are
  //   nullable; for each accumulator, a null flag and an initialized flag,
  //   starting on a new byte so that both are in the same byte; the null flags
  //   of the dependent columns; the probed flag, if any; the free flag.
  // - The accumulators, each aligned as it asks.
  // - The dependent columns.
  // - The raw dependent, if any, 8-byte aligned.
  // - The size of the variable-width data of the row, if there are
  //   variable-width columns or accumulators.
  // - The pointer to the next row with the same keys, if any.
  // - The count, if any.
  //
  // The row is padded to a multiple of the largest alignment. If
  // 'hasNormalizedKey' is set, a normalized key, padded to the same alignment,
  // sits below the row, until disableNormalizedKeys().
  //
  // Offsets are in bytes from the start of the row unless named as bit
  // offsets. An optional part that is absent has offset 0.
  struct Layout {
    // Lays out rows of the given columns, with what 'options' adds. Throws if
    // there are both accumulators and dependent columns.
    Layout(
        const std::vector<TypePtr>& keyTypes,
        const std::vector<Accumulator>& accumulators,
        const std::vector<TypePtr>& dependentTypes,
        const Options& options);

    // The types of the keys and dependent columns, keys first. A container
    // never has both accumulators and dependent columns, so an index into
    // 'types' is also an index into 'rowColumns'.
    std::vector<TypePtr> types;
    std::vector<TypeKind> typeKinds;

    // The offsets and null flags of the keys, accumulators and dependent
    // columns, in that order.
    std::vector<RowColumn> rowColumns;

    // The offset and the size of the flags: null, initialized, probed, free.
    int32_t flagsByteOffset{0};
    int32_t flagBytes{0};

    // The bit offsets of the free and probed flags.
    int32_t freeFlagOffset{0};
    int32_t probedFlagOffset{0};

    // The offset and the size of the raw dependent.
    int32_t rawDependentOffset{0};
    int32_t rawDependentSize{0};

    // The offset of the uint32_t size of a row's variable-width data.
    int32_t rowSizeOffset{0};

    // The offsets of the pointer to the next row with the same keys and of
    // the count.
    int32_t nextOffset{0};
    int32_t countOffset{0};

    // The size of the fixed-width part of a row, a multiple of 'alignment'.
    int32_t fixedRowSize{0};
    int32_t alignment{1};

    // The bytes reserved below each row for a normalized key, if enabled.
    int32_t normalizedKeySize{0};

    // Whether any accumulator allocates memory outside the container.
    bool usesExternalMemory{false};
  };

  // Offset of the pointer to the next free row on a free row.
  static constexpr int32_t kNextFreeOffset = 0;

  template <typename T>
  static inline T valueAt(const char* row, int32_t offset) {
    return *reinterpret_cast<const T*>(row + offset);
  }

  template <typename T>
  static inline T& valueAt(char* row, int32_t offset) {
    return *reinterpret_cast<T*>(row + offset);
  }

  // Copies a string or complex type value from the specified row and column
  // into provided buffer. Stored the size of the data in the first 4 bytes of
  // the buffer. If the value is null, writes zero into the first 4 bytes of
  // destination and returns.
  // @return The number of bytes written to 'destination' including the 4 bytes
  // of the size.
  int32_t extractVariableSizeAt(
      const char* row,
      column_index_t columnIndex,
      char* output) const;

  // Copies a string or complex type value from 'data' into the specified row
  // and column. Expects first 4 bytes in 'data' to contain the size of the
  // string or complex value.
  // @return The number of bytes read from 'data': 4 bytes for size + that many
  // bytes.
  int32_t
  storeVariableSizeAt(const char* data, char* row, column_index_t columnIndex);

  // Copies the rows at the positions in 'rowNumbers' of 'rows'. See the
  // public overload taking a column index.
  static void extractColumn(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      RowColumn column,
      bool columnHasNulls,
      const VectorPtr& result,
      vector_size_t resultOffset);

  template <TypeKind Kind>
  static void extractColumnTyped(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      RowColumn column,
      bool columnHasNulls,
      int32_t resultOffset,
      const VectorPtr& result);

  template <bool useRowNumbers, TypeKind Kind>
  static void extractColumnTypedInternal(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      RowColumn column,
      bool columnHasNulls,
      int32_t resultOffset,
      const VectorPtr& result);

  // Updates the column stats for the values of 'row' going away: if
  // 'setToNull' is true, they become nulls; otherwise, they are removed.
  void removeOrUpdateRowColumnStats(const char* row, bool setToNull);

  char*& nextFree(char* row) const {
    return *reinterpret_cast<char**>(row + kNextFreeOffset);
  }

  uint32_t& variableRowSize(char* row) const {
    VELOX_DCHECK(layout_.rowSizeOffset);
    return *reinterpret_cast<uint32_t*>(row + layout_.rowSizeOffset);
  }

  template <TypeKind Kind>
  inline void storeWithNulls(
      const DecodedVector& decoded,
      vector_size_t rowIndex,
      bool isKey,
      char* row,
      int32_t offset,
      int32_t nullByte,
      uint8_t nullMask,
      column_index_t columnIndex);

  template <TypeKind Kind>
  inline void storeNoNulls(
      const DecodedVector& decoded,
      vector_size_t rowIndex,
      bool isKey,
      char* row,
      int32_t offset);

  template <TypeKind Kind>
  inline void storeWithNullsBatch(
      const DecodedVector& decoded,
      RowSpan rows,
      bool isKey,
      int32_t offset,
      int32_t nullByte,
      uint8_t nullMask,
      column_index_t columnIndex) {
    for (int32_t i = 0; i < rows.size(); ++i) {
      storeWithNulls<Kind>(
          decoded, i, isKey, rows[i], offset, nullByte, nullMask, columnIndex);
      updateColumnStats(rows[i], columnIndex);
    }
  }

  template <TypeKind Kind>
  inline void storeNoNullsBatch(
      const DecodedVector& decoded,
      RowSpan rows,
      bool isKey,
      int32_t offset,
      column_index_t columnIndex) {
    for (int32_t i = 0; i < rows.size(); ++i) {
      storeNoNulls<Kind>(decoded, i, isKey, rows[i], offset);
      updateColumnStats(rows[i], columnIndex);
    }
  }

  template <bool useRowNumbers, typename T>
  static void extractValuesWithNulls(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      RowColumn column,
      int32_t resultOffset,
      FlatVector<T>* result);

  template <bool useRowNumbers, typename T>
  static void extractValuesNoNulls(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      int32_t offset,
      int32_t resultOffset,
      FlatVector<T>* result);

  static HashStringAllocator::InputStream prepareRead(
      const char* row,
      int32_t offset);

  // Writes to 'result' the hashes of the values of the column at
  // 'columnIndex' in 'rows'. If 'mix' is true, mixes each hash into the value
  // already in 'result' instead.
  void hash(
      column_index_t columnIndex,
      RowSpan rows,
      bool mix,
      uint64_t* result) const;

  template <bool typeProvidesCustomComparison, TypeKind Kind>
  void hashTyped(
      const Type* type,
      RowColumn column,
      bool nullable,
      RowSpan rows,
      bool mix,
      uint64_t* result) const;

  template <bool mayHaveNulls, TypeKind Kind>
  inline int compare(
      const char* row,
      RowColumn column,
      const DecodedVector& decoded,
      vector_size_t index,
      CompareFlags flags) const {
    if (decoded.base()->typeUsesCustomComparison()) {
      return compare<true, Kind, mayHaveNulls>(
          row, column, decoded, index, flags);
    } else {
      return compare<false, Kind, mayHaveNulls>(
          row, column, decoded, index, flags);
    }
  }

  template <bool typeProvidesCustomComparison, TypeKind Kind, bool mayHaveNulls>
  inline int compare(
      const char* row,
      RowColumn column,
      const DecodedVector& decoded,
      vector_size_t index,
      CompareFlags flags) const;

  template <bool typeProvidesCustomComparison, TypeKind Kind>
  inline int compare(
      const char* left,
      const char* right,
      const Type* type,
      RowColumn leftColumn,
      RowColumn rightColumn,
      CompareFlags flags) const;

  template <bool typeProvidesCustomComparison, TypeKind Kind>
  inline int compare(
      const char* left,
      const char* right,
      const Type* type,
      RowColumn column,
      CompareFlags flags) const {
    return compare<typeProvidesCustomComparison, Kind>(
        left, right, type, column, column, flags);
  }

  void storeComplexType(
      const DecodedVector& decoded,
      vector_size_t index,
      bool isKey,
      char* row,
      int32_t offset,
      int32_t nullByte = 0,
      uint8_t nullMask = 0);

  template <bool useRowNumbers>
  static void extractComplexType(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      RowColumn column,
      int32_t resultOffset,
      const VectorPtr& result);

  static void extractString(
      StringView value,
      FlatVector<StringView>* values,
      vector_size_t index);

  static int32_t compareStringAsc(
      StringView left,
      const DecodedVector& decoded,
      vector_size_t index);

  static int32_t compareStringAsc(StringView left, StringView right);

  int32_t compareComplexType(
      const char* row,
      int32_t offset,
      const DecodedVector& decoded,
      vector_size_t index,
      CompareFlags flags = CompareFlags()) const;

  int32_t compareComplexType(
      const char* left,
      const char* right,
      const Type* type,
      int32_t offset,
      CompareFlags flags) const;

  int32_t compareComplexType(
      const char* left,
      const char* right,
      const Type* type,
      int32_t leftOffset,
      int32_t rightOffset,
      CompareFlags flags = CompareFlags()) const;

  // Frees the out-of-line data of the values of 'rows' in the column at
  // 'columnIndex'. 'ViewType' is how a row holds a value of the column:
  // StringView for VARCHAR and VARBINARY, std::string_view for ARRAY, MAP and
  // ROW.
  template <typename ViewType>
  void freeVariableWidthValues(column_index_t columnIndex, RowSpan rows);

  // Advances 'iterator' by 'numRows' rows. Its current row is then null if
  // past the end.
  void skip(RowContainerIterator& iterator, int32_t numRows) const;

  // Throws if the rows have parts that extractSerializedRows() does not hold.
  void checkSerializable() const;

  // Frees the out-of-line data of the variable-width values of 'rows'.
  void freeVariableWidthValues(RowSpan rows);

  // Frees the memory the accumulators of 'rows' hold outside the rows.
  void freeAggregates(std::span<char*> rows);

  // Frees the memory 'rows' hold outside the rows and removes them from the
  // row count.
  void freeRowsExtraMemory(std::span<char*> rows);

  // Adds the value of 'row' at 'columnIndex', just stored, to the column stats.
  void updateColumnStats(const char* row, column_index_t columnIndex);

  int32_t& countRef(char* row) const {
    return *reinterpret_cast<int32_t*>(row + layout_.countOffset);
  }

  const std::vector<TypePtr> keyTypes_;
  const bool nullableKeys_;
  // True if use 'listRowsFast'.
  const bool useListRowIndex_;
  const Layout layout_;
  // Allocates the variable-width data of rows and 'rowPointers_'.
  const std::unique_ptr<HashStringAllocator> stringAllocator_;

  // Whether rows can be added. Set to false by createRowPartitions().
  bool mutable_{true};

  std::vector<Accumulator> accumulators_;

  // The stats of the keys and dependent columns, in the order of
  // 'layout_.types'.
  std::vector<RowColumnStats> rowColumnsStats_;

  // The count of entries that have an extra normalized_key_t before the
  // start.
  int64_t numRowsWithNormalizedKey_{0};
  // Extra bytes to reserve before each added row for a normalized key. Set to
  // 0 after deciding not to use normalized keys.
  int normalizedKeySize_;
  // The rows added and not erased.
  uint64_t numRows_{0};
  // Head of linked list of free rows.
  char* firstFreeRow_{nullptr};
  // The erased rows on the free list, reused by newRow().
  uint64_t numFreeRows_{0};

  // Allocates the fixed-width parts of rows, in runs of pages so that rows
  // never move.
  memory::AllocationPool rows_;
  std::vector<char*, StlAllocator<char*>> rowPointers_;
};

/// A comparator of rows stored in the RowContainer compatible with
/// std::priority_queue. Uses specified columns and sorting orders for
/// comparison.
class RowComparator {
 public:
  RowComparator(
      const RowTypePtr& rowType,
      const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys,
      const std::vector<core::SortOrder>& sortingOrders,
      RowContainer* rowContainer);

  /// Returns true if lhs < rhs, false otherwise.
  bool operator()(const char* lhs, const char* rhs);

  /// Returns 0 for equal, < 0 for lhs < rhs, > 0 otherwise.
  int compare(const char* lhs, const char* rhs);

  /// Returns true if decodedVectors[index] < other, false otherwise.
  bool operator()(
      const std::vector<DecodedVector>& decodedVectors,
      vector_size_t index,
      const char* other);

  /// Returns 0 for equal, < 0 for decodedVectors[index] < other,
  /// > 0 otherwise.
  int32_t compare(
      const std::vector<DecodedVector>& decodedVectors,
      vector_size_t index,
      const char* other);

 private:
  std::vector<std::pair<column_index_t, core::SortOrder>> keyInfo_;
  RowContainer* rowContainer_;
};

template <
    RowContainer::ProbedFilter probedFilter,
    int32_t prefetchDistanceBytes>
FOLLY_DISABLE_THREAD_SANITIZER int32_t RowContainer::listRows(
    RowContainerIterator* iter,
    std::span<char*> rows,
    uint64_t maxBytes) const {
  if (rows.empty()) {
    return 0;
  }
  const auto maxRows = static_cast<int32_t>(rows.size());
  int32_t count = 0;
  uint64_t totalBytes = 0;
  auto numAllocations = rows_.numRanges();
  if (iter->allocationIndex == 0 && iter->rowOffset == 0) {
    iter->normalizedKeysLeft = numRowsWithNormalizedKey_;
    iter->normalizedKeySize = layout_.normalizedKeySize;
  }
  int32_t rowSize = layout_.fixedRowSize +
      (iter->normalizedKeysLeft > 0 ? layout_.normalizedKeySize : 0);
  for (auto i = iter->allocationIndex; i < numAllocations; ++i) {
    auto range = rows_.rangeAt(i);
    auto* data = range.data() +
        memory::alignmentPadding(range.data(), layout_.alignment);
    auto limit = range.size() -
        (reinterpret_cast<uintptr_t>(data) -
         reinterpret_cast<uintptr_t>(range.data()));
    auto row = iter->rowOffset;
    while (row + rowSize <= limit) {
      // Tuned and validated on x86 only; unmeasured on other architectures.
#if defined(__x86_64__)
      // Use integer arithmetic to avoid out-of-bounds pointer arithmetic when
      // the target lies past the current range; the hint itself is
      // non-faulting for an invalid target on x86.
      if constexpr (prefetchDistanceBytes > 0) {
        __builtin_prefetch(
            reinterpret_cast<const char*>(
                reinterpret_cast<uintptr_t>(data) + row +
                prefetchDistanceBytes));
      }
#endif
      rows[count++] = data + row +
          (iter->normalizedKeysLeft > 0 ? layout_.normalizedKeySize : 0);
      VELOX_DCHECK_EQ(
          reinterpret_cast<uintptr_t>(rows[count - 1]) % layout_.alignment, 0);
      row += rowSize;
      auto newTotalBytes = totalBytes + rowSize;
      if (--iter->normalizedKeysLeft == 0) {
        rowSize -= layout_.normalizedKeySize;
      }
      if (bits::isBitSet(rows[count - 1], layout_.freeFlagOffset)) {
        --count;
        continue;
      }
      if constexpr (probedFilter == ProbedFilter::kNotProbed) {
        if (bits::isBitSet(rows[count - 1], layout_.probedFlagOffset)) {
          --count;
          continue;
        }
      }
      if constexpr (probedFilter == ProbedFilter::kProbed) {
        if (not(bits::isBitSet(rows[count - 1], layout_.probedFlagOffset))) {
          --count;
          continue;
        }
      }
      totalBytes = newTotalBytes;
      if (layout_.rowSizeOffset) {
        totalBytes += variableRowSize(rows[count - 1]);
      }
      if (count == maxRows || totalBytes > maxBytes) {
        iter->rowOffset = row;
        iter->allocationIndex = i;
        return count;
      }
    }
    iter->rowOffset = 0;
  }
  iter->allocationIndex = std::numeric_limits<int32_t>::max();
  return count;
}

template <>
inline int128_t RowContainer::valueAt<int128_t>(
    const char* row,
    int32_t offset) {
  return HugeInt::deserialize(row + offset);
}

template <>
inline void RowContainer::storeWithNulls<TypeKind::ROW>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool isKey,
    char* row,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    column_index_t columnIndex) {
  storeComplexType(decoded, rowIndex, isKey, row, offset, nullByte, nullMask);
}

template <>
inline void RowContainer::storeNoNulls<TypeKind::ROW>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool isKey,
    char* row,
    int32_t offset) {
  storeComplexType(decoded, rowIndex, isKey, row, offset);
}

template <>
inline void RowContainer::storeWithNulls<TypeKind::ARRAY>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool isKey,
    char* row,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    column_index_t columnIndex) {
  storeComplexType(decoded, rowIndex, isKey, row, offset, nullByte, nullMask);
}

template <>
inline void RowContainer::storeNoNulls<TypeKind::ARRAY>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool isKey,
    char* row,
    int32_t offset) {
  storeComplexType(decoded, rowIndex, isKey, row, offset);
}

template <>
inline void RowContainer::storeWithNulls<TypeKind::MAP>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool isKey,
    char* row,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    column_index_t columnIndex) {
  storeComplexType(decoded, rowIndex, isKey, row, offset, nullByte, nullMask);
}

template <>
inline void RowContainer::storeNoNulls<TypeKind::MAP>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool isKey,
    char* row,
    int32_t offset) {
  storeComplexType(decoded, rowIndex, isKey, row, offset);
}

template <>
inline void RowContainer::storeWithNulls<TypeKind::HUGEINT>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool /*isKey*/,
    char* row,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    column_index_t columnIndex) {
  if (decoded.isNullAt(rowIndex)) {
    row[nullByte] |= nullMask;
    memset(row + offset, 0, sizeof(int128_t));
    return;
  }
  HugeInt::serialize(decoded.valueAt<int128_t>(rowIndex), row + offset);
}

template <>
inline void RowContainer::storeNoNulls<TypeKind::HUGEINT>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool /*isKey*/,
    char* row,
    int32_t offset) {
  HugeInt::serialize(decoded.valueAt<int128_t>(rowIndex), row + offset);
}

template <>
inline void RowContainer::extractColumnTyped<TypeKind::OPAQUE>(
    const char* const* /*rows*/,
    folly::Range<const vector_size_t*> /*rowNumbers*/,
    int32_t /*numRows*/,
    RowColumn /*column*/,
    bool /*columnHasNulls*/,
    int32_t /*resultOffset*/,
    const VectorPtr& /*result*/) {
  VELOX_UNSUPPORTED("RowContainer doesn't support values of type OPAQUE");
}

inline void RowContainer::extractColumn(
    const char* const* rows,
    int32_t numRows,
    RowColumn column,
    bool columnHasNulls,
    const VectorPtr& result,
    vector_size_t resultOffset) {
  VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      extractColumnTyped,
      result->typeKind(),
      rows,
      {},
      numRows,
      column,
      columnHasNulls,
      resultOffset,
      result);
}

inline void RowContainer::extractColumn(
    const char* const* rows,
    folly::Range<const vector_size_t*> rowNumbers,
    RowColumn column,
    bool columnHasNulls,
    const VectorPtr& result,
    vector_size_t resultOffset) {
  VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      extractColumnTyped,
      result->typeKind(),
      rows,
      rowNumbers,
      rowNumbers.size(),
      column,
      columnHasNulls,
      resultOffset,
      result);
}

inline vector_size_t RowContainer::extractNulls(
    const char* const* rows,
    int32_t numRows,
    RowColumn column,
    const BufferPtr& result) {
  VELOX_CHECK_GE(result->size(), bits::nbytes(numRows));
  auto* rawResult = result->asMutable<uint64_t>();
  bits::fillBits(rawResult, 0, numRows, bits::kNotNull);

  const bool nullable = column.nullMask() != 0;
  vector_size_t numNulls = 0;
  for (int32_t i = 0; i < numRows; ++i) {
    const char* row = rows[i];
    if (row == nullptr || (nullable && isNullAt(row, column))) {
      bits::setNull(rawResult, i);
      ++numNulls;
    }
  }
  return numNulls;
}

template <bool mayHaveNulls>
inline int RowContainer::compare(
    const char* row,
    RowColumn column,
    const DecodedVector& decoded,
    vector_size_t index,
    CompareFlags flags) const {
  return VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH_ALL(
      compare,
      mayHaveNulls,
      decoded.base()->typeKind(),
      row,
      column,
      decoded,
      index,
      flags);
}

inline int RowContainer::compare(
    const char* left,
    const char* right,
    column_index_t columnIndex,
    CompareFlags flags) const {
  auto type = layout_.types[columnIndex].get();
  if (type->providesCustomComparison()) {
    return VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH_ALL(
        compare,
        true,
        type->kind(),
        left,
        right,
        type,
        columnAt(columnIndex),
        flags);
  } else {
    return VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH_ALL(
        compare,
        false,
        type->kind(),
        left,
        right,
        type,
        columnAt(columnIndex),
        flags);
  }
}

inline int RowContainer::compare(
    const char* left,
    const char* right,
    int leftColumnIndex,
    int rightColumnIndex,
    CompareFlags flags) const {
  auto leftType = layout_.types[leftColumnIndex].get();
  auto rightType = layout_.types[rightColumnIndex].get();
  VELOX_CHECK(leftType->equivalent(*rightType));

  if (leftType->providesCustomComparison()) {
    return VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH_ALL(
        compare,
        true,
        leftType->kind(),
        left,
        right,
        leftType,
        columnAt(leftColumnIndex),
        columnAt(rightColumnIndex),
        flags);
  } else {
    return VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH_ALL(
        compare,
        false,
        leftType->kind(),
        left,
        right,
        leftType,
        columnAt(leftColumnIndex),
        columnAt(rightColumnIndex),
        flags);
  }
}

inline int32_t RowContainer::compareRows(
    const char* left,
    const char* right,
    const std::vector<CompareFlags>& flags) const {
  VELOX_DCHECK(flags.empty() || flags.size() == keyTypes_.size());
  for (auto i = 0; i < keyTypes_.size(); ++i) {
    auto result =
        compare(left, right, i, flags.empty() ? CompareFlags() : flags[i]);
    if (result) {
      return result;
    }
  }
  return 0;
}

template <TypeKind Kind>
inline void RowContainer::extractColumnTyped(
    const char* const* rows,
    folly::Range<const vector_size_t*> rowNumbers,
    int32_t numRows,
    RowColumn column,
    bool columnHasNulls,
    int32_t resultOffset,
    const VectorPtr& result) {
  if (rowNumbers.size() > 0) {
    extractColumnTypedInternal<true, Kind>(
        rows,
        rowNumbers,
        rowNumbers.size(),
        column,
        columnHasNulls,
        resultOffset,
        result);
  } else {
    extractColumnTypedInternal<false, Kind>(
        rows,
        rowNumbers,
        numRows,
        column,
        columnHasNulls,
        resultOffset,
        result);
  }
}

template <bool useRowNumbers, TypeKind Kind>
inline void RowContainer::extractColumnTypedInternal(
    const char* const* rows,
    folly::Range<const vector_size_t*> rowNumbers,
    int32_t numRows,
    RowColumn column,
    bool columnHasNulls,
    int32_t resultOffset,
    const VectorPtr& result) {
  // Resize the result vector before all copies.
  result->resize(numRows + resultOffset);

  if constexpr (
      Kind == TypeKind::ROW || Kind == TypeKind::ARRAY ||
      Kind == TypeKind::MAP) {
    extractComplexType<useRowNumbers>(
        rows, rowNumbers, numRows, column, resultOffset, result);
    return;
  }
  using T = typename KindToFlatVector<Kind>::HashRowType;
  auto* flatResult = result->as<FlatVector<T>>();
  auto nullMask = column.nullMask();
  auto offset = column.offset();
  if (!nullMask || !columnHasNulls) {
    extractValuesNoNulls<useRowNumbers, T>(
        rows, rowNumbers, numRows, offset, resultOffset, flatResult);
  } else {
    extractValuesWithNulls<useRowNumbers, T>(
        rows, rowNumbers, numRows, column, resultOffset, flatResult);
  }
}

template <TypeKind Kind>
inline void RowContainer::storeWithNulls(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool isKey,
    char* row,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    column_index_t columnIndex) {
  using T = typename TypeTraits<Kind>::NativeType;
  if (decoded.isNullAt(rowIndex)) {
    row[nullByte] |= nullMask;
    // Do not leave an uninitialized value in the case of a
    // null. This is an error with valgrind/asan.
    *reinterpret_cast<T*>(row + offset) = T();
    return;
  }
  if constexpr (std::is_same_v<T, StringView>) {
    RowSizeTracker tracker(row[layout_.rowSizeOffset], *stringAllocator_);
    stringAllocator_->copyMultipart(decoded.valueAt<T>(rowIndex), row, offset);
  } else {
    *reinterpret_cast<T*>(row + offset) = decoded.valueAt<T>(rowIndex);
  }
}

template <TypeKind Kind>
inline void RowContainer::storeNoNulls(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool isKey,
    char* row,
    int32_t offset) {
  using T = typename TypeTraits<Kind>::NativeType;
  if constexpr (std::is_same_v<T, StringView>) {
    RowSizeTracker tracker(row[layout_.rowSizeOffset], *stringAllocator_);
    stringAllocator_->copyMultipart(decoded.valueAt<T>(rowIndex), row, offset);
  } else {
    *reinterpret_cast<T*>(row + offset) = decoded.valueAt<T>(rowIndex);
  }
}

template <bool useRowNumbers, typename T>
inline void RowContainer::extractValuesWithNulls(
    const char* const* rows,
    folly::Range<const vector_size_t*> rowNumbers,
    int32_t numRows,
    RowColumn column,
    int32_t resultOffset,
    FlatVector<T>* result) {
  auto maxRows = numRows + resultOffset;
  VELOX_DCHECK_LE(maxRows, result->size());

  BufferPtr& nullBuffer = result->mutableNulls(maxRows, true);
  auto nulls = nullBuffer->asMutable<uint64_t>();
  BufferPtr valuesBuffer = result->mutableValues();
  [[maybe_unused]] auto values = valuesBuffer->asMutableRange<T>();
  for (int32_t i = 0; i < numRows; ++i) {
    const char* row;
    if constexpr (useRowNumbers) {
      auto rowNumber = rowNumbers[i];
      row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
    } else {
      row = rows[i];
    }
    auto resultIndex = resultOffset + i;
    if (row == nullptr || isNullAt(row, column)) {
      bits::setNull(nulls, resultIndex, true);
    } else {
      bits::setNull(nulls, resultIndex, false);
      if constexpr (std::is_same_v<T, StringView>) {
        extractString(
            valueAt<StringView>(row, column.offset()), result, resultIndex);
      } else {
        values[resultIndex] = valueAt<T>(row, column.offset());
      }
    }
  }
}

template <bool useRowNumbers, typename T>
inline void RowContainer::extractValuesNoNulls(
    const char* const* rows,
    folly::Range<const vector_size_t*> rowNumbers,
    int32_t numRows,
    int32_t offset,
    int32_t resultOffset,
    FlatVector<T>* result) {
  [[maybe_unused]] auto maxRows = numRows + resultOffset;
  VELOX_DCHECK_LE(maxRows, result->size());
  BufferPtr valuesBuffer = result->mutableValues();
  [[maybe_unused]] auto values = valuesBuffer->asMutableRange<T>();
  for (int32_t i = 0; i < numRows; ++i) {
    const char* row;
    if constexpr (useRowNumbers) {
      auto rowNumber = rowNumbers[i];
      row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
    } else {
      row = rows[i];
    }
    auto resultIndex = resultOffset + i;
    if (row == nullptr) {
      result->setNull(resultIndex, true);
    } else {
      result->setNull(resultIndex, false);
      if constexpr (std::is_same_v<T, StringView>) {
        extractString(valueAt<StringView>(row, offset), result, resultIndex);
      } else {
        values[resultIndex] = valueAt<T>(row, offset);
      }
    }
  }
}

template <bool typeProvidesCustomComparison, TypeKind Kind, bool mayHaveNulls>
inline int RowContainer::compare(
    const char* row,
    RowColumn column,
    const DecodedVector& decoded,
    vector_size_t index,
    CompareFlags flags) const {
  if constexpr (Kind == TypeKind::OPAQUE) {
    VELOX_UNSUPPORTED("Comparing Opaque types is not supported.");
  } else if constexpr (Kind == TypeKind::UNKNOWN) {
    return isNullAt(row, column) ? 0 : flags.nullsFirst ? 1 : -1;
  } else {
    using T = typename KindToFlatVector<Kind>::HashRowType;

    if constexpr (mayHaveNulls) {
      bool rowIsNull = isNullAt(row, column);
      bool indexIsNull = decoded.isNullAt(index);
      if (rowIsNull) {
        return indexIsNull ? 0 : flags.nullsFirst ? -1 : 1;
      }
      if (indexIsNull) {
        return flags.nullsFirst ? 1 : -1;
      }
    }

    if constexpr (
        Kind == TypeKind::ROW || Kind == TypeKind::ARRAY ||
        Kind == TypeKind::MAP) {
      return compareComplexType(row, column.offset(), decoded, index, flags);
    } else if constexpr (is_string_kind(Kind)) {
      auto result = compareStringAsc(
          valueAt<StringView>(row, column.offset()), decoded, index);
      return flags.ascending ? result : result * -1;
    } else {
      auto left = valueAt<T>(row, column.offset());
      auto right = decoded.valueAt<T>(index);

      int result;
      if constexpr (typeProvidesCustomComparison) {
        result =
            SimpleVector<T>::template comparePrimitiveAscWithCustomComparison<
                Kind>(decoded.base()->type().get(), left, right);
      } else {
        result = SimpleVector<T>::comparePrimitiveAsc(left, right);
      }

      return flags.ascending ? result : result * -1;
    }
  }
}

template <bool typeProvidesCustomComparison, TypeKind Kind>
inline int RowContainer::compare(
    const char* left,
    const char* right,
    const Type* type,
    RowColumn leftColumn,
    RowColumn rightColumn,
    CompareFlags flags) const {
  if constexpr (Kind == TypeKind::OPAQUE) {
    VELOX_UNSUPPORTED("Comparing Opaque types is not supported.");
  } else {
    using T = typename KindToFlatVector<Kind>::HashRowType;
    bool leftIsNull = isNullAt(left, leftColumn);
    bool rightIsNull = isNullAt(right, rightColumn);
    if (leftIsNull) {
      return rightIsNull ? 0 : flags.nullsFirst ? -1 : 1;
    }
    if (rightIsNull) {
      return flags.nullsFirst ? 1 : -1;
    }

    auto leftOffset = leftColumn.offset();
    auto rightOffset = rightColumn.offset();
    if constexpr (
        Kind == TypeKind::ROW || Kind == TypeKind::ARRAY ||
        Kind == TypeKind::MAP) {
      return compareComplexType(
          left, right, type, leftOffset, rightOffset, flags);
    } else if constexpr (is_string_kind(Kind)) {
      auto leftValue = valueAt<StringView>(left, leftOffset);
      auto rightValue = valueAt<StringView>(right, rightOffset);
      auto result = compareStringAsc(leftValue, rightValue);
      return flags.ascending ? result : result * -1;
    } else {
      auto leftValue = valueAt<T>(left, leftOffset);
      auto rightValue = valueAt<T>(right, rightOffset);

      int result;
      if constexpr (typeProvidesCustomComparison) {
        result =
            SimpleVector<T>::template comparePrimitiveAscWithCustomComparison<
                Kind>(type, leftValue, rightValue);
      } else {
        result = SimpleVector<T>::comparePrimitiveAsc(leftValue, rightValue);
      }

      return flags.ascending ? result : result * -1;
    }
  }
}

template <bool useRowNumbers>
inline void RowContainer::extractComplexType(
    const char* const* rows,
    folly::Range<const vector_size_t*> rowNumbers,
    int32_t numRows,
    RowColumn column,
    int32_t resultOffset,
    const VectorPtr& result) {
  const auto offset = column.offset();

  VELOX_DCHECK_LE(numRows + resultOffset, result->size());
  for (int i = 0; i < numRows; ++i) {
    const char* row;
    if constexpr (useRowNumbers) {
      auto rowNumber = rowNumbers[i];
      row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
    } else {
      row = rows[i];
    }
    auto resultIndex = resultOffset + i;
    if (!row || isNullAt(row, column)) {
      result->setNull(resultIndex, true);
    } else {
      auto stream = prepareRead(row, offset);
      ContainerRowSerde::deserialize(stream, resultIndex, result.get());
    }
  }
}

} // namespace facebook::velox::exec
