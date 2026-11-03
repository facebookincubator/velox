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
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <fmt/format.h>

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/OrderedInteger.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/EliasFanoEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble {

namespace detail::partitioned_elias_fano {

// Where PartitionGroupPlanner put a partition.
struct PartitionPlacement {
  // Whether the partition starts a new group.
  bool newGroup{false};

  // Stored value of the partition's first row: 0 in a new group, otherwise
  // the stored value of the previous partition's last row.
  uint64_t baseShift{0};
};

// Assigns partitions to groups in order, merging a partition into the open
// group when the merged list is no larger than the two lists apart.
class PartitionGroupPlanner {
 public:
  explicit PartitionGroupPlanner(const Encoding::Options& options)
      : options_{options} {}

  // Places the next partition, which has 'numRows' rows whose last value
  // exceeds the first value by 'range'.
  PartitionPlacement add(uint32_t numRows, uint64_t range) {
    const auto singleGroupSize = estimateGroupBytes(numRows, range);
    // The merged range must fit in uint64 for the shifted values to ascend.
    if (numOpenGroupRows_ != 0 &&
        range <= std::numeric_limits<uint64_t>::max() - openGroupRange_) {
      const auto mergedGroupSize = estimateGroupBytes(
          numOpenGroupRows_ + numRows, openGroupRange_ + range);
      if (mergedGroupSize.has_value() && singleGroupSize.has_value() &&
          *mergedGroupSize <= openGroupBytes_ + *singleGroupSize) {
        const auto baseShift = openGroupRange_;
        numOpenGroupRows_ += numRows;
        openGroupRange_ += range;
        openGroupBytes_ = *mergedGroupSize;
        return {.newGroup = false, .baseShift = baseShift};
      }
    }
    hasOversizedGroup_ |= !singleGroupSize.has_value();
    closedGroupBytes_ += openGroupBytes_;
    numOpenGroupRows_ = numRows;
    openGroupRange_ = range;
    openGroupBytes_ = singleGroupSize.value_or(0);
    return {.newGroup = true, .baseShift = 0};
  }

  // Returns the bytes of all groups, or nullopt if one exceeds uint32. Then
  // estimateSize() declines the stream and encode() fails with an
  // incompatible-encoding error, so selection picks another encoding.
  std::optional<uint64_t> totalBytes() const {
    if (hasOversizedGroup_) {
      return std::nullopt;
    }
    return closedGroupBytes_ + openGroupBytes_;
  }

 private:
  // Returns the bytes of a group stream and its varint size, or nullopt if
  // the stream exceeds uint32.
  std::optional<uint64_t> estimateGroupBytes(uint32_t numRows, uint64_t range)
      const {
    const auto size = EliasFanoEncoding<uint64_t>::estimateNonDecreasingSize(
        numRows, range, options_);
    if (!size.has_value()) {
      return std::nullopt;
    }
    return *size + varint::varintSize(*size);
  }

  // Refers to the caller's options, which outlive the planner.
  const Encoding::Options& options_;

  // Makes totalBytes() return nullopt.
  bool hasOversizedGroup_{false};

  // Bytes of the groups before the open one.
  uint64_t closedGroupBytes_{0};

  // Rows, value range and bytes of the open group. A group is open once
  // 'numOpenGroupRows_' is non-zero, since every partition has a row.
  uint32_t numOpenGroupRows_{0};
  uint64_t openGroupRange_{0};
  uint64_t openGroupBytes_{0};
};

// Checks that an offset directory begins at 0 and ascends strictly below
// 'limit'.
inline void checkOffsets(std::span<const uint32_t> offsets, uint32_t limit) {
  NIMBLE_CHECK_FILE_EQ(
      offsets[0], 0, "PartitionedEliasFano directory must start at 0.");
  for (size_t i = 1; i < offsets.size(); ++i) {
    NIMBLE_CHECK_FILE_LT(
        offsets[i - 1],
        offsets[i],
        "PartitionedEliasFano directory must ascend.");
  }
  NIMBLE_CHECK_FILE_LT(
      offsets.back(), limit, "PartitionedEliasFano directory is out of range.");
}

} // namespace detail::partitioned_elias_fano

/// Encodes integers that ascend within partitions, such as sorted runs written
/// one after another. The stream starts a new partition wherever a value drops
/// below the previous one. Adjacent partitions are greedily merged into groups
/// while that does not grow the stream. Each group is one Elias-Fano list of
/// its partitions, shifted to ascend as a whole. Runs of similar density thus
/// share a list header, and runs of different density keep their own
/// lower-bit width.
///
/// A row of partition p stores baseShift(p) + value - base(p). base(p) is the
/// first value of p. baseShift(p) is 0 for the first partition of a group and
/// otherwise the stored value of the last row of p - 1. Values are compared
/// and subtracted in an unsigned form that keeps the logical order: a signed
/// value has its sign bit flipped, so -1 and 0 become 0x7F...F and 0x80...0.
///
/// For example, [10, 20, 5, 9] has two partitions, [10, 20] and [5, 9], which
/// merge into one group:
///
///   partition offsets       : [0, 2]
///   partition bases         : [10, 5]
///   partition group offsets : [0]
///   base shifts             : [0, 10]
///   group 0 values          : [0, 10, 10, 14]
///
/// Row 3 reads stored value 14 and restores 5 + (14 - 10) = 9: the base of its
/// partition plus the distance from the partition's first stored value.
///
/// Data layout (after the standard Encoding prefix):
///
///   varint          : partition count P
///   varint          : group count G
///   varint + bytes  : partition offsets, P uint32 rows (nested encoding)
///   varint + bytes  : partition bases, P physical values (nested encoding)
///   varint + bytes  : partition group offsets, G uint32 partition indexes
///                     (nested encoding)
///   G x (varint + bytes) : one uint64 Elias-Fano stream per group
///
/// A reader rejects a stream unless 1 <= G <= P <= row count, both offset
/// directories begin at 0 and ascend strictly within their range, each
/// directory has its declared data type and row count, and each group is an
/// Elias-Fano stream with the rows of its partitions.
template <typename T>
class PartitionedEliasFanoEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  /// Opens a stream, decoding and validating its partition and group
  /// directories.
  PartitionedEliasFanoEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      const Encoding::Options& options = {});

  /// Rewinds sequential decoding to the first row.
  void reset() final;

  /// Advances sequential decoding by 'rowCount' rows.
  void skip(uint32_t rowCount) final;

  /// Materializes 'rowCount' rows from the sequential position.
  void materialize(uint32_t rowCount, void* buffer) final;

  /// Decodes selected rows through the generic visitor interface.
  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  /// Returns the value at an absolute row without changing sequential state.
  physicalType valueAt(uint32_t row) const;

  /// Materializes an absolute row range without changing sequential state.
  void materializeAt(uint32_t offset, uint32_t rowCount, physicalType* output)
      const;

  /// Returns the first row in [begin, end) whose value is at least 'value', or
  /// 'end'. Stores the value of a returned row before 'end' in 'found', which
  /// spares a caller that compares it a second seek. The rows must lie in one
  /// partition, as a non-decreasing run does.
  uint32_t lowerBound(
      uint32_t begin,
      uint32_t end,
      physicalType value,
      physicalType& found) const;

  /// Returns the number of partitions.
  uint32_t numPartitions() const {
    return static_cast<uint32_t>(partitions_.size());
  }

  /// Returns the number of Elias-Fano groups.
  uint32_t numGroups() const {
    return numGroups_;
  }

  /// Encodes a non-empty span.
  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  /// Estimates the encoded size, or returns nullopt for oversized input, for
  /// input with fewer than two partitions, which EliasFano stores without a
  /// directory, and for partitions shorter than kMinRowsPerPartition on
  /// average.
  static std::optional<uint64_t> estimateSize(
      std::span<const physicalType> values,
      const Encoding::Options& options = {});

  /// Returns the partition and group counts after the common description.
  std::string debugString(int offset) const final;

 private:
  static_assert(
      std::is_integral_v<T> && !std::is_same_v<T, bool>,
      "PartitionedEliasFanoEncoding only supports non-bool integer types.");

  using GroupEncoding = EliasFanoEncoding<uint64_t>;
  using GroupList = GroupEncoding::List;

  // Reads keep about 70 bytes per partition, so shorter partitions cost more
  // memory per row than a per-value encoding saves on disk.
  static constexpr uint32_t kMinRowsPerPartition{64};

  // Calls 'visit(start, numRows, range)' for each partition in row order and
  // stops once it returns false.
  // 'start' is the partition's first row and 'range' its last value minus its
  // first, in toOrdered() form.
  template <typename Visit>
  static void forEachPartition(
      std::span<const physicalType> values,
      Visit&& visit);

  // Returns the index of the partition holding 'row'.
  uint32_t findPartition(uint32_t row) const;

  // Fills 'partitionHints_' and 'hintShift_' from 'partitions_'.
  void buildPartitionHints();

  // What a read needs about a partition, including its group's list, so a
  // read goes from one record, about a cache line, straight to the list's
  // bits.
  struct PartitionInfo {
    // Returns the value that row 'row' of the partition stores in its group.
    uint64_t storedAt(uint32_t row) const {
      return groupList.valueAt(row - groupStartRow);
    }

    // Restores the value the partition stores as 'stored'.
    physicalType restore(uint64_t stored) const {
      return fromOrdered<T>(orderedBase + (stored - shiftedStart));
    }

    // Rows [start, end) of the partition.
    uint32_t start{0};
    uint32_t end{0};

    // First row of the partition's group. Row r of the partition is entry
    // r - groupStartRow of 'groupList'.
    uint32_t groupStartRow{0};

    // First value of the partition in toOrdered() form.
    uint64_t orderedBase{0};

    // Stored value of the partition's first row, its shift in the group.
    uint64_t shiftedStart{0};

    // List of the partition's group. Points into the encoded stream, which
    // outlives the encoding.
    GroupList groupList;
  };

  // Entry b is the index of the partition holding row b << 'hintShift_'.
  // Buckets about as long as an average partition make findPartition() a
  // lookup and a short search.
  Vector<uint32_t> partitionHints_;

  // Log2 of the rows per bucket of 'partitionHints_'.
  uint8_t hintShift_{0};

  // One record per partition, in row order.
  Vector<PartitionInfo> partitions_;

  // Number of Elias-Fano groups.
  uint32_t numGroups_{0};

  // Next row of the stateful Encoding interface.
  uint32_t row_{0};
};

//
// End of class declaration. Implementations follow.
//

template <typename T>
PartitionedEliasFanoEncoding<T>::PartitionedEliasFanoEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const std::function<void*(uint32_t)>& stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options},
      partitionHints_{&pool},
      partitions_{&pool} {
  const auto rowCount = this->rowCount();
  NIMBLE_CHECK_FILE_GT(
      rowCount, 0, "PartitionedEliasFano encoding requires at least one row.");
  auto body = data.substr(this->dataOffset());
  const auto numPartitions = encoding::readVarint32(body);
  const auto numGroups = encoding::readVarint32(body);
  NIMBLE_CHECK_FILE_GT(numGroups, 0, "PartitionedEliasFano has no groups.");
  NIMBLE_CHECK_FILE_LE(
      numGroups,
      numPartitions,
      "PartitionedEliasFano has more groups than partitions.");
  NIMBLE_CHECK_FILE_LE(
      numPartitions,
      rowCount,
      "PartitionedEliasFano has more partitions than rows.");

  const auto readStream = [&]() {
    return EncodingFactory().create(
        pool,
        encoding::readLengthPrefixedBytes(body),
        stringBufferFactory,
        options);
  };
  // Decodes a directory after checking its type, so a stream of another
  // width cannot write past or short of 'output'.
  const auto readDirectory = [&]<typename V>(
                                 Vector<V>& output, uint32_t count) {
    auto encoding = readStream();
    NIMBLE_CHECK_FILE_EQ(
        encoding->dataType(),
        TypeTraits<V>::dataType,
        "PartitionedEliasFano directory has the wrong data type.");
    NIMBLE_CHECK_FILE_EQ(
        encoding->rowCount(),
        count,
        "PartitionedEliasFano directory has the wrong row count.");
    output.resize(count);
    encoding->materialize(count, output.data());
  };

  // Entry p is the first row of partition p.
  Vector<uint32_t> partitionOffsets{&pool};
  readDirectory(partitionOffsets, numPartitions);
  // Entry p is the first value of partition p.
  Vector<physicalType> partitionBases{&pool};
  readDirectory(partitionBases, numPartitions);
  // Entry g is the index of the first partition of group g.
  Vector<uint32_t> partitionGroupOffsets{&pool};
  readDirectory(partitionGroupOffsets, numGroups);
  detail::partitioned_elias_fano::checkOffsets(
      {partitionOffsets.data(), partitionOffsets.size()}, rowCount);
  detail::partitioned_elias_fano::checkOffsets(
      {partitionGroupOffsets.data(), partitionGroupOffsets.size()},
      numPartitions);
  // Ends the last partition and the last group, so partition p covers rows
  // [partitionOffsets[p], partitionOffsets[p + 1]) and group g partitions
  // [partitionGroupOffsets[g], partitionGroupOffsets[g + 1]).
  partitionOffsets.push_back(rowCount);
  partitionGroupOffsets.push_back(numPartitions);

  partitions_.resize(numPartitions);
  for (uint32_t groupIndex = 0; groupIndex < numGroups; ++groupIndex) {
    const auto firstPartition = partitionGroupOffsets[groupIndex];
    const auto endPartition = partitionGroupOffsets[groupIndex + 1];
    const auto groupStartRow = partitionOffsets[firstPartition];
    const auto encoding = readStream();
    NIMBLE_CHECK_FILE_EQ(
        encoding->encodingType(),
        EncodingType::EliasFano,
        "PartitionedEliasFano group is not an EliasFano stream.");
    NIMBLE_CHECK_FILE_EQ(
        encoding->dataType(),
        DataType::Uint64,
        "PartitionedEliasFano group has the wrong data type.");
    NIMBLE_CHECK_FILE_EQ(
        encoding->rowCount(),
        partitionOffsets[endPartition] - groupStartRow,
        "PartitionedEliasFano group has the wrong row count.");
    // The list points into 'data', so it outlives the encoding object.
    const auto groupList = static_cast<const GroupEncoding&>(*encoding).list();
    for (auto partitionIndex = firstPartition; partitionIndex < endPartition;
         ++partitionIndex) {
      auto& partition = partitions_[partitionIndex];
      partition = {
          .start = partitionOffsets[partitionIndex],
          .end = partitionOffsets[partitionIndex + 1],
          .groupStartRow = groupStartRow,
          .orderedBase = toOrdered<T>(partitionBases[partitionIndex]),
          .groupList = groupList,
      };
      partition.shiftedStart = partition.storedAt(partition.start);
    }
  }
  numGroups_ = numGroups;
  NIMBLE_CHECK_FILE(body.empty(), "PartitionedEliasFano has trailing bytes.");
  buildPartitionHints();
}

template <typename T>
void PartitionedEliasFanoEncoding<T>::reset() {
  row_ = 0;
}

template <typename T>
void PartitionedEliasFanoEncoding<T>::skip(uint32_t rowCount) {
  NIMBLE_CHECK_LE(rowCount, this->rowCount() - row_);
  row_ += rowCount;
}

template <typename T>
void PartitionedEliasFanoEncoding<T>::materialize(
    uint32_t rowCount,
    void* buffer) {
  NIMBLE_CHECK_LE(rowCount, this->rowCount() - row_);
  materializeAt(row_, rowCount, static_cast<physicalType*>(buffer));
  row_ += rowCount;
}

template <typename T>
template <typename DecoderVisitor>
void PartitionedEliasFanoEncoding<T>::readWithVisitor(
    DecoderVisitor& visitor,
    ReadWithVisitorParams& params) {
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] { return valueAt(row_++); });
}

template <typename T>
void PartitionedEliasFanoEncoding<T>::buildPartitionHints() {
  const auto rowCount = this->rowCount();
  const auto rowsPerPartition =
      rowCount / static_cast<uint32_t>(partitions_.size());
  // Rounds the bucket size down to a power of two no larger than an average
  // partition, which has at least one row.
  hintShift_ =
      static_cast<uint8_t>(velox::bits::bitsRequired(rowsPerPartition) - 1);
  partitionHints_.resize(
      velox::bits::divRoundUp(uint64_t{rowCount}, uint64_t{1} << hintShift_));
  uint32_t partitionIndex{0};
  for (uint32_t bucket = 0; bucket < partitionHints_.size(); ++bucket) {
    const auto row = bucket << hintShift_;
    while (partitions_[partitionIndex].end <= row) {
      ++partitionIndex;
    }
    partitionHints_[bucket] = partitionIndex;
  }
}

template <typename T>
uint32_t PartitionedEliasFanoEncoding<T>::findPartition(uint32_t row) const {
  const auto bucket = row >> hintShift_;
  const auto last = bucket + 1 < partitionHints_.size()
      ? partitionHints_[bucket + 1]
      : static_cast<uint32_t>(partitions_.size() - 1);
  // The bucket's rows lie in partitions [hint, last], usually one or two.
  const auto found = std::partition_point(
      partitions_.begin() + partitionHints_[bucket],
      partitions_.begin() + last,
      [row](const PartitionInfo& partition) { return partition.end <= row; });
  return static_cast<uint32_t>(found - partitions_.begin());
}

template <typename T>
typename PartitionedEliasFanoEncoding<T>::physicalType
PartitionedEliasFanoEncoding<T>::valueAt(uint32_t row) const {
  NIMBLE_CHECK_LT(row, this->rowCount(), "Reading past end of encoding.");
  const auto& partition = partitions_[findPartition(row)];
  return partition.restore(partition.storedAt(row));
}

template <typename T>
void PartitionedEliasFanoEncoding<T>::materializeAt(
    uint32_t offset,
    uint32_t rowCount,
    physicalType* output) const {
  NIMBLE_CHECK_LE(offset, this->rowCount(), "Invalid encoding position.");
  NIMBLE_CHECK_LE(
      rowCount, this->rowCount() - offset, "Reading past end of encoding.");
  if (rowCount == 0) {
    return;
  }
  constexpr uint32_t kBatchSize{1'024};
  // @lint-ignore CLANGTIDY cppcoreguidelines-pro-type-member-init
  std::array<uint64_t, kBatchSize> shiftedValues;
  const auto end = offset + rowCount;
  auto partitionIndex = findPartition(offset);
  // Each batch stays within one partition, so it restores with one base.
  for (auto row = offset; row < end;) {
    const auto& partition = partitions_[partitionIndex];
    const auto length = std::min({end, partition.end, row + kBatchSize}) - row;
    partition.groupList.materializeAt(
        row - partition.groupStartRow, length, shiftedValues.data());
    for (uint32_t i = 0; i < length; ++i) {
      output[row - offset + i] = partition.restore(shiftedValues[i]);
    }
    row += length;
    if (row == partition.end) {
      ++partitionIndex;
    }
  }
}

template <typename T>
uint32_t PartitionedEliasFanoEncoding<T>::lowerBound(
    uint32_t begin,
    uint32_t end,
    physicalType value,
    physicalType& found) const {
  NIMBLE_CHECK_LE(begin, end, "Invalid lowerBound range.");
  NIMBLE_CHECK_LE(end, this->rowCount(), "Invalid lowerBound range.");
  if (begin == end) {
    return end;
  }
  const auto& partition = partitions_[findPartition(begin)];
  NIMBLE_CHECK_LE(
      end,
      partition.end,
      "PartitionedEliasFano lowerBound range crosses a partition.");
  const auto& groupList = partition.groupList;
  const auto ordered = toOrdered<T>(value);
  if (ordered > partition.orderedBase) {
    // Maps 'value' to the value the partition would store for it.
    const auto target =
        partition.shiftedStart + (ordered - partition.orderedBase);
    // The sum wraps only when 'value' exceeds every value the group can store.
    if (target < partition.shiftedStart) {
      return end;
    }
    // A group's stored values ascend, so the lower bound of the stored target
    // is the lower bound within the partition's rows unless it falls before
    // 'begin'.
    uint64_t stored;
    const auto row =
        partition.groupStartRow + groupList.lowerBound(target, stored);
    if (row >= end) {
      return end;
    }
    if (row >= begin) {
      found = partition.restore(stored);
      return row;
    }
  }
  // Every row from 'begin' on is at least 'value'.
  found = begin == partition.start
      ? fromOrdered<T>(partition.orderedBase)
      : partition.restore(partition.storedAt(begin));
  return begin;
}

template <typename T>
template <typename Visit>
void PartitionedEliasFanoEncoding<T>::forEachPartition(
    std::span<const physicalType> values,
    Visit&& visit) {
  uint32_t start{0};
  const auto numRows = static_cast<uint32_t>(values.size());
  // A partition ends before a drop and at the end of the input.
  for (uint32_t row = 1; row <= numRows; ++row) {
    if (row == numRows ||
        toOrdered<T>(values[row]) < toOrdered<T>(values[row - 1])) {
      if (!visit(
              start,
              row - start,
              toOrdered<T>(values[row - 1]) - toOrdered<T>(values[start]))) {
        return;
      }
      start = row;
    }
  }
}

template <typename T>
std::optional<uint64_t> PartitionedEliasFanoEncoding<T>::estimateSize(
    std::span<const physicalType> values,
    const Encoding::Options& options) {
  if (values.empty() || values.size() > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  // Counts partitions and the bounds of their directories in one pass, without
  // materializing them.
  detail::partitioned_elias_fano::PartitionGroupPlanner planner{options};
  uint32_t numPartitions{0};
  uint32_t numGroups{0};
  uint32_t lastPartitionStart{0};
  // Bounds of the bases as stored: raw physical values, not toOrdered().
  uint64_t minBase{std::numeric_limits<uint64_t>::max()};
  uint64_t maxBase{0};
  forEachPartition(
      values, [&](uint32_t start, uint32_t numRows, uint64_t range) {
        numGroups += planner.add(numRows, range).newGroup ? 1 : 0;
        ++numPartitions;
        lastPartitionStart = start;
        minBase = std::min<uint64_t>(minBase, values[start]);
        maxBase = std::max<uint64_t>(maxBase, values[start]);
        return planner.totalBytes().has_value();
      });
  const auto groupBytes = planner.totalBytes();
  // Fewer than two partitions decline in favor of plain EliasFano, which
  // stores the stream without directories.
  if (!groupBytes.has_value() || numPartitions < 2 ||
      values.size() < uint64_t{kMinRowsPerPartition} * numPartitions) {
    return std::nullopt;
  }
  // Estimates each directory as FixedBitWidth.
  const auto directorySize = [&](uint64_t size) {
    return varint::varintSize(size) + size;
  };
  return EncodingPrefix::serializedSize(
             static_cast<uint32_t>(values.size()), options.useVarintRowCount) +
      varint::varintSize(numPartitions) + varint::varintSize(numGroups) +
      directorySize(
             FixedBitWidthEncoding<uint32_t>::estimateSize(
                 numPartitions, /*minValue=*/0, lastPartitionStart, options)) +
      directorySize(
             FixedBitWidthEncoding<physicalType>::estimateSize(
                 numPartitions, minBase, maxBase, options)) +
      directorySize(
             FixedBitWidthEncoding<uint32_t>::estimateSize(
                 numGroups, /*minValue=*/0, numPartitions - 1, options)) +
      *groupBytes;
}

template <typename T>
std::string_view PartitionedEliasFanoEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_CHECK(
      !values.empty(), "PartitionedEliasFano encoding requires values.");
  NIMBLE_CHECK_LE(
      values.size(),
      std::numeric_limits<uint32_t>::max(),
      "PartitionedEliasFano row count exceeds uint32.");
  auto* pool = &buffer.getMemoryPool();
  const auto rowCount = static_cast<uint32_t>(values.size());
  // Entry p is the first row of partition p.
  ScopedVector<uint32_t> partitionOffsets{/*size=*/0, pool, options.bufferPool};
  // Entry p is the first value of partition p.
  ScopedVector<physicalType> partitionBases{
      /*size=*/0, pool, options.bufferPool};
  // Entry g is the index of the first partition of group g.
  ScopedVector<uint32_t> partitionGroupOffsets{
      /*size=*/0, pool, options.bufferPool};
  // Entry r is the value row r stores in its group's list.
  ScopedVector<uint64_t> shiftedValues{rowCount, pool, options.bufferPool};
  detail::partitioned_elias_fano::PartitionGroupPlanner planner{options};
  forEachPartition(
      values, [&](uint32_t start, uint32_t numRows, uint64_t range) {
        const auto placement = planner.add(numRows, range);
        if (!planner.totalBytes().has_value()) {
          NIMBLE_INCOMPATIBLE_ENCODING(
              "PartitionedEliasFano group exceeds uint32 size.");
        }
        if (placement.newGroup) {
          partitionGroupOffsets->push_back(
              static_cast<uint32_t>(partitionOffsets.size()));
        }
        const auto base = toOrdered<T>(values[start]);
        for (auto row = start; row < start + numRows; ++row) {
          shiftedValues[row] =
              placement.baseShift + (toOrdered<T>(values[row]) - base);
        }
        partitionOffsets->push_back(start);
        partitionBases->push_back(values[start]);
        return true;
      });
  const auto numPartitions = static_cast<uint32_t>(partitionOffsets.size());
  const auto numGroups = static_cast<uint32_t>(partitionGroupOffsets.size());

  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  // The three directories, then one stream per group.
  std::vector<std::string_view> streams;
  streams.reserve(3 + numGroups);
  streams.push_back(selection.template encodeNested<uint32_t>(
      EncodingIdentifiers::PartitionedEliasFano::PartitionOffsets,
      std::span<const uint32_t>(partitionOffsets.data(), numPartitions),
      scopedBuffer.get(),
      options));
  streams.push_back(selection.template encodeNested<physicalType>(
      EncodingIdentifiers::PartitionedEliasFano::PartitionBases,
      std::span<const physicalType>(partitionBases.data(), numPartitions),
      scopedBuffer.get(),
      options));
  streams.push_back(selection.template encodeNested<uint32_t>(
      EncodingIdentifiers::PartitionedEliasFano::PartitionGroupOffsets,
      std::span<const uint32_t>(partitionGroupOffsets.data(), numGroups),
      scopedBuffer.get(),
      options));
  // Ends the last partition and the last group, after the directories above
  // were written without them.
  partitionOffsets->push_back(rowCount);
  partitionGroupOffsets->push_back(numPartitions);
  for (uint32_t groupIndex = 0; groupIndex < numGroups; ++groupIndex) {
    const auto groupBegin = partitionOffsets[partitionGroupOffsets[groupIndex]];
    const auto groupEnd =
        partitionOffsets[partitionGroupOffsets[groupIndex + 1]];
    streams.push_back(
        GroupEncoding::encodeNonDecreasing(
            std::span<const uint64_t>(
                shiftedValues.data() + groupBegin, groupEnd - groupBegin),
            scopedBuffer.get(),
            options));
  }

  // Sums in uint64_t, so a total past uint32 cannot wrap before the check.
  uint64_t encodingSize =
      Encoding::serializePrefixSize(rowCount, options.useVarintRowCount) +
      varint::varintSize(numPartitions) + varint::varintSize(numGroups);
  for (const auto stream : streams) {
    encodingSize += varint::varintSize(stream.size()) + stream.size();
  }
  NIMBLE_CHECK_LE(
      encodingSize,
      std::numeric_limits<uint32_t>::max(),
      "PartitionedEliasFano encoding exceeds uint32 size.");
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::PartitionedEliasFano,
      TypeTraits<T>::dataType,
      rowCount,
      options.useVarintRowCount,
      pos);
  varint::writeVarint(numPartitions, &pos);
  varint::writeVarint(numGroups, &pos);
  for (const auto stream : streams) {
    encoding::writeVarintString(stream, pos);
  }
  NIMBLE_CHECK_EQ(
      pos - reserved,
      encodingSize,
      "PartitionedEliasFano encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string PartitionedEliasFanoEncoding<T>::debugString(int offset) const {
  return Encoding::debugString(offset) +
      fmt::format(
             "\n{}partitions={}, groups={}",
             std::string(offset, ' '),
             partitions_.size(),
             numGroups_);
}

} // namespace facebook::nimble
