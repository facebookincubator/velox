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

#include "velox/dwio/nimble/serializer/Projector.h"

#include <glog/logging.h>

#include <algorithm>
#include <numeric>

#include "folly/io/Cursor.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/serializer/StreamDataParser.h"
#include "velox/dwio/nimble/serializer/StreamDataWriter.h"
#include "velox/dwio/nimble/serializer/legacy/TrailerReader.h"
#include "velox/dwio/nimble/velox/SchemaUtils.h"
#include "velox/type/Type.h"

namespace facebook::nimble::serde {

namespace {

// Lightweight adapter for writing small sections (header, trailer) directly
// into an IOBuf. Satisfies the size()/resize()/data() interface required by
// detail::extend/writeSerializationHeader/writeTrailer. Avoids the std::string
// → IOBuf copy that would occur if writing into a temporary std::string first.
class IOBufSection {
 public:
  explicit IOBufSection(size_t initialCapacity)
      : buf_(folly::IOBuf::create(initialCapacity)) {}

  size_t size() const {
    return size_;
  }

  void resize(size_t newSize) {
    if (newSize > buf_->capacity()) {
      auto newBuf =
          folly::IOBuf::create(std::max(newSize, buf_->capacity() * 2));
      if (size_ > 0) {
        std::memcpy(newBuf->writableData(), buf_->data(), size_);
      }
      buf_ = std::move(newBuf);
    }
    size_ = newSize;
  }

  char* data() {
    return reinterpret_cast<char*>(buf_->writableData());
  }

  // Finalizes the IOBuf by setting its length and returns it.
  std::unique_ptr<folly::IOBuf> build() && {
    buf_->append(size_);
    return std::move(buf_);
  }

 private:
  std::unique_ptr<folly::IOBuf> buf_;
  size_t size_{0};
};

// Reads a varint32 from a Cursor, advancing past the encoded bytes.
uint32_t readVarint32(folly::io::Cursor& cursor) {
  uint32_t value = 0;
  uint32_t shift = 0;
  while (true) {
    auto byte = cursor.read<uint8_t>();
    value |= static_cast<uint32_t>(byte & 0x7f) << shift;
    if (!(byte & 0x80)) {
      return value;
    }
    shift += 7;
  }
}

inline void setHeaderFlags(
    folly::IOBuf& header,
    size_t flagsOffset,
    bool outputRequiresNullBarrier,
    bool streamEncodingUsesVarintRowCount) {
  NIMBLE_CHECK_LT(flagsOffset, header.length(), "Invalid flags byte offset");
  header.writableData()[flagsOffset] = detail::makeFlagsByte(
      outputRequiresNullBarrier,
      streamEncodingUsesVarintRowCount,
      /*streamHasChunkHeader=*/false);
}

inline void updateRequiresNullBarrier(
    bool inputRequiresNullBarrier,
    uint32_t streamSize,
    const std::vector<bool>& rowOrFlatMapNullStreams,
    size_t outputStreamIdx,
    bool& outputRequiresNullBarrier) {
  if (!inputRequiresNullBarrier || streamSize == 0) {
    return;
  }
  NIMBLE_DCHECK_LT(outputStreamIdx, rowOrFlatMapNullStreams.size());
  outputRequiresNullBarrier |= rowOrFlatMapNullStreams[outputStreamIdx];
}

// Forward declaration for recursive calls from per-type helpers.
std::shared_ptr<const Type> updateColumnNames(
    const std::shared_ptr<const Type>& inputType,
    const velox::Type& projectType);

// Updates row column names via positional matching with velox RowType.
std::shared_ptr<const Type> updateRowColumnNames(
    const std::shared_ptr<const Type>& inputType,
    const velox::RowType& veloxRow) {
  const auto& nimbleRow = inputType->asRow();
  std::vector<std::string> newNames;
  std::vector<std::shared_ptr<const Type>> newChildren;
  newNames.reserve(nimbleRow.childrenCount());
  newChildren.reserve(nimbleRow.childrenCount());

  bool changed{false};
  const auto minChildren =
      std::min<size_t>(veloxRow.size(), nimbleRow.childrenCount());
  for (size_t i = 0; i < minChildren; ++i) {
    newNames.emplace_back(veloxRow.nameOf(i));
    changed |= (newNames.back() != nimbleRow.nameAt(i));
    auto newChild =
        updateColumnNames(nimbleRow.childAt(i), *veloxRow.childAt(i));
    changed |= (newChild != nimbleRow.childAt(i));
    newChildren.emplace_back(std::move(newChild));
  }

  if (!changed) {
    return inputType;
  }

  // Keep remaining children as-is (file has more columns than table).
  newNames.insert(
      newNames.end(),
      nimbleRow.names().begin() + minChildren,
      nimbleRow.names().end());
  newChildren.insert(
      newChildren.end(),
      nimbleRow.children().begin() + minChildren,
      nimbleRow.children().end());
  return std::make_shared<RowType>(
      nimbleRow.nullsDescriptor(), std::move(newNames), std::move(newChildren));
}

// Updates column names within array elements.
// Handles both Array and ArrayWithOffsets nimble types.
std::shared_ptr<const Type> updateArrayColumnNames(
    const std::shared_ptr<const Type>& inputType,
    const velox::ArrayType& veloxArray) {
  if (inputType->kind() == Kind::ArrayWithOffsets) {
    const auto& array = inputType->asArrayWithOffsets();
    auto newElements =
        updateColumnNames(array.elements(), *veloxArray.elementType());
    if (newElements == array.elements()) {
      return inputType;
    }
    return std::make_shared<ArrayWithOffsetsType>(
        array.offsetsDescriptor(),
        array.lengthsDescriptor(),
        std::move(newElements));
  }
  const auto& array = inputType->asArray();
  auto newElements =
      updateColumnNames(array.elements(), *veloxArray.elementType());
  if (newElements == array.elements()) {
    return inputType;
  }
  return std::make_shared<ArrayType>(
      array.lengthsDescriptor(), std::move(newElements));
}

// Updates column names within FlatMap value types.
// FlatMap keys are data-determined, not renamed.
std::shared_ptr<const Type> updateFlatMapColumnNames(
    const std::shared_ptr<const Type>& inputType,
    const velox::MapType& veloxMap) {
  const auto& flatMap = inputType->asFlatMap();
  std::vector<std::shared_ptr<const Type>> newChildren;
  newChildren.reserve(flatMap.childrenCount());
  bool changed{false};
  for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
    auto newChild =
        updateColumnNames(flatMap.childAt(i), *veloxMap.valueType());
    changed |= (newChild != flatMap.childAt(i));
    newChildren.emplace_back(std::move(newChild));
  }
  if (!changed) {
    return inputType;
  }
  // Clone names and inMapDescriptors.
  std::vector<std::string> newNames;
  newNames.reserve(flatMap.childrenCount());
  std::vector<std::unique_ptr<StreamDescriptor>> inMapDescriptors;
  inMapDescriptors.reserve(flatMap.childrenCount());
  for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
    newNames.emplace_back(flatMap.nameAt(i));
    const auto& desc = flatMap.inMapDescriptorAt(i);
    inMapDescriptors.emplace_back(
        std::make_unique<StreamDescriptor>(desc.offset(), desc.scalarKind()));
  }
  return std::make_shared<FlatMapType>(
      flatMap.nullsDescriptor(),
      flatMap.keyScalarKind(),
      std::move(newNames),
      std::move(inMapDescriptors),
      std::move(newChildren));
}

// Updates column names within Map/SlidingWindowMap key and value types.
std::shared_ptr<const Type> updateMapColumnNames(
    const std::shared_ptr<const Type>& inputType,
    const velox::MapType& veloxMap) {
  switch (inputType->kind()) {
    case Kind::SlidingWindowMap: {
      const auto& map = inputType->asSlidingWindowMap();
      auto newKeys = updateColumnNames(map.keys(), *veloxMap.keyType());
      auto newValues = updateColumnNames(map.values(), *veloxMap.valueType());
      if (newKeys == map.keys() && newValues == map.values()) {
        return inputType;
      }
      return std::make_shared<SlidingWindowMapType>(
          map.offsetsDescriptor(),
          map.lengthsDescriptor(),
          std::move(newKeys),
          std::move(newValues));
    }
    case Kind::Map: {
      const auto& map = inputType->asMap();
      auto newKeys = updateColumnNames(map.keys(), *veloxMap.keyType());
      auto newValues = updateColumnNames(map.values(), *veloxMap.valueType());
      if (newKeys == map.keys() && newValues == map.values()) {
        return inputType;
      }
      return std::make_shared<MapType>(
          map.lengthsDescriptor(), std::move(newKeys), std::move(newValues));
    }
    default:
      NIMBLE_UNREACHABLE(
          "updateMapColumnNames called with unsupported kind: {}",
          inputType->kind());
  }
}

// Updates column names in nimble schema to match projectType (velox) using
// positional matching. Same algorithm as Reader::updateColumnNames but across
// velox/nimble type trees.
std::shared_ptr<const Type> updateColumnNames(
    const std::shared_ptr<const Type>& inputType,
    const velox::Type& projectType) {
  switch (projectType.kind()) {
    case velox::TypeKind::ROW:
      return updateRowColumnNames(inputType, projectType.asRow());
    case velox::TypeKind::ARRAY:
      return updateArrayColumnNames(inputType, projectType.asArray());
    case velox::TypeKind::MAP:
      if (inputType->isFlatMap()) {
        return updateFlatMapColumnNames(inputType, projectType.asMap());
      }
      return updateMapColumnNames(inputType, projectType.asMap());
    default:
      return inputType;
  }
}

// kLegacyCompact is read-only post the two-array trailer change. Callers that
// still pass it (e.g., not-yet-migrated tests or downstream code we can't
// update in this diff due to directory branching) are silently upgraded to
// kProjection so the round-trip still works on the new wire format. A
// LOG(WARNING) flags the migration so the caller can switch.
Projector::Options upgradeReadOnlyProjectVersion(Projector::Options options) {
  if (options.projectVersion == SerializationVersion::kLegacyCompact) {
    LOG_FIRST_N(WARNING, 10)
        << "Projector constructed with projectVersion=kLegacyCompact "
           "(read-only post the two-array trailer change); silently upgrading "
           "to kProjection. Migrate the caller to pass kProjection explicitly.";
    options.projectVersion = SerializationVersion::kProjection;
  }
  return options;
}

} // namespace

Projector::Projector(
    std::shared_ptr<const Type> inputSchema,
    const std::vector<Subfield>& projectSubfields,
    velox::memory::MemoryPool* pool,
    Options options)
    : pool_(pool),
      options_(upgradeReadOnlyProjectVersion(std::move(options))),
      inputSchema_(std::move(inputSchema)) {
  NIMBLE_CHECK_NOT_NULL(pool_, "Memory pool cannot be null");
  NIMBLE_CHECK_NOT_NULL(inputSchema_, "Input schema cannot be null");
  NIMBLE_CHECK(
      inputSchema_->isRow(),
      "Input schema must be a RowType, got: {}",
      inputSchema_->kind());
  NIMBLE_CHECK(!projectSubfields.empty(), "Must project at least one subfield");
  NIMBLE_CHECK(
      options_.projectVersion == SerializationVersion::kProjection,
      "Projection output version must be kProjection. Got: {}",
      options_.projectVersion);

  // Update inputSchema_ with projectType names for schema evolution.
  if (options_.projectType) {
    inputSchema_ = updateColumnNames(inputSchema_, *options_.projectType);
  }

  for (const auto& subfield : projectSubfields) {
    NIMBLE_CHECK(subfield.valid(), "Invalid subfield: {}", subfield.toString());
  }

  projection_ = buildProjectedNimbleType(inputSchema_.get(), projectSubfields);
  NIMBLE_CHECK_EQ(
      projection_.streamOffsets.size(),
      projection_.rowOrFlatMapNullStreams.size(),
      "Projected stream indices and Row/FlatMap null stream mask must align");

  inputStreamsSorted_ = std::is_sorted(
      projection_.streamOffsets.begin(), projection_.streamOffsets.end());

  if (!inputStreamsSorted_) {
    sortedStreamMappings_.reserve(projection_.streamOffsets.size());
    for (size_t i = 0; i < projection_.streamOffsets.size(); ++i) {
      sortedStreamMappings_.push_back({projection_.streamOffsets[i], i});
    }
    std::sort(
        sortedStreamMappings_.begin(),
        sortedStreamMappings_.end(),
        [](const auto& lhs, const auto& rhs) {
          return lhs.inputStreamIdx < rhs.inputStreamIdx;
        });
  }
}

namespace {

bool isProjectorInputVersion(SerializationVersion version) {
  return version == SerializationVersion::kLegacyCompact ||
      version == SerializationVersion::kLegacySerialization ||
      version == SerializationVersion::kSerialization ||
      version == SerializationVersion::kProjection;
}

// Validates the input version header is a supported compact non-tablet format.
// kLegacyCompact is read via the legacy reader; kLegacySerialization,
// kSerialization, and kProjection use the two-array sparse trailer reader.
// kTablet is rejected because the Projector does not strip chunk headers.
SerializationVersion getAndValidateInputVersion(const folly::IOBuf& input) {
  const auto version = static_cast<SerializationVersion>(*input.data());
  NIMBLE_CHECK(
      isProjectorInputVersion(version),
      "Input must be kLegacyCompact, kLegacySerialization, kSerialization, or kProjection; got: {}",
      version);
  return version;
}

} // namespace

// Projects selected streams from a contiguous IOBuf into the output chain.
// Streams are output in the order of selectedStreamIndices (output stream
// order), which may differ from sorted input order for interleaved schemas
// (e.g., multiple FlatMap columns). Walks selectedStreamIndices in output
// order; for each selected input stream, binary-searches the sparse
// (streamIndices, streamSizes) trailer for its position. Merges adjacent
// output entries that are contiguous in the input into a single zero-copy
// IOBuf clone to minimize IOBuf allocations. The input IOBuf must outlive the
// output chain.
// static
std::vector<uint32_t> Projector::projectStreamsContiguousUnsorted(
    const folly::IOBuf& input,
    size_t dataOffset,
    const std::vector<uint32_t>& streamIndices,
    const std::vector<uint32_t>& streamSizes,
    const std::vector<uint32_t>& selectedStreamIndices,
    bool inputRequiresNullBarrier,
    const std::vector<bool>& rowOrFlatMapNullStreams,
    bool& outputRequiresNullBarrier,
    std::unique_ptr<folly::IOBuf>& output) {
  std::vector<uint32_t> outputSizes(selectedStreamIndices.size(), 0);

  // Precompute byte offset of each sparse pair (input-order prefix sum of
  // sizes). streamOffsets[sparsePosition] is the byte offset of
  // streamIndices[sparsePosition] within input.
  std::vector<size_t> streamOffsets(streamIndices.size());
  size_t offset = dataOffset;
  for (size_t sparsePosition = 0; sparsePosition < streamIndices.size();
       ++sparsePosition) {
    // NOLINTBEGIN(facebook-hte-LocalUncheckedArrayBounds)
    streamOffsets[sparsePosition] = offset;
    offset += streamSizes[sparsePosition];
    // NOLINTEND(facebook-hte-LocalUncheckedArrayBounds)
  }

  // Track the start offset and byte count of a contiguous run.
  size_t runStart = 0;
  size_t numRunBytes = 0;

  auto flushRun = [&]() {
    if (numRunBytes == 0) {
      return;
    }
    auto buf = input.cloneOne();
    buf->trimStart(runStart);
    buf->trimEnd(buf->length() - numRunBytes);
    output->appendToChain(std::move(buf));
    numRunBytes = 0;
  };

  // Walk selectedStreamIndices in output order; binary-search sparse
  // streamIndices for each. Selected streams absent from the sparse trailer or
  // present with size 0 keep outputSizes at 0 and contribute zero bytes, so
  // they can sit inside a run without breaking byte-contiguity for the next
  // present selected stream.
  for (size_t i = 0; i < selectedStreamIndices.size(); ++i) {
    const auto inputStreamIdx = selectedStreamIndices[i];
    const auto it = std::lower_bound(
        streamIndices.begin(), streamIndices.end(), inputStreamIdx);
    if (it == streamIndices.end() || *it != inputStreamIdx) {
      continue;
    }
    const auto sparsePosition = static_cast<size_t>(it - streamIndices.begin());
    // NOLINTBEGIN(facebook-hte-LocalUncheckedArrayBounds)
    const auto streamSize = streamSizes[sparsePosition];
    const auto streamOffset = streamOffsets[sparsePosition];
    outputSizes[i] = streamSize;
    // NOLINTEND(facebook-hte-LocalUncheckedArrayBounds)
    updateRequiresNullBarrier(
        inputRequiresNullBarrier,
        streamSize,
        rowOrFlatMapNullStreams,
        i,
        outputRequiresNullBarrier);
    if (numRunBytes > 0 && streamOffset == runStart + numRunBytes) {
      numRunBytes += streamSize;
    } else {
      flushRun();
      runStart = streamOffset;
      numRunBytes = streamSize;
    }
  }
  flushRun();

  return outputSizes;
}

// Fast path for projectStreamsContiguousUnsorted when inputStreamIndices are
// sorted (no FlatMap key reordering). Single forward two-pointer merge over
// the sparse (streamIndices, streamSizes) trailer and selectedStreamIndices.
// static
std::vector<uint32_t> Projector::projectStreamsContiguousSorted(
    const folly::IOBuf& input,
    size_t dataOffset,
    const std::vector<uint32_t>& streamIndices,
    const std::vector<uint32_t>& streamSizes,
    const std::vector<uint32_t>& selectedStreamIndices,
    bool inputRequiresNullBarrier,
    const std::vector<bool>& rowOrFlatMapNullStreams,
    bool& outputRequiresNullBarrier,
    std::unique_ptr<folly::IOBuf>& output) {
  std::vector<uint32_t> outputSizes(selectedStreamIndices.size(), 0);

  size_t currentOffset = dataOffset;

  // Track the start offset and byte count of a contiguous run.
  size_t runStart = 0;
  size_t numRunBytes = 0;

  auto flushRun = [&]() {
    if (numRunBytes == 0) {
      return;
    }
    // Zero-copy: clone the input and trim to the run's sub-range.
    auto buf = input.cloneOne();
    buf->trimStart(runStart);
    buf->trimEnd(buf->length() - numRunBytes);
    output->appendToChain(std::move(buf));
    numRunBytes = 0;
  };

  size_t sparsePosition = 0;
  size_t selectedPosition = 0;
  while (sparsePosition < streamIndices.size() &&
         selectedPosition < selectedStreamIndices.size()) {
    if (selectedStreamIndices[selectedPosition] <
        streamIndices[sparsePosition]) {
      // Selected stream absent from sparse trailer (size 0); outputSizes
      // already 0.
      ++selectedPosition;
      continue;
    }
    if (selectedStreamIndices[selectedPosition] >
        streamIndices[sparsePosition]) {
      // Non-selected stream; advance past its bytes. The next match's
      // run-extension check will detect the byte-gap and flush via its else
      // branch, so no explicit flush is needed here.
      currentOffset += streamSizes[sparsePosition];
      ++sparsePosition;
      continue;
    }
    // Match.
    // NOLINTBEGIN(facebook-hte-LocalUncheckedArrayBounds)
    const auto streamSize = streamSizes[sparsePosition];
    outputSizes[selectedPosition] = streamSize;
    updateRequiresNullBarrier(
        inputRequiresNullBarrier,
        streamSize,
        rowOrFlatMapNullStreams,
        selectedPosition,
        outputRequiresNullBarrier);
    if (numRunBytes > 0 && currentOffset == runStart + numRunBytes) {
      numRunBytes += streamSize;
    } else {
      flushRun();
      runStart = currentOffset;
      numRunBytes = streamSize;
    }
    currentOffset += streamSize;
    // NOLINTEND(facebook-hte-LocalUncheckedArrayBounds)
    ++sparsePosition;
    ++selectedPosition;
  }
  flushRun();

  return outputSizes;
}

// Fast path for projectStreamsChainedUnsorted when inputStreamIndices are
// sorted (no FlatMap key reordering). Two-pointer merge over the sparse
// (streamIndices, streamSizes) trailer and selectedStreamIndices, with run
// merging for contiguous-in-input selected streams.
// static
std::vector<uint32_t> Projector::projectStreamsChainedSorted(
    folly::io::Cursor& cursor,
    const std::vector<uint32_t>& streamIndices,
    const std::vector<uint32_t>& streamSizes,
    const std::vector<uint32_t>& selectedStreamIndices,
    bool inputRequiresNullBarrier,
    const std::vector<bool>& rowOrFlatMapNullStreams,
    bool& outputRequiresNullBarrier,
    std::unique_ptr<folly::IOBuf>& output) {
  std::vector<uint32_t> outputSizes(selectedStreamIndices.size(), 0);

  // Track how many contiguous selected stream bytes we can merge.
  size_t numRunBytes = 0;

  auto flushRun = [&](std::unique_ptr<folly::IOBuf>& out) {
    if (numRunBytes == 0) {
      return;
    }
    std::unique_ptr<folly::IOBuf> buf;
    cursor.clone(buf, numRunBytes);
    out->appendToChain(std::move(buf));
    numRunBytes = 0;
  };

  size_t sparsePosition = 0;
  size_t selectedPosition = 0;
  while (sparsePosition < streamIndices.size() &&
         selectedPosition < selectedStreamIndices.size()) {
    if (selectedStreamIndices[selectedPosition] <
        streamIndices[sparsePosition]) {
      // Selected stream absent from sparse trailer (size 0); outputSizes
      // already 0.
      ++selectedPosition;
      continue;
    }
    if (selectedStreamIndices[selectedPosition] >
        streamIndices[sparsePosition]) {
      // Non-selected stream; flush any pending run and skip its bytes via
      // cursor. Flush is required for cursor correctness: cursor.skip moves
      // the cursor forward, so we must clone the pending run bytes (which
      // sit at the cursor's current position) before skipping past them.
      flushRun(output);
      cursor.skip(streamSizes[sparsePosition]);
      ++sparsePosition;
      continue;
    }
    // Match: accumulate into the current run.
    // NOLINTBEGIN(facebook-hte-LocalUncheckedArrayBounds)
    const auto streamSize = streamSizes[sparsePosition];
    outputSizes[selectedPosition] = streamSize;
    updateRequiresNullBarrier(
        inputRequiresNullBarrier,
        streamSize,
        rowOrFlatMapNullStreams,
        selectedPosition,
        outputRequiresNullBarrier);
    numRunBytes += streamSize;
    // NOLINTEND(facebook-hte-LocalUncheckedArrayBounds)
    ++sparsePosition;
    ++selectedPosition;
  }
  flushRun(output);

  return outputSizes;
}

// Projects selected streams from a chained IOBuf via cursor.
// Streams are output in the order of selectedStreamIndices (output stream
// order). Walks the sparse (streamIndices, streamSizes) trailer in lockstep
// with the sorted-by-input mapping; merges contiguous-in-input runs that are
// also adjacent in output order into single zero-copy clones, then sorts and
// chains by output index.
// static
std::vector<uint32_t> Projector::projectStreamsChainedUnsorted(
    folly::io::Cursor& cursor,
    const std::vector<uint32_t>& streamIndices,
    const std::vector<uint32_t>& streamSizes,
    const std::vector<StreamMapping>& sortedStreamMappings,
    bool inputRequiresNullBarrier,
    const std::vector<bool>& rowOrFlatMapNullStreams,
    bool& outputRequiresNullBarrier,
    std::unique_ptr<folly::IOBuf>& output) {
  std::vector<uint32_t> outputSizes(sortedStreamMappings.size(), 0);

  // Holds the extracted IOBuf for one or more merged output streams.
  struct OutputStreamBuffer {
    // Start output stream index. When multiple contiguous input streams are
    // merged, this is the index of the first output stream in the merged run.
    size_t outputStreamIdx;
    std::unique_ptr<folly::IOBuf> buf;
  };

  std::vector<OutputStreamBuffer> outputStreamBufs;
  outputStreamBufs.reserve(sortedStreamMappings.size());

  size_t sparsePosition = 0;
  size_t mappingPosition = 0;
  while (sparsePosition < streamIndices.size() &&
         mappingPosition < sortedStreamMappings.size()) {
    const auto wantInputStreamIdx =
        sortedStreamMappings[mappingPosition].inputStreamIdx;
    if (wantInputStreamIdx < streamIndices[sparsePosition]) {
      // Selected input stream absent from sparse trailer (size 0).
      ++mappingPosition;
      continue;
    }
    if (wantInputStreamIdx > streamIndices[sparsePosition]) {
      // Non-selected input stream; skip its bytes via cursor.
      cursor.skip(streamSizes[sparsePosition]);
      ++sparsePosition;
      continue;
    }

    // Match: try to extend a run while consecutive in input AND adjacent in
    // output AND present in sparse.
    const auto runOutputStreamIdx =
        sortedStreamMappings[mappingPosition].outputStreamIdx;
    size_t numRunBytes = 0;
    size_t numRunStreams = 0;
    while (
        mappingPosition + numRunStreams < sortedStreamMappings.size() &&
        sparsePosition + numRunStreams < streamIndices.size() &&
        streamIndices[sparsePosition + numRunStreams] ==
            streamIndices[sparsePosition] + numRunStreams &&
        sortedStreamMappings[mappingPosition + numRunStreams].inputStreamIdx ==
            streamIndices[sparsePosition] + numRunStreams &&
        sortedStreamMappings[mappingPosition + numRunStreams].outputStreamIdx ==
            runOutputStreamIdx + numRunStreams) {
      // NOLINTBEGIN(facebook-hte-LocalUncheckedArrayBounds)
      const auto streamSize = streamSizes[sparsePosition + numRunStreams];
      const auto outputStreamIdx = runOutputStreamIdx + numRunStreams;
      outputSizes[outputStreamIdx] = streamSize;
      updateRequiresNullBarrier(
          inputRequiresNullBarrier,
          streamSize,
          rowOrFlatMapNullStreams,
          outputStreamIdx,
          outputRequiresNullBarrier);
      numRunBytes += streamSize;
      // NOLINTEND(facebook-hte-LocalUncheckedArrayBounds)
      ++numRunStreams;
    }

    if (numRunBytes > 0) {
      std::unique_ptr<folly::IOBuf> buf;
      cursor.clone(buf, numRunBytes);
      outputStreamBufs.push_back({runOutputStreamIdx, std::move(buf)});
    }
    mappingPosition += numRunStreams;
    sparsePosition += numRunStreams;
  }

  // Sort by output stream index and chain in order.
  std::sort(
      outputStreamBufs.begin(),
      outputStreamBufs.end(),
      [](const auto& lhs, const auto& rhs) {
        return lhs.outputStreamIdx < rhs.outputStreamIdx;
      });
  for (auto& streamBuf : outputStreamBufs) {
    output->appendToChain(std::move(streamBuf.buf));
  }
  return outputSizes;
}

folly::IOBuf Projector::buildProjectedOutput(
    const std::vector<uint32_t>& outputStreamSizes,
    std::unique_ptr<folly::IOBuf> output) const {
  IOBufSection trailer(
      detail::estimateTrailerSize(
          options_.projectVersion,
          outputStreamSizes.size(),
          options_.streamIndicesEncodingType,
          options_.streamSizesEncodingType));
  detail::writeTrailer(
      outputStreamSizes,
      options_.streamIndicesEncodingType,
      options_.streamSizesEncodingType,
      trailer);
  output->appendToChain(std::move(trailer).build());
  return std::move(*output);
}

folly::IOBuf Projector::project(const folly::IOBuf& input) const {
  const auto inputVersion = getAndValidateInputVersion(input);

  if (!input.isChained()) {
    return projectContiguous(input, inputVersion);
  }
  return projectChained(input, inputVersion);
}

folly::IOBuf Projector::projectContiguous(
    const folly::IOBuf& input,
    SerializationVersion inputVersion) const {
  const auto* data = reinterpret_cast<const char*>(input.data());
  // Skip version byte (already validated and passed in).
  const auto* pos = data + sizeof(uint8_t);
  const uint32_t rowCount = varint::readVarint32(&pos);
  const auto inputFlags = readHeaderFlags(pos, inputVersion);
  const bool inputRequiresNullBarrier = inputFlags.requiresNullBarrier;

  IOBufSection header(
      estimateSerializationHeaderSize(options_.projectVersion, rowCount));
  const auto flagsOffset =
      writeSerializationHeader(header, options_.projectVersion, rowCount);
  auto output = std::move(header).build();

  // Dispatch on input version: legacy kLegacyCompact blobs are read via the
  // frozen legacy reader (normalized to sparse pairs internally); all newer
  // versions use the new two-array sparse trailer reader.
  auto [streamIndices, streamSizes] = usesLegacyTrailer(inputVersion)
      ? legacy::readLegacyTrailerStreamMetadata(input)
      : detail::readTrailerStreamMetadata(input);

  // Extract selected streams as zero-copy sub-range clones.
  const auto dataOffset = static_cast<size_t>(pos - data);
  bool outputRequiresNullBarrier = false;
  std::vector<uint32_t> outputStreamSizes;
  if (inputStreamsSorted_) {
    outputStreamSizes = projectStreamsContiguousSorted(
        input,
        dataOffset,
        streamIndices,
        streamSizes,
        projection_.streamOffsets,
        inputRequiresNullBarrier,
        projection_.rowOrFlatMapNullStreams,
        outputRequiresNullBarrier,
        output);
  } else {
    outputStreamSizes = projectStreamsContiguousUnsorted(
        input,
        dataOffset,
        streamIndices,
        streamSizes,
        projection_.streamOffsets,
        inputRequiresNullBarrier,
        projection_.rowOrFlatMapNullStreams,
        outputRequiresNullBarrier,
        output);
  }
  setHeaderFlags(
      *output,
      flagsOffset,
      outputRequiresNullBarrier,
      /*streamEncodingUsesVarintRowCount=*/true);

  return buildProjectedOutput(outputStreamSizes, std::move(output));
}

folly::IOBuf Projector::projectChained(
    const folly::IOBuf& input,
    SerializationVersion inputVersion) const {
  folly::io::Cursor cursor(&input);
  // Skip version byte (already validated and passed in).
  cursor.skip(sizeof(uint8_t));
  const uint32_t rowCount = readVarint32(cursor);
  const auto inputFlags = readHeaderFlags(cursor, inputVersion);
  const bool inputRequiresNullBarrier = inputFlags.requiresNullBarrier;

  IOBufSection header(
      estimateSerializationHeaderSize(options_.projectVersion, rowCount));
  const auto flagsOffset =
      writeSerializationHeader(header, options_.projectVersion, rowCount);
  auto output = std::move(header).build();

  // Dispatch on input version: legacy kLegacyCompact blobs are read via the
  // frozen legacy reader (normalized to sparse pairs internally); all newer
  // versions use the new two-array sparse trailer reader.
  auto [streamIndices, streamSizes] = usesLegacyTrailer(inputVersion)
      ? legacy::readLegacyTrailerStreamMetadata(input)
      : detail::readTrailerStreamMetadata(input);

  // Extract selected streams as zero-copy clones via cursor.
  bool outputRequiresNullBarrier = false;
  std::vector<uint32_t> outputStreamSizes;
  if (inputStreamsSorted_) {
    outputStreamSizes = projectStreamsChainedSorted(
        cursor,
        streamIndices,
        streamSizes,
        projection_.streamOffsets,
        inputRequiresNullBarrier,
        projection_.rowOrFlatMapNullStreams,
        outputRequiresNullBarrier,
        output);
  } else {
    outputStreamSizes = projectStreamsChainedUnsorted(
        cursor,
        streamIndices,
        streamSizes,
        sortedStreamMappings_,
        inputRequiresNullBarrier,
        projection_.rowOrFlatMapNullStreams,
        outputRequiresNullBarrier,
        output);
  }
  setHeaderFlags(
      *output,
      flagsOffset,
      outputRequiresNullBarrier,
      /*streamEncodingUsesVarintRowCount=*/true);

  return buildProjectedOutput(outputStreamSizes, std::move(output));
}

folly::IOBuf Projector::project(std::string_view input) const {
  return project(folly::IOBuf::wrapBufferAsValue(input.data(), input.size()));
}

std::vector<folly::IOBuf> Projector::project(
    const std::vector<std::string_view>& inputs) const {
  std::vector<folly::IOBuf> results;
  results.reserve(inputs.size());
  for (const auto& input : inputs) {
    results.emplace_back(project(input));
  }
  return results;
}

std::vector<folly::IOBuf> Projector::project(
    const std::vector<folly::IOBuf>& inputs) const {
  std::vector<folly::IOBuf> results;
  results.reserve(inputs.size());
  for (const auto& input : inputs) {
    results.emplace_back(project(input));
  }
  return results;
}

} // namespace facebook::nimble::serde
