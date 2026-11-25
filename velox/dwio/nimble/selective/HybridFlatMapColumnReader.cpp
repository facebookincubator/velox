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

#include "velox/dwio/nimble/selective/HybridFlatMapColumnReader.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <string>

#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/SelectiveColumnReaderInternal.h"
#include "velox/dwio/nimble/common/HybridFlatMap.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/selective/ChunkedDecoder.h"
#include "velox/dwio/nimble/selective/ColumnReader.h"

namespace facebook::nimble {

using namespace facebook::velox;

namespace {

// Decodes 'numKeys' keys that the file stores as 'FileKey' into 'keys'. A map
// read with a wider integer key type than the file's widens the keys.
template <typename FileKey, typename T>
void decodeHybridFlatMapIntegerKeys(
    ChunkedDecoder& decoder,
    uint64_t numKeys,
    std::vector<T>& keys) {
  if constexpr (std::is_same_v<FileKey, T>) {
    keys.resize(numKeys);
    decoder.decodeNullable(
        /*nulls=*/nullptr, keys.data(), numKeys, /*incomingNulls=*/nullptr);
  } else if constexpr (sizeof(FileKey) > sizeof(T)) {
    NIMBLE_USER_FAIL(
        "Hybrid FlatMap keys cannot be read as a narrower integer type.");
  } else {
    std::vector<FileKey> fileKeys(numKeys);
    decoder.decodeNullable(
        /*nulls=*/nullptr,
        fileKeys.data(),
        numKeys,
        /*incomingNulls=*/nullptr);
    keys.assign(fileKeys.begin(), fileKeys.end());
  }
}

// Decodes a Hybrid FlatMap metadata stream into 'values' and returns its
// number of values, or 0 when the stream is absent. Key-presence bitmaps and
// in-map streams decode into packed bits, and the keys a Default group lists
// in a stripe decode into typed keys. String keys view the decoder's chunk,
// which stays loaded while the decoder lives. The writer never splits these
// streams into chunks, so a second chunk means a corrupt file.
template <typename Values>
uint32_t readHybridFlatMapMetadata(
    ChunkedDecoder* decoder,
    Values& values,
    std::string_view streamName) {
  values.clear();
  if (decoder == nullptr) {
    return 0;
  }
  decoder->ensureLoaded();
  const auto* encoding = decoder->currentEncoding();
  NIMBLE_CHECK_FILE(
      !encoding->isNullable(),
      "Hybrid FlatMap {} stream must not contain nulls.",
      streamName);
  // velox::bits takes int32_t bit positions.
  NIMBLE_CHECK_FILE_LE(
      decoder->remainingValues(),
      std::numeric_limits<int32_t>::max(),
      "Hybrid FlatMap {} count exceeds the supported maximum.",
      streamName);
  const auto numValues = static_cast<uint32_t>(decoder->remainingValues());

  // The container type picks the stream: packed bits for a key-presence bitmap
  // or in-map stream, StringView or integer keys for a key list.
  if constexpr (std::is_same_v<Values, nimble::Vector<uint64_t>>) {
    NIMBLE_CHECK_FILE_EQ(
        encoding->dataType(),
        DataType::Bool,
        "Hybrid FlatMap {} stream must be boolean.",
        streamName);
    values.resize(bits::nwords(numValues), 0);
    decoder->nextBools(values.data(), numValues, /*incomingNulls=*/nullptr);
  } else if constexpr (std::is_same_v<Values, std::vector<StringView>>) {
    NIMBLE_CHECK_FILE_EQ(encoding->dataType(), DataType::String);
    std::vector<std::string_view> views(numValues);
    decoder->decodeNullable(
        /*nulls=*/nullptr, views.data(), numValues, /*incomingNulls=*/nullptr);
    values.reserve(numValues);
    for (const auto& view : views) {
      values.emplace_back(view.data(), view.size());
    }
  } else {
    // The file may store integer keys narrower than the requested key type.
    switch (encoding->dataType()) {
      case DataType::Int8:
        decodeHybridFlatMapIntegerKeys<int8_t>(*decoder, numValues, values);
        break;
      case DataType::Int16:
        decodeHybridFlatMapIntegerKeys<int16_t>(*decoder, numValues, values);
        break;
      case DataType::Int32:
        decodeHybridFlatMapIntegerKeys<int32_t>(*decoder, numValues, values);
        break;
      case DataType::Int64:
        decodeHybridFlatMapIntegerKeys<int64_t>(*decoder, numValues, values);
        break;
      default:
        NIMBLE_FILE_FAIL(
            "Hybrid FlatMap {} stream must hold integer keys: {}",
            streamName,
            encoding->dataType());
    }
  }
  decoder->ensureLoaded();
  NIMBLE_CHECK_FILE_EQ(
      decoder->remainingValues(),
      0,
      "Hybrid FlatMap {} stream must have one encoded chunk per decode run.",
      streamName);
  return numValues;
}

// Returns whether 'key' passes the map keys filter.
template <typename T>
bool passesKeyFilter(const common::ScanSpec& keysSpec, T key) {
  const auto* keyFilter = keysSpec.filter();
  return keyFilter == nullptr || common::applyFilter(*keyFilter, key);
}

// Returns the keys that 'keyFilter' selects, as schema keys, when the filter
// lists its keys, or std::nullopt otherwise. Subscript pushdown builds such
// filters: BytesValues for string keys, and BigintValues or BigintRange for
// integer keys. A range with more than 'maxRangeKeys' keys, the number of keys
// the groups list, holds a key in no group, so it is not listed and the groups
// are found by testing each schema key instead.
std::optional<std::vector<std::string>> selectedKeyList(
    const common::Filter* keyFilter,
    size_t maxRangeKeys) {
  if (keyFilter == nullptr) {
    return std::nullopt;
  }
  std::vector<int64_t> integerKeys;
  switch (keyFilter->kind()) {
    case common::FilterKind::kBytesValues: {
      const auto& keys = keyFilter->as<common::BytesValues>()->values();
      return std::vector<std::string>{keys.begin(), keys.end()};
    }
    case common::FilterKind::kBigintValuesUsingHashTable:
      integerKeys =
          keyFilter->as<common::BigintValuesUsingHashTable>()->values();
      break;
    case common::FilterKind::kBigintValuesUsingBitmask:
      integerKeys = keyFilter->as<common::BigintValuesUsingBitmask>()->values();
      break;
    case common::FilterKind::kBigintRange: {
      const auto* range = keyFilter->as<common::BigintRange>();

      // Wraps to 0 for the whole int64_t range.
      const uint64_t numKeys = static_cast<uint64_t>(range->upper()) -
          static_cast<uint64_t>(range->lower()) + 1;
      if (numKeys == 0 || numKeys > maxRangeKeys) {
        return std::nullopt;
      }
      integerKeys.resize(numKeys);
      std::iota(integerKeys.begin(), integerKeys.end(), range->lower());
      break;
    }
    default:
      return std::nullopt;
  }

  // Schema keys are canonical, so an integer key is the decimal form of its
  // value.
  std::vector<std::string> keys;
  keys.reserve(integerKeys.size());
  for (const auto key : integerKeys) {
    keys.push_back(std::to_string(key));
  }
  return keys;
}

// A Hybrid FlatMap group that holds a key that passes the keys filter: its
// typed catalog keys, the decoders of its metadata streams and the reader of
// its shared value stream. The column reader decodes its layout and values in
// each stripe.
template <typename T>
struct HybridFlatMapGroup {
  explicit HybridFlatMapGroup(memory::MemoryPool* pool)
      : inMapBits{pool}, valueRows{pool} {}

  // Whether the group lists its keys in each stripe instead of in the schema.
  // Loading the stripe then fills keys and selected.
  bool useKeyPresentList{false};
  // Typed catalog keys in key-presence order. StringView keys view the file
  // schema, which the column reader's NimbleData keeps alive. A group that
  // lists its keys replaces them with views of the loaded chunk of
  // keyPresence.
  std::vector<T> keys;
  // Whether each catalog key passes the keys filter.
  std::vector<bool> selected;
  // Null when the stream is absent from the stripe.
  std::unique_ptr<ChunkedDecoder> keyPresence;
  std::unique_ptr<ChunkedDecoder> inMap;
  // Whether the value subtree has a stream in the stripe.
  bool hasValueStreams{false};
  // Null when the group has neither an in-map stream nor value streams in
  // the stripe, and so no values.
  std::unique_ptr<dwio::common::SelectiveColumnReader> valueReader;

  // The members below are set when the stripe is loaded.
  // Key-major in-map bits, which the group's key nodes point into.
  nimble::Vector<uint64_t> inMapBits;
  // Value stream positions of the selected keys' values, ascending. Kept
  // alive because the value reader refers to its last read rows.
  raw_vector<vector_size_t> valueRows;
};

// A selected key of a selected group that is present in the stripe. Its rows
// are a slice of the group's in-map bits, and its values a range of the
// stripe's values, so each batch converts it directly to map entries.
template <typename T>
struct HybridFlatMapKeyNode {
  // Returned as the map key. copyKeyStrings() moves a non-inline StringView
  // key into the column reader's key buffer.
  T key;
  // Key-major in-map bits of the key's group.
  const uint64_t* inMapBits;
  // First bit of the key in inMapBits.
  int32_t inMapStart;
  // Index into the stripe values of the key's first value in a non-null map
  // the column reader has not yet counted. Equals the end of the key's values
  // when no later map holds the key.
  vector_size_t nextValue;
};

// Reads the Hybrid FlatMap schema and builds the selected groups: the groups
// that hold a key that passes the keys filter. Only their selected keys are
// later converted to rows. It enqueues their streams, and streams enqueued
// after the stripe is loaded would never be read, so the column reader calls
// it from its constructor.
template <typename T>
std::vector<HybridFlatMapGroup<T>> makeHybridFlatMapGroups(
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    NimbleParams& params,
    NimbleData& nimbleData,
    const common::ScanSpec& keysSpec,
    common::ScanSpec& valuesSpec,
    memory::MemoryPool& pool) {
  const auto& hybridFlatMapType = params.nimbleType()->asHybridFlatMap();
  const auto& hybridFlatMap = hybridFlatMapType.hybridFlatMap();
  auto& streams = params.streams();

  // A keys filter that lists its keys finds the groups to read in the key
  // index, so the keys of the other groups are never parsed.
  const auto selectedKeys =
      selectedKeyList(keysSpec.filter(), hybridFlatMap.numGroupedKeys());
  const auto holdsSelectedKey = selectedKeys.has_value()
      ? hybridFlatMap.groupsHoldingKeys(*selectedKeys)
      : std::vector<bool>{};
  std::vector<HybridFlatMapGroup<T>> groups;
  for (size_t i = 0; i < hybridFlatMapType.groupCount(); ++i) {
    if (selectedKeys.has_value() && !holdsSelectedKey[i]) {
      continue;
    }
    const auto& schema = hybridFlatMapType.groupAt(i);
    HybridFlatMapGroup<T> group{&pool};
    group.useKeyPresentList =
        hybridFlatMapType.useKeyPresentList(schema.groupId);

    // A Default group that lists its keys has no schema keys, so the loop
    // below adds nothing for it and this initial value decides whether it is
    // read. It holds every key in no explicit group. With a key list, it got
    // here only because one of the listed keys is in no explicit group.
    // Without one, the filter may pass such a key. Either way it is read
    // whenever it has streams.
    bool anySelected{group.useKeyPresentList};

    // Empty for a Default group that lists its keys. Its keys come from each
    // stripe's key list, which loadGroup() decodes and filters.
    group.keys.reserve(schema.groupKeys.size());
    group.selected.reserve(schema.groupKeys.size());
    for (const auto& catalogKey : schema.groupKeys) {
      // Parses the schema key into the type of the requested map key.
      const auto key = HybridFlatMap::parseKey<T>(catalogKey);

      // Tests the key against the map keys filter.
      const bool selected = passesKeyFilter(keysSpec, key);

      // Keeps both in key-presence order, the order of the group's
      // key-presence bits, which loadGroup() indexes them by.
      group.keys.push_back(key);
      group.selected.push_back(selected);

      // A group with schema keys, an explicit group or a bitmap Default, is
      // read when one of its keys passes the filter.
      anySelected |= selected;
    }

    // Keys are unique across groups, so no other group holds the selected
    // keys.
    if (!anySelected) {
      continue;
    }
    group.keyPresence = nimbleData.makeDecoder(
        schema.keyPresenceDescriptor, /*decodeValuesWithNulls=*/false);
    group.inMap = nimbleData.makeDecoder(
        schema.inMapDescriptor, /*decodeValuesWithNulls=*/false);
    group.hasValueStreams = visitPresenceStreamOffsets(
        *schema.valueType,
        [&](offset_size offset) { return streams.hasStream(offset); });
    // Builds the value reader when the group has value streams or in-map
    // rows. A group with in-map rows has values even when its value subtree
    // writes no streams: a ROW whose rows are all non-null and whose fields
    // are all null writes none. The value reader still returns those non-null
    // rows with null fields.
    if (group.hasValueStreams || group.inMap != nullptr) {
      auto childParams = params.makeChildParams(schema.valueType);
      group.valueReader = buildColumnReader(
          requestedType->childAt(1),
          fileType->childAt(1),
          childParams,
          valuesSpec,
          /*isRoot=*/false);
    }
    // The writer omits every stream of a group without data in the stripe.
    if (group.keyPresence == nullptr && group.inMap == nullptr &&
        group.valueReader == nullptr) {
      continue;
    }

    auto groupId = schema.groupId;
    common::testutil::TestValue::adjust(
        "facebook::nimble::makeHybridFlatMapGroups", &groupId);
    groups.push_back(std::move(group));
  }
  return groups;
}

// Reads a Hybrid FlatMap column as a MAP by converting the selected keys of
// the selected groups directly to rows, in three parts:
// 1. makeHybridFlatMapGroups() reads the schema and builds the selected
//    groups, the groups that hold a selected key.
// 2. loadKeyNodes() reads only the selected keys of the selected groups: the
//    in-map bits and values of each selected key in the stripe. Each group
//    stores the values of all its keys in one value stream, key after key, so
//    batches of consecutive rows read scattered value positions. The first
//    getValues() of a stripe that returns a non-null map therefore decodes the
//    group metadata and reads the values of the selected keys for the whole
//    stripe. read() and skip() only move a cursor over the non-null maps.
// 3. calculateOffsets() and copyValues() convert the selected keys directly
//    to the rows of each batch: the map sizes and offsets, then the map keys
//    and values.
template <typename T>
class HybridFlatMapColumnReader final
    : public dwio::common::SelectiveColumnReader {
 public:
  HybridFlatMapColumnReader(
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      NimbleParams& params,
      common::ScanSpec& scanSpec)
      : SelectiveColumnReader(requestedType, fileType, params, scanSpec),
        keysSpec_{scanSpec.getOrCreateChild(
            common::Subfield(common::ScanSpec::kMapKeysFieldName))},
        valuesSpec_{scanSpec.getOrCreateChild(
            common::Subfield(common::ScanSpec::kMapValuesFieldName))} {
    // All keys of a group share one value reader, so a value filter cannot be
    // applied per key.
    NIMBLE_USER_CHECK(
        !valuesSpec_->hasFilter(),
        "Hybrid FlatMap does not support filters on map values.");

    // Extraction pushdown is not implemented for the key-major layout. With a
    // delta update the column reader returns the whole map and the extraction
    // is applied after the update.
    if (scanSpec.extractionType() != common::ScanSpec::ExtractionType::kNone) {
      NIMBLE_USER_CHECK_NOT_NULL(
          scanSpec.deltaUpdate(),
          "Hybrid FlatMap does not support extraction pushdown.");
    }
    NIMBLE_USER_CHECK(
        keysSpec_->extractionType() == common::ScanSpec::ExtractionType::kNone,
        "Hybrid FlatMap does not support extraction pushdown on map keys.");
    NIMBLE_USER_CHECK(
        valuesSpec_->extractionType() ==
            common::ScanSpec::ExtractionType::kNone,
        "Hybrid FlatMap does not support extraction pushdown on map values.");
    keysSpec_->setProjectOut(true);
    valuesSpec_->setProjectOut(true);
    // A filter-only column needs only the nulls stream.
    if (scanSpec.keepValues()) {
      // Part 1: reads the schema and builds the selected groups.
      groups_ = makeHybridFlatMapGroups<T>(
          requestedType_,
          fileType_,
          params,
          formatData().template as<NimbleData>(),
          *keysSpec_,
          *valuesSpec_,
          *pool_);
    }
  }

  uint64_t skip(uint64_t numValues) final {
    const auto numNonNulls =
        formatData_->skipNulls(numValues, /*nullsOnly=*/false);
    advanceNonNullCursor(static_cast<uint32_t>(numNonNulls), "skip");
    return numNonNulls;
  }

  // Reads the map nulls of 'rows' and moves the cursor over the non-null maps
  // of the read range. It decodes nothing from the groups. getValues() later
  // converts the maps it returns, using the range read() records:
  // firstNonNullInReadRange_, the position of the range's first non-null map
  // among the stripe's non-null maps, and numNonNullsInReadRange_, the number
  // of non-null maps in the range.
  //
  // Example: a stripe holds the non-null maps {1:a, 2:b, 7:c}, {5:d, 3:e} and
  // {1:f, 9:g}, and the first read covers rows 0 to 2. read() sets
  // firstNonNullInReadRange_ to 0 and numNonNullsInReadRange_ to 3, and moves
  // the cursor to 3. A filter on another column then drops row 1, so
  // getValues() gets rows 0 and 2, and forEachEntry() walks the in-map bits of
  // non-null maps 0 to 2 to build their maps.
  void read(int64_t offset, const RowSet& rows, const uint64_t* incomingNulls)
      final {
    // Skips to 'offset' and reads the map nulls of rows [0, rows.back()] into
    // nullsInReadRange_.
    prepareRead<char>(offset, rows, incomingNulls);
    if (const auto* filter = scanSpec_->filter()) {
      // A map filter can only test whether the map is null, so it is applied
      // here from the nulls stream alone.
      const auto kind = filter->kind();
      if (kind != common::FilterKind::kIsNull &&
          kind != common::FilterKind::kIsNotNull) {
        NIMBLE_USER_FAIL(
            "Hybrid FlatMap supports only IS NULL and IS NOT NULL map filters: {}",
            filter->toString());
      }
      filterNulls<int32_t>(
          rows, kind == common::FilterKind::kIsNull, /*extractValues=*/false);
    }

    // The range starts at the cursor. Only non-null maps have in-map bits, so
    // the range is counted in non-null maps.
    const vector_size_t numRows = rows.back() + 1;
    firstNonNullInReadRange_ = nextNonNullRow_;
    numNonNullsInReadRange_ = nullsInReadRange_
        ? static_cast<uint32_t>(bits::countNonNulls(
              nullsInReadRange_->as<uint64_t>(), 0, numRows))
        : numRows;
    advanceNonNullCursor(numNonNullsInReadRange_, "read");
    readOffset_ = offset + numRows;
  }

  void getValues(const RowSet& rows, VectorPtr* result) final {
    const auto numRows = static_cast<vector_size_t>(rows.size());
    if (*result != nullptr &&
        (*result)->encoding() == VectorEncoding::Simple::MAP) {
      BaseVector::prepareForReuse(*result, numRows);
    } else {
      *result = BaseVector::create(requestedType_, numRows, pool_);
    }
    auto* map = (*result)->asUnchecked<MapVector>();
    auto* rawOffsets = map->mutableOffsets(numRows)->asMutable<vector_size_t>();
    auto* rawSizes = map->mutableSizes(numRows)->asMutable<vector_size_t>();

    // Part 2: reads the values of the selected keys of the selected groups
    // for the whole stripe, once per stripe. Maps that are all null need no
    // stripe load.
    const bool anyNonNull = anyNonNullMap(rows);
    if (anyNonNull) {
      loadKeyNodes();
    }

    // Part 3: converts the selected keys to the rows of this batch, first the
    // map sizes and offsets, then the map keys and values.
    const auto numEntries =
        calculateOffsets(rows, anyNonNull, rawOffsets, rawSizes);
    BaseVector::prepareForReuse(map->mapKeys(), numEntries);
    if (numEntries > 0) {
      copyValues(rawOffsets, numRows, numEntries, *map);
    } else {
      BaseVector::prepareForReuse(map->mapValues(), 0);
    }
    setComplexNulls(rows, *result);
  }

  void seekToRowGroup(int64_t /*index*/) final {
    NIMBLE_UNREACHABLE("Nimble files have no row groups.");
  }

 private:
  // The base seekTo() skips through skipNulls() when the scan spec reads only
  // nulls, which would not move the non-null cursor.
  bool readsNullsOnly() const final {
    return false;
  }

  // Moves the cursor over 'numNonNulls' non-null maps.
  void advanceNonNullCursor(uint32_t numNonNulls, std::string_view operation) {
    nextNonNullRow_ += numNonNulls;
    NIMBLE_CHECK_FILE_LE(
        nextNonNullRow_,
        nonNullRowBound_,
        "Hybrid FlatMap {} exceeds group row count.",
        operation);
  }

  // Decodes the layout of every selected group and reads the values of its
  // selected keys for the whole stripe, as one key node per selected present
  // key.
  void loadKeyNodes() {
    if (loaded_) {
      return;
    }
    loaded_ = true;
    nimble::Vector<uint64_t> keyPresenceBits{pool_};
    std::vector<VectorPtr> groupValues;
    uint64_t numValues{0};
    for (auto& group : groups_) {
      const auto firstKeyNode = keyNodes_.size();
      auto values = loadGroup(group, keyPresenceBits);
      if (values == nullptr) {
        continue;
      }
      NIMBLE_CHECK_LE(
          numValues + values->size(),
          static_cast<uint64_t>(std::numeric_limits<vector_size_t>::max()),
          "Hybrid FlatMap stripe values exceed the vector size limit.");

      // The stripe values place the groups one after another, so the group's
      // key nodes index its values from the values of the groups before it.
      for (auto i = firstKeyNode; i < keyNodes_.size(); ++i) {
        keyNodes_[i].nextValue += static_cast<vector_size_t>(numValues);
      }
      numValues += values->size();
      groupValues.push_back(std::move(values));
    }
    stripeValues_ = concatenateGroupValues(groupValues, numValues);
    if constexpr (std::is_same_v<T, StringView>) {
      copyKeyStrings();
    }
  }

  // Decodes the metadata and layout of 'group' in the stripe, adds a key node
  // for each selected present key, and returns the values of those keys, or
  // null when the group has none. The key nodes index the returned values.
  // 'keyPresenceBits' is scratch space that the groups share.
  //
  // Example: a read selects keys 1 and 9 in the stripe of the read() example.
  //   group {1, 5}: in-map 1 0 1 | 0 1 0, value stream [a, f, d]
  //     -> key node 1 (inMapStart 0, nextValue 0), returns [a, f]
  //   Default listing 7, 9: in-map 1 0 0 | 0 0 1, value stream [c, g]
  //     -> key node 9 (inMapStart 3, nextValue 0), returns [g]
  // Only the values of the selected keys are read.
  VectorPtr loadGroup(
      HybridFlatMapGroup<T>& group,
      nimble::Vector<uint64_t>& keyPresenceBits) {
    uint32_t numAllKeyPresenceBits{0};
    if (group.useKeyPresentList) {
      readHybridFlatMapMetadata(
          group.keyPresence.get(), group.keys, "key-presence-list");
      group.selected.clear();
      group.selected.reserve(group.keys.size());
      for (const auto& key : group.keys) {
        group.selected.push_back(passesKeyFilter(*keysSpec_, key));
      }
    } else {
      numAllKeyPresenceBits = readHybridFlatMapMetadata(
          group.keyPresence.get(), keyPresenceBits, "key-presence");
    }
    const auto numInMap =
        readHybridFlatMapMetadata(group.inMap.get(), group.inMapBits, "in-map");

    // Lays out the group in one pass over its present keys, and adds a key
    // node for each selected key with the position of its first value in
    // valueRows. Unselected keys are only counted, because their values still
    // take positions in the value stream. For group {1, 5} of the example
    // above, the layout is numNonNullRows 3, presentKeys {0, 1} and
    // valueStarts {0, 2, 3}. Key 5 is counted but gets no key node, so
    // valueRows is [0, 1].
    HybridFlatMapGroupLayout layout;
    vector_size_t numSelectedValues{0};
    decodeHybridFlatMapGroupLayout(
        HybridFlatMapGroupMetadata{
            .numKeys = static_cast<uint32_t>(group.keys.size()),
            .numAllKeyPresenceBits = numAllKeyPresenceBits,
            .numSetKeyPresenceBits = static_cast<uint32_t>(bits::countBits(
                keyPresenceBits.data(),
                0,
                static_cast<int32_t>(numAllKeyPresenceBits))),
            .numInMap = numInMap,
            .listsKeys = group.useKeyPresentList,
            .hasValueStreams = group.hasValueStreams,
        },
        [&](uint32_t key) {
          return bits::isBitSet(keyPresenceBits.data(), key);
        },
        [&](uint32_t key, uint32_t presentKey, uint32_t /*firstValue*/) {
          // Present key p owns the in-map bits [p * N, (p + 1) * N), where N
          // is layout.numNonNullRows. The helper sets N before the first
          // call.
          const auto numRows = static_cast<int32_t>(layout.numNonNullRows);
          const auto begin = static_cast<int32_t>(presentKey) * numRows;
          const auto numKeyValues = static_cast<uint32_t>(
              bits::countBits(group.inMapBits.data(), begin, begin + numRows));

          // A selected key's values follow those of the selected keys
          // before it in valueRows.
          if (group.selected[key]) {
            keyNodes_.push_back(
                HybridFlatMapKeyNode<T>{
                    .key = group.keys[key],
                    .inMapBits = group.inMapBits.data(),
                    .inMapStart = begin,
                    .nextValue = numSelectedValues,
                });
            numSelectedValues += static_cast<vector_size_t>(numKeyValues);
          }
          return numKeyValues;
        },
        // Error messages format the typed key.
        [&](uint32_t key) { return group.keys[key]; },
        layout);
    if (layout.empty()) {
      return nullptr;
    }
    NIMBLE_CHECK_FILE_LE(
        nextNonNullRow_,
        layout.numNonNullRows,
        "Hybrid FlatMap read exceeds group row count.");

    // Later reads and skips check the cursor against the smallest group row
    // count, which covers every group.
    nonNullRowBound_ = std::min(nonNullRowBound_, layout.numNonNullRows);

    // Readers index the last row of a read, so they cannot read no rows.
    if (numSelectedValues == 0) {
      return nullptr;
    }

    // Keys own disjoint ascending ranges of the value stream, so the
    // positions are ascending. They follow the order of the key nodes.
    group.valueRows.resize(numSelectedValues);
    auto* nextRow = group.valueRows.data();
    for (uint32_t presentKey = 0; presentKey < layout.presentKeys.size();
         ++presentKey) {
      if (!group.selected[layout.presentKeys[presentKey]]) {
        continue;
      }
      const auto firstPosition = layout.valueStarts[presentKey];
      const auto endPosition = layout.valueStarts[presentKey + 1];
      std::iota(
          nextRow,
          nextRow + (endPosition - firstPosition),
          static_cast<vector_size_t>(firstPosition));
      nextRow += endPosition - firstPosition;
    }

    // Present keys have in-map rows, so makeHybridFlatMapGroups() built the
    // reader.
    NIMBLE_CHECK_NOT_NULL(
        group.valueReader, "Hybrid FlatMap group has no value reader.");
    const auto& valueType = requestedType_->childAt(1);
    VectorPtr values;
    // Struct readers fill a ROW result in place.
    if (valueType->isRow()) {
      values = BaseVector::create(valueType, 0, pool_);
    }
    group.valueReader->readWithTiming(
        0, group.valueRows, /*incomingNulls=*/nullptr);
    group.valueReader->getValues(group.valueRows, &values);
    return values;
  }

  // Places the 'numValues' values of 'groupValues' in one vector, one group
  // after another, so that the map values of each batch are a dictionary over
  // it. The values of a single group are used without a copy.
  //
  // Example: [a, f] from group {1, 5} and [g] from Default in the loadGroup()
  // example give the stripe values [a, f, g]. loadKeyNodes() has moved the
  // nextValue of key 9 from 0 to 2 to match. A single group's values, such as
  // [g] alone, are used as they are.
  VectorPtr concatenateGroupValues(
      std::vector<VectorPtr>& groupValues,
      uint64_t numValues) {
    if (groupValues.size() <= 1) {
      return groupValues.empty() ? nullptr : std::move(groupValues.front());
    }
    auto values = BaseVector::create(
        requestedType_->childAt(1),
        static_cast<vector_size_t>(numValues),
        pool_);
    vector_size_t offset{0};
    for (auto& group : groupValues) {
      values->copy(group.get(), offset, 0, group->size());
      offset += group->size();
      group.reset();
    }
    return values;
  }

  // Copies the selected keys that do not fit inline into one buffer. Result
  // keys outlive this per-stripe reader and the file schema they view.
  void copyKeyStrings() {
    size_t numBytes{0};
    for (const auto& keyNode : keyNodes_) {
      if (!keyNode.key.isInline()) {
        numBytes += keyNode.key.size();
      }
    }
    if (numBytes == 0) {
      return;
    }
    keyStrings_ = AlignedBuffer::allocate<char>(numBytes, pool_);
    auto* data = keyStrings_->asMutable<char>();
    for (auto& keyNode : keyNodes_) {
      if (keyNode.key.isInline()) {
        continue;
      }
      const auto size = keyNode.key.size();
      memcpy(data, keyNode.key.data(), size);
      keyNode.key = StringView(data, size);
      data += size;
    }
  }

  // Returns whether 'rows' holds a non-null map. Stops at the first one.
  bool anyNonNullMap(const RowSet& rows) const {
    if (nullsInReadRange_ == nullptr) {
      return !rows.empty();
    }
    const auto* nulls = nullsInReadRange_->as<uint64_t>();
    return std::any_of(rows.begin(), rows.end(), [&](vector_size_t row) {
      return !bits::isBitNull(nulls, row);
    });
  }

  // Sets the sizes and offsets of the maps at 'rows', the maps getValues()
  // returns, from the in-map bits of each key node, and returns the number of
  // map entries. Without a non-null map, every map is empty.
  //
  // Example: rows [0, 2] of the read() example and the key nodes of the
  // loadGroup() example.
  //   key 1: maps 0 and 2 -> output rows 0 and 1
  //   key 9: map 2        -> output row 1
  //   sizes [1, 2], offsets [0, 1], 3 entries
  // Map 1 is not returned, so it has no output row.
  vector_size_t calculateOffsets(
      const RowSet& rows,
      bool anyNonNull,
      vector_size_t* rawOffsets,
      vector_size_t* rawSizes) {
    const auto numRows = static_cast<vector_size_t>(rows.size());
    std::fill(rawSizes, rawSizes + numRows, 0);
    if (anyNonNull) {
      mapNonNullRows(rows);
      syncRanks();
      for (const auto& keyNode : keyNodes_) {
        forEachEntry(keyNode, [&](vector_size_t row, vector_size_t /*value*/) {
          ++rawSizes[row];
        });
      }
    }
    vector_size_t numEntries{0};
    for (vector_size_t i = 0; i < numRows; ++i) {
      rawOffsets[i] = numEntries;
      numEntries += rawSizes[i];
    }
    return numEntries;
  }

  // Maps each non-null map of the last read range to its position in 'rows'.
  void mapNonNullRows(const RowSet& rows) {
    outputRowOfNonNull_.assign(numNonNullsInReadRange_, -1);
    const auto* nulls =
        nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
    vector_size_t nonNull{0};
    vector_size_t countedRows{0};
    for (vector_size_t i = 0; i < static_cast<vector_size_t>(rows.size());
         ++i) {
      const auto row = rows[i];
      if (nulls == nullptr) {
        outputRowOfNonNull_[row] = i;
        continue;
      }
      nonNull += static_cast<vector_size_t>(
          bits::countNonNulls(nulls, countedRows, row));
      countedRows = row;
      if (!bits::isBitNull(nulls, row)) {
        outputRowOfNonNull_[nonNull] = i;
      }
    }
  }

  // Counts the values of each key node up to the first non-null map of the
  // last read range. Reads that returned no values are counted here. Every
  // key node is synced at once, so one rankRow_ serves them all.
  void syncRanks() {
    if (rankRow_ == firstNonNullInReadRange_) {
      return;
    }
    for (auto& keyNode : keyNodes_) {
      keyNode.nextValue += bits::countBits(
          keyNode.inMapBits,
          keyNode.inMapStart + static_cast<int32_t>(rankRow_),
          keyNode.inMapStart + static_cast<int32_t>(firstNonNullInReadRange_));
    }
    rankRow_ = firstNonNullInReadRange_;
  }

  // Calls 'emit(row, value)' for each entry of 'keyNode' in the last read
  // range that getValues() returns. 'row' is the position in the getValues()
  // rows and 'value' indexes stripeValues_. This is where the key-major layout
  // turns into rows: each set in-map bit of the key is one entry of one map.
  // calculateOffsets() counts the entries and copyValues() writes them.
  //
  // Example: a stripe holds the non-null maps {1:a, 2:b, 7:c}, {5:d, 3:e} and
  // {1:f, 9:g}. Keys 1 and 5 are in one group, keys 2 and 3 in another, and
  // keys 7 and 9 in a Default group that lists its keys. A read selects keys 1
  // and 9 and returns maps 0 and 2, so stripeValues_ is [a, f, g] and the key
  // nodes are:
  //   key 1: in-map bits 1 0 1, nextValue 0
  //   key 9: in-map bits 0 0 1, nextValue 2
  // Key 1 emits (row 0, value 0) for map 0 and (row 1, value 1) for map 2. Key
  // 9 emits (row 1, value 2) for map 2. The maps are {1:a} and {1:f, 9:g}.
  template <typename Emit>
  void forEachEntry(const HybridFlatMapKeyNode<T>& keyNode, Emit emit) const {
    const auto begin =
        keyNode.inMapStart + static_cast<int32_t>(firstNonNullInReadRange_);
    auto value = keyNode.nextValue;
    bits::forEachSetBit(
        keyNode.inMapBits,
        begin,
        begin + static_cast<int32_t>(numNonNullsInReadRange_),
        [&](int32_t inMapRow) {
          if (const auto row = outputRowOfNonNull_[inMapRow - begin];
              row >= 0) {
            emit(row, value);
          }

          // A map that getValues() does not return still holds the key's
          // value, so the next value index advances anyway.
          ++value;
        });
  }

  // Fills the 'numEntries' map keys and values of 'map', key node after key
  // node. Entries of a map follow group order, then key-presence order, as in
  // the batch reader. The map values are a dictionary over stripeValues_.
  // 'rawOffsets' serve as the next entry of each map while filling, and are
  // restored at the end.
  void copyValues(
      vector_size_t* rawOffsets,
      vector_size_t numRows,
      vector_size_t numEntries,
      MapVector& map) {
    auto* flatKeys = map.mapKeys()->asChecked<FlatVector<T>>();
    flatKeys->resetNulls();
    auto* rawKeys = flatKeys->mutableRawValues();
    if constexpr (std::is_same_v<T, StringView>) {
      if (keyStrings_ != nullptr) {
        flatKeys->setStringBuffers({keyStrings_});
      } else {
        flatKeys->clearStringBuffers();
      }
    }
    BufferPtr indices = allocateIndices(numEntries, pool_);
    auto* rawIndices = indices->asMutable<vector_size_t>();
    for (const auto& keyNode : keyNodes_) {
      forEachEntry(keyNode, [&](vector_size_t row, vector_size_t value) {
        const auto entry = rawOffsets[row]++;
        rawKeys[entry] = keyNode.key;
        rawIndices[entry] = value;
      });
    }

    // Each offset now points past its map's entries, which is where the next
    // map starts.
    std::copy_backward(
        rawOffsets, rawOffsets + numRows - 1, rawOffsets + numRows);
    rawOffsets[0] = 0;
    map.setKeysAndValues(
        map.mapKeys(),
        BaseVector::wrapInDictionary(
            /*nulls=*/nullptr, std::move(indices), numEntries, stripeValues_));
  }

  common::ScanSpec* const keysSpec_;
  common::ScanSpec* const valuesSpec_;
  // Groups that hold a key that passes the keys filter, in schema order.
  std::vector<HybridFlatMapGroup<T>> groups_;
  // Selected keys present in the stripe, in group order, then key-presence
  // order. Set by loadKeyNodes().
  std::vector<HybridFlatMapKeyNode<T>> keyNodes_;
  // Holds the selected keys that do not fit inline in a StringView.
  BufferPtr keyStrings_;
  bool loaded_{false};
  // Non-null maps of the stripe that reads and skips have passed.
  uint32_t nextNonNullRow_{0};
  // Fewest non-null maps of a group with entries in the stripe. Unbounded
  // until loadKeyNodes() decodes the groups.
  uint32_t nonNullRowBound_{std::numeric_limits<uint32_t>::max()};
  // Non-null map up to which HybridFlatMapKeyNode::nextValue is counted.
  uint32_t rankRow_{0};
  // Non-null map index of the first non-null map of the last read range.
  uint32_t firstNonNullInReadRange_{0};
  uint32_t numNonNullsInReadRange_{0};
  // Position in the getValues() rows of each non-null map of the last read
  // range, or -1 when getValues() does not return the map.
  std::vector<vector_size_t> outputRowOfNonNull_;
  // The values of the selected keys of every group in the stripe. Null when
  // no selected key is present.
  VectorPtr stripeValues_;
};

} // namespace

std::unique_ptr<velox::dwio::common::SelectiveColumnReader>
createHybridFlatMapColumnReader(
    const TypePtr& requestedType,
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
    NimbleParams& params,
    velox::common::ScanSpec& scanSpec) {
  VELOX_DCHECK(requestedType->isMap());
  NIMBLE_USER_CHECK(
      !scanSpec.isFlatMapAsStruct(),
      "Hybrid FlatMap does not support struct output.");
  auto kind = requestedType->childAt(0)->kind();
  switch (kind) {
    case TypeKind::TINYINT:
      return std::make_unique<HybridFlatMapColumnReader<int8_t>>(
          requestedType, fileType, params, scanSpec);
    case TypeKind::SMALLINT:
      return std::make_unique<HybridFlatMapColumnReader<int16_t>>(
          requestedType, fileType, params, scanSpec);
    case TypeKind::INTEGER:
      return std::make_unique<HybridFlatMapColumnReader<int32_t>>(
          requestedType, fileType, params, scanSpec);
    case TypeKind::BIGINT:
      return std::make_unique<HybridFlatMapColumnReader<int64_t>>(
          requestedType, fileType, params, scanSpec);
    case TypeKind::VARBINARY:
    case TypeKind::VARCHAR:
      return std::make_unique<HybridFlatMapColumnReader<StringView>>(
          requestedType, fileType, params, scanSpec);
    default:
      VELOX_UNSUPPORTED("Not supported key type: {}", kind);
  }
}

} // namespace facebook::nimble
