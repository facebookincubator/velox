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
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "folly/Likely.h"
#include "folly/ScopeGuard.h"
#include "folly/container/F14Set.h"
#include "velox/buffer/Buffer.h"
#include "velox/dwio/common/ColumnSelector.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/serializer/BatchedStreamDecoder.h"
#include "velox/dwio/nimble/serializer/StreamDataParser.h"
#include "velox/dwio/nimble/velox/Decoder.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/dwio/nimble/velox/SchemaUtils.h"

#include <algorithm>
#include <limits>
#include <optional>

namespace facebook::nimble {

namespace {

const StreamDescriptor& getMainDescriptor(const Type& type) {
  switch (type.kind()) {
    case Kind::Scalar:
      return type.asScalar().scalarDescriptor();
    case Kind::TimestampMicroNano:
      return type.asTimestampMicroNano().microsDescriptor();
    case Kind::Array:
      return type.asArray().lengthsDescriptor();
    case Kind::Map:
      return type.asMap().lengthsDescriptor();
    case Kind::Row:
      return type.asRow().nullsDescriptor();
    case Kind::FlatMap:
      return type.asFlatMap().nullsDescriptor();
    default:
      // ArrayWithOffsets and SlidingWindowMap are not supported.
      NIMBLE_UNSUPPORTED(
          "Schema type {} is not supported.", toString(type.kind()));
  }
}

bool checkColumnProjectionSubfield(
    const RowType& row,
    const Deserializer::Subfield& subfield) {
  const auto& path = subfield.path();
  NIMBLE_USER_CHECK(
      subfield.valid(),
      "Column projection deserialize requires a named subfield path: {}",
      subfield);
  auto childIndex = row.findChild(subfield.baseName());
  NIMBLE_USER_CHECK(
      childIndex.has_value(),
      "Column projection subfield does not exist in schema: {}",
      subfield);
  const auto* nestedType = row.childAt(childIndex.value()).get();
  for (size_t i = 1; i < path.size(); ++i) {
    if (nestedType->isFlatMap()) {
      NIMBLE_USER_CHECK(
          path[i]->is(velox::common::SubfieldKind::kStringSubscript) ||
              path[i]->is(velox::common::SubfieldKind::kLongSubscript),
          "FlatMap projection requires a string or integer key: {}",
          subfield);
      NIMBLE_USER_CHECK_EQ(
          i + 1,
          path.size(),
          "Nested projection inside a FlatMap value is not supported: {}",
          subfield);
      return true;
    }
    NIMBLE_USER_CHECK(
        path[i]->is(velox::common::SubfieldKind::kNestedField),
        "Column projection deserialize only supports named fields. Path: {}, element: {}",
        subfield,
        path[i]->toString());
    NIMBLE_USER_CHECK(
        nestedType->isRow(),
        "Column projection deserialize only supports nested Row fields. Path: {}, type: {}",
        subfield,
        nestedType->kind());
    const auto& nestedName =
        path[i]->asChecked<velox::common::Subfield::NestedField>()->name();
    childIndex = nestedType->asRow().findChild(nestedName);
    NIMBLE_USER_CHECK(
        childIndex.has_value(),
        "Column projection subfield does not exist in schema: {}",
        subfield);
    nestedType = nestedType->asRow().childAt(childIndex.value()).get();
  }
  return false;
}

// One reader operation. Executed as `reader_->skip(numRows)` when
// `skip == true`, or `reader_->next(numRows, ...)` when `skip == false`.
struct DecodeOp {
  bool skip;
  uint32_t numRows;
};

// Turns per-batch rowRanges (in run-local coordinates) into the minimal
// sequence of skip/read ops that visits each range exactly once.
// Adjacent ranges with no gap fold into a single read op. Empty ranges
// are dropped. Returns `[{read, 0}]` when the result would otherwise be
// empty, so callers always emit at least one `reader_->next` and produce
// a non-null output vector.
std::vector<DecodeOp> buildDecodeOps(
    const std::vector<nimble::RowRange>& ranges) {
  std::vector<DecodeOp> ops;
  // Worst case is skip+read per range (all disjoint, non-contiguous).
  ops.reserve(2 * ranges.size());
  uint32_t cursor{0};
  for (const auto& range : ranges) {
    // Empty range = "no rows from this batch". Common when the caller
    // uses the rowRanges overload to skip whole batches, or when a
    // batch's rowCount is 0. Nothing to emit; move on.
    if (range.numRows() == 0) {
      continue;
    }
    if (range.startRow > cursor) {
      ops.push_back({/*skip=*/true, range.startRow - cursor});
    }
    // Fold into the preceding read op when this range is contiguous with
    // it; otherwise start a fresh read (either `ops` is empty or the last
    // op was the skip we just pushed).
    if (!ops.empty() && !ops.back().skip) {
      ops.back().numRows += range.numRows();
    } else {
      ops.push_back({/*skip=*/false, range.numRows()});
    }
    cursor = range.endRow;
  }
  // Every input range was empty (or `ranges` itself was). Emit one
  // zero-length read so the caller still runs a `reader_->next` and
  // produces a non-null empty output vector (see ProjectorFormatTest
  // .emptyInput and equivalents).
  if (ops.empty()) {
    ops.push_back({/*skip=*/false, 0});
  }
  return ops;
}

} // namespace

Deserializer::ProjectedField* Deserializer::ProjectedField::ensureChild(
    const std::string& name) {
  auto& selectedChild = children[name];
  if (selectedChild == nullptr) {
    selectedChild = std::make_unique<ProjectedField>();
  }
  return selectedChild.get();
}

velox::TypePtr Deserializer::buildProjectedType(
    const velox::TypePtr& source,
    const ProjectedField& selected,
    Deserializer::OutputProjection& projection) {
  if (selected.selectWholeField) {
    return source;
  }
  const auto& sourceRow = source->asRow();
  std::vector<std::string> names;
  std::vector<velox::TypePtr> types;
  names.reserve(selected.children.size());
  types.reserve(selected.children.size());
  std::vector<std::string> selectedNames;
  selectedNames.reserve(selected.children.size());
  for (const auto& [name, _] : selected.children) {
    selectedNames.emplace_back(name);
  }
  std::sort(selectedNames.begin(), selectedNames.end());
  for (const auto& name : selectedNames) {
    const auto sourceChannel = sourceRow.getChildIdx(name);
    const auto& selectedChild = *selected.children.at(name);
    projection.identityProjections.emplace_back(sourceChannel, names.size());
    names.emplace_back(name);
    auto& childProjection = projection.childProjections.emplace_back();
    if (selectedChild.selectWholeField) {
      types.emplace_back(sourceRow.childAt(sourceChannel));
    } else {
      types.emplace_back(buildProjectedType(
          sourceRow.childAt(sourceChannel), selectedChild, childProjection));
    }
  }
  return velox::ROW(std::move(names), std::move(types));
}

velox::RowTypePtr Deserializer::buildProjectedType(
    const velox::RowTypePtr& sourceType,
    const std::vector<Deserializer::Subfield>& selectedSubfields,
    Deserializer::OutputProjection& outputProjection) {
  ProjectedField root;
  for (const auto& subfield : selectedSubfields) {
    auto* selected = root.ensureChild(subfield.baseName());
    const auto& path = subfield.path();
    for (size_t i = 1; i < path.size(); ++i) {
      if (path[i]->is(velox::common::SubfieldKind::kStringSubscript) ||
          path[i]->is(velox::common::SubfieldKind::kLongSubscript)) {
        NIMBLE_CHECK_EQ(
            i,
            1,
            "FlatMap key projection is only supported for top-level fields: {}",
            subfield);
        selected->selectWholeField = true;
        break;
      }
      const auto& name =
          path[i]->asChecked<velox::common::Subfield::NestedField>()->name();
      selected = selected->ensureChild(name);
    }
    selected->selectWholeField = true;
  }

  return velox::checkedPointerCast<const velox::RowType>(
      buildProjectedType(sourceType, root, outputProjection));
}

FieldReaderParams Deserializer::createFieldReaderParams() const {
  FieldReaderParams params;
  params.flatMapFeatureSelector = flatMapFeatureSelector_;
  params.decodeExecutor = options_.decodeExecutor;
  params.maxDecodeParallelism = options_.maxDecodeParallelism;
  params.minStreamsPerDecodeUnit = options_.minStreamsPerDecodeUnit;
  if (options_.outputType == nullptr) {
    return params;
  }

  NIMBLE_CHECK(
      schema_->isRow(),
      "outputType requires Row schema root, got {}",
      toString(schema_->kind()));

  const auto& rootRow = schema_->asRow();
  NIMBLE_CHECK_EQ(
      rootRow.childrenCount(),
      options_.outputType->size(),
      "Output type field count must match schema field count");

  for (size_t i = 0; i < rootRow.childrenCount(); ++i) {
    if (!rootRow.childAt(i)->isFlatMap()) {
      continue;
    }
    const auto& outputFieldType = options_.outputType->childAt(i);
    if (outputFieldType->kind() != velox::TypeKind::ROW) {
      continue;
    }

    const auto& columnName = rootRow.nameAt(i);
    params.readFlatMapFieldAsStruct.insert(columnName);

    const auto& rowType = outputFieldType->asRow();
    std::vector<std::string> features;
    features.reserve(rowType.size());
    for (size_t j = 0; j < rowType.size(); ++j) {
      features.push_back(rowType.nameOf(j));
    }
    params.flatMapFeatureSelector[columnName] = FeatureSelection{
        .features = std::move(features),
        .mode = SelectionMode::Include,
    };
  }
  return params;
}

Deserializer::Deserializer(
    std::shared_ptr<const Type> schema,
    velox::memory::MemoryPool* pool)
    : Deserializer{std::move(schema), pool, {}} {}

Deserializer::Deserializer(
    std::shared_ptr<const Type> schema,
    velox::memory::MemoryPool* pool,
    DeserializerOptions options)
    : Deserializer{
          std::move(schema),
          /*selectedSubfields=*/{},
          pool,
          std::move(options)} {}

Deserializer::Deserializer(
    std::shared_ptr<const Type> schema,
    const std::vector<Deserializer::Subfield>& selectedSubfields,
    velox::memory::MemoryPool* pool,
    DeserializerOptions options)
    : schema_{std::move(schema)},
      pool_{pool},
      options_{std::move(options)},
      hasColumnProjection_{!selectedSubfields.empty()} {
  auto veloxType = convertToVeloxType(*schema_);
  if (!hasColumnProjection_) {
    initialize(
        velox::dwio::common::TypeWithId::create(veloxType),
        [](uint32_t) { return true; });
    return;
  }

  initializeColumnProjection(veloxType, selectedSubfields);
}

void Deserializer::initializeColumnProjection(
    const velox::TypePtr& veloxType,
    const std::vector<Deserializer::Subfield>& selectedSubfields) {
  NIMBLE_CHECK(hasColumnProjection_, "Column projection is not enabled");
  const auto rowType =
      velox::checkedPointerCast<const velox::RowType>(veloxType);
  std::vector<std::string> projectedColumnPaths;
  folly::F14FastSet<std::string> selectedSubfieldSet;
  folly::F14FastSet<std::string> projectedColumnPathSet;
  folly::F14FastMap<std::string, folly::F14FastSet<std::string>>
      flatMapFeatureSets;
  projectedColumnPaths.reserve(selectedSubfields.size());
  selectedSubfieldSet.reserve(selectedSubfields.size());
  projectedColumnPathSet.reserve(selectedSubfields.size());
  flatMapFeatureSets.reserve(selectedSubfields.size());
  for (const auto& subfield : selectedSubfields) {
    const bool selectsFlatMapKey =
        checkColumnProjectionSubfield(schema_->asRow(), subfield);
    const auto& path = subfield.path();
    const auto selectedPath = subfield.toString();
    NIMBLE_USER_CHECK(
        selectedSubfieldSet.insert(selectedPath).second,
        "Duplicate column projection subfield: {}",
        subfield);
    std::string columnPath;
    if (selectsFlatMapKey) {
      columnPath = subfield.baseName();
      auto feature = path[1]->is(velox::common::SubfieldKind::kStringSubscript)
          ? path[1]
                ->asChecked<velox::common::Subfield::StringSubscript>()
                ->index()
          : std::to_string(
                path[1]
                    ->asChecked<velox::common::Subfield::LongSubscript>()
                    ->index());
      const bool newFlatMapFeature =
          flatMapFeatureSets[columnPath].insert(feature).second;
      NIMBLE_USER_CHECK(
          newFlatMapFeature, "Duplicate FlatMap projection key: {}", subfield);
      flatMapFeatureSelector_[columnPath].features.emplace_back(
          std::move(feature));
    } else {
      columnPath = subfield.toString();
    }
    const bool newColumnPath = projectedColumnPathSet.insert(columnPath).second;
    if (newColumnPath) {
      projectedColumnPaths.emplace_back(columnPath);
    }
  }

  outputProjection_ = std::make_unique<OutputProjection>();
  outputType_ =
      buildProjectedType(rowType, selectedSubfields, *outputProjection_);
  auto selector = std::make_shared<velox::dwio::common::ColumnSelector>(
      rowType, projectedColumnPaths);
  initialize(selector->getSchemaWithId(), [selector](auto nodeId) {
    return selector->shouldReadNode(nodeId);
  });
}

void Deserializer::initialize(
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& schemaWithId,
    const std::function<bool(uint32_t)>& isSelected) {
  const auto params = createFieldReaderParams();
  parser_ = std::make_unique<serde::StreamDataParser>(pool_, options_);

  std::vector<uint32_t> offsets;
  rootFactory_ = FieldReaderFactory::create(
      params, schema_, schemaWithId, offsets, isSelected, pool_);

  if (hasColumnProjection_) {
    const auto maxSelectedOffset =
        *std::max_element(offsets.begin(), offsets.end());
    selectedStreamOffsetFlags_.resize(maxSelectedOffset + 1, false);
    for (const auto offset : offsets) {
      selectedStreamOffsetFlags_[offset] = true;
    }
  }

  SchemaReader::traverseSchema(schema_, [this](auto depth, auto& type, auto&) {
    createDeserializersForType(type, depth);
  });

  reader_ = rootFactory_->createReader(deserializerMap_);

  // Build flat vector for O(1) stream offset lookup during deserialize().
  uint32_t maxOffset = 0;
  for (const auto& [offset, _] : deserializerMap_) {
    maxOffset = std::max(maxOffset, offset);
  }
  deserializers_.resize(maxOffset + 1, nullptr);
  for (auto& [offset, decoder] : deserializerMap_) {
    deserializers_[offset] = decoder.get();
  }

  if (!inMapChildTypes_.empty()) {
    streamPresentFlags_.resize(maxOffset + 1, false);
    valueOffsetToInMap_.resize(maxOffset + 1, kInvalidInMapOffset);
    // Populate the reverse-lookup table: for each top-level FlatMap child,
    // record its inMap stream offset at every one of its presence-stream
    // anchors. The per-batch in-map inference reads this to map a present
    // child stream back to its owning child without re-walking the schema.
    //
    // Value leaves are not enough for omitted in-map inference: a child present
    // in every row with all-null values may only have null/container/nested
    // in-map streams to prove an omitted all-true in-map stream. Therefore,
    // visitPresenceStreamOffsets records all data-bearing offsets in the child
    // subtree, including Row/FlatMap null streams and nested FlatMap in-map
    // streams.
    // Relies on RowFieldWriter writing every field over the same
    // OrderedRanges, so sibling Row children populate in lockstep; if any
    // sibling's value stream is present in a batch, all are. If a future
    // writer ever made Row children conditionally absent, the in-map
    // inference below would over-attribute presence to keys whose first
    // child was absent but a sibling was present.
    for (const auto& [inMapOffset, childType] : inMapChildTypes_) {
      visitPresenceStreamOffsets(
          *childType,
          [this, _inMapOffset = inMapOffset](offset_size presenceOffset) {
            // This lookup is only indexed by presentStreamOffsets_, which is
            // populated from decoded streams and therefore bounded by
            // deserializerMap_. visitPresenceStreamOffsets() walks schema
            // anchors too, including projected-away streams with no decoder.
            if (presenceOffset >= valueOffsetToInMap_.size()) {
              return false;
            }
            valueOffsetToInMap_[presenceOffset] = _inMapOffset;
            return false;
          });
    }
  }
}

Deserializer::~Deserializer() = default;

void Deserializer::createDeserializersForType(
    const Type& type,
    uint32_t depth) {
  const auto streamOffset = getMainDescriptor(type).offset();
  if (shouldDecodeStream(streamOffset)) {
    deserializerMap_[streamOffset] = std::make_unique<BatchedStreamDecoder>(
        &type,
        /*isInMapStream=*/false,
        options_.bufferPoolCapacity,
        pool_);
  }
  // FlatMap is only supported at depth 1 (top-level columns). Register each
  // child in-map stream so it is decoded like other physical streams.
  if (type.isFlatMap()) {
    NIMBLE_CHECK_EQ(
        depth, 1, "FlatMap is only supported as a top-level column (depth 1)");
    auto& flatMap = type.asFlatMap();
    for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
      const auto inMapOffset = flatMap.inMapDescriptorAt(i).offset();
      if (!shouldDecodeStream(inMapOffset)) {
        continue;
      }
      deserializerMap_[inMapOffset] = std::make_unique<BatchedStreamDecoder>(
          &type,
          /*isInMapStream=*/true,
          options_.bufferPoolCapacity,
          pool_);
      inMapChildTypes_[inMapOffset] = flatMap.childAt(i).get();
    }
  }
}

void Deserializer::deserialize(std::string_view data, velox::VectorPtr& output)
    const {
  deserialize(folly::Range<const std::string_view*>(&data, 1), output);
}

void Deserializer::deserialize(
    const std::vector<std::string_view>& data,
    velox::VectorPtr& output) const {
  deserialize(
      folly::Range<const std::string_view*>(data.data(), data.size()), output);
}

void Deserializer::deserialize(
    const std::vector<std::string_view>& data,
    velox::VectorPtr& output,
    std::vector<uint32_t>& outputRowCounts) const {
  deserializeImpl(
      folly::Range<const std::string_view*>(data.data(), data.size()),
      {},
      output,
      &outputRowCounts);
}

void Deserializer::appendToOutput(
    velox::VectorPtr&& decoded,
    velox::VectorPtr& output) const {
  if (FOLLY_LIKELY(output == nullptr)) {
    output = std::move(decoded);
    return;
  }
  output->append(decoded.get());
}

velox::VectorPtr Deserializer::projectOutput(velox::VectorPtr&& decoded) const {
  if (!hasColumnProjection_) {
    return std::move(decoded);
  }
  NIMBLE_CHECK_NOT_NULL(
      outputProjection_, "Output projection must be initialized");
  NIMBLE_CHECK_NOT_NULL(outputType_, "Output type must be initialized");

  return projectOutput(std::move(decoded), outputType_, *outputProjection_);
}

velox::VectorPtr Deserializer::projectOutput(
    velox::VectorPtr&& source,
    const velox::TypePtr& projectedType,
    const OutputProjection& projection) const {
  auto* decodedRow = source->asChecked<velox::RowVector>();
  const auto& projectedRowType = projectedType->asRow();
  NIMBLE_CHECK_EQ(
      projection.identityProjections.size(), projectedRowType.size());
  NIMBLE_CHECK_EQ(projection.childProjections.size(), projectedRowType.size());
  std::vector<velox::VectorPtr> children(projectedRowType.size());
  for (const auto& identity : projection.identityProjections) {
    const auto inputChannel = identity.inputChannel;
    const auto outputChannel = identity.outputChannel;
    const auto& childProjection = projection.childProjections[outputChannel];
    auto decodedChild = decodedRow->childAt(inputChannel);
    NIMBLE_CHECK_NOT_NULL(
        decodedChild,
        "Projected field was not decoded: {}",
        projectedRowType.nameOf(outputChannel));
    if (childProjection.identityProjections.empty()) {
      children[outputChannel] = std::move(decodedChild);
    } else {
      children[outputChannel] = projectOutput(
          std::move(decodedChild),
          projectedRowType.childAt(outputChannel),
          childProjection);
    }
  }
  return std::make_shared<velox::RowVector>(
      pool_,
      projectedType,
      decodedRow->nulls(),
      decodedRow->size(),
      std::move(children),
      std::nullopt);
}

void Deserializer::decodeRun(DecodeRun& run, velox::VectorPtr& output) const {
  if (FOLLY_UNLIKELY(run.batches == 0)) {
    return;
  }
  const auto ops = buildDecodeOps(runRanges_);
  for (const auto& op : ops) {
    if (op.skip) {
      reader_->skip(op.numRows);
      continue;
    }
    velox::VectorPtr decoded;
    reader_->next(op.numRows, decoded, nullptr);
    decoded = projectOutput(std::move(decoded));
    appendToOutput(std::move(decoded), output);
  }
  run = {};
  runRanges_.clear();
  reader_->reset();
}

void Deserializer::appendStreamSegments(
    uint32_t rowCount,
    uint32_t startRow,
    bool requiresBarrier) const {
  const auto maxStreamOffset = deserializers_.size() - 1;
  const auto version = parser_->version();
  const auto streamEncodingUsesVarintRowCount =
      parser_->streamEncodingUsesVarintRowCount();
  const bool hasInMapChildren = !inMapChildTypes_.empty();
  if (hasInMapChildren) {
    std::fill(streamPresentFlags_.begin(), streamPresentFlags_.end(), false);
    presentStreamOffsets_.clear();
  }
  parser_->iterateStreams([&](uint32_t offset, std::string_view streamData) {
    if (FOLLY_UNLIKELY(offset > maxStreamOffset)) {
      return;
    }
    if (FOLLY_UNLIKELY(!shouldDecodeStream(offset))) {
      return;
    }
    if (hasInMapChildren) {
      if (!streamPresentFlags_[offset]) {
        streamPresentFlags_[offset] = true;
        presentStreamOffsets_.emplace_back(offset);
      }
    }
    auto* decoder = deserializers_[offset];
    NIMBLE_CHECK_NOT_NULL(decoder, "Missing decoder for stream");
    BatchedStreamDecoder::as(decoder)->addBatch(
        startRow, streamData, version, streamEncodingUsesVarintRowCount);
  });

  if (!hasInMapChildren) {
    return;
  }
  const auto presentStreamCount = presentStreamOffsets_.size();
  for (size_t i = 0; i < presentStreamCount; ++i) {
    const auto inMapOffset = valueOffsetToInMap_[presentStreamOffsets_[i]];
    if (inMapOffset == kInvalidInMapOffset ||
        streamPresentFlags_[inMapOffset]) {
      continue;
    }
    auto* decoder = deserializers_[inMapOffset];
    NIMBLE_CHECK_NOT_NULL(decoder, "Missing FlatMap in-map decoder");
    auto* segmentedDecoder = BatchedStreamDecoder::as(decoder);
    if (requiresBarrier) {
      segmentedDecoder->addPresentInMapBatch();
    } else {
      segmentedDecoder->addPresentInMapBatch(startRow, rowCount);
    }
    streamPresentFlags_[inMapOffset] = true;
  }
}

uint32_t Deserializer::appendBatch(
    std::string_view batch,
    std::optional<nimble::RowRange> rowRange,
    DecodeRun& run,
    velox::VectorPtr& output) const {
  const auto rowCount = parser_->initialize(batch);
  const auto requiresBarrier = parser_->requiresNullBarrier();
  if (FOLLY_UNLIKELY(requiresBarrier)) {
    decodeRun(run, output);
  }

  // Rows this batch exposes: the parser's rowRange (kTablet header) or the
  // whole batch if the header didn't encode one.
  nimble::RowRange range =
      parser_->rowRange().value_or(nimble::RowRange{0, rowCount});
  // A caller range narrows within what the batch exposes rather than
  // replacing it, so it is relative to `range.startRow`. For a kTablet
  // batch windowed to [500, 600), caller [0, 50) selects [500, 550). For
  // every other version `range` starts at 0, so the two coincide.
  if (rowRange.has_value()) {
    NIMBLE_USER_CHECK_LE(
        rowRange->startRow,
        rowRange->endRow,
        "rowRange startRow must be <= endRow");
    NIMBLE_USER_CHECK_LE(
        rowRange->endRow,
        range.numRows(),
        "rowRange endRow exceeds the rows this batch exposes");
    range = nimble::RowRange{
        range.startRow + rowRange->startRow, range.startRow + rowRange->endRow};
  }

  appendStreamSegments(rowCount, /*startRow=*/run.rows, requiresBarrier);
  runRanges_.push_back({run.rows + range.startRow, run.rows + range.endRow});
  run.rows += rowCount;
  ++run.batches;
  if (FOLLY_UNLIKELY(requiresBarrier)) {
    decodeRun(run, output);
    parser_->reset();
  }
  return range.numRows();
}

void Deserializer::deserialize(
    folly::Range<const std::string_view*> data,
    velox::VectorPtr& output) const {
  deserializeImpl(data, /*rowRanges=*/{}, output, /*outputRowCounts=*/nullptr);
}

void Deserializer::deserialize(
    const std::vector<std::string_view>& data,
    const std::vector<nimble::RowRange>& rowRanges,
    velox::VectorPtr& output) const {
  deserializeImpl(
      folly::Range<const std::string_view*>(data.data(), data.size()),
      folly::Range<const nimble::RowRange*>(rowRanges.data(), rowRanges.size()),
      output,
      /*outputRowCounts=*/nullptr);
}

void Deserializer::deserializeImpl(
    folly::Range<const std::string_view*> data,
    folly::Range<const nimble::RowRange*> rowRanges,
    velox::VectorPtr& output,
    std::vector<uint32_t>* outputRowCounts) const {
  NIMBLE_USER_CHECK(!data.empty(), "Expected at least one serialized batch");
  if (!rowRanges.empty()) {
    NIMBLE_USER_CHECK_EQ(
        data.size(),
        rowRanges.size(),
        "data and rowRanges must have the same size");
  }
  // `runRanges_` must be empty across `deserialize*` calls — check on
  // entry, clear on exit (including exceptions) via SCOPE_EXIT.
  NIMBLE_CHECK(
      runRanges_.empty(), "runRanges_ must be empty on deserialize entry");
  SCOPE_EXIT {
    runRanges_.clear();
  };

  output = nullptr;
  if (outputRowCounts != nullptr) {
    outputRowCounts->clear();
    outputRowCounts->reserve(data.size());
  }
  DecodeRun run;
  runRanges_.reserve(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    std::optional<nimble::RowRange> rowRange;
    if (!rowRanges.empty()) {
      rowRange = rowRanges[i];
    }
    const auto outputRows = appendBatch(data[i], rowRange, run, output);
    if (outputRowCounts != nullptr) {
      outputRowCounts->emplace_back(outputRows);
    }
  }
  decodeRun(run, output);
  parser_->reset();
}

} // namespace facebook::nimble
