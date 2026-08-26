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
#include "velox/dwio/nimble/writer/Writer.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "folly/container/F14Map.h"
#include "velox/common/base/Counters.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/CpuWallTimer.h"
#include "velox/common/time/Timer.h"
#include "velox/dwio/common/ExecutorBarrier.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryCatalog.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/index/ClusterIndexFactory.h"
#include "velox/dwio/nimble/index/DenseIndexFactory.h"
#include "velox/dwio/nimble/index/HashIndexWriter.h"
#include "velox/dwio/nimble/index/IndexSerialization.h"
#include "velox/dwio/nimble/index/SortedIndexWriter.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/FileProperties.h"
#include "velox/dwio/nimble/tablet/IndexGenerated.h"
#include "velox/dwio/nimble/velox/BufferGrowthPolicy.h"
#include "velox/dwio/nimble/velox/ChunkedStreamWriter.h"
#include "velox/dwio/nimble/velox/FieldWriter.h"
#include "velox/dwio/nimble/velox/LayoutPlanner.h"
#include "velox/dwio/nimble/velox/MetadataGenerated.h"
#include "velox/dwio/nimble/velox/RawSizeUtils.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"
#include "velox/dwio/nimble/velox/SchemaTypes.h"

#include "velox/dwio/nimble/velox/SharedDictionaryWriter.h"
#include "velox/dwio/nimble/velox/StatsGenerated.h"
#include "velox/dwio/nimble/velox/stats/VectorizedStatistics.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"
#include "velox/dwio/nimble/writer/StreamChunker.h"
#include "velox/type/Subfield.h"
#include "velox/type/Type.h"

namespace facebook::nimble {

using velox::dwio::common::TypeWithId;

namespace detail {

class WriterContext : public FieldWriterContext {
 public:
  WriterContext(velox::memory::MemoryPool& memoryPool, WriterOptions options)
      : FieldWriterContext{memoryPool, options.reclaimerFactory(), options.vectorDecoderVisitor},
        options_{std::move(options)},
        hasStripeDictionaryConfig_{hasStripeDictionaryConfig(options_)},
        logger_{
            this->options_.metricsLogger == nullptr
                ? std::make_shared<MetricsLogger>()
                : this->options_.metricsLogger} {
    inputBufferGrowthPolicy_ = this->options_.makeInputBufferGrowthPolicy();
    stringBufferGrowthPolicy_ = this->options_.makeStringBufferGrowthPolicy();
    ignoreTopLevelNulls_ = options_.ignoreTopLevelNulls;
    disableSharedStringBuffers_ = options_.disableSharedStringBuffers;
    maxFlatMapKeys_ = options_.maxFlatMapKeys;
    if (this->options_.encodingExecutor &&
        this->options_.maxEncodeParallelism > 0) {
      setParallelEncoding(
          this->options_.encodingExecutor.get(),
          this->options_.maxEncodeParallelism,
          this->options_.minStreamsPerEncodeUnit);
    }
  }

  const WriterOptions& options() const {
    return options_;
  }

  bool hasStripeDictionaryConfig() const {
    return hasStripeDictionaryConfig_;
  }

  velox::CpuWallTiming& encodingTiming() {
    return encodingTiming_;
  }

  const velox::CpuWallTiming& encodingTiming() const {
    return encodingTiming_;
  }

  velox::CpuWallTiming& writeTiming() {
    return writeTiming_;
  }

  const velox::CpuWallTiming& writeTiming() const {
    return writeTiming_;
  }

  velox::CpuWallTiming& ingestionTiming() {
    return ingestionTiming_;
  }

  const velox::CpuWallTiming& ingestionTiming() const {
    return ingestionTiming_;
  }

  velox::CpuWallTiming& encodingSelectionTiming() {
    return encodingSelectionTiming_;
  }

  const velox::CpuWallTiming& encodingSelectionTiming() const {
    return encodingSelectionTiming_;
  }

  std::shared_ptr<MetricsLogger>& logger() {
    return logger_;
  }

  const std::shared_ptr<MetricsLogger>& logger() const {
    return logger_;
  }

  uint64_t memoryUsed() const {
    return memoryUsed_;
  }

  void setMemoryUsed(uint64_t value) {
    memoryUsed_ = value;
  }

  void updateMemoryUsed(uint64_t value) {
    memoryUsed_ += value;
  }

  uint64_t bytesWritten() const {
    return bytesWritten_;
  }

  void setBytesWritten(uint64_t writtenBytes) {
    bytesWritten_ = writtenBytes;
  }

  uint64_t rowsInFile() const {
    return rowsInFile_;
  }

  void updateRowsInFile(uint64_t numRows) {
    rowsInFile_ += numRows;
  }

  uint32_t rowsInStripe() const {
    return rowsInStripe_;
  }

  void updateRowsInStripe(uint32_t numRows) {
    rowsInStripe_ += numRows;
  }

  uint64_t stripeEncodedPhysicalSize() const {
    return stripeEncodedPhysicalSize_;
  }

  void updateStripeEncodedPhysicalSize(uint64_t updateBytes) {
    stripeEncodedPhysicalSize_ += updateBytes;
  }

  uint64_t stripeEncodedLogicalSize() const {
    return stripeEncodedLogicalSize_;
  }

  void updateStripeEncodedLogicalSize(uint64_t bytes) {
    stripeEncodedLogicalSize_ += bytes;
  }

  uint64_t fileRawSize() const {
    return fileRawBytes_;
  }

  void updateFileRawSize(uint64_t bytes) {
    fileRawBytes_ += bytes;
  }

  std::vector<uint64_t>& rowsPerStripe() {
    return rowsPerStripe_;
  }

  const std::vector<uint64_t>& rowsPerStripe() const {
    return rowsPerStripe_;
  }

  void nextStripe() {
    rowsPerStripe_.push_back(rowsInStripe_);
    memoryUsed_ = 0;
    rowsInStripe_ = 0;
    stripeEncodedPhysicalSize_ = 0;
    stripeEncodedLogicalSize_ = 0;
    ++stripeIndex_;
  }

  size_t getStripeIndex() const {
    return stripeIndex_;
  }

 private:
  static bool hasStripeDictionaryConfig(const WriterOptions& options) {
    for (const auto& columnDictionary :
         options.experimentalSharedDictionaryEncoding.columns) {
      if (columnDictionary.dictionary.scope == SharedDictionaryScope::Stripe) {
        return true;
      }
    }
    for (const auto& flatMap :
         options.experimentalSharedDictionaryEncoding.flatMaps) {
      for (const auto& flatMapKey : flatMap.keys) {
        if (flatMapKey.dictionary.scope == SharedDictionaryScope::Stripe) {
          return true;
        }
      }
    }
    return false;
  }

  const WriterOptions options_;
  const bool hasStripeDictionaryConfig_;
  velox::CpuWallTiming encodingTiming_;
  velox::CpuWallTiming writeTiming_;
  velox::CpuWallTiming ingestionTiming_;
  velox::CpuWallTiming encodingSelectionTiming_;
  std::shared_ptr<MetricsLogger> logger_;
  uint64_t memoryUsed_{0};
  uint64_t bytesWritten_{0};
  uint64_t rowsInFile_{0};
  uint32_t rowsInStripe_{0};
  uint64_t stripeEncodedPhysicalSize_{0};
  uint64_t stripeEncodedLogicalSize_{0};
  uint64_t fileRawBytes_{0};
  std::vector<uint64_t> rowsPerStripe_;
  size_t stripeIndex_{0};
};

} // namespace detail

namespace {

using SchemaAttributeValues = std::vector<std::pair<std::string, std::string>>;
using SchemaAttributes = folly::F14FastMap<uint32_t, SchemaAttributeValues>;

std::string_view asView(const flatbuffers::FlatBufferBuilder& builder) {
  return {
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

const index::ClusterIndexConfig& clusterIndexConfig(
    const WriterOptions& options) {
  NIMBLE_USER_CHECK_NOT_NULL(
      options.clusterIndexConfig,
      "Cluster index key column storage can only be omitted when cluster index is enabled");
  return index::checkedIndexConfig<index::ClusterIndexConfig>(
      *options.clusterIndexConfig);
}

bool omitClusterIndexKeyColumnStorage(const WriterOptions& options) {
  if (!options.experimentalOmitClusterIndexKeyColumnStorage) {
    return false;
  }
  NIMBLE_USER_CHECK_NOT_NULL(
      options.clusterIndexConfig,
      "Cluster index key column storage can only be omitted when cluster index is enabled");
  return true;
}

std::vector<velox::column_index_t> storedInputColumnIndices(
    const velox::TypePtr& type,
    const WriterOptions& options) {
  NIMBLE_USER_CHECK(
      omitClusterIndexKeyColumnStorage(options),
      "storedInputColumnIndices is only used when cluster index key column storage is omitted");

  const auto rowType = velox::asRowType(type);
  std::vector<velox::column_index_t> indices;
  indices.reserve(rowType->size());

  const auto& indexOptions = clusterIndexConfig(options);
  std::unordered_set<std::string> keyColumns;
  keyColumns.reserve(indexOptions.columns.size());
  for (const auto& column : indexOptions.columns) {
    NIMBLE_USER_CHECK(
        rowType->containsChild(column),
        "Cluster index key column '{}' not found in input schema: {}",
        column,
        rowType->toString());
    keyColumns.insert(column);
  }

  for (auto i = 0; i < rowType->size(); ++i) {
    if (keyColumns.find(rowType->nameOf(i)) == keyColumns.end()) {
      indices.push_back(i);
    }
  }
  return indices;
}

velox::RowTypePtr storedDataType(
    const velox::TypePtr& type,
    const WriterOptions& options) {
  if (!omitClusterIndexKeyColumnStorage(options)) {
    return velox::asRowType(type);
  }

  const auto rowType = velox::asRowType(type);
  const auto indices = storedInputColumnIndices(type, options);
  std::vector<std::string> names;
  std::vector<velox::TypePtr> types;
  names.reserve(indices.size());
  types.reserve(indices.size());
  for (auto index : indices) {
    names.push_back(rowType->nameOf(index));
    types.push_back(rowType->childAt(index));
  }
  return velox::ROW(std::move(names), std::move(types));
}

void mapSchemaAttributeNode(
    const velox::dwio::common::TypeWithId& inputNode,
    const velox::dwio::common::TypeWithId& storedNode,
    const SchemaAttributes& inputAttributes,
    SchemaAttributes& remappedAttributes) {
  // schemaAttributes is sparse; nodes without caller-provided attributes are
  // intentionally skipped.
  auto it = inputAttributes.find(inputNode.id());
  if (it != inputAttributes.end()) {
    remappedAttributes[storedNode.id()] = it->second;
  }

  NIMBLE_CHECK_EQ(
      inputNode.size(),
      storedNode.size(),
      "Stored schema subtree must match input schema subtree");
  for (auto i = 0; i < inputNode.size(); ++i) {
    mapSchemaAttributeNode(
        *inputNode.childAt(i),
        *storedNode.childAt(i),
        inputAttributes,
        remappedAttributes);
  }
}

SchemaAttributes remapSchemaAttributes(
    const velox::TypePtr& inputType,
    const velox::TypePtr& storedType,
    const std::vector<velox::column_index_t>& storedInputColumnIndices,
    const SchemaAttributes& inputAttributes) {
  const auto input = velox::dwio::common::TypeWithId::create(inputType);
  const auto stored = velox::dwio::common::TypeWithId::create(storedType);
  SchemaAttributes remappedAttributes;

  auto rootAttributes = inputAttributes.find(input->id());
  if (rootAttributes != inputAttributes.end()) {
    remappedAttributes[stored->id()] = rootAttributes->second;
  }
  for (auto storedIndex = 0; storedIndex < storedInputColumnIndices.size();
       ++storedIndex) {
    mapSchemaAttributeNode(
        *input->childAt(storedInputColumnIndices[storedIndex]),
        *stored->childAt(storedIndex),
        inputAttributes,
        remappedAttributes);
  }
  return remappedAttributes;
}

std::unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>
encodingLayouts(const EncodingLayoutTree& tree) {
  std::unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>
      layouts;
  for (const auto identifier : tree.encodingLayoutIdentifiers()) {
    layouts.emplace(identifier, *tree.encodingLayout(identifier));
  }
  return layouts;
}

EncodingLayoutTree remapEncodingLayoutTree(
    const EncodingLayoutTree& tree,
    const std::vector<velox::column_index_t>& storedInputColumnIndices) {
  std::vector<EncodingLayoutTree> children;
  children.reserve(storedInputColumnIndices.size());
  for (const auto inputIndex : storedInputColumnIndices) {
    if (inputIndex >= tree.childrenCount()) {
      break;
    }
    children.emplace_back(tree.child(inputIndex));
  }
  return EncodingLayoutTree{
      tree.schemaKind(),
      encodingLayouts(tree),
      tree.name(),
      std::move(children)};
}

void validateEncodingLayout(const EncodingLayout& layout) {
  NIMBLE_USER_CHECK(
      !isReadOnlyEncoding(layout.encodingType()),
      "Encoding is read-only and cannot be used for new writes: {}",
      layout.encodingType());
  for (uint32_t index = 0; index < layout.childrenCount(); ++index) {
    const auto& child = layout.child(index);
    if (child.has_value()) {
      validateEncodingLayout(child.value());
    }
  }
}

void validateEncodingLayoutTree(const EncodingLayoutTree& tree) {
  for (const auto identifier : tree.encodingLayoutIdentifiers()) {
    validateEncodingLayout(*tree.encodingLayout(identifier));
  }
  for (uint32_t index = 0; index < tree.childrenCount(); ++index) {
    validateEncodingLayoutTree(tree.child(index));
  }
}

WriterOptions storedWriterOptions(
    const velox::TypePtr& inputType,
    const velox::TypePtr& storedType,
    const std::vector<velox::column_index_t>& storedInputColumnIndices,
    WriterOptions options) {
  if (options.encodingLayoutTree.has_value()) {
    validateEncodingLayoutTree(options.encodingLayoutTree.value());
  }
  if (!omitClusterIndexKeyColumnStorage(options)) {
    return options;
  }

  if (!options.schemaAttributes.empty()) {
    // schemaAttributes is keyed by input TypeWithId ids, while field writers
    // are built from storedDataType(). When key columns are omitted, stored
    // TypeWithId ids can differ from input ids, so attributes such as Iceberg
    // field ids must be translated to the matching stored schema nodes.
    options.schemaAttributes = remapSchemaAttributes(
        inputType,
        storedType,
        storedInputColumnIndices,
        options.schemaAttributes);
  }

  if (options.encodingLayoutTree.has_value()) {
    options.encodingLayoutTree.emplace(remapEncodingLayoutTree(
        *options.encodingLayoutTree, storedInputColumnIndices));
  }

  const bool hasFeatureReordering = options.featureReordering.has_value();
  if (!hasFeatureReordering) {
    return options;
  }

  // Field writers and the layout planner are built from storedDataType().
  // When key columns are omitted, stored column ordinals can differ from input
  // ordinals, so ordinal-based feature reordering configs must be translated to
  // stored ordinals before they reach the writer internals.
  folly::F14FastMap<velox::column_index_t, size_t> storedIndexByInputIndex;
  storedIndexByInputIndex.reserve(storedInputColumnIndices.size());
  for (auto storedIndex = 0; storedIndex < storedInputColumnIndices.size();
       ++storedIndex) {
    storedIndexByInputIndex.emplace(
        storedInputColumnIndices[storedIndex], storedIndex);
  }

  std::vector<std::tuple<size_t, std::vector<int64_t>>> remapped;
  remapped.reserve(options.featureReordering->size());
  for (auto& [originalIndex, keys] : *options.featureReordering) {
    auto it = storedIndexByInputIndex.find(originalIndex);
    NIMBLE_USER_CHECK(
        it != storedIndexByInputIndex.end(),
        "Feature reordering cannot target hidden cluster index key column at index {}",
        originalIndex);
    remapped.emplace_back(it->second, std::move(keys));
  }
  options.featureReordering = std::move(remapped);

  return options;
}

void writeIndexSection(
    const std::vector<index::IndexDescriptor>& descriptors,
    const WriteOptionalSectionFn& writeMetadataFn) {
  if (descriptors.empty()) {
    return;
  }

  flatbuffers::FlatBufferBuilder builder(kInitialFooterSize);
  auto indexes =
      builder.CreateVector<flatbuffers::Offset<serialization::IndexDescriptor>>(
          descriptors.size(), [&builder, &descriptors](size_t i) {
            const auto& descriptor = descriptors[i];
            auto name = builder.CreateString(descriptor.name);
            auto root = serialization::CreateMetadataSection(
                builder,
                descriptor.root.offset(),
                descriptor.root.size(),
                static_cast<serialization::CompressionType>(
                    descriptor.root.compressionType()),
                descriptor.root.uncompressedSize().value_or(
                    descriptor.root.size()));
            return serialization::CreateIndexDescriptor(
                builder, index::toIndexFamily(descriptor.family), name, root);
          });
  builder.Finish(serialization::CreateIndexRoot(builder, indexes));
  writeMetadataFn(std::string(kIndexSection), asView(builder));
}

velox::RuntimeMetric toRuntimeMetric(const std::vector<uint64_t>& values) {
  velox::RuntimeMetric metric;
  for (auto value : values) {
    metric.addValue(value);
  }
  return metric;
}

constexpr uint32_t kInitialSchemaSectionSize = 1 << 20; // 1MB

// When writing null streams, we write the nulls as data, and the stream itself
// is non-nullable. This adapter class is how we expose the nulls as values.
class NullsAsDataStreamData : public StreamData {
 public:
  explicit NullsAsDataStreamData(StreamData& streamData)
      : StreamData(streamData.descriptor()), streamData_{&streamData} {
    streamData_->materialize();
  }

  inline uint32_t rowCount() const override {
    return streamData_->rowCount();
  }

  inline std::string_view data() const override {
    return {
        reinterpret_cast<const char*>(streamData_->nonNulls().data()),
        streamData_->nonNulls().size()};
  }

  inline std::span<const bool> nonNulls() const override {
    return {};
  }

  inline bool hasNulls() const override {
    return false;
  }

  inline bool empty() const override {
    return streamData_->empty();
  }

  inline uint64_t memoryUsed() const override {
    return streamData_->memoryUsed();
  }

  inline void reset() override {
    streamData_->reset();
  }

 private:
  StreamData* const streamData_;
};

class WriterStreamContext : public StreamContext {
 public:
  bool isNullStream() const {
    return isNullStream_;
  }

  void setIsNullStream(bool value) {
    isNullStream_ = value;
  }

  bool isInMapStream() const {
    return isInMapStream_;
  }

  void setIsInMapStream(bool value) {
    isInMapStream_ = value;
  }

  void setFlatMapValueStreamOffsets(std::vector<offset_size> offsets) {
    flatMapValueStreamOffsets_ = std::move(offsets);
  }

  // The layout to replay for this stream: overlaid from the EncodingLayoutTree
  // at setup, or captured from this stream's first encode when
  // encoding-selection caching is enabled. Empty until one of those populates
  // it.
  const EncodingLayout* encoding() const {
    return encoding_.has_value() ? &*encoding_ : nullptr;
  }

  void setEncoding(EncodingLayout value) {
    encoding_.emplace(std::move(value));
  }

  // Stores the shared dictionary configuration selected for this value stream.
  void setSharedDictionaryConfig(const SharedDictionaryConfig& config) {
    sharedDictionaryConfig_ = config;
  }

  // Returns the shared dictionary configuration, if this stream is configured
  // to use one.
  const std::optional<SharedDictionaryConfig>& sharedDictionaryConfig() const {
    return sharedDictionaryConfig_;
  }

  SharedDictionaryWriter* sharedDictionaryWriter() const {
    return sharedDictionaryWriter_.get();
  }

  void setSharedDictionaryWriter(
      std::unique_ptr<SharedDictionaryWriter> writer) const {
    NIMBLE_CHECK_NULL(
        sharedDictionaryWriter_, "Shared dictionary writer already exists.");
    sharedDictionaryWriter_ = std::move(writer);
  }

 private:
  bool isNullStream_{false};
  bool isInMapStream_{false};
  // Value stream descriptor offsets for this in-map stream's flat-map field.
  // Empty for non in-map streams and flat-map values with no reader-visible
  // value stream.
  std::vector<offset_size> flatMapValueStreamOffsets_;
  std::optional<EncodingLayout> encoding_;
  std::optional<SharedDictionaryConfig> sharedDictionaryConfig_;
  mutable std::unique_ptr<SharedDictionaryWriter> sharedDictionaryWriter_;
};

// Context attached to one FlatMap TypeBuilder node. It carries the per-key
// encoding layouts and value dictionary configs that are applied later when the
// writer materializes children for specific flat-map keys.
class FlatmapEncodingLayoutContext : public TypeBuilderContext {
 public:
  using KeyEncodingMap =
      folly::F14FastMap<std::string_view, const EncodingLayoutTree*>;

  struct ValueDictionaryConfig {
    // Relative subfield below the materialized flat-map key value. Empty
    // selects the key value itself.
    std::string valueSubfield;
    SharedDictionaryConfig dictionary;
  };

  // Keyed by the materialized flat-map key name, e.g. "10" or "feature_a".
  using ValueDictionaryMap =
      folly::F14FastMap<std::string, std::vector<ValueDictionaryConfig>>;

  explicit FlatmapEncodingLayoutContext(
      KeyEncodingMap keyEncodings = {},
      ValueDictionaryMap valueDictionaries = {})
      : keyEncodings{std::move(keyEncodings)},
        valueDictionaries{std::move(valueDictionaries)} {}

  const KeyEncodingMap keyEncodings;
  const ValueDictionaryMap valueDictionaries;
};

WriterStreamContext& streamContext(const StreamDescriptorBuilder& descriptor);

Encoding::Options makeEncodingOptions(
    const WriterOptions& writerOptions,
    velox::BufferPool* encodingScratchBufferPool,
    EncodingBufferPool* encodingBufferPool) {
  // WriterOptions carries file-format and selection settings; the scratch
  // pools are per encode call and must be threaded into nested encoders.
  auto encodingOptions = writerOptions.buildEncodingOptions();
  encodingOptions.bufferPool = encodingScratchBufferPool;
  encodingOptions.encodingBufferPool = encodingBufferPool;
  return encodingOptions;
}

bool hasDictionaryConfig(const StreamData& streamData) {
  const auto* streamContext =
      streamData.descriptor().context<WriterStreamContext>();
  return streamContext != nullptr &&
      streamContext->sharedDictionaryConfig().has_value();
}

bool isSharedDictionaryValueType(DataType dataType) {
  switch (dataType) {
    case DataType::Int8:
    case DataType::Uint8:
    case DataType::Int16:
    case DataType::Uint16:
    case DataType::Int32:
    case DataType::Uint32:
    case DataType::Int64:
    case DataType::Uint64:
      return true;
    case DataType::Undefined:
    case DataType::Float:
    case DataType::Double:
    case DataType::Bool:
    case DataType::String:
      return false;
  }
  NIMBLE_UNREACHABLE("Unsupported data type {}.", dataType);
}

template <typename T>
// NOLINTNEXTLINE(facebook-hte-NullableReturn)
TypedSharedDictionaryWriter<T>* sharedDictionaryWriter(
    const StreamData& streamData,
    detail::WriterContext& context,
    Buffer& buffer,
    velox::BufferPool* encodingScratchBufferPool,
    EncodingBufferPool* encodingBufferPool) {
  auto* streamContext = streamData.descriptor().context<WriterStreamContext>();
  if (streamContext == nullptr) {
    return nullptr;
  }

  // Chunked writes revisit the same value stream, so the writer is cached in
  // the stream context. The type check guards against reusing that state with
  // a different physical value type.
  if (auto* dictionaryWriter = streamContext->sharedDictionaryWriter()) {
    NIMBLE_CHECK_EQ(
        dictionaryWriter->dataType(),
        TypeTraits<T>::dataType,
        "Shared dictionary writer has unexpected value type.");
    return velox::checkedPointerCast<TypedSharedDictionaryWriter<T>>(
        dictionaryWriter);
  }

  const auto& config = streamContext->sharedDictionaryConfig();
  if (!config.has_value()) {
    return nullptr;
  }
  const auto& writerOptions = context.options();
  auto encodingOptions = makeEncodingOptions(
      writerOptions, encodingScratchBufferPool, encodingBufferPool);
  SharedDictionaryWriter::Options dictionaryOptions{
      .scope = config->scope,
      .dictionaryId = config->dictionaryId,
      .useExternalAlphabet = config->useExternalAlphabet,
      .alphabetEncodings = config->alphabetEncodings,
      .encodingSelectionPolicyCreator =
          writerOptions.encodingSelectionPolicyCreator,
      .encodingOptions = std::move(encodingOptions),
      .resolver =
          writerOptions.experimentalSharedDictionaryEncoding.externalResolver,
  };
  auto writer = std::make_unique<TypedSharedDictionaryWriter<T>>(
      &buffer.getMemoryPool(), dictionaryOptions);
  auto* writerPtr = writer.get();
  streamContext->setSharedDictionaryWriter(std::move(writer));
  return writerPtr;
}

template <typename T>
std::unique_ptr<EncodingSelectionPolicy<T>> makeEncodingPolicy(
    bool hasEncodingLayout,
    std::optional<EncodingLayout> encodingLayout,
    detail::WriterContext& context,
    Buffer& buffer,
    velox::BufferPool* encodingScratchBufferPool,
    EncodingBufferPool* encodingBufferPool,
    const StreamData& streamData) {
  if (hasDictionaryConfig(streamData)) {
    NIMBLE_USER_CHECK(
        isSharedDictionaryValueType(TypeTraits<T>::dataType),
        "Shared dictionary encoding only supports non-bool integer streams, "
        "got {}.",
        TypeTraits<T>::dataType);
    if constexpr (isIntegralType<T>() && !std::is_same_v<T, bool>) {
      auto* dictionaryWriter = sharedDictionaryWriter<T>(
          streamData,
          context,
          buffer,
          encodingScratchBufferPool,
          encodingBufferPool);
      NIMBLE_CHECK_NOT_NULL(dictionaryWriter);
      return dictionaryWriter->createEncodingPolicy(context.getStripeIndex());
    } else {
      NIMBLE_UNREACHABLE(
          "Shared dictionary type validation accepted unsupported {}.",
          TypeTraits<T>::dataType);
    }
  }
  if (hasEncodingLayout) {
    return std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
        std::move(encodingLayout.value()),
        context.options().compressionOptions,
        context.options().encodingSelectionPolicyCreator);
  }
  return std::unique_ptr<EncodingSelectionPolicy<T>>(
      static_cast<EncodingSelectionPolicy<T>*>(
          context.options()
              .encodingSelectionPolicyCreator(TypeTraits<T>::dataType)
              .release()));
}

void configureDictionary(
    const TypeBuilder& typeBuilder,
    SharedDictionaryConfig config,
    SchemaBuilder& schemaBuilder) {
  switch (typeBuilder.kind()) {
    case Kind::Scalar: {
      const auto& scalar = typeBuilder.asScalar();
      NIMBLE_USER_CHECK(
          isIntegerScalarKind(scalar.scalarDescriptor().scalarKind()),
          "Shared dictionary value must be an integer scalar, got {}.",
          scalar.scalarDescriptor().scalarKind());
      if (config.scope == SharedDictionaryScope::Stripe) {
        NIMBLE_USER_CHECK_EQ(
            config.dictionaryId,
            0,
            "Stripe shared dictionary config must leave dictionaryId unset.");
        config.dictionaryId = schemaBuilder.createSharedDictionaryStream(
            scalar.scalarDescriptor().offset());
      }
      streamContext(scalar.scalarDescriptor())
          .setSharedDictionaryConfig(config);
      return;
    }
    case Kind::Array:
      configureDictionary(
          typeBuilder.asArray().elements(), std::move(config), schemaBuilder);
      return;
    case Kind::ArrayWithOffsets:
      configureDictionary(
          typeBuilder.asArrayWithOffsets().elements(),
          std::move(config),
          schemaBuilder);
      return;
    case Kind::Map:
      configureDictionary(
          typeBuilder.asMap().values(), std::move(config), schemaBuilder);
      return;
    case Kind::SlidingWindowMap:
      configureDictionary(
          typeBuilder.asSlidingWindowMap().values(),
          std::move(config),
          schemaBuilder);
      return;
    case Kind::TimestampMicroNano:
    case Kind::Row:
    case Kind::FlatMap:
      NIMBLE_USER_FAIL(
          "Shared dictionary value must resolve to an integer scalar, array "
          "element, or map value, got {}.",
          typeBuilder.kind());
  }
}

// Resolves [*] to the selected array element or map value type.
const TypeBuilder& allSubscriptValueType(
    const TypeBuilder& typeBuilder,
    std::string_view subfield) {
  switch (typeBuilder.kind()) {
    case Kind::Array:
      return typeBuilder.asArray().elements();
    case Kind::ArrayWithOffsets:
      return typeBuilder.asArrayWithOffsets().elements();
    case Kind::Map:
      return typeBuilder.asMap().values();
    case Kind::SlidingWindowMap:
      return typeBuilder.asSlidingWindowMap().values();
    case Kind::Scalar:
    case Kind::TimestampMicroNano:
    case Kind::Row:
    case Kind::FlatMap:
      NIMBLE_USER_FAIL(
          "Shared dictionary value subfield '{}' cannot apply [*] to {}.",
          subfield,
          typeBuilder.kind());
  }
  NIMBLE_UNREACHABLE("Unknown schema kind: {}.", typeBuilder.kind());
}

const TypeBuilder& resolveFieldPath(
    const TypeBuilder& root,
    const std::string& fieldPath) {
  if (fieldPath.empty()) {
    return root;
  }
  const velox::common::Subfield subfield{fieldPath};
  const auto& path = subfield.path();
  const TypeBuilder* current = &root;
  for (const auto& pathElement : path) {
    if (pathElement->is(velox::common::SubfieldKind::kAllSubscripts)) {
      current = &allSubscriptValueType(*current, fieldPath);
      continue;
    }
    const auto& childName =
        pathElement->asChecked<velox::common::Subfield::NestedField>()->name();
    current = &current->asRow().findChild(childName);
  }
  return *current;
}

void maybeAddFileDictionaryId(
    const SharedDictionaryConfig& config,
    folly::F14FastSet<uint32_t>& fileDictionaryIds) {
  if (config.scope != SharedDictionaryScope::File) {
    return;
  }
  const bool inserted = fileDictionaryIds.insert(config.dictionaryId).second;
  NIMBLE_USER_CHECK(
      inserted,
      "File shared dictionary ID {} is configured for multiple streams. "
      "Cross-stream domains are not supported yet.",
      config.dictionaryId);
}

// Resolves [*] to the selected array element or map value type.
const TypeWithId& allSubscriptValueType(
    const TypeWithId& type,
    const std::string& fieldPath) {
  switch (type.type()->kind()) {
    case velox::TypeKind::ARRAY:
      return *type.childAt(0);
    case velox::TypeKind::MAP:
      return *type.childAt(1);
    case velox::TypeKind::BOOLEAN:
    case velox::TypeKind::TINYINT:
    case velox::TypeKind::SMALLINT:
    case velox::TypeKind::INTEGER:
    case velox::TypeKind::BIGINT:
    case velox::TypeKind::REAL:
    case velox::TypeKind::DOUBLE:
    case velox::TypeKind::VARCHAR:
    case velox::TypeKind::VARBINARY:
    case velox::TypeKind::TIMESTAMP:
    case velox::TypeKind::HUGEINT:
    case velox::TypeKind::ROW:
    case velox::TypeKind::UNKNOWN:
    case velox::TypeKind::FUNCTION:
    case velox::TypeKind::OPAQUE:
    case velox::TypeKind::INVALID:
      NIMBLE_USER_FAIL(
          "Shared dictionary path '{}' cannot apply [*] to {}.",
          fieldPath,
          type.type()->toString());
  }
  VELOX_UNREACHABLE();
}

const TypeWithId& resolveFieldPath(
    const TypeWithId& root,
    const std::string& fieldPath) {
  const velox::common::Subfield subfield{fieldPath};
  const TypeWithId* current = &root;
  for (const auto& pathElement : subfield.path()) {
    if (pathElement->is(velox::common::SubfieldKind::kAllSubscripts)) {
      current = &allSubscriptValueType(*current, fieldPath);
      continue;
    }
    const auto& childName =
        pathElement->asChecked<velox::common::Subfield::NestedField>()->name();
    current = current->childByName(childName).get();
  }
  return *current;
}

const TypeWithId& resolveDictionaryValueType(
    const TypeWithId& type,
    const std::string& fieldPath) {
  switch (type.type()->kind()) {
    case velox::TypeKind::ARRAY:
      return resolveDictionaryValueType(*type.childAt(0), fieldPath);
    case velox::TypeKind::MAP:
      return resolveDictionaryValueType(*type.childAt(1), fieldPath);
    case velox::TypeKind::TINYINT:
    case velox::TypeKind::SMALLINT:
    case velox::TypeKind::INTEGER:
    case velox::TypeKind::BIGINT:
      return type;
    case velox::TypeKind::BOOLEAN:
    case velox::TypeKind::REAL:
    case velox::TypeKind::DOUBLE:
    case velox::TypeKind::VARCHAR:
    case velox::TypeKind::VARBINARY:
    case velox::TypeKind::TIMESTAMP:
    case velox::TypeKind::HUGEINT:
    case velox::TypeKind::ROW:
    case velox::TypeKind::UNKNOWN:
    case velox::TypeKind::FUNCTION:
    case velox::TypeKind::OPAQUE:
    case velox::TypeKind::INVALID:
      NIMBLE_USER_FAIL(
          "Shared dictionary column '{}' must resolve to an integer scalar, "
          "array element, or map value, got {}.",
          fieldPath,
          type.type()->toString());
  }
  VELOX_UNREACHABLE();
}

template <typename T>
std::string_view encode(
    std::optional<EncodingLayout> encodingLayout,
    detail::WriterContext& context,
    Buffer& buffer,
    velox::BufferPool* encodingScratchBufferPool,
    EncodingBufferPool* encodingBufferPool,
    const StreamData& streamData) {
  NIMBLE_DCHECK_EQ(
      streamData.data().size() % sizeof(T),
      0,
      "Unexpected size {}",
      streamData.data().size());
  std::span<const T> data{
      reinterpret_cast<const T*>(streamData.data().data()),
      streamData.data().size() / sizeof(T)};

  // True when replaying a saved layout (an external EncodingLayoutTree layout
  // or one cached from a previous encode), false when running a fresh encoding
  // selection. Exposed as a test-injection point so tests can count how often
  // full selection actually runs (once per stream with the cache on, once per
  // chunk with it off) and force the replay to fail to exercise the fallback.
  const bool hasEncodingLayout = encodingLayout.has_value();
  velox::common::testutil::TestValue::adjust(
      "facebook::nimble::encode", const_cast<bool*>(&hasEncodingLayout));

  auto policy = makeEncodingPolicy<T>(
      hasEncodingLayout,
      std::move(encodingLayout),
      context,
      buffer,
      encodingScratchBufferPool,
      encodingBufferPool,
      streamData);

  auto encodingOptions = makeEncodingOptions(
      context.options(), encodingScratchBufferPool, encodingBufferPool);
  velox::common::testutil::TestValue::adjust(
      "facebook::nimble::Writer::encode", &encodingOptions);

  if (streamData.hasNulls()) {
    std::span<const bool> notNulls = streamData.nonNulls();
    return EncodingFactory::encodeNullable(
        std::move(policy), data, notNulls, buffer, encodingOptions);
  } else {
    return EncodingFactory::encode(
        std::move(policy), data, buffer, encodingOptions);
  }
}

// Encodes streamData by replaying a saved layout -- either an external
// EncodingLayoutTree layout or one cached from this stream's first encode. Only
// entered once the stream has a saved layout to replay (checked below). Replay
// is best-effort: if it throws for any reason (e.g. the layout no longer fits
// this chunk's data), retry once with a fresh selection, letting any failure of
// that retry propagate.
template <typename T>
std::string_view encodeWithFallback(
    const EncodingLayout* encodingLayout,
    detail::WriterContext& context,
    Buffer& buffer,
    velox::BufferPool* encodingScratchBufferPool,
    EncodingBufferPool* encodingBufferPool,
    const StreamData& streamData) {
  NIMBLE_CHECK_NOT_NULL(
      encodingLayout,
      "encodeWithFallback requires a saved encoding layout to replay.");
  try {
    return encode<T>(
        *encodingLayout,
        context,
        buffer,
        encodingScratchBufferPool,
        encodingBufferPool,
        streamData);
  } catch (const std::exception&) {
    // A saved layout can fail to apply to this chunk's data in ways beyond a
    // clean IncompatibleEncoding, so retry on any error rather than keying
    // off a specific (unreliable) error code.
    return encode<T>(
        std::nullopt,
        context,
        buffer,
        encodingScratchBufferPool,
        encodingBufferPool,
        streamData);
  }
}

template <typename T>
std::string_view encodeStreamTyped(
    detail::WriterContext& context,
    Buffer& buffer,
    velox::BufferPool* encodingScratchBufferPool,
    EncodingBufferPool* encodingBufferPool,
    const StreamData& streamData) {
  const auto* writerStreamContext =
      streamData.descriptor().context<WriterStreamContext>();
  const bool hasDictionary = hasDictionaryConfig(streamData);

  // Replay an externally provided (EncodingLayoutTree) or previously captured
  // layout, falling back to a fresh selection if it no longer fits the data.
  // TODO: Replace the exception-based best-effort replay in encodeWithFallback
  // with a non-throwing compatibility check before the replay attempt.
  if (!hasDictionary && writerStreamContext &&
      writerStreamContext->encoding()) {
    return encodeWithFallback<T>(
        writerStreamContext->encoding(),
        context,
        buffer,
        encodingScratchBufferPool,
        encodingBufferPool,
        streamData);
  }

  // No layout to replay: run a fresh selection.
  auto encoded = encode<T>(
      std::nullopt,
      context,
      buffer,
      encodingScratchBufferPool,
      encodingBufferPool,
      streamData);

  // Cache the data layout from this first encode so later chunks/stripes replay
  // it, skipping the full selection cascade. EncodingLayoutCapture::capture()
  // already strips any Nullable/Sentinel wrapper, so the cached layout is the
  // data encoding alone — Nimble re-applies per-chunk nullability at encode
  // time, so it stays valid regardless of a later chunk's nulls.
  if (!hasDictionary && context.options().enableEncodingSelectionCache) {
    streamContext(streamData.descriptor())
        .setEncoding(
            EncodingLayoutCapture::capture(
                encoded, context.options().buildEncodingOptions()));
  }
  return encoded;
}

std::string_view encodeStreamData(
    detail::WriterContext& context,
    Buffer& buffer,
    velox::BufferPool* encodingScratchBufferPool,
    EncodingBufferPool* encodingBufferPool,
    const StreamData& streamData) {
  auto scalarKind = streamData.descriptor().scalarKind();
  switch (scalarKind) {
    case ScalarKind::Bool:
      return encodeStreamTyped<bool>(
          context,
          buffer,
          encodingScratchBufferPool,
          encodingBufferPool,
          streamData);
    case ScalarKind::Int8:
      return encodeStreamTyped<int8_t>(
          context,
          buffer,
          encodingScratchBufferPool,
          encodingBufferPool,
          streamData);
    case ScalarKind::Int16:
      return encodeStreamTyped<int16_t>(
          context,
          buffer,
          encodingScratchBufferPool,
          encodingBufferPool,
          streamData);
    case ScalarKind::UInt16:
      return encodeStreamTyped<uint16_t>(
          context,
          buffer,
          encodingScratchBufferPool,
          encodingBufferPool,
          streamData);
    case ScalarKind::Int32:
      return encodeStreamTyped<int32_t>(
          context,
          buffer,
          encodingScratchBufferPool,
          encodingBufferPool,
          streamData);
    case ScalarKind::UInt32:
      return encodeStreamTyped<uint32_t>(
          context,
          buffer,
          encodingScratchBufferPool,
          encodingBufferPool,
          streamData);
    case ScalarKind::Int64:
      return encodeStreamTyped<int64_t>(
          context,
          buffer,
          encodingScratchBufferPool,
          encodingBufferPool,
          streamData);
    case ScalarKind::Float:
      return encodeStreamTyped<float>(
          context,
          buffer,
          encodingScratchBufferPool,
          encodingBufferPool,
          streamData);
    case ScalarKind::Double:
      return encodeStreamTyped<double>(
          context,
          buffer,
          encodingScratchBufferPool,
          encodingBufferPool,
          streamData);
    case ScalarKind::String:
    case ScalarKind::Binary:
      return encodeStreamTyped<std::string_view>(
          context,
          buffer,
          encodingScratchBufferPool,
          encodingBufferPool,
          streamData);
    default:
      NIMBLE_UNREACHABLE("Unsupported scalar kind {}", toString(scalarKind));
  }
}

template <typename Set>
void findNodeIds(
    const velox::dwio::common::TypeWithId& typeWithId,
    Set& output,
    const std::function<bool(const velox::dwio::common::TypeWithId&)>&
        predicate) {
  if (predicate(typeWithId)) {
    output.insert(typeWithId.id());
  }

  for (const auto& child : typeWithId.getChildren()) {
    findNodeIds(*child, output, predicate);
  }
}

void collectFlatMapValueStreamOffsets(
    const TypeBuilder& type,
    std::vector<offset_size>& offsets) {
  switch (type.kind()) {
    case Kind::Scalar:
      offsets.push_back(type.asScalar().scalarDescriptor().offset());
      return;
    case Kind::TimestampMicroNano:
      offsets.push_back(
          type.asTimestampMicroNano().microsDescriptor().offset());
      offsets.push_back(type.asTimestampMicroNano().nanosDescriptor().offset());
      return;
    case Kind::Array:
      offsets.push_back(type.asArray().lengthsDescriptor().offset());
      collectFlatMapValueStreamOffsets(type.asArray().elements(), offsets);
      return;
    case Kind::ArrayWithOffsets:
      offsets.push_back(type.asArrayWithOffsets().offsetsDescriptor().offset());
      offsets.push_back(type.asArrayWithOffsets().lengthsDescriptor().offset());
      collectFlatMapValueStreamOffsets(
          type.asArrayWithOffsets().elements(), offsets);
      return;
    case Kind::Map:
      offsets.push_back(type.asMap().lengthsDescriptor().offset());
      collectFlatMapValueStreamOffsets(type.asMap().keys(), offsets);
      collectFlatMapValueStreamOffsets(type.asMap().values(), offsets);
      return;
    case Kind::SlidingWindowMap:
      offsets.push_back(type.asSlidingWindowMap().offsetsDescriptor().offset());
      offsets.push_back(type.asSlidingWindowMap().lengthsDescriptor().offset());
      collectFlatMapValueStreamOffsets(
          type.asSlidingWindowMap().keys(), offsets);
      collectFlatMapValueStreamOffsets(
          type.asSlidingWindowMap().values(), offsets);
      return;
    case Kind::Row: {
      const auto& row = type.asRow();
      offsets.push_back(row.nullsDescriptor().offset());
      for (size_t i = 0; i < row.childrenCount(); ++i) {
        collectFlatMapValueStreamOffsets(row.childAt(i), offsets);
      }
      return;
    }
    case Kind::FlatMap: {
      const auto& flatMap = type.asFlatMap();
      offsets.push_back(flatMap.nullsDescriptor().offset());
      for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
        offsets.push_back(flatMap.inMapDescriptorAt(i).offset());
        collectFlatMapValueStreamOffsets(flatMap.childAt(i), offsets);
      }
      return;
    }
    default:
      NIMBLE_UNREACHABLE("Unsupported type kind {}", type.kind());
  }
}

FlatmapEncodingLayoutContext::KeyEncodingMap keyEncodingsForFlatMap(
    const EncodingLayoutTree& encodingLayoutTree) {
  FlatmapEncodingLayoutContext::KeyEncodingMap keyEncodings;
  keyEncodings.reserve(encodingLayoutTree.childrenCount());
  for (auto i = 0; i < encodingLayoutTree.childrenCount(); ++i) {
    const auto& child = encodingLayoutTree.child(i);
    keyEncodings.emplace(child.name(), &child);
  }
  return keyEncodings;
}

folly::F14FastMap<uint32_t, FlatmapEncodingLayoutContext::KeyEncodingMap>
collectFlatMapKeyEncodings(
    const TypeWithId& type,
    const detail::WriterContext& context) {
  folly::F14FastMap<uint32_t, FlatmapEncodingLayoutContext::KeyEncodingMap>
      flatMapKeyEncodings;
  if (!context.options().encodingLayoutTree.has_value() ||
      context.options().flatMapColumns.empty()) {
    return flatMapKeyEncodings;
  }

  const auto& encodingLayoutTree = context.options().encodingLayoutTree.value();
  NIMBLE_CHECK_EQ(
      encodingLayoutTree.schemaKind(),
      Kind::Row,
      "Incompatible encoding layout node. Expecting row node.");
  NIMBLE_CHECK_EQ(
      type.type()->kind(),
      velox::TypeKind::ROW,
      "Flat-map encoding layout collection requires row root type, got {}.",
      type.type()->toString());
  const auto& rowType = type.type()->as<velox::TypeKind::ROW>();
  for (const auto& [columnName, _] : context.options().flatMapColumns) {
    const auto child = type.childByName(columnName);
    for (uint32_t i = 0;
         i < rowType.size() && i < encodingLayoutTree.childrenCount();
         ++i) {
      if (rowType.nameOf(i) != columnName) {
        continue;
      }
      const auto& childLayout = encodingLayoutTree.child(i);
      if (childLayout.schemaKind() == Kind::Map) {
        // Schema evolution - If a map is converted to flatmap, we should not
        // fail, but also not try to replay captured encodings.
        break;
      }
      NIMBLE_CHECK_EQ(
          childLayout.schemaKind(),
          Kind::FlatMap,
          "Incompatible encoding layout node. Expecting flatmap node.");
      const auto nodeId = static_cast<uint32_t>(child->id());
      flatMapKeyEncodings.emplace(nodeId, keyEncodingsForFlatMap(childLayout));
      break;
    }
  }
  return flatMapKeyEncodings;
}

struct DictionaryConfigs {
  folly::F14FastMap<uint32_t, SharedDictionaryConfig> columns;
  folly::F14FastMap<uint32_t, FlatmapEncodingLayoutContext::ValueDictionaryMap>
      flatMaps;
};

// Resolves shared dictionary configs to writer schema targets. Flat-map value
// stream assignment is deferred until each configured key is materialized.
DictionaryConfigs collectDictionaryConfigs(
    const TypeWithId& type,
    const detail::WriterContext& context) {
  DictionaryConfigs configs;
  const auto& dictionaryEncodingConfig =
      context.options().experimentalSharedDictionaryEncoding;
  if (dictionaryEncodingConfig.empty()) {
    return configs;
  }

  NIMBLE_USER_CHECK_EQ(
      type.type()->kind(),
      velox::TypeKind::ROW,
      "Shared dictionary encoding requires a row root type, got {}.",
      type.type()->toString());
  folly::F14FastSet<uint32_t> fileDictionaryIds;
  folly::F14FastSet<uint32_t> columnValueNodeIds;
  configs.columns.reserve(dictionaryEncodingConfig.columns.size());
  for (const auto& columnDictionary : dictionaryEncodingConfig.columns) {
    const auto& fieldType = resolveFieldPath(type, columnDictionary.fieldPath);
    const auto& valueType =
        resolveDictionaryValueType(fieldType, columnDictionary.fieldPath);
    const auto valueNodeId = static_cast<uint32_t>(valueType.id());
    const auto insertedValueNode =
        columnValueNodeIds.insert(valueNodeId).second;
    NIMBLE_USER_CHECK(
        insertedValueNode,
        "Duplicate shared dictionary column configuration for schema value "
        "node {}.",
        valueNodeId);
    NIMBLE_USER_CHECK(
        valueType.type()->isInteger(),
        "Shared dictionary column '{}' must resolve to an integer scalar, "
        "array element, or map value, got {}.",
        columnDictionary.fieldPath,
        valueType.type()->toString());
    maybeAddFileDictionaryId(columnDictionary.dictionary, fileDictionaryIds);
    const auto nodeId = static_cast<uint32_t>(fieldType.id());
    const auto inserted =
        configs.columns.emplace(nodeId, columnDictionary.dictionary).second;
    NIMBLE_USER_CHECK(
        inserted,
        "Duplicate shared dictionary column configuration for schema node {}.",
        nodeId);
  }

  configs.flatMaps.reserve(dictionaryEncodingConfig.flatMaps.size());
  for (const auto& flatMap : dictionaryEncodingConfig.flatMaps) {
    const velox::common::Subfield subfield{flatMap.fieldPath};
    NIMBLE_USER_CHECK_EQ(
        subfield.path().size(),
        1,
        "Shared dictionary flat-map path '{}' must be a top-level writer input "
        "column.",
        flatMap.fieldPath);
    const auto& fieldType = resolveFieldPath(type, flatMap.fieldPath);
    NIMBLE_USER_CHECK_EQ(
        fieldType.type()->kind(),
        velox::TypeKind::MAP,
        "Flat-map dictionary column '{}' must be a map, got {}.",
        flatMap.fieldPath,
        fieldType.type()->toString());
    NIMBLE_USER_CHECK(
        context.hasFlatMapNodeId(fieldType.id()),
        "Flat-map dictionary column '{}' is not configured as a flat map.",
        flatMap.fieldPath);

    const auto nodeId = static_cast<uint32_t>(fieldType.id());
    auto& flatMapKeys = configs.flatMaps[nodeId];
    for (const auto& flatMapKey : flatMap.keys) {
      const auto& config = flatMapKey.dictionary;
      maybeAddFileDictionaryId(config, fileDictionaryIds);
      flatMapKeys[std::to_string(flatMapKey.key)].push_back(
          FlatmapEncodingLayoutContext::ValueDictionaryConfig{
              .valueSubfield = flatMapKey.valueSubfield, .dictionary = config});
    }
  }
  return configs;
}

WriterStreamContext& streamContext(const StreamDescriptorBuilder& descriptor) {
  auto* context = descriptor.context<WriterStreamContext>();
  if (context != nullptr) {
    return *context;
  }
  descriptor.setContext(std::make_unique<WriterStreamContext>());
  return *descriptor.context<WriterStreamContext>();
}

// Applies writer context that depends on the TypeBuilder node just created.
// FlatMap value configs live on the parent FlatMap node so the key-add handler
// can apply them when each keyed value stream is materialized.
void configureAddedType(
    detail::WriterContext& context,
    TypeBuilder& type,
    uint32_t nodeId,
    const DictionaryConfigs& dictionaryConfigs,
    const folly::
        F14FastMap<uint32_t, FlatmapEncodingLayoutContext::KeyEncodingMap>&
            flatMapKeyEncodingLayouts) {
  if (type.kind() == Kind::Row) {
    streamContext(type.asRow().nullsDescriptor()).setIsNullStream(true);
  } else if (type.kind() == Kind::FlatMap) {
    streamContext(type.asFlatMap().nullsDescriptor()).setIsNullStream(true);
    FlatmapEncodingLayoutContext::KeyEncodingMap keyEncodings;
    auto keyEncodingIt = flatMapKeyEncodingLayouts.find(nodeId);
    if (keyEncodingIt != flatMapKeyEncodingLayouts.end()) {
      keyEncodings = keyEncodingIt->second;
    }
    FlatmapEncodingLayoutContext::ValueDictionaryMap valueDictionaries;
    auto dictionaryIt = dictionaryConfigs.flatMaps.find(nodeId);
    if (dictionaryIt != dictionaryConfigs.flatMaps.end()) {
      valueDictionaries = dictionaryIt->second;
    }
    if (!keyEncodings.empty() || !valueDictionaries.empty()) {
      type.setContext(
          std::make_unique<FlatmapEncodingLayoutContext>(
              std::move(keyEncodings), std::move(valueDictionaries)));
    }
  }

  auto columnDictionaryIt = dictionaryConfigs.columns.find(nodeId);
  if (columnDictionaryIt != dictionaryConfigs.columns.end()) {
    configureDictionary(
        type, columnDictionaryIt->second, context.schemaBuilder());
  }

  const auto& schemaAttributes = context.options().schemaAttributes;
  auto attribute = schemaAttributes.find(nodeId);
  if (attribute != schemaAttributes.end()) {
    type.setAttributes(attribute->second);
  }
}

std::unique_ptr<FieldWriter> createRootFieldWriter(
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& type,
    detail::WriterContext& context) {
  if (!context.options().flatMapColumns.empty()) {
    context.reserveFlatMapNodes(context.options().flatMapColumns.size());
    for (const auto& [columnName, keys] : context.options().flatMapColumns) {
      auto nodeId = type->childByName(columnName)->id();
      context.addFlatMapNodeId(nodeId, keys);
    }
  }

  if (!context.options().dictionaryArrayColumns.empty()) {
    context.clearAndReserveDictionaryArrayNodeIds(
        context.options().dictionaryArrayColumns.size());
    for (const auto& column : context.options().dictionaryArrayColumns) {
      findNodeIds(
          *type->childByName(column),
          context.dictionaryArrayNodeIds(),
          [](const velox::dwio::common::TypeWithId& type) {
            return type.type()->kind() == velox::TypeKind::ARRAY;
          });
    }
  }

  if (!context.options().deduplicatedMapColumns.empty()) {
    context.clearAndReserveDeduplicatedMapNodeIds(
        context.options().deduplicatedMapColumns.size());
    for (const auto& column : context.options().deduplicatedMapColumns) {
      findNodeIds(
          *type->childByName(column),
          context.deduplicatedMapNodeIds(),
          [](const TypeWithId& type) {
            return type.type()->kind() == velox::TypeKind::MAP;
          });
    }
  }

  if (context.options().enableStatsCollection) {
    context.initStatsCollectors(type);
  }

  // Stamp per-node attributes (e.g. Iceberg field-ids) keyed by pre-order node
  // id. schemaAttributes uses the same TypeWithId::id() numbering the handler
  // receives, so the lookup is a direct O(1) hit as each TypeBuilder is
  // constructed. Ids with no matching node are simply never looked up.
  auto flatMapKeyEncodingLayouts = collectFlatMapKeyEncodings(*type, context);
  auto dictionaryConfigs = collectDictionaryConfigs(*type, context);

  return FieldWriter::create(
      context,
      type,
      [&context,
       _flatMapKeyEncodingLayouts = std::move(flatMapKeyEncodingLayouts),
       _dictionaryConfigs = std::move(dictionaryConfigs)](
          TypeBuilder& type, uint32_t nodeId) {
        configureAddedType(
            context,
            type,
            nodeId,
            _dictionaryConfigs,
            _flatMapKeyEncodingLayouts);
      });
}

void initializeEncodingLayouts(
    const TypeBuilder& typeBuilder,
    const EncodingLayoutTree& encodingLayoutTree) {
  {
#define SET_STREAM_CONTEXT(builder, descriptor, identifier)           \
  if (auto* encodingLayout = encodingLayoutTree.encodingLayout(       \
          EncodingLayoutTree::StreamIdentifiers::identifier)) {       \
    streamContext(builder.descriptor()).setEncoding(*encodingLayout); \
  }

    if (typeBuilder.kind() == Kind::FlatMap) {
      if (encodingLayoutTree.schemaKind() == Kind::Map) {
        // Schema evolution - If a map is converted to flatmap, we should not
        // fail, but also not try to replay captured encodings.
        return;
      }
      NIMBLE_CHECK_EQ(
          encodingLayoutTree.schemaKind(),
          Kind::FlatMap,
          "Incompatible encoding layout node. Expecting flatmap node.");
      const auto& mapBuilder = typeBuilder.asFlatMap();
      SET_STREAM_CONTEXT(mapBuilder, nullsDescriptor, FlatMap::NullsStream);
      return;
    }

    switch (typeBuilder.kind()) {
      case Kind::Scalar: {
        NIMBLE_CHECK_EQ(
            encodingLayoutTree.schemaKind(),
            Kind::Scalar,
            "Incompatible encoding layout node. Expecting scalar node.");
        SET_STREAM_CONTEXT(
            typeBuilder.asScalar(), scalarDescriptor, Scalar::ScalarStream);
        break;
      }
      case Kind::TimestampMicroNano: {
        NIMBLE_CHECK_EQ(
            encodingLayoutTree.schemaKind(),
            Kind::TimestampMicroNano,
            "Incompatible encoding layout node. Expecting TimestampMicroNano node but got {}.",
            toString(encodingLayoutTree.schemaKind()));
        auto& timestampMicroNanoBuilder = typeBuilder.asTimestampMicroNano();
        SET_STREAM_CONTEXT(
            timestampMicroNanoBuilder,
            microsDescriptor,
            TimestampMicroNano::MicrosStream);
        SET_STREAM_CONTEXT(
            timestampMicroNanoBuilder,
            nanosDescriptor,
            TimestampMicroNano::NanosStream);
        break;
      }
      case Kind::Row: {
        NIMBLE_CHECK_EQ(
            encodingLayoutTree.schemaKind(),
            Kind::Row,
            "Incompatible encoding layout node. Expecting row node.");
        auto& rowBuilder = typeBuilder.asRow();
        SET_STREAM_CONTEXT(rowBuilder, nullsDescriptor, Row::NullsStream);
        for (auto i = 0; i < rowBuilder.childrenCount() &&
             i < encodingLayoutTree.childrenCount();
             ++i) {
          initializeEncodingLayouts(
              rowBuilder.childAt(i), encodingLayoutTree.child(i));
        }
        break;
      }
      case Kind::Array: {
        NIMBLE_CHECK_EQ(
            encodingLayoutTree.schemaKind(),
            Kind::Array,
            "Incompatible encoding layout node. Expecting array node.");
        auto& arrayBuilder = typeBuilder.asArray();
        SET_STREAM_CONTEXT(
            arrayBuilder, lengthsDescriptor, Array::LengthsStream);
        if (encodingLayoutTree.childrenCount() > 0) {
          NIMBLE_CHECK(
              encodingLayoutTree.childrenCount() == 1,
              "Invalid encoding layout tree. Array node should have exactly one child.");
          initializeEncodingLayouts(
              arrayBuilder.elements(), encodingLayoutTree.child(0));
        }
        break;
      }
      case Kind::Map: {
        if (encodingLayoutTree.schemaKind() == Kind::FlatMap) {
          // Schema evolution - If a flatmap is converted to map, we should
          // not fail, but also not try to replay captured encodings.
          return;
        }
        NIMBLE_CHECK_EQ(
            encodingLayoutTree.schemaKind(),
            Kind::Map,
            "Incompatible encoding layout node. Expecting map node.");
        auto& mapBuilder = typeBuilder.asMap();

        SET_STREAM_CONTEXT(mapBuilder, lengthsDescriptor, Map::LengthsStream);
        if (encodingLayoutTree.childrenCount() > 0) {
          NIMBLE_CHECK_EQ(
              encodingLayoutTree.childrenCount(),
              2,
              "Invalid encoding layout tree. Map node should have exactly two children.");
          initializeEncodingLayouts(
              mapBuilder.keys(), encodingLayoutTree.child(0));
          initializeEncodingLayouts(
              mapBuilder.values(), encodingLayoutTree.child(1));
        }

        break;
      }
      case Kind::SlidingWindowMap: {
        NIMBLE_CHECK_EQ(
            encodingLayoutTree.schemaKind(),
            Kind::SlidingWindowMap,
            "Incompatible encoding layout node. Expecting SlidingWindowMap node.");
        auto& mapBuilder = typeBuilder.asSlidingWindowMap();
        SET_STREAM_CONTEXT(
            mapBuilder, offsetsDescriptor, SlidingWindowMap::OffsetsStream);
        SET_STREAM_CONTEXT(
            mapBuilder, lengthsDescriptor, SlidingWindowMap::LengthsStream);
        if (encodingLayoutTree.childrenCount() > 0) {
          NIMBLE_CHECK(
              encodingLayoutTree.childrenCount() == 2,
              "Invalid encoding layout tree. SlidingWindowMap node should have exactly two children.");
          initializeEncodingLayouts(
              mapBuilder.keys(), encodingLayoutTree.child(0));
          initializeEncodingLayouts(
              mapBuilder.values(), encodingLayoutTree.child(1));
        }

        break;
      }
      case Kind::ArrayWithOffsets: {
        NIMBLE_CHECK_EQ(
            encodingLayoutTree.schemaKind(),
            Kind::ArrayWithOffsets,
            "Incompatible encoding layout node. Expecting offset array node.");
        auto& arrayBuilder = typeBuilder.asArrayWithOffsets();
        SET_STREAM_CONTEXT(
            arrayBuilder, offsetsDescriptor, ArrayWithOffsets::OffsetsStream);
        SET_STREAM_CONTEXT(
            arrayBuilder, lengthsDescriptor, ArrayWithOffsets::LengthsStream);
        if (encodingLayoutTree.childrenCount() > 0) {
          NIMBLE_CHECK(
              encodingLayoutTree.childrenCount() == 2,
              "Invalid encoding layout tree. ArrayWithOffset node should have exactly two children.");
          initializeEncodingLayouts(
              arrayBuilder.elements(), encodingLayoutTree.child(0));
        }
        break;
      }
      case Kind::FlatMap: {
        NIMBLE_UNREACHABLE("Flatmap handled already");
      }
    }
#undef SET_STREAM_CONTEXT
  }
}

void configureAddedFlatMapField(
    detail::WriterContext& context,
    const TypeBuilder& flatmap,
    std::string_view fieldKey,
    const TypeBuilder& fieldType) {
  auto& flatmapBuilder = flatmap.asFlatMap();
  auto& inMapContext = streamContext(
      flatmapBuilder.inMapDescriptorAt(flatmapBuilder.childrenCount() - 1));
  inMapContext.setIsInMapStream(true);
  std::vector<offset_size> valueStreamOffsets;
  collectFlatMapValueStreamOffsets(fieldType, valueStreamOffsets);
  inMapContext.setFlatMapValueStreamOffsets(std::move(valueStreamOffsets));

  auto* flatMapContext = flatmap.context<FlatmapEncodingLayoutContext>();
  if (flatMapContext == nullptr) {
    return;
  }

  if (context.options().encodingLayoutTree.has_value()) {
    auto it = flatMapContext->keyEncodings.find(fieldKey);
    if (it != flatMapContext->keyEncodings.end()) {
      initializeEncodingLayouts(fieldType, *it->second);
    }
  }

  auto it = flatMapContext->valueDictionaries.find(std::string{fieldKey});
  if (it == flatMapContext->valueDictionaries.end()) {
    return;
  }
  for (const auto& valueDictionary : it->second) {
    configureDictionary(
        resolveFieldPath(fieldType, valueDictionary.valueSubfield),
        valueDictionary.dictionary,
        context.schemaBuilder());
  }
}
} // namespace

std::unique_ptr<index::IndexWriter> Writer::createClusterIndexWriter(
    const WriterOptions& options,
    const velox::TypePtr& type,
    velox::memory::MemoryPool* pool) {
  if (options.clusterIndexConfig == nullptr) {
    return nullptr;
  }
  const auto& config = *options.clusterIndexConfig;
  NIMBLE_USER_CHECK_EQ(
      config.family,
      index::IndexFamily::Cluster,
      "Cluster index configuration must use the cluster family: {}",
      config.name);
  auto writer =
      index::clusterIndexFactory(config.name).createWriter(config, type, pool);
  NIMBLE_CHECK_NOT_NULL(
      writer, "Cluster index factory returned a null writer: {}", config.name);
  return writer;
}

std::vector<Writer::DenseIndexWriter> Writer::createDenseIndexWriters(
    const WriterOptions& options,
    const velox::TypePtr& type,
    velox::memory::MemoryPool* pool) {
  struct ConfigGroup {
    std::string_view name;
    std::vector<const index::IndexConfig*> configs;
  };

  std::vector<ConfigGroup> groups;
  for (const auto& configPtr : options.denseIndexConfigs) {
    NIMBLE_USER_CHECK_NOT_NULL(configPtr, "Dense index config cannot be null");
    const auto& config = *configPtr;
    NIMBLE_USER_CHECK_EQ(
        config.family,
        index::IndexFamily::Dense,
        "Dense index configuration must use the dense family: {}",
        config.name);
    auto group =
        std::find_if(groups.begin(), groups.end(), [&](const auto& candidate) {
          return candidate.name == config.name;
        });
    if (group == groups.end()) {
      groups.emplace_back(ConfigGroup{config.name, {}});
      group = std::prev(groups.end());
    }
    group->configs.emplace_back(configPtr.get());
  }

  std::vector<DenseIndexWriter> writers;
  writers.reserve(groups.size());
  for (auto& group : groups) {
    const auto* factory = index::denseIndexFactory(group.name);
    NIMBLE_USER_CHECK_NOT_NULL(
        factory, "Unknown dense index factory: {}", group.name);
    auto writer = factory->createWriter(group.configs, type, pool);
    NIMBLE_CHECK_NOT_NULL(
        writer, "Dense index factory returned a null writer: {}", group.name);
    writers.emplace_back(
        DenseIndexWriter{std::string{group.name}, std::move(writer)});
  }
  return writers;
}

Writer::Writer(
    const velox::TypePtr& type,
    std::unique_ptr<velox::WriteFile> file,
    velox::memory::MemoryPool& pool,
    WriterOptions options)
    : storedDataType_{storedDataType(type, options)},
      storedInputColumnIndices_{
          omitClusterIndexKeyColumnStorage(options)
              ? storedInputColumnIndices(type, options)
              : std::vector<velox::column_index_t>{}},
      schema_{velox::dwio::common::TypeWithId::create(storedDataType_)},
      // Tracks storedDataType_ rather than the input type so the metadata's
      // pre-order node ids line up with the stats, which come from the field
      // writers and so only cover the stored columns.
      // TODO(T283224280): report statistics for omitted cluster index key
      // columns too, instead of leaving them out of the file metadata.
      rowType_{velox::asRowType(storedDataType_)},
      spillConfig_{options.spillConfig},
      pool_{MemoryPoolHolder::create(
          pool,
          [&](auto& pool) {
            // Velox rejects a child reclaimer whose parent has none, so only
            // install one when the parent pool participates in arbitration.
            // There, prefer the writer's own: it subsumes a plain
            // exec::MemoryReclaimer and additionally frees memory by flushing a
            // stripe. `this` is still under construction, but the reclaimer
            // only dereferences it once arbitration runs.
            auto reclaimer = options.reclaimerFactory();
            if (pool.reclaimer() != nullptr) {
              if (auto writerReclaimer = makeMemoryReclaimer()) {
                reclaimer = std::move(writerReclaimer);
              }
            }
            return pool.addAggregateChild(
                fmt::format("nimble_writer_{}", folly::Random::rand64()),
                std::move(reclaimer));
          })},
      encodingMemoryPool_{MemoryPoolHolder::create(
          *pool_,
          [&](auto& pool) {
            return pool.addLeafChild(
                "encoding", true, options.reclaimerFactory());
          })},
      context_{std::make_unique<detail::WriterContext>(
          *pool_,
          storedWriterOptions(
              type,
              storedDataType_,
              storedInputColumnIndices_,
              std::move(options)))},
      file_{std::move(file)},
      clusterIndexWriter_{createClusterIndexWriter(
          context_->options(),
          type,
          &(*context_->bufferMemoryPool()))},
      denseIndexWriters_{createDenseIndexWriters(
          context_->options(),
          type,
          &(*context_->bufferMemoryPool()))},
      tabletWriter_{TabletWriter::create(
          file_.get(),
          *encodingMemoryPool_,
          {.layoutPlanner = std::make_unique<DefaultLayoutPlanner>(
               &context_->schemaBuilder(),
               context_->options().featureReordering),
           .metadataCompressionThreshold =
               context_->options().metadataCompressionThreshold.value_or(
                   kMetadataCompressionThreshold),
           .streamDeduplicationEnabled =
               context_->options().enableStreamDeduplication,
           .enableChunkIndex = context_->options().enableChunkIndex,
           .chunkStatsMinAvgChunks = context_->options().chunkStatsMinAvgChunks,
           .stripeGroupEncodingLayout =
               context_->options().experimentalStripeGroupEncodingLayout,
           .stripeGroupEncodingLayoutReadFactors =
               context_->options()
                   .experimentalStripeGroupEncodingLayoutReadFactors,
           .stripeGroupFlushCallback =
               (clusterIndexWriter_ != nullptr || !denseIndexWriters_.empty())
               ? TabletWriter::StripeGroupFlushCallback(
                     [this](
                         const WriteDataFn& writeDataFn,
                         const CreateMetadataSectionFn& createMetadataFn) {
                       if (clusterIndexWriter_ != nullptr) {
                         clusterIndexWriter_->flush(
                             writeDataFn, createMetadataFn);
                       }
                       for (const auto& denseIndex : denseIndexWriters_) {
                         denseIndex.writer->flush(
                             writeDataFn, createMetadataFn);
                       }
                     })
               : nullptr,
           .closeCallback =
               [this](
                   const WriteDataFn& writeDataFn,
                   const CreateMetadataSectionFn& createMetadataFn,
                   const WriteOptionalSectionFn& writeMetadataFn) {
                 writeDictionarySection(writeDataFn, writeMetadataFn);
                 writeProperties(writeMetadataFn);
                 writeIndexes(writeDataFn, createMetadataFn, writeMetadataFn);
               }})},
      bufferPolicy_{
          context_->options().bufferPolicyFactory != nullptr
              ? context_->options().bufferPolicyFactory()
              : nullptr} {
  NIMBLE_CHECK_NOT_NULL(file_);

  // Register handler for dynamically discovered FlatMap keys before creating
  // the writer tree, so that predefined keys also trigger the handler.
  context_->setFlatmapFieldAddedEventHandler([this](
                                                 const TypeBuilder& flatmap,
                                                 std::string_view fieldKey,
                                                 const TypeBuilder& fieldType) {
    configureAddedFlatMapField(*context_, flatmap, fieldKey, fieldType);
  });

  rootWriter_ = createRootFieldWriter(schema_, *context_);

  if (context_->options().encodingLayoutTree.has_value()) {
    initializeEncodingLayouts(
        *rootWriter_->typeBuilder(),
        context_->options().encodingLayoutTree.value());
  }

  setState(State::kRunning);
}

Writer::~Writer() {
  // The reclaimer installed on `pool_` outlives every member it reaches, since
  // `pool_` is destroyed last. Leaving the writer running would let a
  // concurrent arbitration request flush a writer whose members are gone.
  if (isRunning()) {
    setState(State::kAborted);
  }
}

void Writer::write(const velox::VectorPtr& input) {
  checkRunning();
  if (lastException_) {
    std::rethrow_exception(lastException_);
  }

  NIMBLE_CHECK_NOT_NULL(input, "Input must not be null");
  NIMBLE_CHECK_NOT_NULL(file_, "Writer is already closed");
  try {
    // BufferPolicy path: content-driven cutting via a policy that buffers
    // inputs across writes and emits stripe-ready row ranges. Policies
    // that need a specific input shape (e.g. plain RowVector for column
    // inspection) validate internally.
    if (bufferPolicy_ != nullptr) {
      bufferPolicy_->bufferInput(input);
      flushInputBuffers(/*finalize=*/false);
    } else {
      // Legacy FlushPolicy path: append the whole batch to the writer's stream
      // buffers, then consult shouldFlush.
      writeBatch(input);
      evaluateFlushPolicy();
    }
    updateIoStatistics();
  } catch (const std::exception& e) {
    lastException_ = std::current_exception();
    context_->logger()->logException(LogOperation::Write, e.what());
    throw;
  } catch (...) {
    lastException_ = std::current_exception();
    context_->logger()->logException(
        LogOperation::Write,
        folly::to<std::string>(folly::exceptionStr(std::current_exception())));
    throw;
  }
}

bool Writer::flushInputBuffers(bool finalize) {
  if (finalize) {
    bufferPolicy_->finalize();
  }
  bool anyStripeFlushed = false;
  while (true) {
    auto range = bufferPolicy_->flushInput();
    if (range.empty()) {
      break;
    }
    NIMBLE_CHECK_EQ(
        range.inputs.size(),
        range.rowRanges.size(),
        "BufferRange inputs and rowRanges must be parallel arrays");
    for (size_t i = 0; i < range.inputs.size(); ++i) {
      const auto& rowRange = range.rowRanges[i];
      writeBatch(range.inputs[i]->slice(rowRange.startRow, rowRange.numRows()));
    }
    anyStripeFlushed = writeStripe() || anyStripeFlushed;
  }
  return anyStripeFlushed;
}

void Writer::writeBatch(const velox::VectorPtr& input) {
  const auto numRows = input->size();
  const auto storedData = storedDataInput(input);
  // When enableStatsConsistencyCheck is true, compute raw size using
  // RawSizeUtils to verify consistency with column statistics.
  // Otherwise, skip this computation as column statistics will provide
  // the raw size.
  // Skip entirely when stats collection is disabled — there is no
  // writeColumnStats() call to consume this value.
  if (context_->options().enableStatsCollection &&
      context_->options().enableStatsConsistencyCheck) {
    // Calculate raw size using schema information to correctly handle
    // passthrough flatmaps.
    RawSizeContext context;
    const auto rawSize = nimble::getRawSizeFromVector(
        storedData,
        velox::common::Ranges::of(0, numRows),
        context,
        schema_.get(),
        context_->flatMapNodeIds(),
        context_->ignoreTopLevelNulls());
    context_->updateFileRawSize(rawSize);
  }

  {
    velox::CpuWallTimer ingestionTimer{context_->ingestionTiming()};
    rootWriter_->write(storedData, OrderedRanges::of(0, numRows));
  }
  addIndexKey(input);

  uint64_t memoryUsed{0};
  for (const auto& [_, stream] : context_->streams()) {
    memoryUsed += stream->memoryUsed();
  }

  context_->setMemoryUsed(memoryUsed);
  context_->updateRowsInFile(numRows);
  context_->updateRowsInStripe(numRows);
  context_->setBytesWritten(file_->size());
}

velox::VectorPtr Writer::storedDataInput(const velox::VectorPtr& input) const {
  if (!omitClusterIndexKeyColumnStorage(context_->options())) {
    return input;
  }

  auto loaded = velox::BaseVector::loadedVectorShared(input);
  auto rowInput = velox::checkedPointerCast<const velox::RowVector>(loaded);

  std::vector<velox::VectorPtr> children;
  children.reserve(storedInputColumnIndices_.size());
  for (auto index : storedInputColumnIndices_) {
    children.push_back(rowInput->childAt(index));
  }
  return std::make_shared<velox::RowVector>(
      pool_.get(),
      velox::asRowType(storedDataType_),
      loaded->nulls(),
      loaded->size(),
      std::move(children));
}

void Writer::writeMetadata() {
  if (context_->options().metadata.empty()) {
    return;
  }
  auto& metadata = context_->options().metadata;
  auto it = metadata.cbegin();
  flatbuffers::FlatBufferBuilder builder(kInitialSchemaSectionSize);
  auto entries =
      builder.CreateVector<flatbuffers::Offset<serialization::MetadataEntry>>(
          metadata.size(), [&builder, &it](size_t /* i */) {
            auto entry = serialization::CreateMetadataEntry(
                builder,
                builder.CreateString(it->first),
                builder.CreateString(it->second));
            ++it;
            return entry;
          });

  builder.Finish(serialization::CreateMetadata(builder, entries));
  tabletWriter_->writeOptionalSection(
      std::string(kMetadataSection),
      {reinterpret_cast<const char*>(builder.GetBufferPointer()),
       builder.GetSize()});
}

void Writer::writeColumnStats() {
  context_->finalizeFileStatsFromStripes();
  // When enableStatsConsistencyCheck is true, verify that fileRawSize
  // (accumulated via RawSizeUtils) matches the root column statistics.
  if (context_->options().enableStatsConsistencyCheck) {
    NIMBLE_CHECK_EQ(
        context_->fileRawSize(),
        context_->columnStats().front()->getLogicalSize(),
        "Mismatched raw sizes!");
  }

  if (context_->options().enableVectorizedStats) {
    VectorizedFileStats fileStats{
        context_->columnStats(), encodingMemoryPool_.get()};
    Buffer buffer{*encodingMemoryPool_};
    tabletWriter_->writeOptionalSection(
        std::string(kVectorizedStatsSection), fileStats.serialize(buffer));
    VectorizedStripeStats stripeStats{
        context_->stripeStats(), encodingMemoryPool_.get()};
    Buffer stripeStatsBuffer{*encodingMemoryPool_};
    tabletWriter_->writeOptionalSection(
        std::string(kStripeStatsSection),
        stripeStats.serialize(stripeStatsBuffer));
  } else {
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(
        serialization::CreateStats(builder, context_->fileRawSize()));
    tabletWriter_->writeOptionalSection(
        std::string(kStatsSection),
        {reinterpret_cast<const char*>(builder.GetBufferPointer()),
         builder.GetSize()});
  }
}

void Writer::writeSchema() {
  SchemaSerializer serializer;
  tabletWriter_->writeOptionalSection(
      std::string(kSchemaSection),
      serializer.serialize(context_->schemaBuilder()));
}

void Writer::writeStripeDictionaryStreams() {
  if (!context_->hasStripeDictionaryConfig()) {
    return;
  }
  for (const auto& [_, streamData] : context_->streams()) {
    const auto* streamContext =
        streamData->descriptor().context<WriterStreamContext>();
    if (streamContext == nullptr) {
      continue;
    }
    auto* writer = streamContext->sharedDictionaryWriter();
    if (writer == nullptr) {
      continue;
    }
    if (writer->scope() != SharedDictionaryScope::Stripe) {
      continue;
    }
    auto alphabetChunk = writer->encodeAlphabet(*encodingBuffer_);
    if (!alphabetChunk.has_value()) {
      continue;
    }
    const auto streamId = writer->dictionaryId();
    NIMBLE_CHECK_LT(streamId, encodedStreams_.size());
    auto& dictionaryStream = encodedStreams_[streamId];
    NIMBLE_CHECK(dictionaryStream.chunks.empty());
    dictionaryStream.offset = streamId;
    dictionaryStream.chunks.push_back(std::move(alphabetChunk.value()));
  }
}

void Writer::writeDictionarySection(
    const WriteDataFn& writeDataFn,
    const WriteOptionalSectionFn& writeMetadataFn) {
  if (context_->options().experimentalSharedDictionaryEncoding.empty()) {
    return;
  }

  std::vector<SharedDictionaryReference> stripeDictionaryReferences;
  std::vector<SharedDictionaryReference> fileDictionaryReferences;
  std::vector<SharedDictionaryReference> externalDictionaryReferences;
  std::vector<FileDictionary> fileDictionaries;
  for (const auto& [_, streamData] : context_->streams()) {
    const auto* streamContext =
        streamData->descriptor().context<WriterStreamContext>();
    if (streamContext == nullptr) {
      continue;
    }
    auto* writer = streamContext->sharedDictionaryWriter();
    if (writer == nullptr) {
      continue;
    }
    if (!writer->hasUsedDictionary()) {
      continue;
    }
    if (writer->scope() == SharedDictionaryScope::File) {
      auto alphabet = writer->encodeAlphabet(*encodingBuffer_);
      NIMBLE_CHECK(
          alphabet.has_value(),
          "File shared dictionary {} used shared dictionary encoding but "
          "produced no alphabet.",
          writer->dictionaryId());
      const auto [offset, length] = writeDataFn(alphabet->content);
      fileDictionaries.push_back({
          .dictionaryId = writer->dictionaryId(),
          .dataType = writer->dataType(),
          .offset = offset,
          .length = length,
      });
    }
    const SharedDictionaryReference reference{
        .valueStreamId = streamData->descriptor().offset(),
        .dictionaryId = writer->dictionaryId(),
        .dataType = writer->dataType(),
    };
    switch (writer->scope()) {
      case SharedDictionaryScope::Stripe:
        stripeDictionaryReferences.push_back(reference);
        break;
      case SharedDictionaryScope::File:
        fileDictionaryReferences.push_back(reference);
        break;
      case SharedDictionaryScope::External:
        externalDictionaryReferences.push_back(reference);
        break;
    }
  }
  if (stripeDictionaryReferences.empty() && fileDictionaryReferences.empty() &&
      externalDictionaryReferences.empty()) {
    return;
  }

  writeMetadataFn(
      std::string(kDictionarySection),
      SharedDictionaryCatalog::serialize(
          stripeDictionaryReferences,
          fileDictionaryReferences,
          externalDictionaryReferences,
          fileDictionaries));
}

void Writer::addIndexKey(const velox::VectorPtr& input) {
  if (clusterIndexWriter_ != nullptr) {
    clusterIndexWriter_->write(input);
  }
  for (const auto& denseIndex : denseIndexWriters_) {
    denseIndex.writer->write(input);
  }
}

void Writer::writeProperties(const WriteOptionalSectionFn& writeMetadataFn) {
  const bool compactRowCountEncoding =
      context_->options().experimentalCompactRowCountEncoding;
  bool clusterIndexKeyColumnStorageOmitted{false};
  std::vector<std::string> clusterIndexKeyColumnsWithOmittedStorage;
  if (omitClusterIndexKeyColumnStorage(context_->options())) {
    const auto& indexOptions = clusterIndexConfig(context_->options());
    clusterIndexKeyColumnStorageOmitted = true;
    clusterIndexKeyColumnsWithOmittedStorage = indexOptions.columns;
  }

  if (!compactRowCountEncoding && !clusterIndexKeyColumnStorageOmitted) {
    return;
  }

  const auto serialized =
      FileProperties{
          compactRowCountEncoding,
          clusterIndexKeyColumnStorageOmitted,
          std::move(clusterIndexKeyColumnsWithOmittedStorage)}
          .serialize();
  writeMetadataFn(std::string(kPropertiesSection), serialized);
}

void Writer::writeIndexes(
    const WriteDataFn& writeDataFn,
    const CreateMetadataSectionFn& createMetadataFn,
    const WriteOptionalSectionFn& writeMetadataFn) {
  std::vector<index::IndexDescriptor> descriptors;
  for (const auto& denseIndex : denseIndexWriters_) {
    if (auto descriptor =
            denseIndex.writer->close(writeDataFn, createMetadataFn)) {
      NIMBLE_CHECK_EQ(
          descriptor->family,
          index::IndexFamily::Dense,
          "Index writer returned an unexpected family for {}",
          denseIndex.name);
      NIMBLE_CHECK_EQ(
          descriptor->name,
          denseIndex.name,
          "Index writer returned an unexpected name for {}",
          denseIndex.name);
      descriptors.emplace_back(std::move(descriptor.value()));
    }
  }
  if (clusterIndexWriter_ != nullptr) {
    if (auto descriptor =
            clusterIndexWriter_->close(writeDataFn, createMetadataFn)) {
      const auto& config = *context_->options().clusterIndexConfig;
      NIMBLE_CHECK_EQ(
          descriptor->family,
          index::IndexFamily::Cluster,
          "Index writer returned an unexpected family for {}",
          config.name);
      NIMBLE_CHECK_EQ(
          descriptor->name,
          config.name,
          "Index writer returned an unexpected name for {}",
          config.name);
      descriptors.emplace_back(std::move(descriptor.value()));
    }
  }
  writeIndexSection(descriptors, writeMetadataFn);
}

bool Writer::shouldFlush(FlushPolicy* policy) const {
  return policy->shouldFlush(
      StripeProgress{
          .stripeRawSize = context_->memoryUsed(),
          .stripeEncodedSize = context_->stripeEncodedPhysicalSize(),
          .stripeEncodedLogicalSize = context_->stripeEncodedLogicalSize()});
}

bool Writer::shouldChunk(FlushPolicy* policy) const {
  return policy->shouldChunk(
      StripeProgress{
          .stripeRawSize = context_->memoryUsed(),
          .stripeEncodedSize = context_->stripeEncodedPhysicalSize(),
          .stripeEncodedLogicalSize = context_->stripeEncodedLogicalSize()});
}

std::unique_ptr<velox::dwio::common::FileMetadata> Writer::close() {
  checkRunning();
  if (lastException_) {
    std::rethrow_exception(lastException_);
  }

  // After checkRunning() and the lastException_ guard above, file_ is always
  // set: the constructor asserts it, and the paths below null it out only while
  // transitioning to kClosed or storing lastException_, so a later call is
  // rejected by one of those guards before reaching here.
  NIMBLE_CHECK_NOT_NULL(file_);
  try {
    if (bufferPolicy_ != nullptr) {
      // Finalize signals "no more input" so the policy emits any residual
      // range as the tail stripe.
      flushInputBuffers(/*finalize=*/true);
    }
    writeStripe();
    rootWriter_->close();

    writeMetadata();
    if (context_->options().enableStatsCollection) {
      writeColumnStats();
    }
    writeSchema();

    tabletWriter_->close();
    file_->close();
    context_->setBytesWritten(file_->size());
    updateIoStatistics();

    // TODO: compute and populate input size.
    FileCloseMetrics metrics{
        .rowCount = context_->rowsInFile(),
        .stripeCount = context_->getStripeIndex(),
        .fileSize = context_->bytesWritten(),
        .encodingCpuNs = context_->encodingTiming().cpuNanos,
        .encodingWallNs = context_->encodingTiming().wallNanos};
    context_->logger()->logFileClose(metrics);
    file_ = nullptr;

    setState(State::kClosed);
    reportRuntimeStats();
    return buildFileMetadata();
  } catch (const std::exception& e) {
    lastException_ = std::current_exception();
    context_->logger()->logException(LogOperation::Close, e.what());
    file_ = nullptr;
    throw;
  } catch (...) {
    lastException_ = std::current_exception();
    context_->logger()->logException(
        LogOperation::Close,
        folly::to<std::string>(folly::exceptionStr(std::current_exception())));
    file_ = nullptr;
    throw;
  }
}

bool Writer::finish() {
  checkRunning();
  return true;
}

void Writer::abort() {
  checkRunning();
  setState(State::kAborted);
}

void Writer::flush() {
  checkRunning();
  if (lastException_) {
    std::rethrow_exception(lastException_);
  }

  try {
    writeStripe();
    updateIoStatistics();
  } catch (const std::exception& e) {
    lastException_ = std::current_exception();
    context_->logger()->logException(LogOperation::Flush, e.what());
    throw;
  } catch (...) {
    lastException_ = std::current_exception();
    context_->logger()->logException(
        LogOperation::Flush,
        folly::to<std::string>(folly::exceptionStr(std::current_exception())));
    throw;
  }
}

bool Writer::reclaimableBytes(
    const velox::memory::MemoryPool& pool,
    uint64_t& reclaimableBytes) const {
  reclaimableBytes = 0;
  // Only a running writer can flush a stripe, which is the sole way this pool
  // gives memory back. Reporting bytes in any other state makes the arbitrator
  // pick this pool as a candidate, suspend the driver, and then be turned away
  // by the same check in reclaimBytes().
  if (!canReclaim() || !isRunning()) {
    return false;
  }
  const auto reservedBytes = pool.reservedBytes();
  if (reservedBytes < spillConfig_->writerFlushThresholdSize) {
    return false;
  }
  reclaimableBytes = reservedBytes;
  return true;
}

uint64_t Writer::reclaimBytes(
    velox::memory::MemoryPool* pool,
    velox::memory::MemoryReclaimer::Stats& stats) {
  VELOX_CHECK(canReclaim());

  if (!isRunning()) {
    LOG(WARNING) << "Can't reclaim from a not running nimble writer: "
                 << pool->name() << ", state: " << state();
    ++stats.numNonReclaimableAttempts;
    return 0;
  }

  const auto flushThreshold = spillConfig_->writerFlushThresholdSize;
  return velox::memory::MemoryReclaimer::run(
      [&]() {
        int64_t reclaimedBytes{0};
        {
          velox::memory::ScopedReclaimedBytesRecorder recorder(
              pool, &reclaimedBytes);
          const auto reservedBytes = pool->reservedBytes();
          if (reservedBytes < flushThreshold) {
            RECORD_METRIC_VALUE(velox::kMetricMemoryNonReclaimableCount);
            LOG(WARNING) << "Can't reclaim memory from nimble writer pool "
                         << pool->name()
                         << " which doesn't have sufficient memory to flush, "
                            "writer memory usage: "
                         << velox::succinctBytes(reservedBytes)
                         << ", writer flush memory threshold: "
                         << velox::succinctBytes(flushThreshold);
            ++stats.numNonReclaimableAttempts;
          } else {
            flush();
          }
        }
        return reclaimedBytes;
      },
      stats);
}

bool Writer::canReclaim() const {
  return spillConfig_ != nullptr;
}

namespace {

// Copies the values needed for Iceberg manifest statistics out of a
// (non-owning) NIMBLE statistics view into an owned snapshot. Unwraps
// deduplicated statistics to reach the typed min/max of the base column.
NimbleFileMetadata::ColumnStats toColumnStatsSnapshot(
    const ColumnStatistics* stat) {
  NimbleFileMetadata::ColumnStats snapshot;
  if (stat == nullptr) {
    return snapshot;
  }

  snapshot.valueCount = stat->getValueCount();
  snapshot.nullCount = stat->getNullCount();
  snapshot.physicalSize = stat->getPhysicalSize();

  const ColumnStatistics* typed = stat;
  if (const auto* deduplicated = stat->as<DeduplicatedColumnStatistics>()) {
    typed = deduplicated->getBaseStatistics();
  }

  if (const auto* integral = typed->as<IntegralStatistics>()) {
    snapshot.integralMin = integral->getMin();
    snapshot.integralMax = integral->getMax();
  } else if (const auto* floating = typed->as<FloatingPointStatistics>()) {
    snapshot.floatingMin = floating->getMin();
    snapshot.floatingMax = floating->getMax();
  } else if (const auto* string = typed->as<StringStatistics>()) {
    snapshot.stringMin = string->getMin();
    snapshot.stringMax = string->getMax();
  }
  return snapshot;
}

} // namespace

std::unique_ptr<NimbleFileMetadata> Writer::buildFileMetadata() const {
  // Column statistics are not part of the runtime-stat map, so they are read
  // straight from the writer rather than through the stats() projection.
  const auto& columnStats = context_->columnStats();
  if (columnStats.empty()) {
    return nullptr;
  }

  std::vector<NimbleFileMetadata::ColumnStats> snapshots;
  snapshots.reserve(columnStats.size());
  for (const auto* stat : columnStats) {
    snapshots.push_back(toColumnStatsSnapshot(stat));
  }

  const int64_t numRows = columnStats.front() != nullptr
      ? static_cast<int64_t>(
            columnStats.front()->getValueCount() +
            columnStats.front()->getNullCount())
      : 0;
  return std::make_unique<NimbleFileMetadata>(
      rowType_, numRows, std::move(snapshots));
}

// TODO: Drop this push channel in favour of the runtimeStats() pull override
// and report the off-thread encode CPU under
// velox::exec::Operator::kBackgroundCpuTimeNanos so the operator attributes it
// as background time. Both channels publish the same keys today and the
// connector applies the pull last via setRuntimeStat(), so the reported values
// agree; only the pull merges correctly across the rotated writers of one sink.
void Writer::reportRuntimeStats() const {
  using Keys = RuntimeStats;
  const auto stats = runtimeStats();
  const auto reportNanos = [&stats](std::string_view key) {
    velox::addThreadLocalRuntimeStat(
        key,
        velox::RuntimeCounter(
            runtimeStat(stats, key).sum, velox::RuntimeCounter::Unit::kNanos));
  };
  reportNanos(Keys::kEncodingCpuNanos);
  reportNanos(Keys::kEncodingWallNanos);
  reportNanos(Keys::kWriteCpuNanos);
  reportNanos(Keys::kWriteWallNanos);
  reportNanos(Keys::kIngestionCpuNanos);
  reportNanos(Keys::kEncodingSelectionCpuNanos);

  // Distributions are published whole, so they replace rather than accumulate.
  velox::setThreadLocalRuntimeStat(
      Keys::kRowsPerStripe, runtimeStat(stats, Keys::kRowsPerStripe));
  velox::setThreadLocalRuntimeStat(
      Keys::kChunkSizeBytes, runtimeStat(stats, Keys::kChunkSizeBytes));
}

void Writer::updateIoStatistics() {
  // Read the two counters directly rather than through stats(): this runs on
  // every write(), which is too hot to materialize the whole runtime-stat map
  // for two values.
  const auto writeWallTimeNs = context_->writeTiming().wallNanos;
  const auto writtenBytes = context_->bytesWritten();
  if (auto* ioStatistics = context_->options().ioStatistics;
      ioStatistics != nullptr) {
    NIMBLE_CHECK_GE(writeWallTimeNs, reportedWriteWallTimeNs_);
    NIMBLE_CHECK_GE(writtenBytes, reportedBytesWritten_);
    ioStatistics->incWriteIOTimeUs(
        (writeWallTimeNs - reportedWriteWallTimeNs_) / 1'000);
    ioStatistics->incRawBytesWritten(writtenBytes - reportedBytesWritten_);
  }
  reportedWriteWallTimeNs_ = writeWallTimeNs;
  reportedBytesWritten_ = writtenBytes;
}

void Writer::ensureEncodingBuffer() {
  if (encodingBuffer_ == nullptr) {
    encodingBuffer_ = std::make_unique<Buffer>(*encodingMemoryPool_);
  }
}

std::unique_ptr<velox::BufferPool> Writer::makeEncodingScratchBufferPool()
    const {
  const auto maxCachedBuffers =
      context_->options().maxCachedEncodingScratchBuffers;
  NIMBLE_CHECK_GT(maxCachedBuffers, 0);
  return std::make_unique<velox::BufferPool>(maxCachedBuffers);
}

std::unique_ptr<EncodingBufferPool> Writer::makeEncodingBufferPool() const {
  const auto maxCachedBuffers =
      context_->options().maxCachedNestedEncodingBuffers;
  NIMBLE_CHECK_GT(maxCachedBuffers, 0);
  return std::make_unique<EncodingBufferPool>(
      encodingMemoryPool_.get(), maxCachedBuffers);
}

uint32_t Writer::encodingConcurrency(uint32_t taskCount) const {
  if (taskCount == 0) {
    return 0;
  }
  const auto& options = context_->options();
  if (!options.encodingExecutor || options.maxEncodeParallelism == 0) {
    return 1;
  }
  return std::min(taskCount, options.maxEncodeParallelism);
}

void Writer::ensureEncodingScratchBufferPools(uint32_t poolCount) {
  if (context_->options().maxCachedEncodingScratchBuffers == 0) {
    NIMBLE_CHECK(
        encodingScratchBufferPools_.empty(),
        "Encoding scratch buffer pools should not be created when caching is disabled.");
    return;
  }
  while (encodingScratchBufferPools_.size() < poolCount) {
    encodingScratchBufferPools_.emplace_back(makeEncodingScratchBufferPool());
  }
}

void Writer::ensureEncodingBufferPools(uint32_t poolCount) {
  if (context_->options().maxCachedNestedEncodingBuffers == 0) {
    NIMBLE_CHECK(
        encodingBufferPools_.empty(),
        "Encoding buffer pools should not be created when caching is disabled.");
    return;
  }
  while (encodingBufferPools_.size() < poolCount) {
    encodingBufferPools_.emplace_back(makeEncodingBufferPool());
  }
}

velox::BufferPool* Writer::encodingScratchBufferPool(uint32_t index) {
  if (context_->options().maxCachedEncodingScratchBuffers == 0) {
    return nullptr;
  }
  ensureEncodingScratchBufferPools(index + 1);
  NIMBLE_CHECK_LT(index, encodingScratchBufferPools_.size());
  return encodingScratchBufferPools_[index].get();
}

EncodingBufferPool* Writer::encodingBufferPool(uint32_t index) {
  if (context_->options().maxCachedNestedEncodingBuffers == 0) {
    return nullptr;
  }
  ensureEncodingBufferPools(index + 1);
  NIMBLE_CHECK_LT(index, encodingBufferPools_.size());
  return encodingBufferPools_[index].get();
}

void Writer::clearEncodingBuffer() {
  if (encodingBuffer_ == nullptr) {
    return;
  }
  if (velox::memory::underMemoryArbitration()) {
    // Under memory arbitration, free the buffer and pools.
    encodingScratchBufferPools_.clear();
    encodingBufferPools_.clear();
    encodingBuffer_.reset();
  } else {
    // Normal flush: rewind and keep chunks allocated for reuse.
    encodingBuffer_->reset();
  }
}

void Writer::ensureWriteStreams() {
  ensureEncodingBuffer();
  const auto schemaNodeCount = context_->schemaBuilder().nodeCount();
  encodedStreams_.resize(schemaNodeCount);
}

void Writer::resetFieldWriter() {
  rootWriter_->reset();
}

void Writer::writeStreams() {
  std::atomic_uint64_t chunkSize{0};
  std::atomic_uint64_t encodingCpuNanos{0};
  uint64_t encodingWallNanos{0};
  {
    LoggingScope scope{*context_->logger()};
    velox::NanosecondTimer wallTimer{&encodingWallNanos};

    ensureWriteStreams();

    const auto& streams = context_->streams();
    const auto streamCount = static_cast<uint32_t>(streams.size());
    const auto concurrency = encodingConcurrency(streamCount);
    ensureEncodingScratchBufferPools(concurrency);
    ensureEncodingBufferPools(concurrency);

    if (concurrency > 1) {
      const auto& encodingExecutor = context_->options().encodingExecutor;
      NIMBLE_CHECK(
          encodingExecutor,
          "Encoding executor is required for parallel encoding.");
      for (uint32_t start = 0; start < streamCount; start += concurrency) {
        velox::dwio::common::ExecutorBarrier barrier{encodingExecutor};
        const auto batchSize = std::min(concurrency, streamCount - start);
        for (uint32_t index = 0; index < batchSize; ++index) {
          auto& [nodeId, streamData] = streams[start + index];
          auto* encodingScratchBufferPool =
              this->encodingScratchBufferPool(index);
          auto* encodingBufferPool = this->encodingBufferPool(index);
          barrier.add([&,
                       statsCollector = context_->getStatsCollector(nodeId),
                       _streamData = streamData.get(),
                       encodingScratchBufferPool,
                       encodingBufferPool]() {
            uint64_t startCpuNs = velox::process::threadCpuNanos();
            uint64_t streamSize{0};
            processStream(
                *_streamData,
                encodingScratchBufferPool,
                encodingBufferPool,
                streamSize,
                chunkSize);
            if (statsCollector) {
              statsCollector->addPhysicalSize(streamSize);
            }
            encodingCpuNanos.fetch_add(
                velox::process::threadCpuNanos() - startCpuNs,
                std::memory_order_relaxed);
          });
        }
        barrier.waitAll();
      }
    } else {
      auto* encodingScratchBufferPool = this->encodingScratchBufferPool();
      auto* encodingBufferPool = this->encodingBufferPool();
      for (auto& [nodeId, streamData] : streams) {
        auto statsCollector = context_->getStatsCollector(nodeId);
        uint64_t startCpuNs = velox::process::threadCpuNanos();
        uint64_t streamSize{0};
        processStream(
            *streamData,
            encodingScratchBufferPool,
            encodingBufferPool,
            streamSize,
            chunkSize);
        if (statsCollector) {
          statsCollector->addPhysicalSize(streamSize);
        }
        encodingCpuNanos.fetch_add(
            velox::process::threadCpuNanos() - startCpuNs,
            std::memory_order_relaxed);
      }
    }
    resetFieldWriter();
  }

  velox::CpuWallTiming encodingTiming;
  encodingTiming.cpuNanos = encodingCpuNanos.load(std::memory_order_relaxed);
  encodingTiming.wallNanos = encodingWallNanos;
  context_->encodingTiming().add(encodingTiming);
  VLOG(1) << "writeChunk cpu: " << velox::succinctNanos(encodingTiming.cpuNanos)
          << ", wall: " << velox::succinctNanos(encodingWallNanos)
          << ", chunk size: " << velox::succinctBytes(chunkSize);
}

void Writer::encodeStream(
    StreamData& streamData,
    velox::BufferPool* encodingScratchBufferPool,
    EncodingBufferPool* encodingBufferPool,
    uint64_t& streamSize,
    std::atomic_uint64_t& chunkSize) {
  const auto offset = streamData.descriptor().offset();
  NIMBLE_DCHECK_LT(
      offset, encodedStreams_.size(), "Stream offset out of range.");
  auto& encodedStream = encodedStreams_[offset];
  // NOTE: we always expect the stream to be empty as encodeStream is only
  // used in non-chunked mode.
  NIMBLE_CHECK(encodedStream.chunks.empty());
  auto& chunk = encodedStream.chunks.emplace_back();
  const auto chunkBytes = encodeChunk(
      streamData, chunk, encodingScratchBufferPool, encodingBufferPool);
  streamSize += chunkBytes;
  chunkSize += chunkBytes;
  streamData.reset();
}

void Writer::processStream(
    StreamData& streamData,
    velox::BufferPool* encodingScratchBufferPool,
    EncodingBufferPool* encodingBufferPool,
    uint64_t& streamSize,
    std::atomic_uint64_t& chunkSize) {
  const auto offset = streamData.descriptor().offset();
  const auto* context = streamData.descriptor().context<WriterStreamContext>();
  NIMBLE_CHECK(encodedStreams_[offset].chunks.empty());
  if ((context != nullptr) && context->isNullStream()) {
    // For null streams we promote the null values to be written as
    // boolean data.
    if (streamData.hasNullValues()) {
      NullsAsDataStreamData nullsStreamData{streamData};
      encodeStream(
          nullsStreamData,
          encodingScratchBufferPool,
          encodingBufferPool,
          streamSize,
          chunkSize);
    }
  } else if (
      (context != nullptr) && context->isInMapStream() &&
      context_->options().skipConstantFlatMapInMapStreams) {
    // When enabled, skip encoding in-map streams that are all-true (every row
    // has the key) or all-false (no row has the key). The reader distinguishes
    // these by checking value stream presence: all-true keys have value
    // streams, all-false keys do not.
    //
    // NOTE: readers that don't infer missing in-map streams require
    // skipConstantFlatMapInMapStreams to remain false.
    streamData.materialize();
    if (!isConstantBoolStream(streamData.data())) {
      encodeStream(
          streamData,
          encodingScratchBufferPool,
          encodingBufferPool,
          streamSize,
          chunkSize);
    }
  } else {
    streamData.materialize();
    if (!streamData.data().empty()) {
      encodeStream(
          streamData,
          encodingScratchBufferPool,
          encodingBufferPool,
          streamSize,
          chunkSize);
    }
  }
}

bool Writer::encodeStreamChunk(
    StreamData& streamData,
    uint64_t minChunkSize,
    uint64_t maxChunkSize,
    bool ensureFullChunks,
    Stream& encodedStream,
    velox::BufferPool* encodingScratchBufferPool,
    EncodingBufferPool* encodingBufferPool,
    uint64_t& streamBytes,
    std::atomic_uint64_t& chunkBytes,
    std::atomic_uint64_t& logicalBytes) {
  bool writtenChunk{false};
  logicalBytes += streamData.memoryUsed();
  auto& streamChunks = encodedStream.chunks;
  auto chunker = getStreamChunker(
      streamData,
      StreamChunkerOptions{
          .minChunkSize = minChunkSize,
          .maxChunkSize = maxChunkSize,
          .ensureFullChunks = ensureFullChunks,
          .isFirstChunk = streamChunks.empty()});
  uint64_t encodedChunkBytes{0};
  while (auto chunkView = chunker->next()) {
    auto& streamChunk = streamChunks.emplace_back();
    encodedChunkBytes += encodeChunk(
        *chunkView, streamChunk, encodingScratchBufferPool, encodingBufferPool);
    writtenChunk = true;
  }
  streamBytes += encodedChunkBytes;
  chunkBytes += encodedChunkBytes;
  // Compact erases processed stream data to reclaim memory.
  chunker->compact();
  logicalBytes -= streamData.memoryUsed();
  return writtenChunk;
}

uint32_t Writer::encodeChunk(
    const StreamData& chunkView,
    Chunk& chunk,
    velox::BufferPool* encodingScratchBufferPool,
    EncodingBufferPool* encodingBufferPool) {
  std::string_view encoded = encodeStreamData(
      *context_,
      *encodingBuffer_,
      encodingScratchBufferPool,
      encodingBufferPool,
      chunkView);
  NIMBLE_DCHECK(!encoded.empty());
  if (encoded.empty()) {
    return 0;
  }
  uint32_t chunkBytes{0};
  chunk.rowCount = chunkView.rowCount();
  // Per-chunk null count, precomputed by the chunker.
  chunk.nullCount = static_cast<uint32_t>(chunkView.numNulls());
  ChunkedStreamWriter chunkWriter{
      *encodingBuffer_, context_->options().chunkCompression};
  for (auto& buffer : chunkWriter.encode(encoded)) {
    chunkBytes += buffer.size();
    chunk.content.push_back(std::move(buffer));
  }
  return chunkBytes;
}

bool Writer::writeChunks(
    std::span<const uint32_t> streamIndices,
    bool ensureFullChunks,
    bool lastChunk) {
  std::atomic_uint64_t chunkBytes{0};
  std::atomic_uint64_t logicalBytes{0};
  std::atomic_bool writtenChunk{false};
  std::atomic_uint64_t encodingCpuNanos{0};
  uint64_t encodingWallNanos{0};
  {
    LoggingScope scope{*context_->logger()};
    velox::NanosecondTimer wallTimer{&encodingWallNanos};
    const auto& options = context_->options();
    const auto minChunkSize = lastChunk ? 0 : options.minStreamChunkRawSize;
    const auto schemaNodeCount = context_->schemaBuilder().nodeCount();
    const auto maxChunkSize = schemaNodeCount > options.largeSchemaThreshold
        ? options.wideSchemaMaxStreamChunkRawSize
        : options.maxStreamChunkRawSize;
    ensureWriteStreams();

    const auto& streams = context_->streams();
    const auto streamCount = static_cast<uint32_t>(streamIndices.size());
    const auto concurrency = encodingConcurrency(streamCount);
    ensureEncodingScratchBufferPools(concurrency);
    ensureEncodingBufferPools(concurrency);

    if (concurrency > 1) {
      const auto& encodingExecutor = context_->options().encodingExecutor;
      NIMBLE_CHECK(
          encodingExecutor,
          "Encoding executor is required for parallel encoding.");
      for (uint32_t start = 0; start < streamCount; start += concurrency) {
        velox::dwio::common::ExecutorBarrier barrier{encodingExecutor};
        const auto batchSize = std::min(concurrency, streamCount - start);
        for (uint32_t index = 0; index < batchSize; ++index) {
          const auto streamIndex = streamIndices[start + index];
          auto& [nodeId, streamData] = streams[streamIndex];
          const auto offset = streamData->descriptor().offset();
          auto* encodedStream = &encodedStreams_[offset];
          auto* encodingScratchBufferPool =
              this->encodingScratchBufferPool(index);
          auto* encodingBufferPool = this->encodingBufferPool(index);
          barrier.add([&,
                       streamDataPtr = streamData.get(),
                       encodedStream,
                       encodingScratchBufferPool,
                       encodingBufferPool,
                       statsCollector = context_->getStatsCollector(nodeId)] {
            uint64_t startCpuNs = velox::process::threadCpuNanos();
            uint64_t streamSize = 0;
            if (encodeStreamChunk(
                    *streamDataPtr,
                    minChunkSize,
                    maxChunkSize,
                    ensureFullChunks,
                    *encodedStream,
                    encodingScratchBufferPool,
                    encodingBufferPool,
                    streamSize,
                    chunkBytes,
                    logicalBytes)) {
              writtenChunk = true;
            }
            if (statsCollector) {
              statsCollector->addPhysicalSize(streamSize);
            }
            encodingCpuNanos.fetch_add(
                velox::process::threadCpuNanos() - startCpuNs,
                std::memory_order_relaxed);
          });
        }
        barrier.waitAll();
      }
    } else {
      auto* encodingScratchBufferPool = this->encodingScratchBufferPool();
      auto* encodingBufferPool = this->encodingBufferPool();
      for (auto streamIndex : streamIndices) {
        auto& [nodeId, streamData] = streams[streamIndex];
        const auto offset = streamData->descriptor().offset();
        auto statsCollector = context_->getStatsCollector(nodeId);
        uint64_t startCpuNs = velox::process::threadCpuNanos();
        uint64_t streamSize = 0;
        if (encodeStreamChunk(
                *streamData,
                minChunkSize,
                maxChunkSize,
                ensureFullChunks,
                encodedStreams_[offset],
                encodingScratchBufferPool,
                encodingBufferPool,
                streamSize,
                chunkBytes,
                logicalBytes)) {
          writtenChunk = true;
        }
        if (statsCollector) {
          statsCollector->addPhysicalSize(streamSize);
        }
        encodingCpuNanos.fetch_add(
            velox::process::threadCpuNanos() - startCpuNs,
            std::memory_order_relaxed);
      }
    }

    if (lastChunk) {
      resetFieldWriter();
    }

    context_->updateStripeEncodedPhysicalSize(chunkBytes);
    context_->updateStripeEncodedLogicalSize(logicalBytes);
    context_->updateMemoryUsed(-logicalBytes);
  }

  velox::CpuWallTiming encodingTiming;
  encodingTiming.cpuNanos = encodingCpuNanos.load(std::memory_order_relaxed);
  encodingTiming.wallNanos = encodingWallNanos;
  context_->encodingTiming().add(encodingTiming);
  if (writtenChunk) {
    context_->recordChunkSize(chunkBytes);
  }
  VLOG(1) << "writeChunk cpu: " << velox::succinctNanos(encodingTiming.cpuNanos)
          << ", wall: " << velox::succinctNanos(encodingWallNanos)
          << ", chunk size: " << velox::succinctBytes(chunkBytes);
  return writtenChunk;
}

bool Writer::flushChunks(
    const std::vector<uint32_t>& indices,
    bool ensureFullChunks,
    FlushPolicy* flushPolicy) {
  const size_t indicesCount = indices.size();
  const auto batchSize = context_->options().chunkedStreamBatchSize;
  for (size_t index = 0; index < indicesCount; index += batchSize) {
    const size_t currentBatchSize = std::min(batchSize, indicesCount - index);
    std::span<const uint32_t> batchIndices(
        indices.begin() + index, currentBatchSize);
    // Stop attempting chunking once streams are too small to chunk or
    // memory pressure is relieved.
    if (!writeChunks(batchIndices, ensureFullChunks) ||
        !shouldChunk(flushPolicy)) {
      return false;
    }
  }
  return true;
}

bool Writer::writeStripe() {
  if (context_->rowsInStripe() == 0) {
    return false;
  }

  if (context_->options().enableChunking) {
    // Chunk all streams.
    std::vector<uint32_t> streamIndices(context_->streams().size());
    std::iota(streamIndices.begin(), streamIndices.end(), 0);
    writeChunks(streamIndices, /*ensureFullChunks=*/false, /*lastChunk=*/true);
  } else {
    writeStreams();
  }
  if (context_->options().enableStatsCollection) {
    context_->finalizeStripeStatsCollectors();
  }

  writeStripeDictionaryStreams();

  uint64_t stripeSize{0};
  {
    LoggingScope scope{*context_->logger()};

    size_t nonEmptyCount{0};
    for (auto i = 0; i < encodedStreams_.size(); ++i) {
      auto& source = encodedStreams_[i];
      if (!source.chunks.empty()) {
        source.offset = i;
        if (nonEmptyCount != i) {
          encodedStreams_[nonEmptyCount] = std::move(source);
        }
        ++nonEmptyCount;
      }
    }
    encodedStreams_.resize(nonEmptyCount);

    const uint64_t startSize = tabletWriter_->size();
    {
      velox::CpuWallTimer writeTimer{context_->writeTiming()};
      tabletWriter_->writeStripe(
          context_->rowsInStripe(), std::move(encodedStreams_));
    }
    stripeSize = tabletWriter_->size() - startSize;
    clearEncodingBuffer();
    // TODO: once chunked string fields are supported, move string buffer
    // reset to writeStreams()
    context_->resetStringBuffer();
  }

  NIMBLE_CHECK_LT(
      stripeSize,
      std::numeric_limits<uint32_t>::max(),
      "unexpected stripe size");

  VLOG(1) << "on disk stripe size: " << velox::succinctBytes(stripeSize);

  StripeFlushMetrics metrics{
      .inputSize = context_->stripeEncodedPhysicalSize(),
      .rowCount = context_->rowsInStripe(),
      .stripeSize = stripeSize,
      .trackedMemory = context_->memoryUsed(),
  };
  context_->logger()->logStripeFlush(metrics);
  context_->nextStripe();
  return true;
}

bool Writer::evaluateFlushPolicy() {
  // NOTE that flush policy factory is stateful, so we need to get a new
  // policy every time we check.
  auto flushPolicy = context_->options().flushPolicyFactory();
  if (context_->options().enableChunking && shouldChunk(flushPolicy.get())) {
    // Relieve memory pressure by chunking streams above max size.
    const auto& streams = context_->streams();
    std::vector<uint32_t> streamIndices;
    const auto streamCount = streams.size();
    streamIndices.reserve(streamCount);

    // Determine size threshold for soft chunking based on schema width.
    const auto& options = context_->options();
    const auto maxChunkSize = streamCount > options.largeSchemaThreshold
        ? options.wideSchemaMaxStreamChunkRawSize
        : options.maxStreamChunkRawSize;
    for (auto streamIndex = 0; streamIndex < streams.size(); ++streamIndex) {
      if (streams[streamIndex].second->memoryUsed() >= maxChunkSize) {
        streamIndices.push_back(streamIndex);
      }
    }

    // Soft chunking.
    const bool continueChunking = flushChunks(
        streamIndices, /*ensureFullChunks=*/true, flushPolicy.get());
    // Hard chunking when chunking streams above maxChunkSize fails to
    // relieve memory pressure.
    if (continueChunking) {
      // Relieve memory pressure by chunking small streams.
      // Sort streams for chunking based on raw memory usage.
      // TODO(T240072104): Improve performance by bucketing the streams
      // by size (by most significant bit) instead of sorting them.
      // Only sort streams above minChunkSize.
      streamIndices.resize(streams.size());
      std::iota(streamIndices.begin(), streamIndices.end(), 0);
      std::sort(
          streamIndices.begin(),
          streamIndices.end(),
          [&](const uint32_t& a, const uint32_t& b) {
            return streams[a].second->memoryUsed() >
                streams[b].second->memoryUsed();
          });
      flushChunks(streamIndices, /*ensureFullChunks=*/false, flushPolicy.get());
    }
  }

  if (!shouldFlush(flushPolicy.get())) {
    return false;
  }
  return writeStripe();
}

namespace {

velox::RuntimeMetric nanosMetric(uint64_t value) {
  return velox::RuntimeMetric(
      velox::saturateCast(value), velox::RuntimeCounter::Unit::kNanos);
}

velox::RuntimeMetric bytesMetric(uint64_t value) {
  return velox::RuntimeMetric(
      velox::saturateCast(value), velox::RuntimeCounter::Unit::kBytes);
}

velox::RuntimeMetric countMetric(uint64_t value) {
  return velox::RuntimeMetric(velox::saturateCast(value));
}

} // namespace

folly::F14FastMap<std::string, velox::RuntimeMetric> Writer::runtimeStats()
    const {
  using Keys = Writer::RuntimeStats;
  const auto& tabletStats = tabletWriter_->stats();
  return {
      {std::string{Keys::kWrittenBytes}, bytesMetric(context_->bytesWritten())},
      {std::string{Keys::kInputBytes}, bytesMetric(context_->fileRawSize())},
      {std::string{Keys::kWriteCpuNanos},
       nanosMetric(context_->writeTiming().cpuNanos)},
      {std::string{Keys::kWriteWallNanos},
       nanosMetric(context_->writeTiming().wallNanos)},
      {std::string{Keys::kIngestionCpuNanos},
       nanosMetric(context_->ingestionTiming().cpuNanos)},
      {std::string{Keys::kEncodingCpuNanos},
       nanosMetric(context_->encodingTiming().cpuNanos)},
      {std::string{Keys::kEncodingWallNanos},
       nanosMetric(context_->encodingTiming().wallNanos)},
      {std::string{Keys::kEncodingSelectionCpuNanos},
       nanosMetric(context_->encodingSelectionTiming().cpuNanos)},
      // Distributions are already accumulated as metrics, so they are published
      // verbatim rather than collapsed to a single value.
      {std::string{Keys::kRowsPerStripe},
       toRuntimeMetric(context_->rowsPerStripe())},
      {std::string{Keys::kChunkSizeBytes}, context_->chunkSizeStats()},
      {std::string{Keys::kDuplicateStreamCount},
       countMetric(tabletStats.duplicateStreamCount)},
      {std::string{Keys::kDuplicateStreamBytes},
       bytesMetric(tabletStats.duplicateStreamBytes)},
  };
}

std::vector<ColumnStatistics*> Writer::columnStats() const {
  return context_->columnStats();
}

velox::RuntimeMetric runtimeStat(
    const folly::F14FastMap<std::string, velox::RuntimeMetric>& stats,
    std::string_view key) {
  // F14 keyed by std::string has no heterogeneous lookup, so the key is
  // materialized here, as velox::exec::TableWriter does when it reads these
  // maps.
  const auto it = stats.find(std::string{key});
  return it != stats.end() ? it->second : velox::RuntimeMetric{};
}

} // namespace facebook::nimble
