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
#include <iomanip>
#include <locale>

#include "velox/common/file/FileSystems.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/index/ClusterIndex.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/FileLayout.h"
#include "velox/dwio/nimble/tools/EncodingUtilities.h"
#include "velox/dwio/nimble/tools/NimbleDslLib.h"
#include "velox/dwio/nimble/velox/StreamLabels.h"
#include "velox/dwio/nimble/velox/VeloxReader.h"
#include "velox/dwio/nimble/velox/stats/ColumnStatistics.h"
#include "velox/dwio/nimble/velox/stats/VectorizedStatistics.h"

namespace facebook::nimble::tools {

#undef RED
#define RED(enableColor) (enableColor ? "\033[31m" : "")
#undef GREEN
#define GREEN(enableColor) (enableColor ? "\033[32m" : "")
#undef YELLOW
#define YELLOW(enableColor) (enableColor ? "\033[33m" : "")
#undef BLUE
#define BLUE(enableColor) (enableColor ? "\033[34m" : "")
#undef MAGENTA
#define MAGENTA(enableColor) (enableColor ? "\033[35m" : "")
#undef CYAN
#define CYAN(enableColor) (enableColor ? "\033[36m" : "")
#undef BOLD
#define BOLD(enableColor) (enableColor ? "\033[1m" : "")
#undef DIM
#define DIM(enableColor) (enableColor ? "\033[2m" : "")
#undef RESET_COLOR
#define RESET_COLOR(enableColor) (enableColor ? "\033[0m" : "")

namespace {

enum class Alignment {
  Left,
  Right,
};

class TableFormatter {
 public:
  TableFormatter(
      std::ostream& ostream,
      bool enableColor,
      std::vector<std::tuple<std::string, uint8_t, Alignment>> fields,
      const std::string& separator = "\t")
      : ostream_{ostream},
        enableColor_{enableColor},
        fields_{std::move(fields)},
        separator_{separator} {
    // Print header in bold yellow.
    ostream << BOLD(enableColor) << YELLOW(enableColor);
    for (size_t idx = 0; idx < fields_.size(); ++idx) {
      const auto& field = fields_[idx];
      ostream << (std::get<2>(field) == Alignment::Right ? std::right
                                                         : std::left)
              << std::setw(std::get<1>(field)) << std::get<0>(field)
              << (idx + 1 < fields_.size() ? separator_ : "");
    }
    ostream << RESET_COLOR(enableColor) << std::endl;

    // Print separator line in dim.
    ostream << DIM(enableColor);
    for (size_t i = 0; i < fields_.size(); ++i) {
      ostream << std::string(std::get<1>(fields_[i]), '-');
      if (i != fields_.size() - 1) {
        ostream << std::string(separator_.size(), '-');
      }
    }
    ostream << RESET_COLOR(enableColor) << std::endl;
  }

  void writeRow(const std::vector<std::string>& values) {
    if (fields_.empty()) {
      return;
    }
    for (size_t i = 0; i < values.size() && i < fields_.size(); ++i) {
      const auto& val = values[i];
      // Colorize cell values.
      if (val == "NULL") {
        ostream_ << RED(enableColor_);
      } else if (val == "N/A") {
        ostream_ << DIM(enableColor_);
      }
      ostream_ << (std::get<2>(fields_[i]) == Alignment::Right ? std::right
                                                               : std::left)
               << std::setw(std::get<1>(fields_[i])) << val;
      if (val == "NULL" || val == "N/A") {
        ostream_ << RESET_COLOR(enableColor_);
      }
      ostream_ << (i != values.size() - 1 ? separator_ : "");
    }
    ostream_ << std::endl;
  }

 private:
  std::ostream& ostream_;
  bool enableColor_;
  std::vector<std::tuple<std::string, uint8_t, Alignment>> fields_;
  const std::string separator_;
};

template <typename T>
auto commaSeparated(T value) {
  try {
    return fmt::format(std::locale("en_US.UTF-8"), "{:L}", value);
  } catch (const std::runtime_error&) {
    return fmt::format("{}", value);
  }
}

struct GroupingKey {
  EncodingType encodingType;
  DataType dataType;
  std::optional<CompressionType> compressinType;
};

struct GroupingKeyHash {
  size_t operator()(const GroupingKey& key) const {
    size_t h1 = std::hash<EncodingType>()(key.encodingType);
    size_t h2 = std::hash<DataType>()(key.dataType);
    size_t h3 = std::hash<std::optional<CompressionType>>()(key.compressinType);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

struct GroupingKeyEqual {
  bool operator()(const GroupingKey& lhs, const GroupingKey& rhs) const {
    return lhs.encodingType == rhs.encodingType &&
        lhs.dataType == rhs.dataType &&
        lhs.compressinType == rhs.compressinType;
  }
};

struct EncodingHistogramValue {
  size_t count;
  size_t bytes;
};

struct HistogramRowCompare {
  size_t operator()(
      const std::unordered_map<GroupingKey, EncodingHistogramValue>::
          const_iterator& lhs,
      const std::unordered_map<GroupingKey, EncodingHistogramValue>::
          const_iterator& rhs) const {
    const auto lhsEncoding = lhs->first.encodingType;
    const auto rhsEncoding = rhs->first.encodingType;
    const auto lhsSize = lhs->second.bytes;
    const auto rhsSize = rhs->second.bytes;
    if (lhsEncoding != rhsEncoding) {
      return lhsEncoding < rhsEncoding;
    } else {
      return lhsSize > rhsSize;
    }
  }
};

constexpr uint32_t kContentBufferSize = 1000;

template <typename T>
void printScalarData(
    std::ostream& ostream,
    velox::memory::MemoryPool& pool,
    Encoding& stream,
    uint32_t rowCount,
    const std::string& separator) {
  nimble::Vector<T> buffer(&pool);
  nimble::Vector<char> nulls(&pool);
  buffer.resize(rowCount);
  nulls.resize((nimble::FixedBitArray::bufferSize(rowCount, 1)));
  nulls.zero_out();
  uint32_t nonNullCount = rowCount;
  if (stream.isNullable()) {
    nonNullCount = stream.materializeNullable(
        rowCount, buffer.data(), [&]() { return nulls.data(); });
  } else {
    stream.materialize(rowCount, buffer.data());
  }

  if (nonNullCount == rowCount) {
    for (uint32_t i = 0; i < rowCount; ++i) {
      ostream << folly::to<std::string>(buffer[i]) << separator;
    }
  } else {
    for (uint32_t i = 0; i < rowCount; ++i) {
      if (velox::bits::isBitSet(
              reinterpret_cast<const uint8_t*>(nulls.data()), i) == 0) {
        ostream << "NULL" << separator;
      } else {
        ostream << folly::to<std::string>(buffer[i]) << separator;
      }
    }
  }
}

void printScalarType(
    std::ostream& ostream,
    velox::memory::MemoryPool& pool,
    Encoding& stream,
    uint32_t rowCount,
    const std::string& separator) {
  switch (stream.dataType()) {
#define CASE(KIND, cppType)                                               \
  case DataType::KIND: {                                                  \
    printScalarData<cppType>(ostream, pool, stream, rowCount, separator); \
    break;                                                                \
  }
    CASE(Int8, int8_t);
    CASE(Uint8, uint8_t);
    CASE(Int16, int16_t);
    CASE(Uint16, uint16_t);
    CASE(Int32, int32_t);
    CASE(Uint32, uint32_t);
    CASE(Int64, int64_t);
    CASE(Uint64, uint64_t);
    CASE(Float, float);
    CASE(Double, double);
    CASE(Bool, bool);
    CASE(String, std::string_view);
#undef CASE
    case DataType::Undefined: {
      NIMBLE_UNREACHABLE(
          fmt::format("Undefined type for stream: {}", stream.dataType()));
    }
  }
}

std::string getMinMaxString(const ColumnStatistics& stat) {
  switch (stat.getType()) {
    case StatType::INTEGRAL: {
      const auto* s = stat.as<IntegralStatistics>();
      auto minVal = s->getMin();
      return minVal.has_value() ? std::to_string(*minVal) : "N/A";
    }
    case StatType::FLOATING_POINT: {
      const auto* s = stat.as<FloatingPointStatistics>();
      auto minVal = s->getMin();
      return minVal.has_value() ? fmt::format("{:.4g}", *minVal) : "N/A";
    }
    case StatType::STRING: {
      const auto* s = stat.as<StringStatistics>();
      auto minVal = s->getMin();
      return minVal.has_value() ? *minVal : "N/A";
    }
    case StatType::DEFAULT:
    case StatType::DEDUPLICATED:
      return "N/A";
  }
}

std::string getMaxString(const ColumnStatistics& stat) {
  switch (stat.getType()) {
    case StatType::INTEGRAL: {
      const auto* s = stat.as<IntegralStatistics>();
      auto maxVal = s->getMax();
      return maxVal.has_value() ? std::to_string(*maxVal) : "N/A";
    }
    case StatType::FLOATING_POINT: {
      const auto* s = stat.as<FloatingPointStatistics>();
      auto maxVal = s->getMax();
      return maxVal.has_value() ? fmt::format("{:.4g}", *maxVal) : "N/A";
    }
    case StatType::STRING: {
      const auto* s = stat.as<StringStatistics>();
      auto maxVal = s->getMax();
      return maxVal.has_value() ? *maxVal : "N/A";
    }
    case StatType::DEFAULT:
    case StatType::DEDUPLICATED:
      return "N/A";
  }
}

std::string statTypeName(StatType type) {
  switch (type) {
    case StatType::INTEGRAL:
      return "INTEGRAL";
    case StatType::FLOATING_POINT:
      return "FLOAT";
    case StatType::STRING:
      return "STRING";
    case StatType::DEDUPLICATED:
      return "DEDUP";
    case StatType::DEFAULT:
      return "DEFAULT";
  }
}

} // namespace

NimbleDslLib::NimbleDslLib(
    std::ostream& ostream,
    bool enableColors,
    const std::string& filePath)
    : pool_{velox::memory::deprecatedAddDefaultLeafMemoryPool()},
      file_{velox::filesystems::getFileSystem(filePath, nullptr)
                ->openFileForRead(filePath)},
      enableColors_{enableColors},
      ostream_{ostream},
      dumpLib_{file_, enableColors, ostream} {}

NimbleDslLib::NimbleDslLib(
    std::ostream& ostream,
    bool enableColors,
    std::shared_ptr<velox::ReadFile> file)
    : pool_{velox::memory::deprecatedAddDefaultLeafMemoryPool()},
      file_{std::move(file)},
      enableColors_{enableColors},
      ostream_{ostream},
      dumpLib_{file_, enableColors, ostream} {}

void NimbleDslLib::select(
    const std::vector<std::string>& columns,
    uint64_t limit,
    uint64_t offset,
    std::optional<uint32_t> stripeId) {
  TabletReader::Options tabletOptions;
  tabletOptions.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  auto tablet = TabletReader::create(file_, pool_.get(), tabletOptions);

  // First create a reader without projection to get the full schema.
  VeloxReader schemaReader{tablet, *pool_};
  const auto& fullType = schemaReader.type();

  // If a stripe is specified, compute the row range.
  uint64_t effectiveOffset = offset;
  uint64_t maxRows = limit;
  if (stripeId.has_value()) {
    auto stripe = stripeId.value();
    if (stripe >= tablet->stripeCount()) {
      ostream_ << RED(enableColors_) << "Error: Stripe " << stripe
               << " out of range. File has " << tablet->stripeCount()
               << " stripes." << RESET_COLOR(enableColors_) << std::endl;
      return;
    }
    // Compute the starting row for the given stripe.
    uint64_t stripeStartRow = 0;
    for (uint32_t i = 0; i < stripe; ++i) {
      stripeStartRow += tablet->stripeRowCount(i);
    }
    uint64_t stripeRows = tablet->stripeRowCount(stripe);
    effectiveOffset = stripeStartRow + offset;
    maxRows = std::min(limit, stripeRows - std::min(offset, stripeRows));
  }

  // Build column selector for projection.
  std::shared_ptr<const velox::dwio::common::ColumnSelector> selector;
  if (!columns.empty()) {
    // Validate column names.
    for (const auto& col : columns) {
      if (!fullType->containsChild(col)) {
        ostream_ << RED(enableColors_) << "Error: Unknown column '" << col
                 << "'. Use DESCRIBE to "
                 << "see available columns." << RESET_COLOR(enableColors_)
                 << std::endl;
        return;
      }
    }
    selector = std::make_shared<velox::dwio::common::ColumnSelector>(
        fullType, columns);
  }

  VeloxReader reader{tablet, *pool_, selector};
  const auto& projectedType = reader.type();

  if (effectiveOffset > 0) {
    reader.seekToRow(effectiveOffset);
  }

  // Build column indices to extract from the result vector.
  // When using projection, the result RowVector may have children indexed
  // by the full schema. Map projected column names to their indices.
  std::vector<std::string> displayNames;
  displayNames.reserve(projectedType->size());
  for (uint32_t i = 0; i < projectedType->size(); ++i) {
    displayNames.push_back(projectedType->nameOf(i));
  }

  // Print header.
  std::vector<std::tuple<std::string, uint8_t, Alignment>> fields;
  fields.reserve(displayNames.size());
  for (const auto& name : displayNames) {
    fields.emplace_back(name, 15, Alignment::Left);
  }
  TableFormatter formatter{ostream_, enableColors_, std::move(fields)};

  // Read and print rows.
  constexpr uint64_t kBatchSize = 1000;
  uint64_t rowsPrinted = 0;
  velox::VectorPtr result;
  while (rowsPrinted < maxRows && reader.next(kBatchSize, result)) {
    auto* rowVector = result->as<velox::RowVector>();
    auto resultType =
        std::dynamic_pointer_cast<const velox::RowType>(rowVector->type());
    auto numRows = std::min(
        static_cast<uint64_t>(rowVector->size()), maxRows - rowsPrinted);
    for (uint64_t row = 0; row < numRows; ++row) {
      std::vector<std::string> values;
      auto vecRow = static_cast<velox::vector_size_t>(row);
      for (const auto& colName : displayNames) {
        auto idx = resultType->getChildIdxIfExists(colName);
        if (!idx.has_value()) {
          values.emplace_back("N/A");
          continue;
        }
        auto child = rowVector->childAt(idx.value());
        if (!child || child->isNullAt(vecRow)) {
          values.emplace_back("NULL");
        } else {
          values.emplace_back(child->toString(vecRow));
        }
      }
      formatter.writeRow(values);
    }
    rowsPrinted += numRows;
  }
  ostream_ << DIM(enableColors_) << "(" << rowsPrinted << " rows)"
           << RESET_COLOR(enableColors_) << std::endl;
}

void NimbleDslLib::describe() {
  TabletReader::Options tabletOptions;
  tabletOptions.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  auto tablet = TabletReader::create(file_, pool_.get(), tabletOptions);
  VeloxReader reader{tablet, *pool_};
  const auto& veloxType = reader.type();

  TableFormatter formatter{
      ostream_,
      enableColors_,
      {{"Column", 25, Alignment::Left},
       {"Type", 20, Alignment::Left},
       {"Stream", 10, Alignment::Left}}};

  // Walk the nimble schema to find stream offsets for each top-level column.
  const auto& schema = reader.schema();
  // The root is a Row type. Each direct child is a top-level column.
  if (!schema->isRow()) {
    return;
  }
  const auto& rootRow = schema->asRow();
  for (uint32_t i = 0; i < rootRow.childrenCount(); ++i) {
    auto childName = std::string(rootRow.nameAt(i));
    const auto& childType = rootRow.childAt(i);

    // Build stream offset string.
    std::string offsets;
    switch (childType->kind()) {
      case Kind::Scalar:
        offsets =
            std::to_string(childType->asScalar().scalarDescriptor().offset());
        break;
      case Kind::Array:
        offsets =
            std::to_string(childType->asArray().lengthsDescriptor().offset());
        break;
      case Kind::Map:
        offsets =
            std::to_string(childType->asMap().lengthsDescriptor().offset());
        break;
      case Kind::Row:
        offsets = std::to_string(childType->asRow().nullsDescriptor().offset());
        break;
      case Kind::FlatMap:
        offsets =
            std::to_string(childType->asFlatMap().nullsDescriptor().offset());
        break;
      case Kind::TimestampMicroNano: {
        const auto& ts = childType->asTimestampMicroNano();
        offsets = "m:" + std::to_string(ts.microsDescriptor().offset()) +
            ",n:" + std::to_string(ts.nanosDescriptor().offset());
        break;
      }
      case Kind::ArrayWithOffsets: {
        const auto& a = childType->asArrayWithOffsets();
        offsets = "o:" + std::to_string(a.offsetsDescriptor().offset()) +
            ",l:" + std::to_string(a.lengthsDescriptor().offset());
        break;
      }
      case Kind::SlidingWindowMap: {
        const auto& s = childType->asSlidingWindowMap();
        offsets = "o:" + std::to_string(s.offsetsDescriptor().offset()) +
            ",l:" + std::to_string(s.lengthsDescriptor().offset());
        break;
      }
    }

    // Get the velox type string for display.
    std::string typeStr;
    if (i < veloxType->size()) {
      typeStr = veloxType->childAt(i)->toString();
    } else {
      typeStr = toString(childType->kind());
    }

    formatter.writeRow(
        {childName,
         std::string(GREEN(enableColors_)) + typeStr +
             RESET_COLOR(enableColors_),
         std::string(BLUE(enableColors_)) + "[" + offsets + "]" +
             RESET_COLOR(enableColors_)});
  }
}

void NimbleDslLib::showSchema() {
  dumpLib_.emitSchema(/*collapseFlatMap=*/false);
}

void NimbleDslLib::showInfo() {
  dumpLib_.emitInfo();
}

void NimbleDslLib::showStats() {
  TabletReader::Options tabletOptions;
  tabletOptions.preloadOptionalSections = {
      std::string(kVectorizedStatsSection)};
  tabletOptions.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  auto tablet = TabletReader::create(file_, pool_.get(), tabletOptions);
  VeloxReader reader{tablet, *pool_};

  auto statsSection =
      tablet->loadOptionalSection(tabletOptions.preloadOptionalSections[0]);
  if (!statsSection.has_value()) {
    ostream_ << YELLOW(enableColors_)
             << "No vectorized statistics available in this file."
             << RESET_COLOR(enableColors_) << std::endl;
    return;
  }

  auto fileStats =
      VectorizedFileStats::deserialize(statsSection->content(), *pool_);
  if (!fileStats) {
    ostream_ << RED(enableColors_) << "Failed to deserialize statistics."
             << RESET_COLOR(enableColors_) << std::endl;
    return;
  }

  auto columnStats =
      fileStats->toColumnStatistics(reader.type(), reader.schema());

  // Build labels from schema for readable column paths.
  StreamLabels labels{reader.schema()};

  TableFormatter formatter{
      ostream_,
      enableColors_,
      {{"Column", 25, Alignment::Left},
       {"Type", 10, Alignment::Left},
       {"Values", 12, Alignment::Right},
       {"Nulls", 10, Alignment::Right},
       {"Min", 15, Alignment::Left},
       {"Max", 15, Alignment::Left},
       {"LogicalSize", 14, Alignment::Right},
       {"PhysicalSize", 14, Alignment::Right}}};

  for (size_t i = 0; i < columnStats.size(); ++i) {
    const auto& stat = columnStats[i];
    if (!stat) {
      continue;
    }
    formatter.writeRow({
        std::string(labels.streamLabel(static_cast<uint32_t>(i))),
        statTypeName(stat->getType()),
        commaSeparated(stat->getValueCount()),
        commaSeparated(stat->getNullCount()),
        getMinMaxString(*stat),
        getMaxString(*stat),
        commaSeparated(stat->getLogicalSize()),
        commaSeparated(stat->getPhysicalSize()),
    });
  }
}

void NimbleDslLib::showStripes() {
  dumpLib_.emitStripes(/*noHeader=*/false);
}

void NimbleDslLib::showStreams(std::optional<uint32_t> stripeId) {
  dumpLib_.emitStreams(
      /*noHeader=*/false,
      /*flatmapKeys=*/true,
      /*rawSize=*/false,
      /*showInMapStream=*/false,
      stripeId);
}

void NimbleDslLib::showHistogram(
    bool topLevel,
    std::optional<uint32_t> stripeId) {
  dumpLib_.emitHistogram(topLevel, /*noHeader=*/false, stripeId);
}

void NimbleDslLib::showContent(
    uint32_t streamId,
    std::optional<uint32_t> stripeId) {
  try {
    dumpLib_.emitContent(streamId, stripeId, "\n");
  } catch (const std::exception& e) {
    ostream_ << RED(enableColors_) << "Error: " << e.what()
             << RESET_COLOR(enableColors_) << std::endl;
  }
}

void NimbleDslLib::showFileLayout() {
  dumpLib_.emitFileLayout(/*noHeader=*/false);
}

void NimbleDslLib::showIndex() {
  dumpLib_.emitIndex();
}

void NimbleDslLib::showStripesMetadata() {
  dumpLib_.emitStripesMetadata(/*noHeader=*/false);
}

void NimbleDslLib::showStripeGroupsMetadata() {
  dumpLib_.emitStripeGroupsMetadata(/*noHeader=*/false);
}

void NimbleDslLib::showOptionalSections() {
  dumpLib_.emitOptionalSectionsMetadata(/*noHeader=*/false);
}

void NimbleDslLib::showEncoding(std::optional<uint32_t> stripeId) {
  TabletReader::Options tabletOptions;
  tabletOptions.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  auto tablet = TabletReader::create(file_, pool_.get(), tabletOptions);
  VeloxReader reader{tablet, *pool_};
  StreamLabels labels{reader.schema()};

  TableFormatter formatter{
      ostream_,
      enableColors_,
      {{"Stripe Id", 9, Alignment::Left},
       {"Stream Id", 9, Alignment::Left},
       {"Stream Label", 25, Alignment::Left},
       {"Encoding", 60, Alignment::Left}}};

  if (tablet->stripeCount() == 0) {
    return;
  }

  uint32_t startStripe = stripeId.value_or(0);
  uint32_t endStripe = stripeId.value_or(tablet->stripeCount() - 1);

  std::optional<StripeIdentifier> stripeIdentifier;
  for (uint32_t i = startStripe; i <= endStripe; ++i) {
    stripeIdentifier = tablet->stripeIdentifier(i);
    std::vector<uint32_t> streamIdentifiers(
        tablet->streamCount(stripeIdentifier.value()));
    std::iota(streamIdentifiers.begin(), streamIdentifiers.end(), 0);
    auto streams = tablet->load(
        stripeIdentifier.value(),
        {streamIdentifiers.cbegin(), streamIdentifiers.cend()});
    for (uint32_t j = 0; j < streams.size(); ++j) {
      auto& stream = streams[j];
      if (stream) {
        InMemoryChunkedStream chunkedStream{*pool_, std::move(stream)};
        auto encodingLabel = getStreamInputLabel(chunkedStream);
        formatter.writeRow({
            std::to_string(i),
            std::to_string(j),
            std::string(labels.streamLabel(j)),
            encodingLabel,
        });
      }
    }
  }
}

} // namespace facebook::nimble::tools
