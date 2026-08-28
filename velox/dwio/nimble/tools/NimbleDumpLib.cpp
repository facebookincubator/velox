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
#include <algorithm>
#include <functional>
#include <locale>
#include <numeric>
#include <ostream>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <zstd.h>
#include "folly/cli/NestedCommandLineApp.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingUtils.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/FileLayout.h"
#include "velox/dwio/nimble/tools/EncodingUtilities.h"
#include "velox/dwio/nimble/tools/NimbleDumpLib.h"
#include "velox/dwio/nimble/velox/BatchReader.h"
#include "velox/dwio/nimble/velox/StatsGenerated.h"
#include "velox/dwio/nimble/velox/stats/ColumnStatistics.h"
#include "velox/dwio/nimble/velox/stats/VectorizedStatistics.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"

namespace facebook::nimble::tools {
#undef RED
#define RED(enableColor) (enableColor ? "\033[31m" : "")
#undef GREEN
#define GREEN(enableColor) (enableColor ? "\033[32m" : "")
#undef YELLOW
#define YELLOW(enableColor) (enableColor ? "\033[33m" : "")
#undef BLUE
#define BLUE(enableColor) (enableColor ? "\033[34m" : "")
#undef PURPLE
#define PURPLE(enableColor) (enableColor ? "\033[35m" : "")
#undef CYAN
#define CYAN(enableColor) (enableColor ? "\033[36m" : "")
#undef RESET_COLOR
#define RESET_COLOR(enableColor) (enableColor ? "\033[0m" : "")

namespace {

constexpr uint32_t kBufferSize = 1000;
constexpr int kRowCountOffset = 2;
constexpr int kPrefixSize = 6;
constexpr int kCompressionTypeSize = 1;

uint64_t getRawDataSize(
    velox::memory::MemoryPool& memoryPool,
    std::string_view encodingStr) {
  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, &memoryPool));
    return buffer->asMutable<void>();
  };

  auto encoding = EncodingFactory().create(
      memoryPool, encodingStr, stringBufferFactory, Encoding::Options{});
  EncodingType encodingType = encoding->encodingType();
  DataType dataType = encoding->dataType();
  uint32_t rowCount = encoding->rowCount();

  if (encodingType == EncodingType::Sentinel) {
    NIMBLE_UNSUPPORTED("Sentinel encoding is not supported");
  }

  if (encodingType == EncodingType::Nullable) {
    auto pos = encodingStr.data() + kPrefixSize;
    auto nonNullsSize = encoding::readUint32(pos);
    // Sum of the null count and size of the non-null child encoding.
    return getRawDataSize(memoryPool, {pos, nonNullsSize}) + rowCount;
  }

  if (dataType != DataType::String) {
    auto typeSize = nimble::detail::dataTypeSize(dataType);
    return typeSize * rowCount;
  }

  auto pos = encodingStr.data() + kPrefixSize; // Skip the prefix.
  uint64_t result = 0;

  switch (encodingType) {
    case EncodingType::Trivial: {
      pos += kCompressionTypeSize;
      auto lengthsSize = encoding::readUint32(pos);
      auto lengths = EncodingFactory().create(
          memoryPool,
          {pos, lengthsSize},
          stringBufferFactory,
          Encoding::Options{});
      std::vector<uint32_t> buffer(rowCount);
      lengths->materialize(rowCount, buffer.data());
      result += std::accumulate(buffer.begin(), buffer.end(), 0u);
      break;
    }

    case EncodingType::Constant: {
      auto valueSize = encoding::readUint32(pos);
      result += rowCount * valueSize;
      break;
    }

    case EncodingType::MainlyConstant: {
      auto isCommonSize = encoding::readUint32(pos);
      pos += isCommonSize;
      auto otherValuesSize = encoding::readUint32(pos);
      auto otherValuesOffset = pos;
      auto otherValuesCount = encoding::peek<uint32_t>(pos + kRowCountOffset);
      pos += otherValuesSize;
      auto constantValueSize = encoding::readUint32(pos);
      result += (rowCount - otherValuesCount) * constantValueSize;
      result +=
          getRawDataSize(memoryPool, {otherValuesOffset, otherValuesSize});
      break;
    }

    case EncodingType::Dictionary: {
      auto alphabetSize = encoding::readUint32(pos);
      auto alphabetCount = encoding::peek<uint32_t>(pos + kRowCountOffset);
      auto alphabet = EncodingFactory().create(
          memoryPool,
          {pos, alphabetSize},
          stringBufferFactory,
          Encoding::Options{});
      std::vector<std::string_view> alphabetBuffer(alphabetCount);
      alphabet->materialize(alphabetCount, alphabetBuffer.data());

      pos += alphabetSize;
      auto indicesSize = encodingStr.length() - (pos - encodingStr.data());
      auto indices = EncodingFactory().create(
          memoryPool,
          {pos, indicesSize},
          stringBufferFactory,
          Encoding::Options{});
      std::vector<uint32_t> indicesBuffer(rowCount);
      indices->materialize(rowCount, indicesBuffer.data());
      for (int i = 0; i < rowCount; ++i) {
        result += alphabetBuffer[indicesBuffer[i]].size();
      }
      break;
    }

    case EncodingType::RLE: {
      auto runLengthsSize = encoding::readUint32(pos);
      auto runLengthsCount = encoding::peek<uint32_t>(pos + kRowCountOffset);
      auto runLengths = EncodingFactory().create(
          memoryPool,
          {pos, runLengthsSize},
          stringBufferFactory,
          Encoding::Options{});
      std::vector<uint32_t> runLengthsBuffer(runLengthsCount);
      runLengths->materialize(runLengthsCount, runLengthsBuffer.data());

      pos += runLengthsSize;
      auto runValuesSize = encodingStr.length() - (pos - encodingStr.data());
      auto runValues = EncodingFactory().create(
          memoryPool,
          {pos, runValuesSize},
          stringBufferFactory,
          Encoding::Options{});
      std::vector<std::string_view> runValuesBuffer(runLengthsCount);
      runValues->materialize(runLengthsCount, runValuesBuffer.data());

      for (int i = 0; i < runLengthsCount; ++i) {
        result += runLengthsBuffer[i] * runValuesBuffer[i].size();
      }
      break;
    }

    default:
      NIMBLE_UNSUPPORTED("Encoding type does not support strings.");
  }
  return result;
}

struct GroupingKey {
  EncodingType encodingType;
  DataType dataType;
  std::optional<CompressionType> compressinType{};
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

enum class Alignment {
  Left,
  Right,
};

class TableFormatter {
 public:
  TableFormatter(
      std::ostream& ostream,
      bool enableColor,
      std::vector<std::tuple<
          std::string /* Title */,
          uint8_t /* Width */,
          Alignment /* Horizontal Alignment */
          >> fields,
      bool noHeader = false,
      const std::string& separator = "\t")
      : ostream_{ostream}, fields_{std::move(fields)}, separator_{separator} {
    if (!noHeader) {
      ostream << YELLOW(enableColor);
      for (const auto& field : fields_) {
        ostream << (std::get<2>(field) == Alignment::Right ? std::right
                                                           : std::left)
                << std::setw(std::get<1>(field)) << std::get<0>(field)
                << ((&field != &fields_.back()) ? separator_ : "");
      }
      ostream << RESET_COLOR(enableColor) << std::endl;
    }
  }

  void writeRow(const std::vector<std::string>& values) {
    assert(values.size() == fields_.size());
    for (auto i = 0; i < values.size(); ++i) {
      ostream_ << (std::get<2>(fields_[i]) == Alignment::Right ? std::right
                                                               : std::left)
               << std::setw(std::get<1>(fields_[i])) << values[i]
               << (i != values.size() - 1 ? separator_ : "");
    }
    ostream_ << std::endl;
  }

 private:
  std::ostream& ostream_;
  std::vector<std::tuple<
      std::string /* Title */,
      uint8_t /* Width */,
      Alignment /* Horizontal Alignment */
      >>
      fields_;
  const std::string separator_;
};

void traverseTablet(
    velox::memory::MemoryPool& memoryPool,
    const TabletReader& tabletReader,
    std::optional<int32_t> stripeIndex,
    const std::function<void(uint32_t /* stripeId */)>& stripeVisitor = nullptr,
    const std::function<void(
        ChunkedStream& /*stream*/,
        uint32_t /*stripeId*/,
        uint32_t /* streamId*/)>& streamVisitor = nullptr) {
  if (tabletReader.stripeCount() == 0) {
    return;
  }

  uint32_t startStripe = stripeIndex ? *stripeIndex : 0;
  uint32_t endStripe =
      stripeIndex ? *stripeIndex : tabletReader.stripeCount() - 1;
  // Stripe identifier internally is holding on to a reference counted cache of
  // stripe groups. We must hold on to it across loop iterations in order to
  // maintain the items in the cache.
  std::optional<StripeIdentifier> stripeIdentifier;
  for (uint32_t i = startStripe; i <= endStripe; ++i) {
    if (stripeVisitor) {
      stripeVisitor(i);
    }
    if (streamVisitor) {
      stripeIdentifier = tabletReader.stripeIdentifier(i);
      std::vector<uint32_t> streamIdentifiers(
          tabletReader.streamCount(stripeIdentifier.value()));
      std::iota(streamIdentifiers.begin(), streamIdentifiers.end(), 0);
      auto streams = tabletReader.load(
          stripeIdentifier.value(),
          {streamIdentifiers.cbegin(), streamIdentifiers.cend()});
      for (uint32_t j = 0; j < streams.size(); ++j) {
        auto& stream = streams[j];
        if (stream) {
          InMemoryChunkedStream chunkedStream{memoryPool, std::move(stream)};
          streamVisitor(chunkedStream, i, j);
        }
      }
    }
  }
}

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
    // If all values are non-null, the returned nulls bitmap is not populated
    // and we should not use it. We should just read all values, ignoring the
    // nulls bitmap.
    for (uint32_t i = 0; i < rowCount; ++i) {
      // Have to use folly::to as Int8 was getting converted to char.
      ostream << folly::to<std::string>(buffer[i]) << separator;
    }
  } else {
    for (uint32_t i = 0; i < rowCount; ++i) {
      assert(stream.isNullable());
      if (velox::bits::isBitSet(
              reinterpret_cast<const uint8_t*>(nulls.data()), i) == 0) {
        ostream << "NULL" << separator;
      } else {
        // Have to use folly::to as Int8 was getting converted to char.
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

template <typename T>
auto commaSeparated(T value) {
  try {
    return fmt::format(std::locale("en_US.UTF-8"), "{:L}", value);
  } catch (const std::runtime_error&) {
    return fmt::format("{}", value);
  }
}

} // namespace

NimbleDumpLib::NimbleDumpLib(
    const std::string& filePath,
    bool enableColors,
    std::ostream& ostream)
    : pool_{velox::memory::deprecatedAddDefaultLeafMemoryPool()},
      file_{velox::filesystems::getFileSystem(filePath, nullptr)
                ->openFileForRead(filePath)},
      ostream_{ostream},
      enableColors_{enableColors} {}

NimbleDumpLib::NimbleDumpLib(
    std::shared_ptr<velox::ReadFile> file,
    bool enableColors,
    std::ostream& ostream)
    : pool_{velox::memory::deprecatedAddDefaultLeafMemoryPool()},
      file_{std::move(file)},
      ostream_{ostream},
      enableColors_{enableColors} {}

void NimbleDumpLib::emitInfo() {
  TabletReader::Options options;
  options.preloadOptionalSections = {std::string(kStatsSection)};
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  const auto tablet = TabletReader::create(file_, pool_.get(), options);
  ostream_ << CYAN(enableColors_) << "Nimble File "
           << RESET_COLOR(enableColors_) << "Version " << tablet->majorVersion()
           << "." << tablet->minorVersion() << std::endl;
  ostream_ << "File Size: " << commaSeparated(tablet->fileSize()) << std::endl;
  ostream_ << "Checksum: " << tablet->checksum() << " ["
           << nimble::toString(tablet->checksumType()) << "]" << std::endl;
  ostream_ << "Postscript Size: " << commaSeparated(kPostscriptSize)
           << std::endl;
  ostream_ << "Footer Size: " << commaSeparated(tablet->footerSize()) << " ("
           << tablet->footerCompressionType() << ")" << std::endl;
  ostream_ << "Stripes Metadata Size: ";
  auto stripesMetadata = tablet->stripesMetadata();
  if (!stripesMetadata) {
    ostream_ << "0" << std::endl;
  } else {
    ostream_ << commaSeparated(stripesMetadata->size()) << " ("
             << stripesMetadata->compressionType() << ")" << std::endl;
  }
  auto stripeGroupsMetadata = tablet->stripeGroupsMetadata();
  ostream_ << "Stripe Groups Metadata Size: "
           << commaSeparated(
                  std::transform_reduce(
                      stripeGroupsMetadata.begin(),
                      stripeGroupsMetadata.end(),
                      0,
                      std::plus{},
                      [](const MetadataSection& metadataSection) {
                        return metadataSection.size();
                      }))
           << std::endl;
  ostream_ << "Optional Sections Size: "
           << commaSeparated(
                  std::transform_reduce(
                      tablet->optionalSections().begin(),
                      tablet->optionalSections().end(),
                      0,
                      std::plus{},
                      [](const std::pair<std::string, MetadataSection>& entry) {
                        return entry.second.size();
                      }))
           << std::endl;
  ostream_ << "Stripe Count: " << commaSeparated(tablet->stripeCount())
           << std::endl;
  ostream_ << "Row Count: " << commaSeparated(tablet->tabletRowCount())
           << std::endl;

  BatchReader reader{tablet, *pool_};

  auto statsSection = tablet->loadOptionalSection(std::string(kStatsSection));
  ostream_ << "Raw Data Size: ";
  if (statsSection.has_value()) {
    auto rawSize = flatbuffers::GetRoot<nimble::serialization::Stats>(
                       statsSection->content().data())
                       ->raw_size();
    ostream_ << commaSeparated(rawSize) << std::endl;
    const auto compressionRate = (double)rawSize / tablet->fileSize();
    ostream_ << "Compression Rate: " << fmt::format("{:.2f}x", compressionRate)
             << std::endl;
  } else {
    ostream_ << "N/A" << std::endl;
  }

  auto& metadata = reader.metadata();
  if (!metadata.empty()) {
    ostream_ << "Metadata:";
    for (const auto& pair : metadata) {
      ostream_ << std::endl << "  " << pair.first << ": " << pair.second;
    }
  }
  ostream_ << std::endl;
}

void NimbleDumpLib::emitSchema(bool collapseFlatMap) {
  TabletReader::Options options;
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  auto tablet = TabletReader::create(file_, pool_.get(), options);
  BatchReader reader{tablet, *pool_};

  auto emitOffsets = [](const Type& type) {
    std::string offsets;
    switch (type.kind()) {
      case Kind::Scalar: {
        offsets =
            folly::to<std::string>(type.asScalar().scalarDescriptor().offset());
        break;
      }
      case Kind::TimestampMicroNano: {
        auto& timestamp = type.asTimestampMicroNano();
        offsets = "m:" +
            folly::to<std::string>(timestamp.microsDescriptor().offset()) +
            ",n:" +
            folly::to<std::string>(timestamp.nanosDescriptor().offset());
        break;
      }
      case Kind::Array: {
        offsets =
            folly::to<std::string>(type.asArray().lengthsDescriptor().offset());
        break;
      }
      case Kind::Map: {
        offsets =
            folly::to<std::string>(type.asMap().lengthsDescriptor().offset());
        break;
      }
      case Kind::Row: {
        offsets =
            folly::to<std::string>(type.asRow().nullsDescriptor().offset());
        break;
      }
      case Kind::FlatMap: {
        offsets =
            folly::to<std::string>(type.asFlatMap().nullsDescriptor().offset());
        break;
      }
      case Kind::ArrayWithOffsets: {
        offsets = "o:" +
            folly::to<std::string>(
                      type.asArrayWithOffsets().offsetsDescriptor().offset()) +
            ",l:" +
            folly::to<std::string>(
                      type.asArrayWithOffsets().lengthsDescriptor().offset());
        break;
      }
      case Kind::SlidingWindowMap: {
        offsets = "o:" +
            folly::to<std::string>(
                      type.asSlidingWindowMap().offsetsDescriptor().offset()) +
            ",l:" +
            folly::to<std::string>(
                      type.asSlidingWindowMap().lengthsDescriptor().offset());
        break;
      }
    }

    return offsets;
  };

  bool skipping = false;
  SchemaReader::traverseSchema(
      reader.schema(),
      [&](uint32_t level,
          const Type& type,
          const SchemaReader::NodeInfo& info) {
        auto parentType = info.parentType;
        if (parentType != nullptr && parentType->isFlatMap()) {
          auto childrenCount = parentType->asFlatMap().childrenCount();
          if (childrenCount > 2 && collapseFlatMap) {
            if (info.placeInSibling == 1) {
              ostream_ << std::string(
                              (std::basic_string<char>::size_type)level * 2,
                              ' ')
                       << "..." << std::endl;
              skipping = true;
            } else if (info.placeInSibling == childrenCount - 1) {
              skipping = false;
            }
          }
        }
        if (!skipping) {
          ostream_ << std::string(
                          (std::basic_string<char>::size_type)level * 2, ' ')
                   << "[" << emitOffsets(type) << "] " << info.name << " : ";
          ostream_ << toString(type.kind());
          if (type.isScalar()) {
            ostream_ << "<"
                     << toString(
                            type.asScalar().scalarDescriptor().scalarKind())
                     << ">";
          } else if (type.isFlatMap()) {
            ostream_ << "<" << toString(type.asFlatMap().keyScalarKind())
                     << ">";
          }

          // Surface per-node schema attributes (e.g. Iceberg `iceberg.id`
          // field-ids) so they are visible when inspecting a dumped schema.
          const auto& attributes = type.attributes();
          if (!attributes.empty()) {
            ostream_ << " {";
            for (size_t i = 0; i < attributes.size(); ++i) {
              if (i > 0) {
                ostream_ << ", ";
              }
              ostream_ << attributes[i].first << "=" << attributes[i].second;
            }
            ostream_ << "}";
          }
          ostream_ << std::endl;
        }
      });
}

void NimbleDumpLib::emitStripes(bool noHeader) {
  TabletReader::Options options;
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  const auto tabletReader = TabletReader::create(file_, pool_.get(), options);
  TableFormatter formatter(
      ostream_,
      enableColors_,
      {{"Stripe Id", 7, Alignment::Left},
       {"Stripe Offset", 15, Alignment::Right},
       {"Stripe Size", 15, Alignment::Right},
       {"Row Count", 10, Alignment::Right}},
      noHeader);
  // Stripe identifier internally is holding on to a reference counted cache of
  // stripe groups. We must hold on to it across loop iterations in order to
  // maintain the items in the cache.
  std::optional<StripeIdentifier> stripeIdentifier;
  std::vector<TabletReader::StreamLocation> locationsScratch;
  for (auto i = 0; i < tabletReader->stripeCount(); ++i) {
    stripeIdentifier = tabletReader->stripeIdentifier(i);
    locationsScratch.resize(
        tabletReader->streamCount(stripeIdentifier.value()));
    tabletReader->streamLocations(stripeIdentifier.value(), locationsScratch);
    auto stripeSize = std::accumulate(
        locationsScratch.begin(),
        locationsScratch.end(),
        0UL,
        [](auto size, const auto& location) { return size + location.size; });
    formatter.writeRow({
        folly::to<std::string>(i),
        commaSeparated(tabletReader->stripeOffset(i)),
        commaSeparated(stripeSize),
        commaSeparated(tabletReader->stripeRowCount(i)),
    });
  }
}

void NimbleDumpLib::emitStreams(
    bool noHeader,
    bool showStreamLabels,
    bool showStreamRawSize,
    bool showInMapStream,
    std::optional<uint32_t> stripeId) {
  TabletReader::Options options;
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  auto tabletReader = TabletReader::create(file_, pool_.get(), options);

  std::vector<std::tuple<std::string, uint8_t, Alignment>> fields;
  fields.emplace_back("Stripe Id", 9, Alignment::Left);
  fields.emplace_back("Stream Id", 9, Alignment::Left);
  fields.emplace_back("Stream Offset", 13, Alignment::Right);
  fields.emplace_back("Stream Length", 13, Alignment::Right);
  if (showStreamRawSize) {
    fields.emplace_back("Raw Stream Size", 16, Alignment::Right);
  }
  fields.emplace_back("Item Count", 13, Alignment::Right);
  if (showStreamLabels) {
    fields.emplace_back("Stream Label", 16, Alignment::Left);
  }
  if (showInMapStream) {
    fields.emplace_back("InMap Stream", 14, Alignment::Left);
  }
  fields.emplace_back("Type", 30, Alignment::Left);

  TableFormatter formatter(
      ostream_, enableColors_, std::move(fields), noHeader);

  std::optional<StreamLabels> labels{};
  std::unordered_set<uint32_t> inMapStreams;
  if (showStreamLabels || showInMapStream) {
    BatchReader reader{tabletReader, *pool_};
    if (showStreamLabels) {
      labels.emplace(reader.schema());
    }
    if (showInMapStream) {
      BatchReader inMapReader{tabletReader, *pool_};
      SchemaReader::traverseSchema(
          inMapReader.schema(),
          [&](auto /*level*/, const Type& type, auto /*info*/) {
            if (type.kind() == Kind::FlatMap) {
              auto& map = type.asFlatMap();
              for (size_t i = 0; i < map.childrenCount(); ++i) {
                inMapStreams.insert(map.inMapDescriptorAt(i).offset());
              }
            }
          });
    }
  }

  traverseTablet(
      *pool_,
      *tabletReader,
      stripeId,
      nullptr /* stripeVisitor */,
      [&](ChunkedStream& stream, uint32_t stripeId, uint32_t streamId) {
        auto stripeIdentifier = tabletReader->stripeIdentifier(stripeId);
        uint32_t itemCount = 0;
        uint64_t rawStreamSize = 0;
        while (stream.hasNext()) {
          auto chunk = stream.nextChunk();
          itemCount += *reinterpret_cast<const uint32_t*>(chunk.data() + 2);
          if (showStreamRawSize) {
            rawStreamSize += getRawDataSize(*pool_, chunk);
          }
        }

        stream.reset();
        std::vector<std::string> values;
        values.push_back(folly::to<std::string>(stripeId));
        values.push_back(folly::to<std::string>(streamId));
        values.push_back(
            folly::to<std::string>(
                tabletReader->streamOffset(stripeIdentifier, streamId)));
        values.push_back(
            folly::to<std::string>(
                tabletReader->streamSize(stripeIdentifier, streamId)));
        if (showStreamRawSize) {
          values.push_back(folly::to<std::string>(rawStreamSize));
        }
        values.push_back(folly::to<std::string>(itemCount));
        if (showStreamLabels) {
          values.emplace_back(labels->streamLabel(streamId));
        }
        if (showInMapStream) {
          values.emplace_back(inMapStreams.contains(streamId) ? "T" : "F");
        }
        values.push_back(getStreamInputLabel(stream));
        formatter.writeRow(values);
      });
}

void NimbleDumpLib::emitHistogram(
    bool topLevel,
    bool noHeader,
    std::optional<uint32_t> stripeId) {
  TabletReader::Options options;
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  auto tabletReader = TabletReader::create(file_, pool_.get(), options);
  std::unordered_map<
      GroupingKey,
      EncodingHistogramValue,
      GroupingKeyHash,
      GroupingKeyEqual>
      encodingHistogram;
  const std::unordered_map<std::string, CompressionType> compressionMap{
      {toString(CompressionType::Uncompressed), CompressionType::Uncompressed},
      {toString(CompressionType::Zstd), CompressionType::Zstd},
      {toString(CompressionType::MetaInternal), CompressionType::MetaInternal},
  };
  traverseTablet(
      *pool_,
      *tabletReader,
      stripeId,
      nullptr,
      [&](ChunkedStream& stream, auto /*stripeIndex*/, auto /*streamIndex*/) {
        while (stream.hasNext()) {
          traverseEncodings(
              stream.nextChunk(),
              [&](EncodingType encodingType,
                  DataType dataType,
                  uint32_t level,
                  uint32_t /* index */,
                  std::string /*nestedEncodingName*/,
                  std::unordered_map<EncodingPropertyType, EncodingProperty>
                      properties) {
                GroupingKey key{
                    .encodingType = encodingType, .dataType = dataType};
                const auto& compression =
                    properties.find(EncodingPropertyType::Compression);
                if (compression != properties.end()) {
                  key.compressinType =
                      compressionMap.at(compression->second.value);
                }
                auto& value = encodingHistogram[key];
                ++value.count;

                const auto& encodedSize =
                    properties.find(EncodingPropertyType::EncodedSize);
                if (encodedSize != properties.end()) {
                  value.bytes += folly::to<uint32_t>(encodedSize->second.value);
                }

                return !(topLevel && level == 1);
              });
        }
      });

  TableFormatter formatter(
      ostream_,
      enableColors_,
      {{"Encoding Type", 17, Alignment::Left},
       {"Data Type", 13, Alignment::Left},
       {"Compression", 15, Alignment::Left},
       {"Instance Count", 15, Alignment::Right},
       {"Storage Bytes", 15, Alignment::Right},
       {"Storage %", 10, Alignment::Right}},
      noHeader);

  std::vector<
      std::unordered_map<GroupingKey, EncodingHistogramValue>::const_iterator>
      rows;
  for (auto it = encodingHistogram.begin(); it != encodingHistogram.end();
       ++it) {
    rows.emplace_back(it);
  }
  std::sort(rows.begin(), rows.end(), HistogramRowCompare{});
  const auto fileSize = tabletReader->fileSize();

  for (const auto& it : rows) {
    formatter.writeRow({
        toString(it->first.encodingType),
        toString(it->first.dataType),
        it->first.compressinType ? toString(*it->first.compressinType) : "",
        commaSeparated(it->second.count),
        commaSeparated(it->second.bytes),
        fmt::format("{:.2f}", it->second.bytes * 100.0 / fileSize),
    });
  }
}

void NimbleDumpLib::emitContent(
    uint32_t streamId,
    std::optional<uint32_t> stripeId,
    const std::string& separator) {
  TabletReader::Options options;
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  auto tabletReader = TabletReader::create(file_, pool_.get(), options);

  uint32_t maxStreamCount;
  bool found = false;
  traverseTablet(*pool_, *tabletReader, stripeId, [&](uint32_t stripeId) {
    auto stripeIdentifier = tabletReader->stripeIdentifier(stripeId);
    maxStreamCount =
        std::max(maxStreamCount, tabletReader->streamCount(stripeIdentifier));
    if (streamId >= tabletReader->streamCount(stripeIdentifier)) {
      return;
    }

    found = true;

    auto streams = tabletReader->load(stripeIdentifier, std::vector{streamId});

    std::vector<velox::BufferPtr> newStringBuffers;
    const auto stringBufferFactory = [&](uint32_t totalLength) {
      auto& buffer = newStringBuffers.emplace_back(
          velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
      return buffer->asMutable<void>();
    };
    if (auto& stream = streams[0]) {
      InMemoryChunkedStream chunkedStream{*pool_, std::move(stream)};
      while (chunkedStream.hasNext()) {
        auto encoding = EncodingFactory().create(
            *pool_,
            chunkedStream.nextChunk(),
            stringBufferFactory,
            Encoding::Options{});
        uint32_t totalRows = encoding->rowCount();
        while (totalRows > 0) {
          auto currentReadSize = std::min(kBufferSize, totalRows);
          printScalarType(
              ostream_, *pool_, *encoding, currentReadSize, separator);
          totalRows -= currentReadSize;
        }
      }
    }
  });

  if (!found) {
    throw folly::ProgramExit(
        -1,
        fmt::format(
            "Stream identifier {} is out of bound. Must be between 0 and {}\n",
            streamId,
            maxStreamCount));
  }
}

void NimbleDumpLib::emitBinary(
    std::function<std::unique_ptr<std::ostream>()> outputFactory,
    uint32_t streamId,
    uint32_t stripeId) {
  TabletReader::Options options;
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  auto tabletReader = TabletReader::create(file_, pool_.get(), options);
  auto stripeIdentifier = tabletReader->stripeIdentifier(stripeId);
  if (streamId >= tabletReader->streamCount(stripeIdentifier)) {
    throw folly::ProgramExit(
        -1,
        fmt::format(
            "Stream identifier {} is out of bound. Must be between 0 and {}\n",
            streamId,
            tabletReader->streamCount(stripeIdentifier)));
  }

  auto streams = tabletReader->load(stripeIdentifier, std::vector{streamId});

  if (auto& stream = streams[0]) {
    auto output = outputFactory();
    output->write(stream->getStream().data(), stream->getStream().size());
    output->flush();
  }
}

void traverseEncodingLayout(
    const std::optional<EncodingLayout>& node,
    const std::optional<EncodingLayout>& parentNode,
    uint32_t& nodeId,
    uint32_t parentId,
    uint32_t level,
    uint8_t childIndex,
    const std::function<void(
        const std::optional<EncodingLayout>&,
        const std::optional<EncodingLayout>&,
        uint32_t,
        uint32_t,
        uint32_t,
        uint8_t)>& visitor) {
  auto currentNodeId = nodeId;
  visitor(node, parentNode, currentNodeId, parentId, level, childIndex);

  if (node.has_value()) {
    for (int i = 0; i < node->childrenCount(); ++i) {
      traverseEncodingLayout(
          node->child(i), node, ++nodeId, currentNodeId, level + 1, i, visitor);
    }
  }
}

void traverseEncodingLayoutTree(
    const EncodingLayoutTree& node,
    const EncodingLayoutTree& parentNode,
    uint32_t& nodeId,
    uint32_t parentId,
    uint32_t level,
    uint8_t childIndex,
    const std::function<void(
        const EncodingLayoutTree&,
        const EncodingLayoutTree&,
        uint32_t,
        uint32_t,
        uint32_t,
        uint8_t)>& visitor) {
  auto currentNodeId = nodeId;
  visitor(node, parentNode, currentNodeId, parentId, level, childIndex);

  for (int i = 0; i < node.childrenCount(); ++i) {
    traverseEncodingLayoutTree(
        node.child(i), node, ++nodeId, currentNodeId, level + 1, i, visitor);
  }
}

std::string getEncodingLayoutLabel(
    const std::optional<nimble::EncodingLayout>& root) {
  std::string label;
  uint32_t currentLevel = 0;
  std::unordered_map<nimble::EncodingType, std::vector<std::string>>
      identifierNames{
          {nimble::EncodingType::Dictionary, {"Alphabet", "Indices"}},
          {nimble::EncodingType::MainlyConstant, {"IsCommon", "OtherValues"}},
          {nimble::EncodingType::Nullable, {"Data", "Nulls"}},
          {nimble::EncodingType::RLE, {"RunLengths", "RunValues"}},
          {nimble::EncodingType::SparseBool, {"Indices"}},
          {nimble::EncodingType::Trivial, {"Lengths"}},
      };

  auto getIdentifierName = [&](nimble::EncodingType encodingType,
                               uint8_t identifier) {
    auto it = identifierNames.find(encodingType);
    LOG(INFO) << (it == identifierNames.end()) << ", "
              << (it != identifierNames.end()
                      ? (int)(identifier >= it->second.size())
                      : -1);
    return it == identifierNames.end() || identifier >= it->second.size()
        ? "Unknown"
        : it->second[identifier];
  };

  uint32_t id = 0;
  traverseEncodingLayout(
      root,
      root,
      id,
      id,
      0,
      (uint8_t)0,
      [&](const std::optional<nimble::EncodingLayout>& node,
          const std::optional<nimble::EncodingLayout>& parentNode,
          uint32_t /* nodeId */,
          uint32_t /* parentId */,
          uint32_t level,
          uint8_t identifier) {
        if (!node.has_value()) {
          label += "N/A";
          return true;
        }

        if (level > currentLevel) {
          label += "[" +
              getIdentifierName(parentNode->encodingType(), identifier) + ":";

        } else if (level < currentLevel) {
          label += "]";
        }

        if (identifier > 0) {
          label += "," +
              getIdentifierName(parentNode->encodingType(), identifier) + ":";
        }

        currentLevel = level;

        label += toString(node->encodingType()) + "{" +
            toString(node->compressionType()) + "}";

        return true;
      });

  while (currentLevel-- > 0) {
    label += "]";
  }

  return label;
}

void NimbleDumpLib::emitLayout(bool noHeader, bool compressed) {
  auto size = file_->size();
  std::string buffer;
  buffer.resize(size);
  file_->pread(0, size, buffer.data());
  if (compressed) {
    auto const decompressedSize =
        ZSTD_getFrameContentSize(buffer.data(), buffer.size());
    NIMBLE_CHECK(
        decompressedSize != ZSTD_CONTENTSIZE_ERROR &&
            decompressedSize != ZSTD_CONTENTSIZE_UNKNOWN,
        "Decompress failed during `emitLayout`: unable to determine decompressed size");
    std::string uncompressed;
    uncompressed.resize(decompressedSize);
    auto const ret = ZSTD_decompress(
        uncompressed.data(), uncompressed.size(), buffer.data(), buffer.size());
    NIMBLE_CHECK(!ZSTD_isError(ret), "Decompress failed during `emitLayout`");
    buffer = std::move(uncompressed);
  }

  auto layout = nimble::EncodingLayoutTree::create(buffer);

  TableFormatter formatter(
      ostream_,
      enableColors_,
      {
          {"Node Id", 11, Alignment::Left},
          {"Parent Id", 11, Alignment::Left},
          {"Node Type", 15, Alignment::Left},
          {"Node Name", 17, Alignment::Left},
          {"Encoding Layout", 20, Alignment::Left},
      },
      noHeader);

  uint32_t id = 0;
  traverseEncodingLayoutTree(
      layout,
      layout,
      id,
      id,
      0,
      0,
      [&](const EncodingLayoutTree& node,
          const EncodingLayoutTree& /* parentNode */,
          uint32_t nodeId,
          uint32_t parentId,
          uint32_t /* level */,
          uint8_t /* identifier */) {
        auto identifiers = node.encodingLayoutIdentifiers();
        std::sort(identifiers.begin(), identifiers.end());

        std::string encodingLayout;
        for (auto identifier : identifiers) {
          if (!encodingLayout.empty()) {
            encodingLayout += "|";
          }
          encodingLayout += folly::to<std::string>(identifier) + ":" +
              getEncodingLayoutLabel(*node.encodingLayout(identifier));
        }

        formatter.writeRow(
            {folly::to<std::string>(nodeId),
             folly::to<std::string>(parentId),
             toString(node.schemaKind()),
             std::string(node.name()),
             std::move(encodingLayout)});
      });
}

void NimbleDumpLib::emitStripesMetadata(bool noHeader) {
  TabletReader::Options options;
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  auto tabletReader = TabletReader::create(file_, pool_.get(), options);
  TableFormatter formatter(
      ostream_,
      enableColors_,
      {
          {"Offset", 15, Alignment::Left},
          {"Size", 15, Alignment::Left},
          {"Compression Type", 18, Alignment::Left},
      },
      noHeader);
  auto stripesMetadata = tabletReader->stripesMetadata();
  if (!stripesMetadata) {
    return;
  }
  formatter.writeRow({
      commaSeparated(stripesMetadata->offset()),
      commaSeparated(stripesMetadata->size()),
      toString(stripesMetadata->compressionType()),
  });
}

void NimbleDumpLib::emitFileLayout(bool noHeader) {
  struct Entry {
    std::string name;
    std::string compression;
    uint64_t offset;
    uint64_t size;
  };

  auto layout = FileLayout::create(file_, pool_.get());
  std::vector<Entry> entries;

  // Stripes metadata section
  if (!layout.stripeGroups.empty()) {
    entries.push_back({
        "Stripes Metadata",
        toString(layout.stripes.compressionType()),
        layout.stripes.offset(),
        layout.stripes.size(),
    });
  }

  // Stripe groups
  for (size_t i = 0; i < layout.stripeGroups.size(); ++i) {
    const auto& metadata = layout.stripeGroups[i];
    entries.push_back({
        fmt::format("Stripe Group {}", i),
        toString(metadata.compressionType()),
        metadata.offset(),
        metadata.size(),
    });
  }

  // Per-stripe info (includes stripe group index in name)
  for (size_t i = 0; i < layout.stripesInfo.size(); ++i) {
    const auto& stripeInfo = layout.stripesInfo[i];
    entries.push_back({
        fmt::format("Stripe {} (Group {})", i, stripeInfo.stripeGroupIndex),
        "NA",
        stripeInfo.offset,
        stripeInfo.size,
    });
  }

  // Optional sections
  for (const auto& [name, metadata] : layout.optionalSections) {
    entries.push_back({
        fmt::format("Optional Section {}", name),
        toString(metadata.compressionType()),
        metadata.offset(),
        metadata.size(),
    });
  }

  // Footer
  entries.push_back({
      "File Footer",
      toString(layout.footer.compressionType()),
      layout.footer.offset(),
      layout.footer.size(),
  });

  // Postscript
  entries.push_back({
      "File Postscript",
      "NA",
      layout.fileSize - kPostscriptSize,
      kPostscriptSize,
  });

  std::sort(
      entries.begin(), entries.end(), [](const Entry& lhs, const Entry& rhs) {
        return lhs.offset < rhs.offset;
      });

  TableFormatter formatter(
      ostream_,
      enableColors_,
      {
          {"Offset", 15, Alignment::Right},
          {"Size", 15, Alignment::Right},
          {"Compression", 12, Alignment::Left},
          {"Object Name", 30, Alignment::Left},
      },
      noHeader);

  for (const auto& entry : entries) {
    formatter.writeRow({
        commaSeparated(entry.offset),
        commaSeparated(entry.size),
        entry.compression,
        entry.name,
    });
  }
}

void NimbleDumpLib::emitStripeGroupsMetadata(bool noHeader) {
  TabletReader::Options options;
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  auto tabletReader = TabletReader::create(file_, pool_.get(), options);
  TableFormatter formatter(
      ostream_,
      enableColors_,
      {
          {"Group Id", 10, Alignment::Left},
          {"Offset", 15, Alignment::Left},
          {"Size", 15, Alignment::Left},
          {"Compression Type", 18, Alignment::Left},
      },
      noHeader);
  auto stripeGroupsMetadata = tabletReader->stripeGroupsMetadata();
  for (auto i = 0; i < stripeGroupsMetadata.size(); ++i) {
    const auto& metadata = stripeGroupsMetadata[i];
    formatter.writeRow({
        commaSeparated(i),
        commaSeparated(metadata.offset()),
        commaSeparated(metadata.size()),
        toString(metadata.compressionType()),
    });
  }
}

void NimbleDumpLib::emitOptionalSectionsMetadata(bool noHeader) {
  struct NamedMetdataSection {
    std::string name;
    nimble::MetadataSection metadata;
  };

  TabletReader::Options options;
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  auto tabletReader = TabletReader::create(file_, pool_.get(), options);
  std::vector<NamedMetdataSection> sections;
  sections.reserve(tabletReader->optionalSections().size());
  for (const auto& [name, metadata] : tabletReader->optionalSections()) {
    sections.push_back({name, metadata});
  }
  std::sort(
      sections.begin(),
      sections.end(),
      [](const NamedMetdataSection& lhs, const NamedMetdataSection& rhs) {
        return lhs.metadata.offset() < rhs.metadata.offset();
      });

  TableFormatter formatter(
      ostream_,
      enableColors_,
      {{"Name", 20, Alignment::Left},
       {"Compression", 12, Alignment::Left},
       {"Offset", 15, Alignment::Right},
       {"Size", 15, Alignment::Right}},
      noHeader);
  for (const auto& namedSection : sections) {
    formatter.writeRow({
        namedSection.name,
        toString(namedSection.metadata.compressionType()),
        commaSeparated(namedSection.metadata.offset()),
        commaSeparated(namedSection.metadata.size()),
    });
  }
}

void NimbleDumpLib::emitIndex() {
  TabletReader::Options options;
  options.loadClusterIndex = true;
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>())
      .setIndexIoStats(std::make_shared<velox::io::IoStatistics>());
  auto tabletReader = TabletReader::create(file_, pool_.get(), options);
  if (!tabletReader->hasClusterIndex()) {
    ostream_ << "Index: Not configured" << std::endl;
    return;
  }

  const auto* index = tabletReader->clusterIndex();
  const auto indexLayout = index->layout();

  ostream_ << CYAN(enableColors_) << "Index" << RESET_COLOR(enableColors_)
           << std::endl;
  ostream_ << "Index Columns: ";
  for (size_t i = 0; i < indexLayout.indexColumns.size(); ++i) {
    if (i > 0) {
      ostream_ << ", ";
    }
    ostream_ << indexLayout.indexColumns[i] << " ("
             << (indexLayout.sortOrders[i].ascending ? "ASC" : "DESC")
             << " NULLS LAST)";
  }
  ostream_ << std::endl;
  ostream_ << "Number of Partitions: "
           << commaSeparated(indexLayout.numPartitions) << std::endl;

  if (!indexLayout.partitions.empty()) {
    ostream_ << "Index Groups:" << std::endl;
    TableFormatter groupFormatter(
        ostream_,
        enableColors_,
        {{"Group", 8, Alignment::Right},
         {"Compression", 15, Alignment::Left},
         {"Offset", 15, Alignment::Right},
         {"Size", 15, Alignment::Right}},
        /*noHeader=*/false);
    for (size_t i = 0; i < indexLayout.partitions.size(); ++i) {
      const auto& part = indexLayout.partitions[i];
      groupFormatter.writeRow({
          std::to_string(i),
          toString(part.metadataSection.compressionType()),
          commaSeparated(part.metadataSection.offset()),
          commaSeparated(part.metadataSection.size()),
      });
    }

    ostream_ << "Partition Details:" << std::endl;
    TableFormatter partitionFormatter(
        ostream_,
        enableColors_,
        {{"Partition", 10, Alignment::Right},
         {"Chunks", 10, Alignment::Right},
         {"Rows", 12, Alignment::Right},
         {"Key Stream", 15, Alignment::Right},
         {"Metadata", 15, Alignment::Right},
         {"Key Offset", 15, Alignment::Right}},
        /*noHeader=*/false);
    uint64_t totalChunks = 0;
    uint64_t totalKeyStreamBytes = 0;
    uint64_t totalMetadataBytes = 0;
    for (size_t i = 0; i < indexLayout.partitions.size(); ++i) {
      const auto& part = indexLayout.partitions[i];
      partitionFormatter.writeRow({
          std::to_string(i),
          commaSeparated(part.numChunks),
          commaSeparated(part.numRows),
          commaSeparated(part.keyStreamRegion.length),
          commaSeparated(part.metadataSizeBytes),
          commaSeparated(part.keyStreamRegion.offset),
      });
      totalChunks += part.numChunks;
      totalKeyStreamBytes += part.keyStreamRegion.length;
      totalMetadataBytes += part.metadataSizeBytes;
    }
    ostream_ << "Total Chunks: " << commaSeparated(totalChunks)
             << ", Key Stream: " << commaSeparated(totalKeyStreamBytes)
             << " bytes, Metadata: " << commaSeparated(totalMetadataBytes)
             << " bytes" << std::endl;
  }
}

namespace {

std::string statTypeToString(StatType type) {
  switch (type) {
    case StatType::DEFAULT:
      return "DEFAULT";
    case StatType::INTEGRAL:
      return "INTEGRAL";
    case StatType::FLOATING_POINT:
      return "FLOATING_POINT";
    case StatType::STRING:
      return "STRING";
    case StatType::DEDUPLICATED:
      return "DEDUPLICATED";
    default:
      return fmt::format("UNKNOWN: {}", static_cast<int>(type));
  }
}

std::pair<std::string, std::string> formatMinMax(ColumnStatistics* stat) {
  switch (stat->getType()) {
    case StatType::INTEGRAL: {
      auto* integralStat = stat->as<IntegralStatistics>();
      auto min = integralStat->getMin();
      auto max = integralStat->getMax();
      return std::make_pair(
          min.has_value() ? std::to_string(min.value()) : "N/A",
          max.has_value() ? std::to_string(max.value()) : "N/A");
    }
    case StatType::FLOATING_POINT: {
      auto* floatStat = stat->as<FloatingPointStatistics>();
      auto min = floatStat->getMin();
      auto max = floatStat->getMax();
      return std::make_pair(
          min.has_value() ? fmt::format("{:.6g}", min.value()) : "N/A",
          max.has_value() ? fmt::format("{:.6g}", max.value()) : "N/A");
    }
    case StatType::STRING: {
      auto* stringStat = stat->as<StringStatistics>();
      auto min = stringStat->getMin();
      auto max = stringStat->getMax();
      return std::make_pair(
          min.has_value() ? min.value() : "N/A",
          max.has_value() ? max.value() : "N/A");
    }
    default:
      return std::make_pair(std::string("N/A"), std::string("N/A"));
  }
}

// Helper to traverse velox schema and build schema node paths.
// For RowType: children use their field names (e.g., root.column_name)
// For ArrayType: child node is named __element__
// For MapType: children are named __key__ and __value__
void buildSchemaNodePaths(
    const velox::TypePtr& schema,
    const std::string& currentPath,
    std::vector<std::string>& schemaNodePaths) {
  schemaNodePaths.push_back(currentPath);

  switch (schema->kind()) {
    case velox::TypeKind::TINYINT:
    case velox::TypeKind::SMALLINT:
    case velox::TypeKind::INTEGER:
    case velox::TypeKind::BIGINT:
    case velox::TypeKind::REAL:
    case velox::TypeKind::DOUBLE:
    case velox::TypeKind::VARCHAR:
    case velox::TypeKind::VARBINARY:
    case velox::TypeKind::BOOLEAN:
    case velox::TypeKind::TIMESTAMP:
      // Leaf types - no children
      break;
    case velox::TypeKind::ROW: {
      const auto& rowType = schema->asRow();
      for (size_t i = 0; i < rowType.size(); ++i) {
        std::string childPath = currentPath + "." + rowType.nameOf(i);
        buildSchemaNodePaths(schema->childAt(i), childPath, schemaNodePaths);
      }
      break;
    }
    case velox::TypeKind::ARRAY: {
      std::string childPath = currentPath + ".__element__";
      buildSchemaNodePaths(schema->childAt(0), childPath, schemaNodePaths);
      break;
    }
    case velox::TypeKind::MAP: {
      std::string keyPath = currentPath + ".__key__";
      std::string valuePath = currentPath + ".__value__";
      buildSchemaNodePaths(schema->childAt(0), keyPath, schemaNodePaths);
      buildSchemaNodePaths(schema->childAt(1), valuePath, schemaNodePaths);
      break;
    }
    default:
      break;
  }
}

std::vector<std::string> getSchemaNodePaths(const velox::TypePtr& schema) {
  std::vector<std::string> schemaNodePaths;
  buildSchemaNodePaths(schema, "root", schemaNodePaths);
  return schemaNodePaths;
}

} // namespace

void NimbleDumpLib::emitStats(bool noHeader) {
  TabletReader::Options options;
  options.preloadOptionalSections = {
      std::string(kVectorizedStatsSection), std::string(kStatsSection)};
  options.ioOptions.emplace(pool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  auto tabletReader = TabletReader::create(file_, pool_.get(), options);

  auto vectorizedStatsSection =
      tabletReader->loadOptionalSection(std::string(kVectorizedStatsSection));

  if (vectorizedStatsSection.has_value()) {
    ostream_ << "Vectorized Stats:" << std::endl;
    auto fileStats = VectorizedFileStats::deserialize(
        vectorizedStatsSection->content(), *pool_);

    BatchReader reader{tabletReader, *pool_};
    auto columnStats =
        fileStats->toColumnStatistics(reader.type(), reader.schema());

    auto schemaNodePaths = getSchemaNodePaths(reader.type());

    TableFormatter formatter(
        ostream_,
        enableColors_,
        {{"index", 8, Alignment::Right},
         {"schema_node", 50, Alignment::Left},
         {"stat_type", 15, Alignment::Left},
         {"value_count", 15, Alignment::Right},
         {"null_count", 12, Alignment::Right},
         {"logical_size", 15, Alignment::Right},
         {"physical_size", 15, Alignment::Right},
         {"min", 20, Alignment::Right},
         {"max", 20, Alignment::Right}},
        noHeader);

    NIMBLE_CHECK_EQ(columnStats.size(), schemaNodePaths.size());

    for (size_t i = 0; i < columnStats.size(); ++i) {
      auto* stat = columnStats[i].get();
      auto [min, max] = formatMinMax(stat);
      formatter.writeRow({
          std::to_string(i),
          schemaNodePaths[i],
          statTypeToString(stat->getType()),
          commaSeparated(stat->getValueCount()),
          commaSeparated(stat->getNullCount()),
          commaSeparated(stat->getLogicalSize()),
          commaSeparated(stat->getPhysicalSize()),
          min,
          max,
      });
    }
    return;
  }

  auto statsSection =
      tabletReader->loadOptionalSection(std::string(kStatsSection));

  if (statsSection.has_value()) {
    ostream_ << "Legacy Stats:" << std::endl;
    auto* stats = flatbuffers::GetRoot<nimble::serialization::Stats>(
        statsSection->content().data());

    TableFormatter formatter(
        ostream_,
        enableColors_,
        {{"raw_size", 20, Alignment::Right}},
        noHeader);

    formatter.writeRow({commaSeparated(stats->raw_size())});
    return;
  }

  ostream_ << "No stats section found in file." << std::endl;
}

} // namespace facebook::nimble::tools
