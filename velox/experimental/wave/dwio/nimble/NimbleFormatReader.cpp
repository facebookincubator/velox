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

#include "velox/experimental/wave/dwio/nimble/NimbleFormatReader.h"

#include <algorithm>
#include <optional>

#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/experimental/wave/dwio/StructColumnReader.h"
#include "velox/experimental/wave/dwio/decode/ReaderKernel.h"

// NOLINTNEXTLINE(facebook-hte-PreventUseOfDeclareGflag)
DECLARE_int32(wave_reader_rows_per_tb);

namespace facebook::velox::wave {

using common::Subfield;

const std::string* NimbleStripe::stream(uint32_t streamId) const {
  auto it = streams.find(streamId);
  return it == streams.end() ? nullptr : &it->second;
}

namespace {

using nimble::DataType;
using nimble::EncodingType;

bool isEncoding(const NimbleNode& node, EncodingType encoding) {
  return node.encoding == static_cast<uint8_t>(encoding);
}

// Returns 'bits' as the signed or unsigned integer of 'dataType', nullopt for
// non-integer types.
std::optional<int64_t> integerValue(uint8_t dataType, int64_t bits) {
  switch (
      static_cast<DataType>(dataType)) { // NOLINT(clang-diagnostic-switch-enum)
    case DataType::Int32:
      return static_cast<int32_t>(bits);
    case DataType::Uint32:
      return static_cast<uint32_t>(bits);
    case DataType::Int64:
    case DataType::Uint64:
      return bits;
    default:
      return std::nullopt;
  }
}

// Size of the aux data of a Nullable or MainlyConstant node. See
// RankBitmap.
int32_t rankBitmapSize(int32_t numRows) {
  constexpr int32_t kWindow = 256;
  const int32_t numWindows = bits::divRoundUp(numRows, kWindow);
  return static_cast<int32_t>(
      bits::roundUp(numWindows * sizeof(int32_t), sizeof(int64_t)) +
      numWindows * kWindow / 8);
}

} // namespace

NimbleFormatData::NimbleFormatData(
    int32_t totalRows,
    const std::string* stream,
    WaveTypeKind kind,
    const common::Filter* filter,
    memory::MemoryPool* pool,
    ColumnRole role,
    std::vector<const FormatData*> levels,
    bool isNested)
    : FormatData(role, std::move(levels)),
      totalRows_(totalRows),
      data_(stream),
      kind_(kind),
      isNested_(isNested) {
  if (!data_) {
    return;
  }
  stream_ = NimbleStream::parse(
      std::string_view(*data_).substr(
          0, data_->size() - NimbleStripe::kStreamPadding),
      pool,
      role_ == ColumnRole::kStructNulls);
  if (!stream_.decompressed.empty()) {
    // The decompressed payloads follow the stream in one device buffer.
    withDecompressed_.reserve(
        stream_.decompressedOffset() + stream_.decompressed.size() +
        NimbleStripe::kStreamPadding);
    withDecompressed_.append(*data_, 0, stream_.streamSize);
    withDecompressed_.resize(stream_.decompressedOffset(), 0);
    withDecompressed_.append(stream_.decompressed);
    withDecompressed_.resize(
        withDecompressed_.size() + NimbleStripe::kStreamPadding, 0);
    stream_.decompressed.clear();
    data_ = &withDecompressed_;
  }
  if (levels_.empty() && !isNested_) {
    // The rows of a column in a nullable struct are the non-null rows of the
    // struct. The rows of a column below an array or map are elements.
    VELOX_CHECK_EQ(
        stream_.numRows,
        totalRows_,
        "Nimble stream and stripe row counts differ");
  }
  if (role_ == ColumnRole::kStructNulls) {
    auxSize_ = rankBitmapSize(stream_.numRows);
  } else {
    accessorsId_ = accessorListId(stream_.accessorList());
    prepareNodes_ = stream_.prepareNodes;
  }
  if (filter) {
    hasFilter_ = true;
    setFilterPreparation(*filter);
  }
  for (auto nodeIdx : prepareNodes_) {
    auto& node = stream_.nodes[nodeIdx];
    node.aux = auxSize_;
    auxSize_ += bits::roundUp(auxBytes(node), sizeof(int64_t));
  }
  if (role_ == ColumnRole::kLengths) {
    // Start of each row's elements, end of the last, null flag of each row.
    lengthsAux_ = auxSize_;
    auxSize_ += static_cast<int32_t>(bits::roundUp(
        (stream_.numRows + 1) * sizeof(int32_t) + stream_.numRows,
        sizeof(int64_t)));
  }
  const auto chunkBytes = stream_.chunks.size() * sizeof(ChunkDesc);
  nodesOffset_ =
      static_cast<int64_t>(bits::roundUp(chunkBytes, sizeof(int64_t)));
  descriptors_.resize(nodesOffset_ + stream_.nodes.size() * sizeof(NimbleNode));
  memcpy(descriptors_.data(), stream_.chunks.data(), chunkBytes);
  memcpy(
      descriptors_.data() + nodesOffset_,
      stream_.nodes.data(),
      stream_.nodes.size() * sizeof(NimbleNode));
  buffers_.add(Staging(
      data_->data(), static_cast<int32_t>(data_->size()), common::Region{}));
  buffers_.add(Staging(
      descriptors_.data(),
      static_cast<int32_t>(descriptors_.size()),
      common::Region{}));
}

int32_t NimbleFormatData::auxBytes(const NimbleNode& node) const {
  if (isEncoding(node, EncodingType::RLE)) {
    return static_cast<int32_t>(
        stream_.nodes[node.children[0]].numRows * sizeof(int32_t));
  }
  if (isEncoding(node, EncodingType::Dictionary)) {
    // A bit per alphabet entry, written 32 bits per warp.
    return bits::roundUp(stream_.nodes[node.children[0]].numRows, 256) / 8;
  }
  if (isEncoding(node, EncodingType::Trivial)) {
    // Strings: the offset of each string and the end of the last.
    return static_cast<int32_t>((node.numRows + 1) * sizeof(int32_t));
  }
  return rankBitmapSize(node.numRows);
}

int64_t NimbleFormatData::sumOfLengths(memory::MemoryPool& pool) const {
  VELOX_CHECK(role_ == ColumnRole::kLengths);
  if (!data_) {
    return 0;
  }
  const std::string_view stream(data_->data(), stream_.streamSize);
  nimble::EncodingFactory factory;
  std::vector<uint32_t> lengths;
  int64_t sum = 0;
  int64_t offset = 0;
  while (offset < stream.size()) {
    const char* pos = stream.data() + offset;
    const auto header = nimble::readChunkHeader(pos);
    offset += nimble::kChunkHeaderSize;
    auto encoding =
        factory.create(pool, stream.substr(offset, header.length), nullptr);
    lengths.resize(encoding->rowCount());
    encoding->materialize(
        static_cast<uint32_t>(lengths.size()), lengths.data());
    for (auto length : lengths) {
      sum += length;
    }
    offset += header.length;
  }
  return sum;
}

void NimbleFormatData::setFilterPreparation(const common::Filter& filter) {
  if (filter.kind() != common::FilterKind::kBigintRange) {
    return;
  }
  for (const auto& chunk : stream_.chunks) {
    int32_t top = chunk.root;
    if (isEncoding(stream_.nodes[top], EncodingType::Nullable)) {
      top = stream_.nodes[top].children[0];
    }
    auto& node = stream_.nodes[top];
    if (isEncoding(node, EncodingType::Constant) ||
        isEncoding(node, EncodingType::MainlyConstant)) {
      // The single value is tested here instead of on every row.
      if (auto value = integerValue(node.dataType, node.value)) {
        node.flags = NimbleNodeFlags::kHasVerdict |
            (filter.testInt64(*value) ? NimbleNodeFlags::kPasses : 0);
      }
    } else if (isEncoding(node, EncodingType::Dictionary)) {
      // The filter is tested on the alphabet by a prepare op.
      const auto& alphabet = stream_.nodes[node.children[0]];
      const bool isScalar = isEncoding(alphabet, EncodingType::Trivial) ||
          isEncoding(alphabet, EncodingType::FixedBitWidth) ||
          isEncoding(alphabet, EncodingType::Constant);
      if (isScalar && integerValue(alphabet.dataType, 0).has_value()) {
        prepareNodes_.push_back(top);
      }
    }
  }
}

void NimbleFormatData::setPointers(
    GpuDecode::Nimble& nimble,
    int32_t firstChunk,
    int32_t numChunks,
    SplitStaging& staging) const {
  auto& column = nimble.column;
  column.aux = deviceAux_;
  column.numChunks = numChunks;
  buffers_.setPointer(kData, column.base, 0, staging);
  buffers_.setPointer(
      kDescriptors,
      column.chunks,
      firstChunk * static_cast<int64_t>(sizeof(ChunkDesc)),
      staging);
  buffers_.setPointer(kDescriptors, nimble.nodes, nodesOffset_, staging);
}

void NimbleFormatData::griddize(
    ColumnOp& op,
    int32_t /*blockSize*/,
    int32_t /*numBlocks*/,
    ResultStaging& deviceStaging,
    ResultStaging& /*resultStaging*/,
    SplitStaging& staging,
    DecodePrograms& programs,
    ReadStream& stream) {
  if (griddized_) {
    return;
  }
  griddized_ = true;
  if (!data_) {
    return;
  }
  std::vector<int32_t> nodes = prepareNodes_;
  if (role_ == ColumnRole::kStructNulls) {
    // One op makes the bitmap of all chunks.
    nodes.push_back(GpuDecode::Nimble::kStructNulls);
  } else if (role_ == ColumnRole::kLengths) {
    nodes.push_back(GpuDecode::Nimble::kLengths);
  }
  if (nodes.empty()) {
    return;
  }
  buffers_.stage(staging);
  allocateAux(stream, auxSize_);
  // The prepare ops of a stream run in one TB in order. A node may read the
  // aux data of the nodes below it, which are before it in 'nodes'.
  programs.programs.emplace_back();
  auto& program = programs.programs.back();
  for (auto nodeIdx : nodes) {
    // The alphabet step has the filter of the column.
    auto step = makePrepareStep(op, hasFilter_, deviceStaging, staging, stream);
    step->step = DecodeStep::kNimblePrepare;
    step->encoding = DecodeStep::kNimble;
    auto& nimble = step->data.nimble;
    setPointers(
        nimble, 0, static_cast<int32_t>(stream_.chunks.size()), staging);
    nimble.column.accessorsId = 0;
    nimble.node = nodeIdx;
    nimble.column.numLevels = 0;
    if (nodeIdx == GpuDecode::Nimble::kLengths) {
      nimble.column.streamRows = stream_.numRows;
      nimble.column.offsets =
          reinterpret_cast<int32_t*>(deviceAux_ + lengthsAux_);
    }
    program.push_back(std::move(step));
  }
}

void NimbleFormatData::setNestedDecode(GpuDecode& step, SplitStaging& staging) {
  step.encoding = DecodeStep::kNimble;
  auto& nimble = step.data.nimble;
  if (data_) {
    setPointers(
        nimble, 0, static_cast<int32_t>(stream_.chunks.size()), staging);
  }
  nimble.column.accessorsId = accessorsId_;
  nimble.node = 0;
}

void NimbleFormatData::setDecode(
    GpuDecode& step,
    ColumnOp& op,
    int32_t blockIdx,
    SplitStaging& staging) {
  VELOX_CHECK_NOT_NULL(data_);
  const auto rowsPerBlock = FLAGS_wave_reader_rows_per_tb;
  const auto numRows = static_cast<int32_t>(op.rows.size());
  const int32_t firstRow = currentRow_ + blockIdx * rowsPerBlock;
  const int32_t lastRow =
      currentRow_ + std::min(numRows, (blockIdx + 1) * rowsPerBlock) - 1;
  step.encoding = DecodeStep::kNimble;
  auto& nimble = step.data.nimble;
  nimble.column.accessorsId = accessorsId_;
  nimble.node = 0;
  setLevels(nimble.column);
  const auto [firstChunk, numChunks] =
      tbChunks(stream_.chunks, firstRow, lastRow);
  setPointers(nimble, firstChunk, numChunks, staging);
}

namespace {

nimble::DataType nimbleDataType(TypeKind kind) {
  switch (kind) { // NOLINT(clang-diagnostic-switch-enum)
    case TypeKind::INTEGER:
      return DataType::Int32;
    case TypeKind::BIGINT:
      return DataType::Int64;
    case TypeKind::REAL:
      return DataType::Float;
    case TypeKind::DOUBLE:
      return DataType::Double;
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      return DataType::String;
    default:
      VELOX_NYI(
          "Missing column of type {} in the Wave Nimble reader",
          TypeKindName::toName(kind));
  }
}

class EncodingWriter {
 public:
  explicit EncodingWriter(std::string& out) : out_(out) {}

  template <typename T>
  void append(T value) {
    out_.append(reinterpret_cast<const char*>(&value), sizeof(T));
  }

  void prefix(EncodingType encoding, DataType dataType, uint32_t numRows) {
    append(static_cast<uint8_t>(encoding));
    append(static_cast<uint8_t>(dataType));
    append(numRows);
  }

 private:
  std::string& out_;
};

} // namespace

int32_t NimbleFormatParams::rowBound() {
  const auto* lengths = nestedLengths();
  if (!lengths) {
    return stripe_.numRows;
  }
  auto [it, isNew] = numElements_.emplace(lengths, 0);
  if (isNew) {
    it->second =
        static_cast<const NimbleFormatData*>(lengths)->sumOfLengths(pool());
  }
  VELOX_CHECK_LE(it->second, std::numeric_limits<int32_t>::max());
  return static_cast<int32_t>(it->second);
}

const std::string* NimbleFormatParams::nullStream(
    DataType dataType,
    bool isNulls) {
  // A stream of 0 rows has no chunks. 1 row makes a chunk that is never read.
  const uint32_t numRows = std::max<int32_t>(1, rowBound());
  std::string encoding;
  EncodingWriter writer(encoding);
  if (!isNulls) {
    // Nullable over 0 non-null values.
    std::string values;
    EncodingWriter valuesWriter(values);
    valuesWriter.prefix(EncodingType::Constant, dataType, 0);
    switch (dataType) { // NOLINT(clang-diagnostic-switch-enum)
      case DataType::Int32:
      case DataType::Uint32:
      case DataType::Float:
      case DataType::String:
        // A String constant is its size and bytes.
        valuesWriter.append<uint32_t>(0);
        break;
      default:
        valuesWriter.append<uint64_t>(0);
        break;
    }
    writer.prefix(EncodingType::Nullable, dataType, numRows);
    writer.append<uint32_t>(static_cast<uint32_t>(values.size()));
    encoding.append(values);
  }
  // The non-null flags.
  writer.prefix(EncodingType::Constant, DataType::Bool, numRows);
  writer.append<uint8_t>(0);
  auto& stream = stripe_.nullStreams.emplace_back();
  EncodingWriter streamWriter(stream);
  streamWriter.append<uint32_t>(static_cast<uint32_t>(encoding.size()));
  streamWriter.append(
      static_cast<uint8_t>(nimble::CompressionType::Uncompressed));
  stream.append(encoding);
  stream.resize(stream.size() + NimbleStripe::kStreamPadding, 0);
  return &stream;
}

std::unique_ptr<FormatData> NimbleFormatParams::toFormatData(
    const std::shared_ptr<const dwio::common::TypeWithId>& type,
    const velox::common::ScanSpec& scanSpec,
    OperandId /*operand*/) {
  const auto kind = type->type()->kind();
  if (inMapStream_ != kNoInMap ||
      (nimbleType_ && (nimbleType_->isRow() || nimbleType_->isFlatMap())) ||
      (!nimbleType_ && kind == TypeKind::ROW)) {
    // The stripe has the nulls stream of a struct or flat map below the root
    // if it has nulls. A key of a flat map has an in-map stream unless the
    // key is in all rows. A struct that is not in the file is null.
    const std::string* stream = nullptr;
    if (inMapStream_ >= 0) {
      stream = stripe_.stream(inMapStream_);
    } else if (inMapStream_ == kMissingKey) {
      stream = nullptr;
    } else if (!nimbleType_) {
      stream = nullStream(DataType::Bool, true);
    } else {
      stream = stripe_.stream(
          nimbleType_->isRow()
              ? nimbleType_->asRow().nullsDescriptor().offset()
              : nimbleType_->asFlatMap().nullsDescriptor().offset());
    }
    return std::make_unique<NimbleFormatData>(
        stripe_.numRows,
        stream,
        WaveTypeKind::UNKNOWN,
        nullptr,
        &pool(),
        ColumnRole::kStructNulls,
        levels(),
        isNested());
  }
  if (kind == TypeKind::ARRAY || kind == TypeKind::MAP) {
    VELOX_CHECK(
        !nimbleType_ || nimbleType_->isArray() || nimbleType_->isMap(),
        "Only arrays without offsets and maps are supported in the Wave Nimble reader");
    const std::string* stream = nullptr;
    if (nimbleType_) {
      stream = stripe_.stream(
          nimbleType_->isArray()
              ? nimbleType_->asArray().lengthsDescriptor().offset()
              : nimbleType_->asMap().lengthsDescriptor().offset());
    }
    if (!stream) {
      // All rows are null.
      stream = nullStream(DataType::Uint32, false);
    }
    return std::make_unique<NimbleFormatData>(
        stripe_.numRows,
        stream,
        WaveTypeKind::INTEGER,
        nullptr,
        &pool(),
        ColumnRole::kLengths,
        levels(),
        isNested());
  }
  const auto* filter = scanSpec.filter();
  if (isNested()) {
    // Keys of maps are not null. Other filters below arrays and maps are not
    // supported.
    VELOX_CHECK(
        !filter || filter->kind() == common::FilterKind::kIsNotNull,
        "Filters below arrays and maps are not supported in the Wave Nimble reader");
    filter = nullptr;
  }
  const std::string* stream = nullptr;
  if (nimbleType_) {
    stream =
        stripe_.stream(nimbleType_->asScalar().scalarDescriptor().offset());
  }
  if (!stream) {
    // The column is not in the file or has no non-null value in the stripe.
    stream = nullStream(nimbleDataType(kind), false);
  }
  return std::make_unique<NimbleFormatData>(
      stripe_.numRows,
      stream,
      static_cast<WaveTypeKind>(kind),
      filter,
      &pool(),
      ColumnRole::kValues,
      levels(),
      isNested());
}

namespace {

// Returns the child 'name' of 'row', nullptr if 'row' is nullptr or has no
// child 'name'.
const nimble::Type* FOLLY_NULLABLE
nimbleChild(const nimble::Type* row, const std::string& name) {
  if (!row) {
    return nullptr;
  }
  const auto& names = row->asRow().names();
  auto it = std::find(names.begin(), names.end(), name);
  return it == names.end() ? nullptr
                           : row->asRow().childAt(it - names.begin()).get();
}

class NimbleStructColumnReader : public StructColumnReader {
 public:
  NimbleStructColumnReader(
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      const nimble::Type* nimbleType,
      NimbleFormatParams& params,
      common::ScanSpec& scanSpec,
      Path& path,
      const DefinesMap& defines,
      bool isRoot)
      : StructColumnReader(
            requestedType,
            fileType,
            structOperand(defines, path, isRoot, params.isNested()),
            params,
            scanSpec,
            isRoot),
        nimbleType_(nimbleType),
        params_(params) {
    const bool hasNulls =
        formatData_->hasNulls() && formatData_->numEncodedRows() > 0;
    if (hasNulls) {
      params.pushLevel(formatData_.get());
    }
    addChildren(scanSpec, path, defines);
    if (hasNulls) {
      params.popLevel();
    }
  }

 protected:
  std::unique_ptr<ColumnReader> makeChild(
      common::ScanSpec& childSpec,
      int32_t fileChildIdx,
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      Path& path,
      const DefinesMap& defines) override {
    return NimbleFormatReader::build(
        requestedType,
        fileType,
        fileChildIdx < 0 ? nullptr
                         : nimbleChild(nimbleType_, childSpec.fieldName()),
        params_,
        childSpec,
        path,
        defines);
  }

 private:
  const nimble::Type* const nimbleType_;
  NimbleFormatParams& params_;
};

// Reads the values of one key of a flat map. The in-map stream of the key is
// the nulls of the reader. The only child is the value column.
class NimbleFlatMapKeyReader : public StructColumnReader {
 public:
  // 'keyIdx' is -1 for a key that is not in the file. Its values are null.
  NimbleFlatMapKeyReader(
      const TypePtr& valueType,
      const std::shared_ptr<const dwio::common::TypeWithId>& mapFileType,
      const nimble::FlatMapType& flatMap,
      int32_t keyIdx,
      NimbleFormatParams& params,
      common::ScanSpec& valuesSpec,
      std::vector<std::unique_ptr<Subfield::PathElement>>& path,
      const DefinesMap& defines)
      : StructColumnReader(
            valueType,
            mapFileType,
            nullptr,
            withInMap(
                params,
                keyIdx < 0 ? NimbleFormatParams::kMissingKey
                           : static_cast<int32_t>(
                                 flatMap.inMapDescriptorAt(keyIdx).offset())),
            valuesSpec,
            false) {
    params.setInMapStream(NimbleFormatParams::kNoInMap);
    const bool hasNulls = formatData_->numEncodedRows() > 0;
    if (hasNulls) {
      params.pushLevel(formatData_.get());
    }
    addChild(
        NimbleFormatReader::build(
            valueType,
            mapFileType->childAt(1),
            keyIdx < 0 ? nullptr : flatMap.childAt(keyIdx).get(),
            params,
            valuesSpec,
            path,
            defines));
    if (hasNulls) {
      params.popLevel();
    }
  }

 private:
  static NimbleFormatParams& withInMap(
      NimbleFormatParams& params,
      int32_t streamId) {
    params.setInMapStream(streamId);
    return params;
  }
};

// Returns the key of the map element at the end of 'path' and whether 'path'
// is 'prefix' followed by a subscript.
std::optional<int64_t> subscriptKey(
    const std::vector<std::unique_ptr<Subfield::PathElement>>& path,
    const std::vector<std::unique_ptr<Subfield::PathElement>>& prefix) {
  if (path.size() != prefix.size() + 1) {
    return std::nullopt;
  }
  for (auto i = 0; i < prefix.size(); ++i) {
    if (!(*path[i] == *prefix[i])) {
      return std::nullopt;
    }
  }
  auto* subscript =
      dynamic_cast<const Subfield::LongSubscript*>(path.back().get());
  if (!subscript) {
    return std::nullopt;
  }
  return subscript->index();
}

// Reads a flat map as the struct of the keys that the scan reads. For a flat
// map read as a struct, these are the fields of the requested struct.
// Otherwise these are the keys that the scan defines operands for, e.g.
// element_at(m, 12). The nulls of the flat map are the nulls of the struct.
class NimbleFlatMapColumnReader : public StructColumnReader {
 public:
  NimbleFlatMapColumnReader(
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      const nimble::Type& nimbleType,
      NimbleFormatParams& params,
      common::ScanSpec& scanSpec,
      std::vector<std::unique_ptr<Subfield::PathElement>>& path,
      const DefinesMap& defines)
      : StructColumnReader(
            requestedType,
            fileType,
            nullptr,
            params,
            scanSpec,
            false) {
    const auto& flatMap = nimbleType.asFlatMap();
    const bool hasNulls = formatData_->numEncodedRows() > 0;
    if (hasNulls) {
      params.pushLevel(formatData_.get());
    }
    if (requestedType->isRow()) {
      addStructFields(
          flatMap, requestedType->asRow(), params, scanSpec, path, defines);
    } else {
      addSubscripts(flatMap, params, scanSpec, path, defines);
    }
    if (hasNulls) {
      params.popLevel();
    }
  }

 private:
  void addStructFields(
      const nimble::FlatMapType& flatMap,
      const RowType& requestedRow,
      NimbleFormatParams& params,
      common::ScanSpec& scanSpec,
      std::vector<std::unique_ptr<Subfield::PathElement>>& path,
      const DefinesMap& defines) {
    VELOX_CHECK(
        scanSpec.isFlatMapAsStruct(),
        "A map read as a struct must be a flat map as struct");
    const auto stableChildren = scanSpec.stableChildren();
    for (const auto& childSpec : *stableChildren) {
      const auto& name = childSpec->fieldName();
      auto childIdx = requestedRow.getChildIdxIfExists(name);
      VELOX_CHECK(
          childIdx.has_value(), "No field {} in flat map as struct", name);
      path.push_back(std::make_unique<Subfield::NestedField>(name));
      addChild(
          std::make_unique<NimbleFlatMapKeyReader>(
              requestedRow.childAt(childIdx.value()),
              fileType_,
              flatMap,
              findKey(flatMap, name),
              params,
              *childSpec,
              path,
              defines));
      path.pop_back();
      childSpec->setSubscript(static_cast<int64_t>(children_.size()) - 1);
    }
  }

  void addSubscripts(
      const nimble::FlatMapType& flatMap,
      NimbleFormatParams& params,
      common::ScanSpec& scanSpec,
      std::vector<std::unique_ptr<Subfield::PathElement>>& path,
      const DefinesMap& defines) {
    auto* valuesSpec =
        scanSpec.childByName(common::ScanSpec::kMapValuesFieldName);
    VELOX_CHECK_NOT_NULL(valuesSpec);
    const auto& valueType = requestedType_->childAt(1);
    for (const auto& [value, operand] : defines) {
      if (!value.subfield) {
        continue;
      }
      auto key = subscriptKey(value.subfield->path(), path);
      if (!key.has_value()) {
        continue;
      }
      const auto keyIdx = findKey(flatMap, std::to_string(*key));
      path.push_back(std::make_unique<Subfield::LongSubscript>(*key));
      addChild(
          std::make_unique<NimbleFlatMapKeyReader>(
              valueType,
              fileType_,
              flatMap,
              keyIdx,
              params,
              *valuesSpec,
              path,
              defines));
      path.pop_back();
    }
  }

  // Returns the index of 'key' in 'flatMap', -1 if the file has no 'key'.
  static int32_t findKey(
      const nimble::FlatMapType& flatMap,
      const std::string& key) {
    for (auto i = 0; i < flatMap.childrenCount(); ++i) {
      if (flatMap.nameAt(i) == key) {
        return i;
      }
    }
    return -1;
  }
};

// Reads an array or a map. The children read the elements of the selected
// rows.
class NimbleRepeatedColumnReader : public RepeatedColumnReader {
 public:
  NimbleRepeatedColumnReader(
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      const nimble::Type* nimbleType,
      NimbleFormatParams& params,
      common::ScanSpec& scanSpec,
      Path& path,
      const DefinesMap& defines)
      : RepeatedColumnReader(
            requestedType,
            fileType,
            params,
            scanSpec,
            path,
            defines),
        nimbleType_(nimbleType),
        params_(params) {
    addElements(params, scanSpec, path, defines);
  }

 protected:
  std::unique_ptr<ColumnReader> makeChild(
      common::ScanSpec& childSpec,
      int32_t fileChildIdx,
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      Path& path,
      const DefinesMap& defines) override {
    const nimble::Type* child = nullptr;
    if (nimbleType_ && nimbleType_->isArray()) {
      child = nimbleType_->asArray().elements().get();
    } else if (nimbleType_) {
      const auto& map = nimbleType_->asMap();
      child = fileChildIdx == 0 ? map.keys().get() : map.values().get();
    }
    // NOLINTNEXTLINE(facebook-hte-NullableDereference)
    return NimbleFormatReader::build(
        requestedType, fileType, child, params_, childSpec, path, defines);
  }

 private:
  const nimble::Type* const nimbleType_;
  NimbleFormatParams& params_;
};

} // namespace

// static
std::unique_ptr<ColumnReader> NimbleFormatReader::build(
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    const nimble::Type* nimbleType,
    NimbleFormatParams& params,
    common::ScanSpec& scanSpec,
    std::vector<std::unique_ptr<Subfield::PathElement>>& path,
    const DefinesMap& defines,
    bool isRoot) {
  params.setNimbleType(nimbleType);
  switch (fileType->type()->kind()) { // NOLINT(clang-diagnostic-switch-enum)
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      VELOX_CHECK(!nimbleType || nimbleType->isScalar());
      return std::make_unique<ColumnReader>(
          requestedType,
          fileType,
          ColumnReader::scalarOperand(defines, path, params),
          params,
          scanSpec);
    case TypeKind::ARRAY:
      VELOX_CHECK(
          !nimbleType || nimbleType->isArray(),
          "Only arrays without offsets are supported in the Wave Nimble reader");
      return std::make_unique<NimbleRepeatedColumnReader>(
          requestedType, fileType, nimbleType, params, scanSpec, path, defines);
    case TypeKind::ROW:
      VELOX_CHECK(!nimbleType || nimbleType->isRow());
      return std::make_unique<NimbleStructColumnReader>(
          requestedType,
          fileType,
          nimbleType,
          params,
          scanSpec,
          path,
          defines,
          isRoot);
    case TypeKind::MAP:
      if (!nimbleType || nimbleType->isMap()) {
        VELOX_CHECK(
            requestedType->isMap(),
            "Maps read as structs must be flat maps in the Wave Nimble reader");
        return std::make_unique<NimbleRepeatedColumnReader>(
            requestedType,
            fileType,
            nimbleType,
            params,
            scanSpec,
            path,
            defines);
      }
      VELOX_CHECK(
          nimbleType->isFlatMap(),
          "Only maps and flat maps are supported in the Wave Nimble reader");
      VELOX_CHECK(
          !params.isNested(),
          "Flat maps below arrays and maps are not supported in the Wave Nimble reader");
      return std::make_unique<NimbleFlatMapColumnReader>(
          requestedType,
          fileType,
          *nimbleType,
          params,
          scanSpec,
          path,
          defines);
    default:
      VELOX_NYI(
          "Type not supported in the Wave Nimble reader: {}",
          fileType->type()->toString());
  }
}

// static
bool NimbleFormatReader::isKeySelected(
    const common::ScanSpec* scanSpec,
    const std::string& key) {
  if (!scanSpec) {
    return true;
  }
  if (scanSpec->isFlatMapAsStruct()) {
    return scanSpec->childByName(key) != nullptr;
  }
  auto* keysSpec = scanSpec->childByName(common::ScanSpec::kMapKeysFieldName);
  if (!keysSpec || !keysSpec->filter()) {
    return true;
  }
  const auto* filter = keysSpec->filter();
  if (auto value = folly::tryTo<int64_t>(key); value.hasValue()) {
    return filter->testInt64(value.value());
  }
  return filter->testBytes(key.data(), static_cast<int32_t>(key.size()));
}

// static
void NimbleFormatReader::addStreamIds(
    const nimble::Type& type,
    const common::ScanSpec* scanSpec,
    std::vector<uint32_t>& streamIds) {
  auto child = [&](const std::string& name) -> const common::ScanSpec* {
    return scanSpec ? scanSpec->childByName(name) : nullptr;
  };
  if (type.isScalar()) {
    streamIds.push_back(type.asScalar().scalarDescriptor().offset());
    return;
  }
  if (type.isArray()) {
    streamIds.push_back(type.asArray().lengthsDescriptor().offset());
    addStreamIds(
        *type.asArray().elements(),
        child(common::ScanSpec::kArrayElementsFieldName),
        streamIds);
    return;
  }
  if (type.isMap()) {
    streamIds.push_back(type.asMap().lengthsDescriptor().offset());
    addStreamIds(
        *type.asMap().keys(),
        child(common::ScanSpec::kMapKeysFieldName),
        streamIds);
    addStreamIds(
        *type.asMap().values(),
        child(common::ScanSpec::kMapValuesFieldName),
        streamIds);
    return;
  }
  if (type.isFlatMap()) {
    const auto& flatMap = type.asFlatMap();
    streamIds.push_back(flatMap.nullsDescriptor().offset());
    const bool asStruct = scanSpec && scanSpec->isFlatMapAsStruct();
    for (auto i = 0; i < flatMap.childrenCount(); ++i) {
      const auto& key = flatMap.nameAt(i);
      if (!isKeySelected(scanSpec, key)) {
        continue;
      }
      streamIds.push_back(flatMap.inMapDescriptorAt(i).offset());
      addStreamIds(
          *flatMap.childAt(i),
          asStruct ? child(key) : child(common::ScanSpec::kMapValuesFieldName),
          streamIds);
    }
    return;
  }
  VELOX_CHECK(
      type.isRow(),
      "Only scalars, structs, arrays and maps are supported in Wave Nimble");
  const auto& row = type.asRow();
  streamIds.push_back(row.nullsDescriptor().offset());
  for (auto i = 0; i < row.childrenCount(); ++i) {
    const common::ScanSpec* childSpec = nullptr;
    if (scanSpec) {
      childSpec = scanSpec->childByName(row.nameAt(i));
      if (!childSpec) {
        // The reader reads only the fields in the ScanSpec.
        continue;
      }
    }
    // NOLINTNEXTLINE(facebook-hte-NullableDereference)
    addStreamIds(*row.childAt(i), childSpec, streamIds);
  }
}

} // namespace facebook::velox::wave
