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

#include "velox/experimental/wave/vector/WaveVector.h"

#include <gflags/gflags.h>

#include "velox/common/base/SimdUtil.h"
#include "velox/experimental/wave/common/Cuda.h"
#include "velox/experimental/wave/common/StringView.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/VectorTypeUtils.h"

DEFINE_int64(
    wave_max_string_copy_bytes,
    1LL << 30,
    "Maximum bytes of device memory referenced by the strings of a vector "
    "that are copied to host when making a Velox vector");

namespace facebook::velox::wave {

static_assert(sizeof(StringView) == sizeof(velox::StringView));

template <TypeKind Kind>
static int32_t kindSize() {
  return sizeof(typename KindToFlatVector<Kind>::WrapperType);
}

int32_t waveTypeKindSize(WaveTypeKind waveKind) {
  TypeKind kind = static_cast<TypeKind>(waveKind);
  if (is_string_kind(kind)) {
    return sizeof(StringView);
  }
  if (kind == TypeKind::UNKNOWN) {
    return sizeof(UnknownValue);
  }

  return VELOX_DYNAMIC_TYPE_DISPATCH(kindSize, kind);
}

WaveVector::WaveVector(
    const TypePtr& type,
    GpuArena& arena,
    std::vector<std::unique_ptr<WaveVector>> children,
    bool notNull)
    : type_(type),
      kind_(type_->kind()),
      arena_(&arena),
      children_(std::move(children)) {
  encoding_ = encodingOf(kind_);
  VELOX_CHECK(
      encoding_ != VectorEncoding::Simple::FLAT,
      "Vector with children must be a ROW, ARRAY or MAP: {}",
      kind_);
  if (!children_.empty()) {
    size_ = children_[0]->size();
  }
}

// static
VectorEncoding::Simple WaveVector::encodingOf(TypeKind kind) {
  switch (kind) { // NOLINT(clang-diagnostic-switch-enum)
    case TypeKind::ROW:
      return VectorEncoding::Simple::ROW;
    case TypeKind::ARRAY:
      return VectorEncoding::Simple::ARRAY;
    case TypeKind::MAP:
      return VectorEncoding::Simple::MAP;
    default:
      return VectorEncoding::Simple::FLAT;
  }
}

void WaveVector::resize(
    vector_size_t size,
    bool nullable,
    WaveBufferPtr* backing,
    int64_t* backingOffset) {
  VELOX_CHECK_GE(size, 0);
  auto capacity = values_ ? values_->capacity() : 0;
  size_ = size;

  // Round to next block. A cardinality expanding plan can produce up to full
  // block for a partial block.
  size = bits::roundUp(size, kBlockSize);
  int32_t bytesNeeded = backingSize(type_, size, nullable);
  if (bytesNeeded > capacity) {
    if (backing) {
      values_ = WaveBufferView<WaveBufferPtr>::create(
          (*backing)->as<uint8_t>() + *backingOffset, bytesNeeded, *backing);
      *backingOffset += bytesNeeded;
    } else {
      values_ = arena_->allocateBytes(bytesNeeded);
    }
  }
  if (nullable && bytesNeeded > 0) {
    nulls_ = values_->as<uint8_t>() + bytesNeeded - size;
  } else {
    nulls_ = nullptr;
  }
}

void WaveVector::toOperand(Operand* operand) const {
  operand->size = size_;
  operand->nulls = nulls_;
  if (encoding_ == VectorEncoding::Simple::CONSTANT) {
    operand->indexMask = 0;
    operand->base = values_->as<uint64_t>();
    operand->indices = nullptr;
    return;
  }
  operand->indexMask = ~0;
  operand->indices = nullptr;
  if (encoding_ == VectorEncoding::Simple::FLAT) {
    operand->base = values_->as<int64_t>();
  } else {
    // A complex vector is written by readers, not by instructions.
    operand->base = values_ ? values_->as<int64_t>() : nullptr;
  }
}

void toBits(uint64_t* words, int32_t numBytes) {
  auto data = reinterpret_cast<uint8_t*>(words);
  auto zero = xsimd::broadcast<uint8_t>(0);
  for (auto i = 0; i < numBytes; i += xsimd::batch<uint8_t>::size) {
    auto flags = xsimd::batch<uint8_t>::load_unaligned(data + i) != zero;
    uint32_t bits = simd::toBitMask(flags);
    reinterpret_cast<uint32_t*>(words)[i / sizeof(flags)] = bits;
  }
}

namespace {
class NoReleaser {
 public:
  void addRef() const {};
  void release() const {};
};

template <TypeKind kind>
static VectorPtr toVeloxTyped(
    vector_size_t size,
    velox::memory::MemoryPool* pool,
    const TypePtr& type,
    const WaveBufferPtr& values,
    const uint8_t* nulls) {
  using T = typename TypeTraits<kind>::NativeType;

  BufferPtr nullsView;
  if (nulls) {
    nullsView = BufferView<NoReleaser>::create(nulls, size, NoReleaser());
    toBits(
        const_cast<uint64_t*>(nullsView->as<uint64_t>()),
        nullsView->capacity());
  }
  BufferPtr valuesView;
  if (values) {
    valuesView = VeloxWaveBufferView::create(values);
  }

  return std::make_shared<FlatVector<T>>(
      pool,
      type,
      std::move(nullsView),
      size,
      std::move(valuesView),
      std::vector<BufferPtr>());
}

bool isDenselyFilled(const BlockStatus* status, int32_t numBlocks) {
  for (int32_t i = 0; i < numBlocks; ++i) {
    if (status[i].numRows != kBlockSize) {
      return false;
    }
  }
  return true;
}
} // namespace

int32_t statusNumRows(const BlockStatus* status, int32_t numBlocks) {
  int32_t numRows = 0;
  for (auto i = 0; i < numBlocks; ++i) {
    numRows += status[i].numRows;
  }
  return numRows;
}

// static
int32_t WaveVector::alignment(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::ARRAY:
    case TypeKind::MAP:
    case TypeKind::ROW:
      return sizeof(void*);
    default:
      return type->cppSizeInBytes();
  }
}

//    static
int64_t
WaveVector::backingSize(const TypePtr& type, int32_t size, bool nullable) {
  int64_t bytes;
  if (type->kind() == TypeKind::VARCHAR ||
      type->kind() == TypeKind::VARBINARY) {
    bytes = sizeof(StringView) * size;
  } else if (type->kind() == TypeKind::ARRAY || type->kind() == TypeKind::MAP) {
    // Offsets and sizes.
    bytes = 2 * static_cast<int64_t>(sizeof(int32_t)) * size;
  } else if (type->kind() == TypeKind::ROW) {
    bytes = 0;
  } else {
    bytes = type->cppSizeInBytes() * size;
  }
  return bits::roundUp(bytes, sizeof(void*)) + (nullable ? size : 0);
}

BufferPtr WaveVector::veloxNulls(int32_t size) {
  if (!nulls_) {
    return nullptr;
  }
  auto nulls = BufferView<NoReleaser>::create(nulls_, size, NoReleaser());
  toBits(
      const_cast<uint64_t*>(nulls->as<uint64_t>()),
      static_cast<int32_t>(nulls->capacity()));
  return nulls;
}

VectorPtr WaveVector::stringsToVelox(
    memory::MemoryPool* pool,
    int32_t size,
    const std::vector<bool>* selected) {
  auto result =
      BaseVector::create<FlatVector<velox::StringView>>(type_, size, pool);
  auto* strings = values_->as<StringView>();
  // Strings that are not inline point to device memory. The range of device
  // memory that they reference is copied to host at once.
  const char* low = nullptr;
  const char* high = nullptr;
  auto isSkipped = [&](int32_t i) {
    return (selected && !(*selected)[i]) || (nulls_ && nulls_[i] == kNull);
  };
  for (auto i = 0; i < size; ++i) {
    if (isSkipped(i) || strings[i].isInline()) {
      continue;
    }
    const char* data = strings[i].data();
    if (!low || data < low) {
      low = data;
    }
    if (!high || data + strings[i].size() > high) {
      high = data + strings[i].size();
    }
  }
  const char* hostChars = nullptr;
  if (low) {
    VELOX_CHECK_LE(
        high - low,
        FLAGS_wave_max_string_copy_bytes,
        "Strings of a vector reference too much device memory, likely corrupt "
        "string views");
    auto hostCopy = AlignedBuffer::allocate<char>(high - low, pool);
    Stream stream;
    stream.deviceToHostAsync(
        hostCopy->asMutable<char>(), low, hostCopy->size());
    stream.wait();
    hostChars = hostCopy->as<char>();
    result->addStringBuffer(hostCopy);
  }
  // The views have the layout of velox::StringView. Only the pointers of
  // strings that are not inline change.
  auto* rawValues = result->mutableRawValues();
  for (auto i = 0; i < size; ++i) {
    if (isSkipped(i)) {
      result->setNull(i, true);
      continue;
    }
    StringView string = strings[i];
    if (!string.isInline()) {
      string.setData(hostChars + (string.data() - low));
    }
    memcpy(&rawValues[i], &string, sizeof(string));
  }
  return result;
}

VectorPtr WaveVector::toVeloxBase(
    memory::MemoryPool* pool,
    int32_t size,
    const std::vector<bool>* selected) {
  if (size == 0) {
    return BaseVector::create(type_, 0, pool);
  }
  switch (kind_) { // NOLINT(clang-diagnostic-switch-enum)
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      return stringsToVelox(pool, size, selected);
    case TypeKind::ARRAY:
    case TypeKind::MAP: {
      std::vector<VectorPtr> children;
      children.reserve(type_->size());
      for (auto i = 0; i < type_->size(); ++i) {
        children.push_back(
            i < children_.size()
                ? children_[i]->toVeloxBase(pool, children_[i]->size())
                : BaseVector::create(type_->childAt(i), 0, pool));
      }
      const auto bytes = static_cast<int32_t>(size * sizeof(int32_t));
      auto offsets = VeloxWaveBufferView::create(values_, 0, bytes);
      auto sizes = VeloxWaveBufferView::create(
          values_,
          static_cast<int64_t>(bits::roundUp(size_, kBlockSize)) *
              static_cast<int64_t>(sizeof(int32_t)),
          bytes);
      VELOX_CHECK_EQ(children.size(), kind_ == TypeKind::ARRAY ? 1 : 2);
      if (kind_ == TypeKind::ARRAY) {
        return std::make_shared<ArrayVector>(
            pool,
            type_,
            veloxNulls(size),
            size,
            std::move(offsets),
            std::move(sizes),
            children.at(0));
      }
      return std::make_shared<MapVector>(
          pool,
          type_,
          veloxNulls(size),
          size,
          std::move(offsets),
          std::move(sizes),
          children.at(0),
          children.at(1));
    }
    case TypeKind::ROW: {
      std::vector<VectorPtr> children;
      children.reserve(children_.size());
      for (auto& child : children_) {
        children.push_back(child->toVeloxBase(pool, size));
      }
      return std::make_shared<RowVector>(
          pool, type_, veloxNulls(size), size, std::move(children));
    }
    default:
      // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
      return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          toVeloxTyped, kind_, size, pool, type_, values_, nulls_);
  }
}

VectorPtr WaveVector::toVelox(
    memory::MemoryPool* pool,
    int32_t numBlocks,
    const BlockStatus* status,
    const Operand* operand) {
  if (!status || !operand) {
    return toVeloxBase(pool, size_);
  }
  auto operandIndices = operand->indices;
  // If there is a wrap, any row can be referenced, so the vector is
  // with size_ of WaveVector. If there is no wrap, it is mapped to
  // the end of the last BlockStatus. If the blocks are densely filled
  // we have the vector at right size with no wrap on top.
  int32_t filledSize = operandIndices
      ? size_
      : (kBlockSize * (numBlocks - 1)) + status[numBlocks - 1].numRows;

  // Translate the BlockStatus and indices in Operand to a host side dictionary
  // wrap.
  int maxRow = std::min<int32_t>(filledSize, numBlocks * kBlockSize);
  numBlocks = bits::roundUp(maxRow, kBlockSize) / kBlockSize;
  int numActive = statusNumRows(status, numBlocks);
  if (!operandIndices) {
    // Vector sizes are >= active in status because they are allocated
    // before the row count in status becomes known. There is always
    // space up to the end of a full block in a partly filled vector.
    VELOX_CHECK_LE(
        numActive,
        bits::roundUp(size_, kBlockSize),
        "If there is no indirection in Operand, vector size must be >= BlockStatus");
    // If all blocks except last are full we return base without wrap.
    if (isDenselyFilled(status, numBlocks)) {
      return toVeloxBase(pool, filledSize);
    }
  }
  auto indices = AlignedBuffer::allocate<vector_size_t>(numActive, pool);
  auto rawIndices = indices->asMutable<vector_size_t>();
  int32_t fill = 0;
  for (auto block = 0; block < numBlocks; ++block) {
    auto blockIndices = operandIndices ? operandIndices[block] : nullptr;
    if (!blockIndices) {
      for (auto i = 0; i < status[block].numRows; ++i) {
        rawIndices[fill++] = block * kBlockSize + i;
      }
    } else {
      memcpy(
          rawIndices + fill,
          blockIndices,
          status[block].numRows * sizeof(int32_t));
      fill += status[block].numRows;
    }
  }
  // Rows that are not selected may never have been written. Strings must not
  // follow their pointers.
  std::vector<bool> selected(filledSize, false);
  for (auto i = 0; i < numActive; ++i) {
    selected[rawIndices[i]] = true;
  }
  auto base = toVeloxBase(pool, filledSize, &selected);
  return BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, numActive, base);
}

} // namespace facebook::velox::wave
