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

#include "velox/vector/ComplexVector.h"
#include <sstream>
#include "velox/vector/SimpleVector.h"

namespace facebook {
namespace velox {

// Up to # of elements to show as debug string for `toString()`.
constexpr vector_size_t LIMIT_ELEMENTS_TO_SHOW = 5;

std::string stringifyTruncatedElementList(
    vector_size_t start,
    vector_size_t size,
    vector_size_t limit,
    std::string_view delimiter,
    const std::function<void(std::stringstream&, vector_size_t)>&
        stringifyElementCB) {
  std::stringstream out;
  if (size == 0) {
    return "<empty>";
  }
  out << size << " elements starting at " << start << " {";

  bool first = true;
  const vector_size_t limitedSize = std::min(size, limit);
  for (vector_size_t i = 0; i < limitedSize; ++i) {
    if (first) {
      first = false;
    } else {
      out << delimiter;
    }
    stringifyElementCB(out, start + i);
  }

  if (size > limitedSize) {
    if (!first) {
      out << delimiter;
    }
    out << "...";
  }
  out << "}";
  return out.str();
}

// static
std::shared_ptr<RowVector> RowVector::createEmpty(
    std::shared_ptr<const Type> type,
    velox::memory::MemoryPool* pool) {
  VELOX_CHECK(type->isRow());
  return std::static_pointer_cast<RowVector>(BaseVector::create(type, 0, pool));
}

bool RowVector::equalValueAt(
    const BaseVector* other,
    vector_size_t index,
    vector_size_t otherIndex) const {
  bool isNull = isNullAt(index);
  bool otherNull = other->isNullAt(otherIndex);
  if (isNull && otherNull) {
    return true;
  }
  if (isNull || otherNull) {
    return false;
  }
  auto otherRow = other->wrappedVector()->as<RowVector>();
  if (otherRow->encoding() != VectorEncoding::Simple::ROW) {
    return false;
  }
  if (children_.size() != otherRow->children_.size()) {
    return false;
  }
  auto wrappedIndex = other->wrappedIndex(otherIndex);
  for (int32_t i = 0; i < children_.size(); ++i) {
    BaseVector* child = children_[i].get();
    BaseVector* otherChild = otherRow->loadedChildAt(i).get();
    if (!child && !otherChild) {
      continue;
    }
    if (!child || !otherChild) {
      return false;
    }
    if (child->typeKind() != otherChild->typeKind()) {
      return false;
    }
    if (!child->equalValueAt(otherChild, index, wrappedIndex)) {
      return false;
    }
  }
  return true;
}

int32_t RowVector::compare(
    const BaseVector* other,
    vector_size_t index,
    vector_size_t otherIndex,
    CompareFlags flags) const {
  auto otherRow = other->wrappedVector()->as<RowVector>();
  if (otherRow->encoding() != VectorEncoding::Simple::ROW) {
    VELOX_CHECK(
        false,
        "Compare of ROW and non-ROW {} and {}",
        BaseVector::toString(),
        otherRow->BaseVector::toString());
  }

  bool isNull = isNullAt(index);
  auto wrappedOtherIndex = other->wrappedIndex(otherIndex);
  bool otherNull = otherRow->isNullAt(wrappedOtherIndex);
  if (isNull) {
    if (otherNull) {
      return 0;
    }
    return flags.nullsFirst ? -1 : 1;
  }
  if (otherNull) {
    return flags.nullsFirst ? 1 : -1;
  }

  auto compareSize = std::min(children_.size(), otherRow->children_.size());
  for (int32_t i = 0; i < compareSize; ++i) {
    BaseVector* child = children_[i].get();
    BaseVector* otherChild = otherRow->loadedChildAt(i).get();
    if (!child && !otherChild) {
      continue;
    }
    if (!child || !otherChild) {
      return child ? 1 : -1; // Absent child counts as less.
    }
    if (child->typeKind() != otherChild->typeKind()) {
      VELOX_CHECK(
          false,
          "Compare of different child types: {} and {}",
          BaseVector::toString(),
          other->BaseVector::toString());
    }
    auto result = child->compare(otherChild, index, wrappedOtherIndex, flags);
    if (result) {
      return result;
    }
  }
  return children_.size() - otherRow->children_.size();
}

void RowVector::appendToChildren(
    const RowVector* source,
    vector_size_t sourceIndex,
    vector_size_t count,
    vector_size_t index) {
  for (int32_t i = 0; i < children_.size(); ++i) {
    auto& child = children_[i];
    child->copy(source->loadedChildAt(i).get(), index, sourceIndex, count);
  }
}

void RowVector::copy(
    const BaseVector* source,
    vector_size_t targetIndex,
    vector_size_t sourceIndex,
    vector_size_t count) {
  auto sourceValue = source->wrappedVector();
  if (sourceValue->isConstantEncoding()) {
    // A null constant does not have a value vector, so wrappedVector
    // returns the constant.
    VELOX_CHECK(sourceValue->isNullAt(0));
    for (auto i = 0; i < count; ++i) {
      setNull(targetIndex + i, true);
    }
    return;
  }
  if (childrenSize_ == 0) {
    return;
  }
  VELOX_CHECK_EQ(sourceValue->encoding(), VectorEncoding::Simple::ROW);
  auto sourceAsRow = sourceValue->asUnchecked<RowVector>();
  VELOX_CHECK(children_.size() && children_[0]);
  VELOX_DCHECK(BaseVector::length_ >= targetIndex + count);
  vector_size_t childSize = this->childSize();
  auto rowType = type()->as<TypeKind::ROW>();
  SelectivityVector allRows;
  for (int32_t i = 0; i < children_.size(); ++i) {
    auto& child = children_[i];
    if (child->isConstantEncoding()) {
      if (!allRows.size()) {
        // Initialize 'allRows' on first use.
        allRows.resize(childSize);
        allRows.clearAll();
      }
      BaseVector::ensureWritable(allRows, rowType.childAt(i), pool(), &child);
    } else {
      // Non-constants will become writable at their original size.
      BaseVector::ensureWritable(
          SelectivityVector::empty(), rowType.childAt(i), pool(), &child);
    }
    if (childSize < targetIndex + count) {
      child->resize(targetIndex + count);
    }
  }
  // Shortcut for insert of non-null at end of children.
  if (!source->mayHaveNulls() && targetIndex == childSize) {
    if (sourceAsRow == source) {
      appendToChildren(sourceAsRow, sourceIndex, count, targetIndex);
    } else {
      for (int32_t i = 0; i < count; ++i) {
        appendToChildren(
            sourceAsRow,
            source->wrappedIndex(sourceIndex + i),
            1,
            childSize + i);
      }
    }
    return;
  }
  auto setNotNulls = mayHaveNulls() || source->mayHaveNulls();
  for (int32_t i = 0; i < count; ++i) {
    auto childIndex = targetIndex + i;
    if (source->isNullAt(sourceIndex + i)) {
      setNull(childIndex, true);
    } else {
      if (setNotNulls) {
        setNull(childIndex, false);
      }
      vector_size_t wrappedIndex = source->wrappedIndex(sourceIndex + i);
      for (int32_t j = 0; j < children_.size(); ++j) {
        childAt(j)->copy(
            sourceAsRow->loadedChildAt(j).get(), childIndex, wrappedIndex, 1);
      }
    }
  }
}

void RowVector::move(vector_size_t source, vector_size_t target) {
  VELOX_CHECK_LT(source, size());
  VELOX_CHECK_LT(target, size());
  if (source != target) {
    for (auto& child : children_) {
      if (child) {
        child->move(source, target);
      }
    }
  }
}

uint64_t RowVector::hashValueAt(vector_size_t index) const {
  if (isNullAt(index)) {
    return BaseVector::kNullHash;
  }
  uint64_t hash = BaseVector::kNullHash;
  bool isFirst = true;
  for (auto i = 0; i < childrenSize(); ++i) {
    auto& child = children_[i];
    if (child) {
      auto childHash = child->hashValueAt(index);
      hash = isFirst ? childHash : bits::hashMix(hash, childHash);
      isFirst = false;
    }
  }
  return hash;
}

std::unique_ptr<SimpleVector<uint64_t>> RowVector::hashAll() const {
  VELOX_NYI();
}

std::string RowVector::toString(vector_size_t index) const {
  if (isNullAt(index)) {
    return "null";
  }
  std::stringstream out;
  out << "{ [child at " << index << "]: ";
  for (int32_t i = 0; i < children_.size(); ++i) {
    out << (children_[i] ? children_[i]->toString(index) : "<not set>");
    out << ((i < children_.size() - 1) ? ", " : "}");
  }
  return out.str();
}

void RowVector::ensureWritable(const SelectivityVector& rows) {
  for (int i = 0; i < childrenSize_; i++) {
    if (children_[i]) {
      BaseVector::ensureWritable(
          rows, children_[i]->type(), BaseVector::pool_, &children_[i]);
    }
  }
  BaseVector::ensureWritable(rows);
}

bool ArrayVector::equalValueAt(
    const BaseVector* other,
    vector_size_t index,
    vector_size_t otherIndex) const {
  bool isNull = isNullAt(index);
  bool otherNull = other->isNullAt(otherIndex);
  if (isNull && otherNull) {
    return true;
  }
  if (isNull || otherNull) {
    return false;
  }
  auto otherValue = other->wrappedVector();
  if (otherValue->encoding() != VectorEncoding::Simple::ARRAY) {
    return false;
  }
  auto otherArray = otherValue->asUnchecked<ArrayVector>();
  auto wrappedIndex = other->wrappedIndex(otherIndex);
  if (rawSizes_[index] != otherArray->rawSizes_[wrappedIndex]) {
    return false;
  }
  if (rawSizes_[index] == 0) {
    return true;
  }
  auto otherElements = otherArray->elements_.get();
  if (elements_->typeKind() != otherElements->typeKind()) {
    return false;
  }

  auto offset = rawOffsets_[index];
  auto otherOffset = otherArray->rawOffsets_[wrappedIndex];
  for (int32_t i = 0; i < rawSizes_[index]; ++i) {
    if (!elements_->equalValueAt(otherElements, offset + i, otherOffset + i)) {
      return false;
    }
  }
  return true;
}
namespace {
int compareArrays(
    const BaseVector& left,
    const BaseVector& right,
    IndexRange leftRange,
    IndexRange rightRange,
    CompareFlags flags) {
  if (flags.equalsOnly && leftRange.size != rightRange.size) {
    // return early if not caring about collation order.
    return 1;
  }
  auto compareSize = std::min(leftRange.size, rightRange.size);
  for (auto i = 0; i < compareSize; ++i) {
    auto result =
        left.compare(&right, leftRange.begin + i, rightRange.begin + i, flags);
    if (result) {
      return result;
    }
  }
  int result = leftRange.size - rightRange.size;
  return flags.ascending ? result : result * -1;
}

int compareArrays(
    const BaseVector& left,
    const BaseVector& right,
    folly::Range<const vector_size_t*> leftRange,
    folly::Range<const vector_size_t*> rightRange,
    CompareFlags flags) {
  if (flags.equalsOnly && leftRange.size() != rightRange.size()) {
    // return early if not caring about collation order.
    return 1;
  }
  auto compareSize = std::min(leftRange.size(), rightRange.size());
  for (auto i = 0; i < compareSize; ++i) {
    auto result = left.compare(&right, leftRange[i], rightRange[i], flags);
    if (result) {
      return result;
    }
  }
  int result = leftRange.size() - rightRange.size();
  return flags.ascending ? result : result * -1;
}
} // namespace

int32_t ArrayVector::compare(
    const BaseVector* other,
    vector_size_t index,
    vector_size_t otherIndex,
    CompareFlags flags) const {
  bool isNull = isNullAt(index);
  bool otherNull = other->isNullAt(otherIndex);
  if (isNull) {
    if (otherNull) {
      return 0;
    }
    return flags.nullsFirst ? -1 : 1;
  }
  if (otherNull) {
    return flags.nullsFirst ? 1 : -1;
  }
  auto otherValue = other->wrappedVector();
  auto wrappedOtherIndex = other->wrappedIndex(otherIndex);
  VELOX_CHECK_EQ(
      VectorEncoding::Simple::ARRAY,
      otherValue->encoding(),
      "Compare of ARRAY and non-ARRAY: {} and {}",
      BaseVector::toString(),
      other->BaseVector::toString());

  auto otherArray = otherValue->asUnchecked<ArrayVector>();
  auto otherElements = otherArray->elements_.get();
  if (elements_->typeKind() != otherElements->typeKind()) {
    VELOX_CHECK(
        false,
        "Compare of arrays of different element type: {} and {}",
        BaseVector::toString(),
        otherArray->BaseVector::toString());
  }
  return compareArrays(
      *elements_,
      *otherArray->elements_,
      IndexRange{rawOffsets_[index], rawSizes_[index]},
      IndexRange{
          otherArray->rawOffsets_[wrappedOtherIndex],
          otherArray->rawSizes_[wrappedOtherIndex]},
      flags);
}

namespace {
uint64_t hashArray(
    uint64_t hash,
    const BaseVector& elements,
    vector_size_t offset,
    vector_size_t size) {
  for (auto i = 0; i < size; ++i) {
    auto elementHash = elements.hashValueAt(offset + i);
    hash = bits::commutativeHashMix(hash, elementHash);
  }
  return hash;
}
} // namespace

uint64_t ArrayVector::hashValueAt(vector_size_t index) const {
  if (isNullAt(index)) {
    return BaseVector::kNullHash;
  }
  return hashArray(
      BaseVector::kNullHash, *elements_, rawOffsets_[index], rawSizes_[index]);
}

std::unique_ptr<SimpleVector<uint64_t>> ArrayVector::hashAll() const {
  VELOX_NYI();
}

void ArrayVector::copy(
    const BaseVector* source,
    vector_size_t targetIndex,
    vector_size_t sourceIndex,
    vector_size_t count) {
  auto sourceValue = source->wrappedVector();
  if (sourceValue->isConstantEncoding()) {
    // A null constant does not have a value vector, so wrappedVector
    // returns the constant.
    VELOX_CHECK(sourceValue->isNullAt(0));
    for (auto i = 0; i < count; ++i) {
      setNull(targetIndex + i, true);
    }
    return;
  }
  VELOX_CHECK_EQ(sourceValue->encoding(), VectorEncoding::Simple::ARRAY);
  auto sourceArray = sourceValue->asUnchecked<ArrayVector>();
  VELOX_DCHECK(BaseVector::length_ >= targetIndex + count);
  BaseVector::ensureWritable(
      SelectivityVector::empty(), elements_->type(), pool(), &elements_);
  auto setNotNulls = mayHaveNulls() || source->mayHaveNulls();
  auto wantWidth = type()->isFixedWidth() ? type()->fixedElementsWidth() : 0;
  for (int32_t i = 0; i < count; ++i) {
    if (source->isNullAt(sourceIndex + i)) {
      setNull(targetIndex + i, true);
    } else {
      if (setNotNulls) {
        setNull(targetIndex + i, false);
      }
      vector_size_t wrappedIndex = source->wrappedIndex(sourceIndex + i);
      vector_size_t copySize = sourceArray->sizeAt(wrappedIndex);
      vector_size_t childSize = elements_->size();
      if (copySize > 0) {
        // If we are populating a FixedSizeArray we validate here that
        // the entries we are populating are the correct sizes.
        if (wantWidth != 0) {
          VELOX_CHECK_EQ(
              copySize,
              wantWidth,
              "Invalid length element at index {}, wrappedIndex {}",
              i,
              wrappedIndex);
        }
        elements_->resize(childSize + copySize);
        elements_->copy(
            sourceArray->elements_.get(),
            childSize,
            sourceArray->offsetAt(wrappedIndex),
            copySize);
      }
      setOffsetAndSize(targetIndex + i, childSize, copySize);
    }
  }
}

void ArrayVector::move(vector_size_t source, vector_size_t target) {
  VELOX_CHECK_LT(source, size());
  VELOX_CHECK_LT(target, size());
  if (source != target) {
    if (isNullAt(source)) {
      setNull(target, true);
    } else {
      offsets_->asMutable<vector_size_t>()[target] = rawOffsets_[source];
      sizes_->asMutable<vector_size_t>()[target] = rawSizes_[source];
    }
  }
}

std::string ArrayVector::toString(vector_size_t index) const {
  if (isNullAt(index)) {
    return "null";
  }

  return stringifyTruncatedElementList(
      rawOffsets_[index],
      rawSizes_[index],
      LIMIT_ELEMENTS_TO_SHOW,
      ", ",
      [this](std::stringstream& ss, vector_size_t index) {
        ss << elements_->toString(index);
      });
}

void ArrayVector::ensureWritable(const SelectivityVector& rows) {
  auto newSize = std::max<vector_size_t>(rows.size(), BaseVector::length_);
  if (offsets_ && !offsets_->unique()) {
    BufferPtr newOffsets =
        AlignedBuffer::allocate<vector_size_t>(newSize, BaseVector::pool_);
    auto rawNewOffsets = newOffsets->asMutable<vector_size_t>();

    // Copy the whole buffer. An alternative could be
    // (1) fill the buffer with zeros and copy over elements not in "rows";
    // (2) or copy over elements not in "rows" and mark "rows" elements as null
    // Leaving offsets or sizes of "rows" elements unspecified leaves the
    // vector in unusable state.
    memcpy(
        rawNewOffsets,
        rawOffsets_,
        byteSize<vector_size_t>(BaseVector::length_));

    offsets_ = std::move(newOffsets);
    rawOffsets_ = offsets_->as<vector_size_t>();
  }

  if (sizes_ && !sizes_->unique()) {
    BufferPtr newSizes =
        AlignedBuffer::allocate<vector_size_t>(newSize, BaseVector::pool_);
    auto rawNewSizes = newSizes->asMutable<vector_size_t>();
    memcpy(
        rawNewSizes, rawSizes_, byteSize<vector_size_t>(BaseVector::length_));

    sizes_ = std::move(newSizes);
    rawSizes_ = sizes_->asMutable<vector_size_t>();
  }

  // Vectors are write-once and nested elements are append only,
  // hence, all values already written must be preserved.
  BaseVector::ensureWritable(
      SelectivityVector::empty(),
      type()->childAt(0),
      BaseVector::pool_,
      &elements_);
  BaseVector::ensureWritable(rows);
}

bool MapVector::equalValueAt(
    const BaseVector* other,
    vector_size_t index,
    vector_size_t otherIndex) const {
  bool isNull = isNullAt(index);
  bool otherNull = other->isNullAt(otherIndex);
  if (isNull && otherNull) {
    return true;
  }
  if (isNull || otherNull) {
    return false;
  }
  auto otherValue = other->wrappedVector();
  if (otherValue->encoding() != VectorEncoding::Simple::MAP) {
    return false;
  }
  auto otherMap = otherValue->asUnchecked<MapVector>();
  auto wrappedIndex = other->wrappedIndex(otherIndex);
  if (rawSizes_[index] != otherMap->rawSizes_[wrappedIndex]) {
    return false;
  }
  if (rawSizes_[index] == 0) {
    return true;
  }
  auto otherKeys = otherMap->keys_.get();
  auto otherValues = otherMap->values_.get();
  if (keys_->typeKind() != otherKeys->typeKind()) {
    return false;
  }
  if (values_->typeKind() != otherValues->typeKind()) {
    return false;
  }

  auto offset = rawOffsets_[index];
  auto otherOffset = otherMap->rawOffsets_[wrappedIndex];
  int32_t mapSize = rawSizes_[index];
  std::vector<bool> othersMatch(mapSize, false);
  for (int32_t i = 0; i < mapSize; ++i) {
    bool found = false;
    for (int32_t j = 0; j < mapSize; ++j) {
      if (!othersMatch[j] &&
          keys_->equalValueAt(otherKeys, offset + i, otherOffset + j) &&
          values_->equalValueAt(otherValues, offset + i, otherOffset + j)) {
        found = true;
        othersMatch[j] = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }

  return true;
}

int32_t MapVector::compare(
    const BaseVector* other,
    vector_size_t index,
    vector_size_t otherIndex,
    CompareFlags flags) const {
  bool isNull = isNullAt(index);
  bool otherNull = other->isNullAt(otherIndex);
  if (isNull) {
    if (otherNull) {
      return 0;
    }
    return flags.nullsFirst ? -1 : 1;
  }
  if (otherNull) {
    return flags.nullsFirst ? 1 : -1;
  }
  auto otherValue = other->wrappedVector();
  auto wrappedOtherIndex = other->wrappedIndex(otherIndex);
  VELOX_CHECK_EQ(
      VectorEncoding::Simple::MAP,
      otherValue->encoding(),
      "Compare of MAP and non-MAP: {} and {}",
      BaseVector::toString(),
      otherValue->BaseVector::toString());
  auto otherMap = otherValue->as<MapVector>();

  if (keys_->typeKind() != otherMap->keys_->typeKind() ||
      values_->typeKind() != otherMap->values_->typeKind()) {
    VELOX_CHECK(
        false,
        "Compare of maps of different key/value types: {} and {}",
        BaseVector::toString(),
        otherMap->BaseVector::toString());
  }
  auto leftIndices = sortedKeyIndices(index);
  auto rightIndices = otherMap->sortedKeyIndices(wrappedOtherIndex);
  auto result =
      compareArrays(*keys_, *otherMap->keys_, leftIndices, rightIndices, flags);
  if (result) {
    return result;
  }
  return compareArrays(
      *values_, *otherMap->values_, leftIndices, rightIndices, flags);
}

uint64_t MapVector::hashValueAt(vector_size_t index) const {
  if (isNullAt(index)) {
    return BaseVector::kNullHash;
  }
  auto offset = rawOffsets_[index];
  auto size = rawSizes_[index];
  // We use a commutative hash mix, thus we do not sort first.
  return hashArray(
      hashArray(BaseVector::kNullHash, *keys_, offset, size),
      *values_,
      offset,
      size);
}

std::unique_ptr<SimpleVector<uint64_t>> MapVector::hashAll() const {
  VELOX_NYI();
}

vector_size_t MapVector::reserveMap(vector_size_t offset, vector_size_t size) {
  auto keySize = keys_->size();
  keys_->resize(keySize + size);
  values_->resize(keySize + size);
  offsets_->asMutable<vector_size_t>()[offset] = keySize;
  sizes_->asMutable<vector_size_t>()[offset] = size;
  return keySize;
}

void MapVector::copy(
    const BaseVector* source,
    vector_size_t targetIndex,
    vector_size_t sourceIndex,
    vector_size_t count) {
  auto sourceValue = source->wrappedVector();
  if (sourceValue->isConstantEncoding()) {
    // A null constant does not have a value vector, so wrappedVector
    // returns the constant.
    VELOX_CHECK(sourceValue->isNullAt(0));
    for (auto i = 0; i < count; ++i) {
      setNull(targetIndex + i, true);
    }
    return;
  }
  VELOX_CHECK_EQ(sourceValue->encoding(), VectorEncoding::Simple::MAP);
  VELOX_DCHECK(BaseVector::length_ >= targetIndex + count);
  auto sourceMap = sourceValue->asUnchecked<MapVector>();
  BaseVector::ensureWritable(
      SelectivityVector::empty(), keys_->type(), pool(), &keys_);
  auto setNotNulls = mayHaveNulls() || source->mayHaveNulls();
  for (int32_t i = 0; i < count; ++i) {
    if (source->isNullAt(sourceIndex + i)) {
      setNull(targetIndex + i, true);
    } else {
      if (setNotNulls) {
        setNull(targetIndex + i, false);
      }
      vector_size_t wrappedIndex = source->wrappedIndex(sourceIndex + i);
      vector_size_t copySize = sourceMap->sizeAt(wrappedIndex);
      // Call reserveMap also for 0 size, since this writes the offset/size.
      vector_size_t childSize = reserveMap(targetIndex + i, copySize);
      if (copySize > 0) {
        keys_->copy(
            sourceMap->keys_.get(),
            childSize,
            sourceMap->offsetAt(wrappedIndex),
            copySize);
        values_->copy(
            sourceMap->values_.get(),
            childSize,
            sourceMap->offsetAt(wrappedIndex),
            copySize);
      }
    }
  }
}

void MapVector::move(vector_size_t source, vector_size_t target) {
  VELOX_CHECK_LT(source, size());
  VELOX_CHECK_LT(target, size());
  if (source != target) {
    if (isNullAt(source)) {
      setNull(target, true);
    } else {
      offsets_->asMutable<vector_size_t>()[target] = rawOffsets_[source];
      sizes_->asMutable<vector_size_t>()[target] = rawSizes_[source];
    }
  }
}

bool MapVector::isSorted(vector_size_t index) const {
  if (isNullAt(index)) {
    return true;
  }
  auto offset = rawOffsets_[index];
  auto size = rawSizes_[index];
  for (auto i = 1; i < size; ++i) {
    if (keys_->compare(keys_.get(), offset + i - 1, offset + i) >= 0) {
      return false;
    }
  }
  return true;
}

// static
void MapVector::canonicalize(
    const std::shared_ptr<MapVector>& map,
    bool useStableSort) {
  if (map->sortedKeys_) {
    return;
  }
  // This is not safe if 'this' is referenced from other
  // threads. The keys and values do not have to be uniquely owned
  // since they are not mutated but rather transposed, which is
  // non-destructive.
  VELOX_CHECK(map.unique());
  BufferPtr indices;
  folly::Range<vector_size_t*> indicesRange;
  for (auto i = 0; i < map->BaseVector::length_; ++i) {
    if (map->isSorted(i)) {
      continue;
    }
    if (!indices) {
      indices = map->elementIndices();
      indicesRange = folly::Range<vector_size_t*>(
          indices->asMutable<vector_size_t>(), map->keys_->size());
    }
    auto offset = map->rawOffsets_[i];
    auto size = map->rawSizes_[i];
    if (useStableSort) {
      std::stable_sort(
          indicesRange.begin() + offset,
          indicesRange.begin() + offset + size,
          [&](vector_size_t left, vector_size_t right) {
            return map->keys_->compare(map->keys_.get(), left, right) < 0;
          });
    } else {
      std::sort(
          indicesRange.begin() + offset,
          indicesRange.begin() + offset + size,
          [&](vector_size_t left, vector_size_t right) {
            return map->keys_->compare(map->keys_.get(), left, right) < 0;
          });
    }
  }
  if (indices) {
    map->keys_ = BaseVector::transpose(indices, std::move(map->keys_));
    map->values_ = BaseVector::transpose(indices, std::move(map->values_));
  }
  map->sortedKeys_ = true;
}

std::vector<vector_size_t> MapVector::sortedKeyIndices(
    vector_size_t index) const {
  std::vector<vector_size_t> indices(rawSizes_[index]);
  std::iota(indices.begin(), indices.end(), rawOffsets_[index]);
  if (!sortedKeys_) {
    std::sort(
        indices.begin(),
        indices.end(),
        [&](vector_size_t left, vector_size_t right) {
          return keys_->compare(keys_.get(), left, right) < 0;
        });
  }
  return indices;
}

BufferPtr MapVector::elementIndices() const {
  auto numElements = keys_->size();
  BufferPtr buffer =
      AlignedBuffer::allocate<vector_size_t>(numElements, BaseVector::pool_);
  auto data = buffer->asMutable<vector_size_t>();
  auto range = folly::Range(data, numElements);
  std::iota(range.begin(), range.end(), 0);
  return buffer;
}

std::string MapVector::toString(vector_size_t index) const {
  if (isNullAt(index)) {
    return "null";
  }
  return stringifyTruncatedElementList(
      rawOffsets_[index],
      rawSizes_[index],
      LIMIT_ELEMENTS_TO_SHOW,
      ",\n ",
      [this](std::stringstream& ss, vector_size_t index) {
        ss << keys_->toString(index) << " = " << values_->toString(index);
      });
}

void MapVector::ensureWritable(const SelectivityVector& rows) {
  auto newSize = std::max<vector_size_t>(rows.size(), BaseVector::length_);
  if (offsets_ && !offsets_->unique()) {
    BufferPtr newOffsets =
        AlignedBuffer::allocate<vector_size_t>(newSize, BaseVector::pool_);
    auto rawNewOffsets = newOffsets->asMutable<vector_size_t>();

    // Copy the whole buffer. An alternative could be
    // (1) fill the buffer with zeros and copy over elements not in "rows";
    // (2) or copy over elements not in "rows" and mark "rows" elements as null
    // Leaving offsets or sizes of "rows" elements unspecified leaves the
    // vector in unusable state.
    memcpy(
        rawNewOffsets,
        rawOffsets_,
        byteSize<vector_size_t>(BaseVector::length_));

    offsets_ = std::move(newOffsets);
    rawOffsets_ = offsets_->as<vector_size_t>();
  }

  if (sizes_ && !sizes_->unique()) {
    BufferPtr newSizes =
        AlignedBuffer::allocate<vector_size_t>(newSize, BaseVector::pool_);
    auto rawNewSizes = newSizes->asMutable<vector_size_t>();
    memcpy(
        rawNewSizes, rawSizes_, byteSize<vector_size_t>(BaseVector::length_));

    sizes_ = std::move(newSizes);
    rawSizes_ = sizes_->as<vector_size_t>();
  }

  // Vectors are write-once and nested elements are append only,
  // hence, all values already written must be preserved.
  BaseVector::ensureWritable(
      SelectivityVector::empty(),
      type()->childAt(0),
      BaseVector::pool_,
      &keys_);
  BaseVector::ensureWritable(
      SelectivityVector::empty(),
      type()->childAt(1),
      BaseVector::pool_,
      &values_);
  BaseVector::ensureWritable(rows);
}

} // namespace velox
} // namespace facebook
