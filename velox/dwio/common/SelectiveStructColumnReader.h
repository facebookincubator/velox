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

#pragma once

#include "velox/dwio/common/SelectiveColumnReaderInternal.h"

namespace facebook::velox::dwio::common {

template <typename T, typename KeyNode, typename FormatData>
class SelectiveFlatMapColumnReaderHelper;

class SelectiveStructColumnReaderBase : public SelectiveColumnReader {
 public:
  void resetFilterCaches() override {
    for (auto& child : children_) {
      child->resetFilterCaches();
    }
  }

  uint64_t skip(uint64_t numValues) override;

  void next(uint64_t numValues, VectorPtr& result, const Mutation*) override;

  void filterRowGroups(
      uint64_t rowGroupSize,
      const dwio::common::StatsContext& context,
      FormatData::FilterRowGroupsResult&) const override;

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override;

  void getValues(RowSet rows, VectorPtr* result) override;

  uint64_t numReads() const {
    return numReads_;
  }

  vector_size_t lazyVectorReadOffset() const {
    return lazyVectorReadOffset_;
  }

  /// Advance field reader to the row group closest to specified offset by
  /// calling seekToRowGroup.
  virtual void advanceFieldReader(
      SelectiveColumnReader* reader,
      vector_size_t offset) = 0;

  // Returns the nulls bitmap from reading this. Used in LazyVector loaders.
  const uint64_t* nulls() const {
    return nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
  }

  void setReadOffsetRecursive(vector_size_t readOffset) override {
    readOffset_ = readOffset;
    for (auto& child : children_) {
      child->setReadOffsetRecursive(readOffset);
    }
  }

  void setIsTopLevel() override {
    isTopLevel_ = true;
    if (!formatData_->hasNulls()) {
      for (auto& child : children_) {
        child->setIsTopLevel();
      }
    }
  }

  const std::vector<SelectiveColumnReader*>& children() const override {
    return children_;
  }

  // Sets 'rows' as the set of rows for which 'this' or its children
  // may be loaded as LazyVectors. When a struct is loaded as lazy,
  // its children will be lazy if the struct does not add nulls. The
  // children will reference the struct reader, whih must have a live
  // and up-to-date set of rows for which children can be loaded.
  void setLoadableRows(RowSet rows) {
    setOutputRows(rows);
    inputRows_ = outputRows_;
  }

  const std::string& debugString() const {
    return debugString_;
  }

 protected:
  template <typename T, typename KeyNode, typename FormatData>
  friend class SelectiveFlatMapColumnReaderHelper;

  // The subscript of childSpecs will be set to this value if the column is
  // constant (either explicitly or because it's missing).
  static constexpr int32_t kConstantChildSpecSubscript = -1;

  SelectiveStructColumnReaderBase(
      const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      FormatParams& params,
      velox::common::ScanSpec& scanSpec,
      bool isRoot = false)
      : SelectiveColumnReader(fileType->type(), fileType, params, scanSpec),
        requestedType_(requestedType),
        debugString_(
            getExceptionContext().message(VeloxException::Type::kSystem)),
        isRoot_(isRoot) {}

  // Records the number of nulls added by 'this' between the end
  // position of each child reader and the end of the range of
  // 'read(). This must be done also if a child is not read so that we
  // know how much to skip when seeking forward within the row group.
  void recordParentNullsInChildren(vector_size_t offset, RowSet rows);

  bool hasMutation() const override {
    return hasMutation_;
  }

  // Returns true if we'll return a constant for that childSpec (i.e. we don't
  // need to read it).
  bool isChildConstant(const velox::common::ScanSpec& childSpec) const;

  const std::shared_ptr<const dwio::common::TypeWithId> requestedType_;

  std::vector<SelectiveColumnReader*> children_;

  // Sequence number of output batch. Checked against ColumnLoaders
  // created by 'this' to verify they are still valid at load.
  uint64_t numReads_ = 0;

  vector_size_t lazyVectorReadOffset_;

  // Dense set of rows to read in next().
  raw_vector<vector_size_t> rows_;

  const Mutation* mutation_ = nullptr;

  // After read() call mutation_ could go out of scope.  Need to keep this
  // around for lazy columns.
  bool hasMutation_ = false;

  // Context information obtained from ExceptionContext. Stored here
  // so that LazyVector readers under this can add this to their
  // ExceptionContext. Allows contextualizing reader errors to split
  // and query. Set at construction, which takes place on first
  // use. If no ExceptionContext is in effect, this is "".
  const std::string debugString_;

  // Whether or not this is the root Struct that represents entire rows of the
  // table.
  const bool isRoot_;
};

struct SelectiveStructColumnReader : SelectiveStructColumnReaderBase {
  using SelectiveStructColumnReaderBase::SelectiveStructColumnReaderBase;

  void addChild(std::unique_ptr<SelectiveColumnReader> child) {
    children_.push_back(child.get());
    childrenOwned_.push_back(std::move(child));
  }

 private:
  // Store the actual child readers.  In `children_` we only kept the raw
  // pointers and do not have ownership.
  std::vector<std::unique_ptr<SelectiveColumnReader>> childrenOwned_;
};

template <typename T, typename KeyNode, typename FormatData>
class SelectiveFlatMapColumnReaderHelper {
 public:
  SelectiveFlatMapColumnReaderHelper(
      SelectiveStructColumnReaderBase& reader,
      std::vector<KeyNode>&& keyNodes)
      : reader_(reader), keyNodes_(std::move(keyNodes)) {
    reader_.children_.resize(keyNodes_.size());
    for (int i = 0; i < keyNodes_.size(); ++i) {
      reader_.children_[i] = keyNodes_[i].reader.get();
      reader_.children_[i]->setIsFlatMapValue(true);
    }
    if (auto type = reader_.requestedType_->type()->childAt(1); type->isRow()) {
      childValues_ = BaseVector::create(type, 0, &reader_.memoryPool_);
    }
  }

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls);

  void getValues(RowSet rows, VectorPtr* result);

 private:
  MapVector& prepareResult(VectorPtr& result, vector_size_t size) {
    if (result && result->encoding() == VectorEncoding::Simple::MAP &&
        result.unique()) {
      result->resetDataDependentFlags(nullptr);
      result->resize(size);
    } else {
      VLOG(1) << "Reallocating result MAP vector of size " << size;
      result = BaseVector::create(
          reader_.requestedType_->type(), size, &reader_.memoryPool_);
    }
    return *result->asUnchecked<MapVector>();
  }

  vector_size_t
  calculateOffsets(RowSet rows, vector_size_t* offsets, vector_size_t* sizes);

  template <TypeKind kKind>
  void copyValues(
      RowSet rows,
      FlatVector<T>* flatKeys,
      vector_size_t* rawOffsets,
      BaseVector& values);

  SelectiveStructColumnReaderBase& reader_;
  std::vector<KeyNode> keyNodes_;
  VectorPtr childValues_;
  DecodedVector decodedChildValues_;
  std::vector<const uint64_t*> inMaps_;
  std::vector<uint64_t> columnRowBits_;
  std::vector<BaseVector::CopyRange> copyRanges_;
};

template <typename T, typename KeyNode, typename FormatData>
void SelectiveFlatMapColumnReaderHelper<T, KeyNode, FormatData>::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  reader_.numReads_ = reader_.scanSpec_->newRead();
  reader_.prepareRead<char>(offset, rows, incomingNulls);
  VELOX_DCHECK(!reader_.hasMutation());
  auto activeRows = rows;
  auto* mapNulls = reader_.nullsInReadRange_
      ? reader_.nullsInReadRange_->as<uint64_t>()
      : nullptr;
  if (reader_.scanSpec_->filter()) {
    auto kind = reader_.scanSpec_->filter()->kind();
    VELOX_CHECK(
        kind == velox::common::FilterKind::kIsNull ||
        kind == velox::common::FilterKind::kIsNotNull);
    reader_.filterNulls<int32_t>(
        rows, kind == velox::common::FilterKind::kIsNull, false);
    if (reader_.outputRows_.empty()) {
      for (auto* child : reader_.children_) {
        child->addParentNulls(offset, mapNulls, rows);
      }
      return;
    }
    activeRows = reader_.outputRows_;
  }
  // Separate the loop to be cache friendly.
  for (auto* child : reader_.children_) {
    reader_.advanceFieldReader(child, offset);
  }
  for (auto* child : reader_.children_) {
    child->read(offset, activeRows, mapNulls);
    child->addParentNulls(offset, mapNulls, rows);
  }
  reader_.lazyVectorReadOffset_ = offset;
  reader_.readOffset_ = offset + rows.back() + 1;
}

template <typename T, typename KeyNode, typename FormatData>
vector_size_t
SelectiveFlatMapColumnReaderHelper<T, KeyNode, FormatData>::calculateOffsets(
    RowSet rows,
    vector_size_t* offsets,
    vector_size_t* sizes) {
  auto* nulls = reader_.nullsInReadRange_
      ? reader_.nullsInReadRange_->as<uint64_t>()
      : nullptr;
  inMaps_.resize(reader_.children_.size());
  for (int k = 0; k < reader_.children_.size(); ++k) {
    auto& data =
        static_cast<const FormatData&>(reader_.children_[k]->formatData());
    inMaps_[k] = data.inMap();
    if (!inMaps_[k]) {
      inMaps_[k] = nulls;
    }
  }
  columnRowBits_.resize(bits::nwords(reader_.children_.size() * rows.size()));
  std::fill(columnRowBits_.begin(), columnRowBits_.end(), 0);
  std::fill(sizes, sizes + rows.size(), 0);
  for (int k = 0; k < reader_.children_.size(); ++k) {
    if (inMaps_[k]) {
      for (vector_size_t i = 0; i < rows.size(); ++i) {
        if (bits::isBitSet(inMaps_[k], rows[i])) {
          bits::setBit(columnRowBits_.data(), i + k * rows.size());
          ++sizes[i];
        }
      }
    } else {
      bits::fillBits(
          columnRowBits_.data(), k * rows.size(), (k + 1) * rows.size(), true);
      for (vector_size_t i = 0; i < rows.size(); ++i) {
        ++sizes[i];
      }
    }
  }
  vector_size_t numNestedRows = 0;
  for (vector_size_t i = 0; i < rows.size(); ++i) {
    if (!reader_.returnReaderNulls_ && nulls &&
        bits::isBitNull(nulls, rows[i])) {
      bits::setNull(reader_.rawResultNulls_, i);
      reader_.anyNulls_ = true;
    }
    offsets[i] = numNestedRows;
    numNestedRows += sizes[i];
  }
  return numNestedRows;
}

template <typename T, typename KeyNode, typename FormatData>
template <TypeKind kKind>
void SelectiveFlatMapColumnReaderHelper<T, KeyNode, FormatData>::copyValues(
    RowSet rows,
    FlatVector<T>* flatKeys,
    vector_size_t* rawOffsets,
    BaseVector& values) {
  // String values are not copied directly because currently we don't have
  // them in production so no need to optimize.
  constexpr bool kDirectCopy =
      TypeKind::TINYINT <= kKind && kKind <= TypeKind::DOUBLE;
  using ValueType = typename TypeTraits<kKind>::NativeType;
  T* rawKeys = flatKeys->mutableRawValues();
  [[maybe_unused]] size_t strKeySize;
  [[maybe_unused]] char* rawStrKeyBuffer;
  if constexpr (std::is_same_v<T, StringView>) {
    strKeySize = 0;
    for (int k = 0; k < reader_.children_.size(); ++k) {
      if (!keyNodes_[k].key.get().isInline()) {
        strKeySize += keyNodes_[k].key.get().size();
      }
    }
    if (strKeySize > 0) {
      auto buf =
          AlignedBuffer::allocate<char>(strKeySize, &reader_.memoryPool_);
      rawStrKeyBuffer = buf->template asMutable<char>();
      flatKeys->addStringBuffer(buf);
      strKeySize = 0;
      for (int k = 0; k < reader_.children_.size(); ++k) {
        auto& s = keyNodes_[k].key.get();
        if (!s.isInline()) {
          memcpy(&rawStrKeyBuffer[strKeySize], s.data(), s.size());
          strKeySize += s.size();
        }
      }
      strKeySize = 0;
    }
  }
  [[maybe_unused]] ValueType* targetValues;
  [[maybe_unused]] uint64_t* targetNulls;
  if constexpr (kDirectCopy) {
    VELOX_CHECK(values.isFlatEncoding());
    auto* flat = values.asUnchecked<FlatVector<ValueType>>();
    targetValues = flat->mutableRawValues();
    targetNulls = flat->mutableRawNulls();
  }
  for (int k = 0; k < reader_.children_.size(); ++k) {
    [[maybe_unused]] StringView strKey;
    if constexpr (std::is_same_v<T, StringView>) {
      strKey = keyNodes_[k].key.get();
      if (!strKey.isInline()) {
        strKey = {
            &rawStrKeyBuffer[strKeySize], static_cast<int32_t>(strKey.size())};
        strKeySize += strKey.size();
      }
    }
    reader_.children_[k]->getValues(rows, &childValues_);
    if constexpr (kDirectCopy) {
      decodedChildValues_.decode(*childValues_);
    }
    const auto begin = k * rows.size();
    bits::forEachSetBit(
        columnRowBits_.data(),
        begin,
        begin + rows.size(),
        [&](vector_size_t i) {
          i -= begin;
          if constexpr (std::is_same_v<T, StringView>) {
            rawKeys[rawOffsets[i]] = strKey;
          } else {
            rawKeys[rawOffsets[i]] = keyNodes_[k].key.get();
          }
          if constexpr (kDirectCopy) {
            targetValues[rawOffsets[i]] =
                decodedChildValues_.valueAt<ValueType>(i);
            bits::setNull(
                targetNulls, rawOffsets[i], decodedChildValues_.isNullAt(i));
          } else {
            copyRanges_.push_back({
                .sourceIndex = i,
                .targetIndex = rawOffsets[i],
                .count = 1,
            });
          }
          ++rawOffsets[i];
        });
    if constexpr (!kDirectCopy) {
      values.copyRanges(childValues_.get(), copyRanges_);
      copyRanges_.clear();
    }
  }
}

template <typename T, typename KeyNode, typename FormatData>
void SelectiveFlatMapColumnReaderHelper<T, KeyNode, FormatData>::getValues(
    RowSet rows,
    VectorPtr* result) {
  auto& mapResult = prepareResult(*result, rows.size());
  auto* rawOffsets = mapResult.mutableOffsets(rows.size())
                         ->template asMutable<vector_size_t>();
  auto* rawSizes =
      mapResult.mutableSizes(rows.size())->template asMutable<vector_size_t>();
  auto numNestedRows = calculateOffsets(rows, rawOffsets, rawSizes);
  auto& keys = mapResult.mapKeys();
  auto& values = mapResult.mapValues();
  BaseVector::prepareForReuse(keys, numNestedRows);
  BaseVector::prepareForReuse(values, numNestedRows);
  auto* flatKeys = keys->template asFlatVector<T>();
  VELOX_DYNAMIC_TYPE_DISPATCH(
      copyValues, values->typeKind(), rows, flatKeys, rawOffsets, *values);
  VELOX_CHECK_EQ(rawOffsets[rows.size() - 1], numNestedRows);
  std::copy_backward(
      rawOffsets, rawOffsets + rows.size() - 1, rawOffsets + rows.size());
  rawOffsets[0] = 0;
  result->get()->setNulls(reader_.resultNulls());
}

} // namespace facebook::velox::dwio::common
