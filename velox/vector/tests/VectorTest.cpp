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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include "velox/common/memory/ByteStream.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/BiasVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/LazyVector.h"
#include "velox/vector/VectorTypeUtils.h"
#include "velox/vector/tests/VectorMaker.h"

using namespace facebook::velox;
using facebook::velox::ComplexType;

// LazyVector loader for testing. Minimal implementation that documents the API
// contract.
class TestingLoader : public VectorLoader {
 public:
  explicit TestingLoader(VectorPtr data) : data_(data) {}

  void loadInternal(RowSet rows, ValueHook* hook, VectorPtr* result) override {
    if (hook) {
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          applyHook, data_->typeKind(), rows, hook);
      return;
    }
    *result = data_;
  }

 private:
  template <TypeKind Kind>
  void applyHook(RowSet rows, ValueHook* hook) {
    using T = typename TypeTraits<Kind>::NativeType;
    auto values = data_->as<SimpleVector<T>>();
    bool acceptsNulls = hook->acceptsNulls();
    for (auto i = 0; i < rows.size(); ++i) {
      auto row = rows[i];
      if (values->isNullAt(row)) {
        if (acceptsNulls) {
          hook->addNull(i);
        }
      } else {
        T value = values->valueAt(row);
        hook->addValue(i, &value);
      }
    }
  }
  VectorPtr data_;
};

struct NonPOD {
  static int alive;

  int x;
  NonPOD(int x = 123) : x(x) {
    ++alive;
  }
  ~NonPOD() {
    --alive;
  }
  bool operator==(const NonPOD& other) const {
    return x == other.x;
  }
};

int NonPOD::alive = 0;

class VectorTest : public testing::Test {
 protected:
  void SetUp() override {
    pool_ = memory::getDefaultScopedMemoryPool();
    mappedMemory_ = memory::MappedMemory::getInstance();
    if (!isRegisteredVectorSerde()) {
      facebook::velox::serializer::presto::PrestoVectorSerde::
          registerVectorSerde();
    }
  }

  template <typename T>
  T testValue(int32_t i, BufferPtr& space) {
    if constexpr (std::is_same_v<T, std::shared_ptr<void>>) {
      return std::make_shared<NonPOD>(i);
    } else {
      return i;
    }
  }

  template <TypeKind KIND>
  VectorPtr createScalar(TypePtr type, vector_size_t size, bool withNulls) {
    using T = typename TypeTraits<KIND>::NativeType;
    BufferPtr buffer;
    VectorPtr base = BaseVector::create(type, size, pool_.get());
    auto flat = std::dynamic_pointer_cast<FlatVector<T>>(base);
    for (int32_t i = 0; i < flat->size(); ++i) {
      if (withNulls && i % 3 == 0) {
        flat->setNull(i, true);
      } else {
        flat->set(i, testValue<T>(i, buffer));
      }
    }
    return base;
  }

  template <TypeKind Kind, TypeKind BiasKind>
  VectorPtr createBias(vector_size_t size, bool withNulls) {
    using T = typename TypeTraits<Kind>::NativeType;
    using TBias = typename TypeTraits<BiasKind>::NativeType;
    BufferPtr buffer;
    BufferPtr values = AlignedBuffer::allocate<TBias>(size, pool_.get());
    values->setSize(size * sizeof(TBias));
    BufferPtr nulls;
    uint64_t* rawNulls = nullptr;
    if (withNulls) {
      int32_t bytes = BaseVector::byteSize<bool>(size);
      nulls = AlignedBuffer::allocate<char>(bytes, pool_.get());
      rawNulls = nulls->asMutable<uint64_t>();
      memset(rawNulls, bits::kNotNullByte, bytes);
      nulls->setSize(bytes);
    }
    auto rawValues = values->asMutable<TBias>();
    int32_t numNulls = 0;
    constexpr int32_t kBias = 100;
    for (int32_t i = 0; i < size; ++i) {
      if (withNulls && i % 3 == 0) {
        ++numNulls;
        bits::setNull(rawNulls, i);
      } else {
        rawValues[i] = testValue<TBias>(i, buffer) - kBias;
      }
    }
    folly::F14FastMap<std::string, std::string> metadata;
    metadata[BiasVector<T>::BIAS_VALUE] = folly::to<std::string>(kBias);
    return std::make_shared<BiasVector<T>>(
        pool_.get(),
        nulls,
        size,
        BiasKind,
        std::move(values),
        std::move(metadata),
        std::nullopt,
        numNulls,
        false,
        size * sizeof(T));
  }

  VectorPtr createRow(int32_t numRows, bool withNulls) {
    auto childType =
        ROW({"child_bigint", "child_string"}, {BIGINT(), VARCHAR()});
    auto parentType =
        ROW({"parent_bigint", "parent_row"}, {BIGINT(), childType});
    auto baseRow = BaseVector::create(parentType, numRows, pool_.get());
    auto row = baseRow->as<RowVector>();
    EXPECT_EQ(row->size(), numRows);
    EXPECT_EQ(row->nulls(), nullptr);

    std::vector<VectorPtr> nested = {
        createScalar<TypeKind::BIGINT>(BIGINT(), numRows, withNulls),
        createScalar<TypeKind::VARCHAR>(VARCHAR(), numRows, withNulls)};
    auto childRow = std::make_shared<RowVector>(
        pool_.get(),
        childType,
        BufferPtr(nullptr),
        numRows,
        std::move(nested),
        0 /*nullCount*/);
    BufferPtr nulls;
    if (withNulls) {
      nulls = AlignedBuffer::allocate<bool>(numRows, pool_.get());
      int32_t childCounter = 0;
      auto rawNulls = nulls->asMutable<uint64_t>();
      for (int32_t i = 0; i < numRows; ++i) {
        bits::setNull(rawNulls, i, i % 8 == 0);
      }
    }
    std::vector<VectorPtr> parentFields = {
        createScalar<TypeKind::BIGINT>(BIGINT(), numRows, withNulls), childRow};
    return std::make_shared<RowVector>(
        pool_.get(),
        parentType,
        nulls,
        numRows,
        std::move(parentFields),
        BaseVector::countNulls(nulls, numRows));
  }

  // Sets the position corresponding to a null to kNullIndex in
  // 'offsets' and 0 in 'sizes'. 'sizes' can be nullptr in the case of
  // a RowVector or DictionaryVector. Returns true if any null was found.
  static bool setOffsetsAndSizesByNulls(
      vector_size_t size,
      BufferPtr* nulls,
      BufferPtr* offsets,
      BufferPtr* sizes) {
    VELOX_CHECK(*offsets);
    VELOX_CHECK((*offsets)->capacity() >= size * sizeof(vector_size_t));
    VELOX_CHECK(!sizes || (*sizes)->capacity() >= size * sizeof(vector_size_t));
    bool hasNulls = false;
    auto rawOffsets = (*offsets)->asMutable<vector_size_t>();
    auto rawSizes = sizes ? (*sizes)->asMutable<vector_size_t>() : nullptr;
    if (*nulls) {
      bits::forEachUnsetBit(
          (*nulls)->as<uint64_t>(),
          0,
          size,
          [rawOffsets, rawSizes, &hasNulls](int32_t row) {
            rawOffsets[row] = 0;
            if (rawSizes) {
              rawSizes[row] = 0;
            }
            hasNulls = true;
          });
    }
    return hasNulls;
  }

  int32_t createRepeated(
      int32_t numRows,
      bool withNulls,
      BufferPtr* nulls,
      BufferPtr* offsets,
      BufferPtr* sizes,
      int forceWidth) {
    int32_t offsetBytes = numRows * sizeof(vector_size_t);
    *offsets = AlignedBuffer::allocate<char>(offsetBytes, pool_.get());
    *sizes = AlignedBuffer::allocate<char>(offsetBytes, pool_.get());
    (*offsets)->setSize(offsetBytes);
    (*sizes)->setSize(offsetBytes);
    int32_t offset = 0;
    if (withNulls) {
      int32_t bytes = BaseVector::byteSize<bool>(numRows);
      *nulls = AlignedBuffer::allocate<char>(bytes, pool_.get());
      memset(
          (*nulls)->asMutable<uint64_t>(),
          bits::kNotNullByte,
          (*nulls)->capacity());
      (*nulls)->setSize(bytes);
    }
    for (int32_t i = 0; i < numRows; ++i) {
      int32_t size = (forceWidth > 0) ? forceWidth : i % 7;
      if (withNulls && i % 5 == 0) {
        bits::setNull((*nulls)->asMutable<uint64_t>(), i, true);
        if (forceWidth == 0) {
          // in forceWidth case, the size of a null element is still
          // the same as a non-null element.
          (*offsets)->asMutable<vector_size_t>()[i] = 0;
          (*sizes)->asMutable<vector_size_t>()[i] = 0;
          continue;
        }
      }
      (*offsets)->asMutable<vector_size_t>()[i] = offset;
      (*sizes)->asMutable<vector_size_t>()[i] = size;
      offset += size;
    }
    return offset;
  }

  VectorPtr createArray(int32_t numRows, bool withNulls) {
    BufferPtr nulls;
    BufferPtr offsets;
    BufferPtr sizes;
    int32_t numElements =
        createRepeated(numRows, withNulls, &nulls, &offsets, &sizes, 0);
    VectorPtr elements = createRow(numElements, withNulls);
    return std::make_shared<ArrayVector>(
        pool_.get(),
        ARRAY(elements->type()),
        nulls,
        numRows,
        offsets,
        sizes,
        elements,
        BaseVector::countNulls(nulls, numRows));
  }

  VectorPtr createFixedSizeArray(int32_t numRows, bool withNulls, int width) {
    BufferPtr nulls;
    BufferPtr offsets;
    BufferPtr sizes;
    int32_t numElements =
        createRepeated(numRows, withNulls, &nulls, &offsets, &sizes, width);
    VectorPtr elements = createRow(numElements, withNulls);
    return std::make_shared<ArrayVector>(
        pool_.get(),
        FIXED_SIZE_ARRAY(width, elements->type()),
        nulls,
        numRows,
        offsets,
        sizes,
        elements,
        BaseVector::countNulls(nulls, numRows));
  }

  VectorPtr createMap(int32_t numRows, bool withNulls);

  template <TypeKind KIND>
  void testFlat(TypePtr type, vector_size_t size) {
    testFlat<KIND>(type, size, false);
    testFlat<KIND>(type, size, true);
  }

  template <TypeKind KIND>
  void testFlat(TypePtr type, vector_size_t size, bool withNulls) {
    using T = typename TypeTraits<KIND>::NativeType;
    BufferPtr buffer;
    VectorPtr base = BaseVector::create(type, size, pool_.get());
    auto flat = std::dynamic_pointer_cast<FlatVector<T>>(base);
    ASSERT_NE(flat.get(), nullptr);
    EXPECT_EQ(flat->size(), size);
    EXPECT_GE(flat->values()->size(), BaseVector::byteSize<T>(size));
    EXPECT_EQ(flat->nulls(), nullptr);
    flat->resize(size * 2);
    EXPECT_EQ(flat->size(), size * 2);
    EXPECT_GE(flat->values()->capacity(), BaseVector::byteSize<T>(size * 2));
    EXPECT_EQ(flat->nulls(), nullptr);
    if (withNulls) {
      flat->setNull(size * 2 - 1, true);
      EXPECT_EQ(flat->rawNulls()[0], bits::kNotNull64);
      EXPECT_GE(
          flat->nulls()->capacity(), BaseVector::byteSize<bool>(flat->size()));
      EXPECT_TRUE(flat->isNullAt(size * 2 - 1));
    }
    // Test that the null is cleared.
    flat->set(size * 2 - 1, testValue<T>(size, buffer));
    EXPECT_FALSE(flat->isNullAt(size * 2 - 1));

    if (withNulls) {
      // Check that new elEments are initialized as not null after downsize
      // and upsize.
      flat->setNull(size * 2 - 1, true);
      flat->resize(size * 2 - 1);
      flat->resize(size * 2);
      EXPECT_FALSE(flat->isNullAt(size * 2 - 1));
    }
    // Fill, the values at size * 2 - 1 gets assigned a second time.
    for (int32_t i = 0; i < flat->size(); ++i) {
      if (withNulls && i % 3 == 0) {
        flat->setNull(i, true);
      } else {
        flat->set(i, testValue<T>(i, buffer));
      }
    }
    for (int32_t i = 0; i < flat->size(); ++i) {
      if (withNulls && i % 3 == 0) {
        EXPECT_TRUE(flat->isNullAt(i));
      } else {
        EXPECT_FALSE(flat->isNullAt(i));
        if constexpr (KIND == TypeKind::OPAQUE) {
          EXPECT_TRUE(flat->valueAt(i));
          EXPECT_EQ(
              *std::static_pointer_cast<NonPOD>(flat->valueAt(i)),
              *std::static_pointer_cast<NonPOD>(testValue<T>(i, buffer)));
        } else {
          EXPECT_EQ(flat->valueAt(i), testValue<T>(i, buffer));
        }
      }
    }
    testCopy(flat, numIterations_);

    // Check that type kind is preserved in cases where T is the same with.
    EXPECT_EQ(
        flat->typeKind(), BaseVector::wrapInConstant(100, 0, flat)->typeKind());
  }

  static SelectivityVector selectEven(vector_size_t size) {
    SelectivityVector even(size);
    for (auto i = 0; i < size; ++i) {
      even.setValid(i, i % 2 == 0);
    }
    even.updateBounds();
    return even;
  }

  static SelectivityVector selectOdd(vector_size_t size) {
    SelectivityVector odd(size);
    for (auto i = 0; i < size; ++i) {
      odd.setValid(i, i % 2 != 0);
    }
    odd.updateBounds();
    return odd;
  }

  void testCopyEncoded(VectorPtr source) {
    auto kind = source->typeKind();
    auto isSmall = kind == TypeKind::BOOLEAN || kind == TypeKind::TINYINT;
    bool maybeConstant = false;
    bool isSequence = false;
    auto sourcePtr = source.get();
    for (;;) {
      auto encoding = sourcePtr->encoding();
      maybeConstant = encoding == VectorEncoding::Simple::CONSTANT ||
          encoding == VectorEncoding::Simple::LAZY;
      isSequence = encoding == VectorEncoding::Simple::SEQUENCE;
      if (maybeConstant || isSequence) {
        break;
      }
      if (encoding != VectorEncoding::Simple::DICTIONARY &&
          encoding != VectorEncoding::Simple::SEQUENCE) {
        break;
      }
      sourcePtr = sourcePtr->valueVector().get();
    }
    auto sourceSize = source->size();
    auto target = BaseVector::create(source->type(), sourceSize, pool_.get());
    // Writes target out of sequence by copying the first half of
    // source to odd positions and the second half to even positions.
    auto even = selectEven(sourceSize);
    auto odd = selectOdd(sourceSize);
    std::vector<vector_size_t> evenSource(sourceSize, INT32_MAX);
    std::vector<vector_size_t> oddSource(sourceSize, INT32_MAX);

    for (auto i = 0; i < sourceSize; ++i) {
      if (i % 2 == 0) {
        evenSource[i] = i / 2;
      } else {
        oddSource[i] = (sourceSize / 2) + (i / 2);
      }
    }
    target->copy(source.get(), even, &evenSource[0]);
    target->copy(source.get(), odd, &oddSource[0]);
    // We check that a wrapped source tests the same via equalValueAt
    // on the wrapping and via the translation of DecodedVector.
    SelectivityVector allRows(sourceSize);
    DecodedVector decoded(*source, allRows);
    auto flatNulls = source->flatRawNulls(allRows);
    auto base = decoded.base();
    auto nulls = decoded.nulls();
    auto indices = decoded.indices();
    auto nullIndices = decoded.nullIndices();
    for (int32_t i = 0; i < sourceSize; ++i) {
      if (i % 2 == 0) {
        auto sourceIdx = evenSource[i];
        EXPECT_TRUE(target->equalValueAt(source.get(), i, sourceIdx))
            << "at " << i << ": " << target->toString(i) << " vs. "
            << source->toString(sourceIdx);

        // We check the same with 'decoded'.
        if (nulls &&
            bits::isBitNull(
                nulls, nullIndices ? nullIndices[sourceIdx] : sourceIdx)) {
          EXPECT_TRUE(decoded.isNullAt(sourceIdx));
          EXPECT_TRUE(bits::isBitNull(flatNulls, sourceIdx));
          EXPECT_TRUE(target->isNullAt(i));
        } else {
          EXPECT_FALSE(flatNulls && bits::isBitNull(flatNulls, sourceIdx));
          EXPECT_FALSE(decoded.isNullAt(sourceIdx));
          EXPECT_FALSE(target->isNullAt(i));
          EXPECT_TRUE(target->equalValueAt(
              base, i, indices ? indices[sourceIdx] : sourceIdx));
        }
      } else {
        EXPECT_TRUE(target->equalValueAt(source.get(), i, oddSource[i]));
        // We check the same with 'decoded'.
        auto sourceIdx = oddSource[i];
        if (nulls &&
            bits::isBitNull(
                nulls, nullIndices ? nullIndices[sourceIdx] : sourceIdx)) {
          EXPECT_TRUE(bits::isBitNull(flatNulls, sourceIdx));
          EXPECT_TRUE(target->isNullAt(i));
        } else {
          EXPECT_FALSE(flatNulls && bits::isBitNull(flatNulls, sourceIdx));
          EXPECT_FALSE(target->isNullAt(i));
          EXPECT_TRUE(target->equalValueAt(
              base, i, indices ? indices[sourceIdx] : sourceIdx));
        }
      }
      if (i > 1 && i < sourceSize - 1 && !target->isNullAt(i) && !isSmall &&
          !maybeConstant && !isSequence && source->isScalar()) {
        EXPECT_FALSE(target->equalValueAt(source.get(), i, i));
      }
    }
    target->resize(target->size() * 2);
    // Nulls must be clear after resize.
    for (int32_t i = target->size() / 2; i < target->size(); ++i) {
      EXPECT_FALSE(target->isNullAt(i));
    }
    // Copy all of source to second half of target.
    target->copy(source.get(), sourceSize, 0, sourceSize);
    for (int32_t i = 0; i < sourceSize; ++i) {
      EXPECT_TRUE(target->equalValueAt(source.get(), sourceSize + i, i));
    }
    // Check that uninitialized is copyable.
    target->resize(target->size() + 100);
    target->copy(target.get(), target->size() - 50, target->size() - 100, 50);
    if (kind != TypeKind::OPAQUE) {
      testSerialization(source);
    } else {
      // No support for serialization of opaque types yet, make sure something
      // throws
      EXPECT_THROW(testSerialization(source), std::exception);
    }
  }

  void testMove(
      const VectorPtr& vector,
      vector_size_t source,
      vector_size_t target) {
    // Save 'source' row in a temp vector.
    auto temp = BaseVector::create(vector->type(), 1, pool_.get());
    temp->copy(vector.get(), 0, source, 1);

    // Confirm that source and target rows are not the same.
    ASSERT_TRUE(temp->equalValueAt(vector.get(), 0, source));
    ASSERT_FALSE(temp->equalValueAt(vector.get(), 0, target));

    vector->move(source, target);

    // Verify that source and target rows are now the same and equal the
    // original source row.
    ASSERT_TRUE(temp->equalValueAt(vector.get(), 0, source));
    ASSERT_TRUE(temp->equalValueAt(vector.get(), 0, target));
  }

  void testCopy(VectorPtr source, int level) {
    testCopyEncoded(source);
    if (level == 0) {
      return;
    }
    auto sourceSize = source->size();
    BufferPtr dictionaryNulls;
    uint64_t* rawNulls = nullptr;
    if (source->mayHaveNulls()) {
      dictionaryNulls = AlignedBuffer::allocate<bool>(
          sourceSize, pool_.get(), bits::kNotNull);
      rawNulls = dictionaryNulls->asMutable<uint64_t>();
    }
    BufferPtr indices =
        AlignedBuffer::allocate<vector_size_t>(sourceSize, pool_.get());
    indices->setSize(sourceSize * sizeof(vector_size_t));
    for (int32_t i = 0; i < sourceSize; ++i) {
      indices->asMutable<vector_size_t>()[i] = sourceSize - i - 1;
      if (rawNulls && (i + level) % 11 == 0) {
        bits::setNull(rawNulls, i);
      }
    }
    auto inDictionary = BaseVector::wrapInDictionary(
        dictionaryNulls, indices, sourceSize, source);
    testCopy(inDictionary, level - 1);
    BufferPtr lengths =
        AlignedBuffer::allocate<vector_size_t>(sourceSize, pool_.get());
    lengths->setSize(sourceSize * sizeof(vector_size_t));
    for (int32_t i = 0; i < sourceSize; ++i) {
      lengths->asMutable<vector_size_t>()[i] = 1;
    }
    auto inSequence = BaseVector::wrapInSequence(lengths, sourceSize, source);
    testCopy(inSequence, level - 1);

    auto constant = BaseVector::wrapInConstant(20, 10 + level, source);
    testCopy(constant, level - 1);
    if (source->mayHaveNulls() &&
        source->encoding() != VectorEncoding::Simple::LAZY) {
      int32_t firstNull = 0;
      for (; firstNull < sourceSize; ++firstNull) {
        if (source->isNullAt(firstNull)) {
          break;
        }
      }
      constant = BaseVector::wrapInConstant(firstNull + 20, firstNull, source);
      testCopy(constant, level - 1);
    }
    auto lazy = std::make_shared<LazyVector>(
        source->pool(),
        source->type(),
        sourceSize,
        std::make_unique<TestingLoader>(source));
    testCopy(lazy, level - 1);
  }

  void prepareInput(ByteStream* input, std::string& string) {
    // Put 'string' in 'input' in many pieces.
    std::vector<ByteRange> ranges;
    int32_t size = string.size();
    for (int32_t i = 0; i < 10; ++i) {
      int32_t start = i * (size / 10);
      int32_t end = (i == 9) ? size : (i + 1) * (size / 10);
      ranges.emplace_back();
      ranges.back().buffer = reinterpret_cast<uint8_t*>(&string[start]);
      ranges.back().size = end - start;
      ranges.back().position = 0;
    }
    input->resetInput(std::move(ranges));
  }

  void checkSizes(
      BaseVector* source,
      std::vector<vector_size_t>& sizes,
      std::string& string) {
    int32_t size = string.size();
    int32_t total = 0;
    for (auto size : sizes) {
      total += size;
    }
    // Approx. 10 bytes for encoding, 4 for length, 1 for null presence. This
    // can vary.
    total -= 15;
    float tolerance = source->mayHaveNulls() ? 0.3 : 0.2;
    switch (source->typeKind()) {
      case TypeKind::BOOLEAN:
        tolerance = 10;
        break;
      case TypeKind::TINYINT:
        tolerance = 2;
        break;
      case TypeKind::SMALLINT:
        tolerance = 1.5;
        break;
      case TypeKind::TIMESTAMP:
        // The size is reported as 16 bytes but Presto wire format is 8 bytes.
        tolerance = 2.2;
        break;
      default:
        break;
    }
    if (total > 1024) {
      EXPECT_GE(total, size * (1 - tolerance));
      EXPECT_LE(total, size * (1 + tolerance));
    }
  }

  RowVectorPtr makeRowVector(VectorPtr vector) {
    std::vector<std::string> names = {"c"};
    std::vector<TypePtr> types = {vector->type()};
    auto rowType = ROW(std::move(names), std::move(types));

    std::vector<VectorPtr> children = {vector};
    return std::make_shared<RowVector>(
        vector->pool(), rowType, BufferPtr(nullptr), vector->size(), children);
  }

  void testSerialization(VectorPtr source) {
    // Serialization functions expect loaded vectors.
    source = BaseVector::loadedVectorShared(source);
    auto sourceRow = makeRowVector(source);
    auto sourceRowType =
        std::dynamic_pointer_cast<const RowType>(sourceRow->type());
    VectorStreamGroup even(mappedMemory_);
    even.createStreamTree(sourceRowType, source->size() / 4);

    VectorStreamGroup odd(mappedMemory_);
    odd.createStreamTree(sourceRowType, source->size() / 3);
    std::vector<IndexRange> evenIndices;
    std::vector<IndexRange> oddIndices;
    std::vector<vector_size_t> evenSizes;
    std::vector<vector_size_t> oddSizes;
    for (vector_size_t i = 0; i < source->size(); ++i) {
      if (i % 2 == 0) {
        evenIndices.push_back(IndexRange{i, 1});
      } else {
        oddIndices.push_back(IndexRange{i, 1});
      }
    }
    evenSizes.resize(evenIndices.size());
    oddSizes.resize(oddIndices.size());
    std::vector<vector_size_t*> evenSizePointers(evenSizes.size());
    std::vector<vector_size_t*> oddSizePointers(oddSizes.size());
    for (int32_t i = 0; i < evenSizes.size(); ++i) {
      evenSizePointers[i] = &evenSizes[i];
    }
    for (int32_t i = 0; i < oddSizes.size(); ++i) {
      oddSizePointers[i] = &oddSizes[i];
    }

    VectorStreamGroup::estimateSerializedSize(
        source, evenIndices, evenSizePointers.data());
    VectorStreamGroup::estimateSerializedSize(
        source, oddIndices, oddSizePointers.data());
    even.append(
        sourceRow, folly::Range(evenIndices.data(), evenIndices.size() / 2));
    even.append(
        sourceRow,
        folly::Range(
            &evenIndices[evenIndices.size() / 2],
            evenIndices.size() - evenIndices.size() / 2));
    odd.append(
        sourceRow, folly::Range(oddIndices.data(), oddIndices.size() / 2));
    odd.append(
        sourceRow,
        folly::Range(
            &oddIndices[oddIndices.size() / 2],
            oddIndices.size() - oddIndices.size() / 2));
    std::stringstream evenStream;
    std::stringstream oddStream;
    OutputStream eventOutputStream(&evenStream);
    OutputStream oddOutputStream(&oddStream);
    even.flush(&eventOutputStream);
    odd.flush(&oddOutputStream);
    ByteStream input;
    auto evenString = evenStream.str();
    checkSizes(source.get(), evenSizes, evenString);
    prepareInput(&input, evenString);

    RowVectorPtr resultRow;
    VectorStreamGroup::read(&input, pool_.get(), sourceRowType, &resultRow);
    VectorPtr result = resultRow->childAt(0);
    switch (source->encoding()) {
      case VectorEncoding::Simple::FLAT:
      case VectorEncoding::Simple::ROW:
      case VectorEncoding::Simple::ARRAY:
      case VectorEncoding::Simple::MAP: {
        // Test that retained sizes are in reasonable bounds if vectors
        // are not wrapped.
        auto originalRetained = source->retainedSize();
        auto deserializedRetained = resultRow->retainedSize();
        EXPECT_LE(deserializedRetained, originalRetained)
            << "Deserializing half is larger than original";
        EXPECT_GT(deserializedRetained, originalRetained / 8)
            << "Deserialized half is less than 1/8 the of original size";
        break;
      }
      default:
        // skip non-flat encodings
        break;
    }
    for (int32_t i = 0; i < evenIndices.size(); ++i) {
      EXPECT_TRUE(result->equalValueAt(source.get(), i, evenIndices[i].begin))
          << "at " << i << ", " << source->encoding();
    }
    auto oddString = oddStream.str();
    prepareInput(&input, oddString);
    VectorStreamGroup::read(&input, pool_.get(), sourceRowType, &resultRow);
    result = resultRow->childAt(0);
    for (int32_t i = 0; i < oddIndices.size(); ++i) {
      EXPECT_TRUE(result->equalValueAt(source.get(), i, oddIndices[i].begin))
          << "at " << i << ", " << source->encoding();
    }
  }

  std::unique_ptr<memory::ScopedMemoryPool> pool_;
  memory::MappedMemory* mappedMemory_;

  size_t vectorSize_{100};
  size_t numIterations_{3};
};

template <>
StringView VectorTest::testValue(int32_t n, BufferPtr& buffer) {
  if (!buffer || buffer->capacity() < 1000) {
    buffer = AlignedBuffer::allocate<char>(1000, pool_.get());
  }
  std::stringstream out;
  out << n;
  for (int32_t i = 0; i < n % 20; ++i) {
    out << " " << i * i;
  }
  std::string str = out.str();
  EXPECT_LE(str.size(), buffer->capacity());
  memcpy(buffer->asMutable<char>(), str.data(), str.size());
  return StringView(buffer->as<char>(), str.size());
}

template <>
bool VectorTest::testValue(int32_t i, BufferPtr& space) {
  return (i % 2) == 1;
}

template <>
Timestamp VectorTest::testValue(int32_t i, BufferPtr& space) {
  // Return even milliseconds.
  return Timestamp(i * 1000, (i % 1000) * 1000000);
}

template <>
Date VectorTest::testValue(int32_t i, BufferPtr& space) {
  return Date(i);
}

VectorPtr VectorTest::createMap(int32_t numRows, bool withNulls) {
  BufferPtr nulls;
  BufferPtr offsets;
  BufferPtr sizes;
  int32_t numElements =
      createRepeated(numRows, withNulls, &nulls, &offsets, &sizes, 0);
  VectorPtr elements = createRow(numElements, withNulls);
  auto keysBase = BaseVector::create(VARCHAR(), 7, pool_.get());
  auto flatKeys = keysBase->as<FlatVector<StringView>>();
  for (int32_t i = 0; i < keysBase->size(); ++i) {
    BufferPtr buffer;
    flatKeys->set(i, testValue<StringView>(1000 + i, buffer));
  }

  auto indicesBytes = elements->size() * sizeof(vector_size_t);
  auto indices = AlignedBuffer::allocate<char>(indicesBytes, pool_.get());
  indices->setSize(indicesBytes);
  auto rawSizes = sizes->as<vector_size_t>();
  int32_t offset = 0;
  for (int32_t i = 0; i < numRows; ++i) {
    int32_t size = rawSizes[i];
    for (int32_t index = 0; index < size; ++index) {
      indices->asMutable<vector_size_t>()[offset++] = index;
    }
  }
  VELOX_CHECK(offset == elements->size());
  auto keys = BaseVector::wrapInDictionary(
      BufferPtr(nullptr),
      std::move(indices),
      elements->size(),
      std::move(keysBase));
  return std::make_shared<MapVector>(
      pool_.get(),
      MAP(VARCHAR(), elements->type()),
      nulls,
      numRows,
      offsets,
      sizes,
      keys,
      elements,
      BaseVector::countNulls(nulls, numRows));
}

TEST_F(VectorTest, createInt) {
  testFlat<TypeKind::BIGINT>(BIGINT(), vectorSize_);
  testFlat<TypeKind::INTEGER>(INTEGER(), vectorSize_);
  testFlat<TypeKind::SMALLINT>(SMALLINT(), vectorSize_);
  testFlat<TypeKind::TINYINT>(TINYINT(), vectorSize_);
}

TEST_F(VectorTest, createDouble) {
  testFlat<TypeKind::REAL>(REAL(), vectorSize_);
  testFlat<TypeKind::DOUBLE>(DOUBLE(), vectorSize_);
}

TEST_F(VectorTest, createStr) {
  testFlat<TypeKind::VARCHAR>(VARCHAR(), vectorSize_);
  testFlat<TypeKind::VARBINARY>(VARBINARY(), vectorSize_);
}

TEST_F(VectorTest, createOther) {
  testFlat<TypeKind::BOOLEAN>(BOOLEAN(), vectorSize_);
  testFlat<TypeKind::TIMESTAMP>(TIMESTAMP(), vectorSize_);
  testFlat<TypeKind::DATE>(DATE(), vectorSize_);
}

TEST_F(VectorTest, createOpaque) {
  NonPOD::alive = 0;
  testFlat<TypeKind::OPAQUE>(OPAQUE<int>(), vectorSize_);
  EXPECT_EQ(NonPOD::alive, 0);
}

TEST_F(VectorTest, getOrCreateEmpty) {
  auto empty = BaseVector::getOrCreateEmpty(nullptr, VARCHAR(), pool_.get());
  EXPECT_NE(empty, nullptr);
  EXPECT_EQ(empty->size(), 0);
  EXPECT_EQ(empty->type(), VARCHAR());
}

TEST_F(VectorTest, bias) {
  auto base =
      createBias<TypeKind::BIGINT, TypeKind::INTEGER>(vectorSize_, false);
  testCopy(base, 4);
  base = createBias<TypeKind::INTEGER, TypeKind::SMALLINT>(vectorSize_, true);
  testCopy(base, 4);
}

TEST_F(VectorTest, row) {
  auto baseRow = createRow(vectorSize_, false);
  testCopy(baseRow, numIterations_);
  baseRow = createRow(vectorSize_, true);
  testCopy(baseRow, numIterations_);
  auto allNull =
      BaseVector::createNullConstant(baseRow->type(), 50, pool_.get());
  testCopy(allNull, numIterations_);

  testMove(baseRow, 5, 23);
}

TEST_F(VectorTest, array) {
  auto baseArray = createArray(vectorSize_, false);
  testCopy(baseArray, numIterations_);
  baseArray = createArray(vectorSize_, true);
  testCopy(baseArray, numIterations_);
  auto allNull =
      BaseVector::createNullConstant(baseArray->type(), 50, pool_.get());
  testCopy(allNull, numIterations_);

  testMove(baseArray, 5, 23);
}

TEST_F(VectorTest, fixedSizeArray) {
  const int width = 7;
  auto baseArray = createFixedSizeArray(vectorSize_, false, width);
  testCopy(baseArray, numIterations_);
  baseArray = createFixedSizeArray(vectorSize_, true, width);
  testCopy(baseArray, numIterations_);
  auto allNull =
      BaseVector::createNullConstant(baseArray->type(), 50, pool_.get());
  testCopy(allNull, numIterations_);

  testMove(baseArray, 5, 23);
}

TEST_F(VectorTest, map) {
  auto baseMap = createMap(vectorSize_, false);
  testCopy(baseMap, numIterations_);
  baseMap = createRow(vectorSize_, true);
  testCopy(baseMap, numIterations_);
  auto allNull =
      BaseVector::createNullConstant(baseMap->type(), 50, pool_.get());
  testCopy(allNull, numIterations_);

  testMove(baseMap, 5, 23);
}

TEST_F(VectorTest, unknown) {
  auto vector =
      BaseVector::createConstant(variant(TypeKind::UNKNOWN), 123, pool_.get());
  ASSERT_EQ(TypeKind::UNKNOWN, vector->typeKind());
  ASSERT_EQ(123, vector->size());
  for (auto i = 0; i < vector->size(); i++) {
    ASSERT_TRUE(vector->isNullAt(i));
  }

  auto copy = BaseVector::create(BIGINT(), 10, pool_.get());
  copy->copy(vector.get(), 0, 0, 1);

  SelectivityVector rows(1);
  copy->copy(vector.get(), rows, nullptr);
}

TEST_F(VectorTest, copyBoolAllNullFlatVector) {
  const vector_size_t size = 1'000;
  BufferPtr nulls =
      AlignedBuffer::allocate<bool>(size, pool_.get(), bits::kNull);
  auto allNulls = std::make_shared<FlatVector<bool>>(
      pool_.get(), nulls, size, BufferPtr(nullptr), std::vector<BufferPtr>{});

  auto vectorMaker = std::make_unique<test::VectorMaker>(pool_.get());
  auto copy = vectorMaker->flatVector<bool>(
      size, [](auto row) { return row % 2 == 0; });

  // Copy first 10 rows from allNulls.
  SelectivityVector rows(10);
  copy->copy(allNulls.get(), rows, nullptr);

  for (auto i = 0; i < size; i++) {
    ASSERT_EQ(copy->isNullAt(i), i < 10) << "at " << i;
    if (i >= 10) {
      ASSERT_EQ(copy->valueAt(i), i % 2 == 0) << "at " << i;
    }
  }
}

TEST_F(VectorTest, wrapInConstant) {
  auto vectorMaker = std::make_unique<test::VectorMaker>(pool_.get());

  // wrap flat vector
  const vector_size_t size = 1'000;
  auto flatVector = vectorMaker->flatVector<int32_t>(
      size, [](auto row) { return row; }, test::VectorMaker::nullEvery(7));

  auto constVector = std::dynamic_pointer_cast<ConstantVector<int32_t>>(
      BaseVector::wrapInConstant(size, 5, flatVector));
  EXPECT_EQ(constVector->valueVector(), nullptr);
  for (auto i = 0; i < size; i++) {
    ASSERT_FALSE(constVector->isNullAt(i));
    ASSERT_EQ(5, constVector->valueAt(i));
  }

  constVector = std::dynamic_pointer_cast<ConstantVector<int32_t>>(
      BaseVector::wrapInConstant(size, 7, flatVector));
  EXPECT_EQ(constVector->valueVector(), nullptr);
  for (auto i = 0; i < size; i++) {
    ASSERT_TRUE(constVector->isNullAt(i));
  }

  // wrap dictionary vector
  BufferPtr indices = AlignedBuffer::allocate<vector_size_t>(size, pool_.get());
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (auto i = 0; i < size; i++) {
    rawIndices[i] = 2 * i % size;
  }

  BufferPtr nulls = AlignedBuffer::allocate<bool>(size, pool_.get());
  auto rawNulls = nulls->asMutable<uint64_t>();
  for (auto i = 0; i < size; i++) {
    bits::setNull(rawNulls, i, i % 11 == 0);
  }

  auto dictVector =
      BaseVector::wrapInDictionary(nulls, indices, size, flatVector);

  constVector = std::dynamic_pointer_cast<ConstantVector<int32_t>>(
      BaseVector::wrapInConstant(size, 5, dictVector));
  EXPECT_EQ(constVector->valueVector(), nullptr);
  for (auto i = 0; i < size; i++) {
    ASSERT_FALSE(constVector->isNullAt(i));
    ASSERT_EQ(10, constVector->valueAt(i));
  }

  constVector = std::dynamic_pointer_cast<ConstantVector<int32_t>>(
      BaseVector::wrapInConstant(size, 11, dictVector));
  EXPECT_EQ(constVector->valueVector(), nullptr);
  for (auto i = 0; i < size; i++) {
    ASSERT_TRUE(constVector->isNullAt(i));
  }
}

TEST_F(VectorTest, setFlatVectorStringView) {
  auto vector = BaseVector::create(VARCHAR(), 1, pool_.get());
  auto flat = vector->asFlatVector<StringView>();
  flat->set(0, StringView("This string is too long to be inlined"));
  EXPECT_EQ(
      flat->valueAt(0).getString(), "This string is too long to be inlined");
}

TEST_F(VectorTest, resizeAtConstruction) {
  using T = int64_t;
  const size_t realSize = 10;

  vector_size_t oldByteSize = BaseVector::byteSize<T>(realSize);
  BufferPtr values = AlignedBuffer::allocate<char>(oldByteSize, pool_.get());

  EXPECT_EQ(oldByteSize, values->size());
  EXPECT_GE(values->capacity(), values->size());

  // Resize the vector.
  const size_t newByteSize = oldByteSize / 2;
  const size_t curCapacity = values->capacity();
  values->setSize(newByteSize);

  // Check it has been resized.
  EXPECT_EQ(newByteSize, values->size());
  EXPECT_EQ(curCapacity, values->capacity());

  // Now create a FlatVector with the resized buffer.
  auto flat = std::make_shared<FlatVector<T>>(
      pool_.get(),
      BufferPtr(nullptr),
      realSize,
      std::move(values),
      std::vector<BufferPtr>());

  // Check that the vector was resized back to the old size.
  EXPECT_EQ(oldByteSize, flat->values()->size());
}

TEST_F(VectorTest, resizeStringAsciiness) {
  auto vectorMaker = std::make_unique<test::VectorMaker>(pool_.get());
  std::vector<std::string> stringInput = {"hellow", "how", "are"};
  auto flatVector = vectorMaker->flatVector(stringInput);
  auto stringVector = flatVector->as<SimpleVector<StringView>>();
  SelectivityVector rows(stringInput.size());
  stringVector->computeAndSetIsAscii(rows);
  ASSERT_TRUE(stringVector->isAscii(rows).value());
  stringVector->resize(2);
  ASSERT_FALSE(stringVector->isAscii(rows));
}

TEST_F(VectorTest, copyAscii) {
  auto maker = std::make_unique<test::VectorMaker>(pool_.get());
  std::vector<std::string> stringData = {"a", "b", "c"};
  auto source = maker->flatVector(stringData);
  SelectivityVector all(stringData.size());
  source->setAllIsAscii(true);

  auto other = maker->flatVector(stringData);
  other->copy(source.get(), all, nullptr);
  auto ascii = other->isAscii(all);
  ASSERT_TRUE(ascii.has_value());
  ASSERT_TRUE(ascii.value());

  // Copy over asciness from a vector without asciiness set.
  source->invalidateIsAscii();
  other->copy(source.get(), all, nullptr);
  // Ensure isAscii returns nullopt
  ascii = other->isAscii(all);
  ASSERT_FALSE(ascii.has_value());

  // Now copy over some rows from vector without ascii compute on it.
  other->setAllIsAscii(false);
  SelectivityVector some(all.size(), false);
  some.setValid(1, true);
  some.updateBounds();
  other->copy(source.get(), some, nullptr);
  // We will have invalidated all.
  ascii = other->isAscii(all);
  ASSERT_FALSE(ascii.has_value());

  std::vector<std::string> moreData = {"a", "b", "c", "d"};
  auto largerSource = maker->flatVector(moreData);
  vector_size_t sourceMappings[] = {3, 2, 1, 0};
  some.setAll();
  some.setValid(0, false);
  some.updateBounds();

  other->setAllIsAscii(false);
  largerSource->setAllIsAscii(true);
  other->copy(largerSource.get(), some, sourceMappings);
  ascii = other->isAscii(all);
  ASSERT_TRUE(ascii.has_value());
  ASSERT_FALSE(ascii.value());
}

TEST_F(VectorTest, compareNan) {
  auto vectorMaker = std::make_unique<test::VectorMaker>(pool_.get());

  // Double input.
  {
    std::vector<double> doubleInput = {1.1, std::nan("nan"), 0.9};

    auto flatVector1 = vectorMaker->flatVector(doubleInput);
    auto flatVector2 = vectorMaker->flatVector(doubleInput);

    for (size_t i = 0; i < doubleInput.size(); ++i) {
      EXPECT_TRUE(flatVector1->equalValueAt(flatVector2.get(), i, i));
    }
  }

  // Float input.
  {
    std::vector<float> floatInput = {1.1, std::nanf("nan"), 0.9};

    auto flatVector1 = vectorMaker->flatVector(floatInput);
    auto flatVector2 = vectorMaker->flatVector(floatInput);

    for (size_t i = 0; i < floatInput.size(); ++i) {
      EXPECT_TRUE(flatVector1->equalValueAt(flatVector2.get(), i, i));
    }
  }
}

class VectorCreateConstantTest : public VectorTest {
 public:
  template <TypeKind KIND>
  void testPrimitiveConstant(typename TypeTraits<KIND>::NativeType val) {
    using TCpp = typename TypeTraits<KIND>::NativeType;
    auto var = variant::create<KIND>(val);

    auto baseVector = BaseVector::createConstant(var, size_, pool_.get());
    auto simpleVector = baseVector->template as<SimpleVector<TCpp>>();
    ASSERT_TRUE(simpleVector != nullptr);

    ASSERT_EQ(KIND, simpleVector->typeKind());
    ASSERT_EQ(size_, simpleVector->size());
    ASSERT_TRUE(simpleVector->isScalar());

    for (auto i = 0; i < simpleVector->size(); i++) {
      if constexpr (std::is_same<TCpp, StringView>::value) {
        ASSERT_EQ(
            var.template value<KIND>(), std::string(simpleVector->valueAt(i)));
      } else {
        ASSERT_EQ(var.template value<KIND>(), simpleVector->valueAt(i));
      }
    }

    auto expectedStr = fmt::format(
        "[CONSTANT {}: {} value, {} size]",
        KIND,
        baseVector->toString(0),
        size_);
    EXPECT_EQ(expectedStr, baseVector->toString());
    for (auto i = 1; i < baseVector->size(); ++i) {
      EXPECT_EQ(baseVector->toString(0), baseVector->toString(i));
    }
  }

  template <TypeKind KIND>
  void testComplexConstant(const TypePtr& type, const VectorPtr& vector) {
    ASSERT_EQ(KIND, type->kind());
    auto baseVector = BaseVector::wrapInConstant(size_, 0, vector);
    auto simpleVector = baseVector->template as<
        SimpleVector<typename KindToFlatVector<KIND>::WrapperType>>();
    ASSERT_TRUE(simpleVector != nullptr);

    ASSERT_EQ(KIND, simpleVector->typeKind());
    ASSERT_EQ(size_, simpleVector->size());
    ASSERT_FALSE(simpleVector->isScalar());

    for (auto i = 0; i < simpleVector->size(); i++) {
      ASSERT_TRUE(vector->equalValueAt(baseVector.get(), 0, i));
    }

    auto expectedStr = fmt::format(
        "[CONSTANT {}: {} value, {} size]",
        type->toString(),
        vector->toString(0),
        size_);
    EXPECT_EQ(expectedStr, baseVector->toString());
    for (auto i = 1; i < baseVector->size(); ++i) {
      EXPECT_EQ(baseVector->toString(0), baseVector->toString(i));
    }
  }

  template <TypeKind KIND>
  void testNullConstant(const TypePtr& type) {
    ASSERT_EQ(KIND, type->kind());
    auto baseVector = BaseVector::createNullConstant(type, size_, pool_.get());
    auto simpleVector = baseVector->template as<
        SimpleVector<typename KindToFlatVector<KIND>::WrapperType>>();
    ASSERT_TRUE(simpleVector != nullptr);

    ASSERT_EQ(KIND, simpleVector->typeKind());
    ASSERT_EQ(size_, simpleVector->size());
    ASSERT_EQ(simpleVector->isScalar(), TypeTraits<KIND>::isPrimitiveType);

    for (auto i = 0; i < simpleVector->size(); i++) {
      ASSERT_TRUE(simpleVector->isNullAt(i));
    }

    auto expectedStr = fmt::format(
        "[CONSTANT {}: null value, {} size]", type->toString(), size_);
    EXPECT_EQ(expectedStr, baseVector->toString());
    for (auto i = 1; i < baseVector->size(); ++i) {
      EXPECT_EQ(baseVector->toString(0), baseVector->toString(i));
    }
  }

 protected:
  // Theoretical "size" of the constant vector created.
  const size_t size_{23};
};

TEST_F(VectorCreateConstantTest, scalar) {
  testPrimitiveConstant<TypeKind::BIGINT>(-123456789);
  testPrimitiveConstant<TypeKind::INTEGER>(98765);
  testPrimitiveConstant<TypeKind::SMALLINT>(1234);
  testPrimitiveConstant<TypeKind::TINYINT>(123);

  testPrimitiveConstant<TypeKind::BOOLEAN>(true);
  testPrimitiveConstant<TypeKind::BOOLEAN>(false);

  testPrimitiveConstant<TypeKind::REAL>(99.98);
  testPrimitiveConstant<TypeKind::DOUBLE>(12.345);

  testPrimitiveConstant<TypeKind::VARCHAR>(StringView("hello world"));
  testPrimitiveConstant<TypeKind::VARBINARY>(StringView("my binary buffer"));
}

TEST_F(VectorCreateConstantTest, complex) {
  {
    auto type = ARRAY(INTEGER());
    test::VectorMaker maker{pool_.get()};
    testComplexConstant<TypeKind::ARRAY>(
        type,
        maker.arrayVector<int32_t>(
            1, [](auto) { return 10; }, [](auto i) { return i; }));
  }
  {
    auto type = MAP(INTEGER(), REAL());
    test::VectorMaker maker{pool_.get()};
    testComplexConstant<TypeKind::MAP>(
        type,
        maker.mapVector<int32_t, float>(
            1,
            [](auto) { return 10; },
            [](auto i) { return i; },
            [](auto i) { return i; }));
  }
  {
    auto type = ROW({{"c0", INTEGER()}});
    test::VectorMaker maker{pool_.get()};
    testComplexConstant<TypeKind::ROW>(
        type, maker.rowVector({maker.flatVector<int32_t>(1, [](auto i) {
          return i;
        })}));
  }
}

TEST_F(VectorCreateConstantTest, null) {
  testNullConstant<TypeKind::BIGINT>(BIGINT());
  testNullConstant<TypeKind::INTEGER>(INTEGER());
  testNullConstant<TypeKind::SMALLINT>(SMALLINT());
  testNullConstant<TypeKind::TINYINT>(TINYINT());

  testNullConstant<TypeKind::BOOLEAN>(BOOLEAN());

  testNullConstant<TypeKind::REAL>(REAL());
  testNullConstant<TypeKind::DOUBLE>(DOUBLE());

  testNullConstant<TypeKind::TIMESTAMP>(TIMESTAMP());
  testNullConstant<TypeKind::DATE>(DATE());

  testNullConstant<TypeKind::VARCHAR>(VARCHAR());
  testNullConstant<TypeKind::VARBINARY>(VARBINARY());

  testNullConstant<TypeKind::ROW>(ROW({BIGINT(), REAL()}));
  testNullConstant<TypeKind::ARRAY>(ARRAY(DOUBLE()));
  testNullConstant<TypeKind::MAP>(MAP(INTEGER(), DOUBLE()));
}

class TestingHook : public ValueHook {
 public:
  TestingHook(RowSet rows, SimpleVector<int64_t>* values)
      : rows_(rows), values_(values) {}

  bool acceptsNulls() const override {
    return false;
  }

  void addValue(vector_size_t row, const void* value) override {
    if (values_->isNullAt(rows_[row])) {
      ++errors_;
    } else {
      auto original = values_->valueAt(rows_[row]);
      if (original != *reinterpret_cast<const int64_t*>(value)) {
        ++errors_;
      }
    }
  }
  int32_t errors() const {
    return errors_;
  }

 private:
  RowSet rows_;
  SimpleVector<int64_t>* values_;
  int32_t errors_ = 0;
};

TEST_F(VectorTest, valueHook) {
  VectorPtr values = createScalar<TypeKind::BIGINT>(BIGINT(), 1000, true);
  auto lazy = std::make_shared<LazyVector>(
      pool_.get(),
      BIGINT(),
      values->size(),
      std::make_unique<TestingLoader>(values));
  std::vector<vector_size_t> rows;
  for (int i = 0; i < values->size(); i += 2) {
    rows.push_back(i);
  }
  TestingHook hook(rows, values->as<SimpleVector<int64_t>>());
  // The hook is called with the row number being an index 'i' into
  // 'rows' and the value being the value at rows[i]. The hook is not
  // called if the value at the row is null because the hook does not
  // accept nulls.  In this way, for the use case of aggregation, the
  // row number is directly usable as an index into an array of groups
  // to update. The caller will compose the RowSet based on what rows
  // have been filtered out beteen creating the LazyVector and the
  // point of loading. The hook itself is concerned with the loaded
  // subset of rows and the position of each row within this set.
  lazy->load(rows, &hook);
  EXPECT_EQ(hook.errors(), 0);
}

TEST_F(VectorTest, byteSize) {
  constexpr vector_size_t count = std::numeric_limits<vector_size_t>::max();
  constexpr uint64_t expected = count * sizeof(int64_t);
  EXPECT_EQ(BaseVector::byteSize<int64_t>(count), expected);
}

TEST_F(VectorTest, constantDictionary) {
  auto vectorMaker = std::make_unique<test::VectorMaker>(pool_.get());
  auto flatVector = vectorMaker->flatVector<int32_t>({1, 2, 3, 4});

  // Repeat each row twice: 1, 1, 2, 2, 3, 3, 4, 4.
  auto dictionarySize = flatVector->size() * 2;
  auto indices = allocateIndices(dictionarySize, pool_.get());
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (auto i = 0; i < dictionarySize; i++) {
    rawIndices[i] = i / 2;
  }
  auto dictionaryVector = BaseVector::wrapInDictionary(
      nullptr, indices, dictionarySize, flatVector);

  {
    SelectivityVector rows(dictionarySize);
    ASSERT_FALSE(dictionaryVector->isConstant(rows));
  }

  // First two rows are the same.
  {
    SelectivityVector rows(2);
    ASSERT_TRUE(dictionaryVector->isConstant(rows));
  }

  // Single row is always constant.
  {
    SelectivityVector rows(dictionarySize, false);
    rows.setValid(3, true);
    rows.updateBounds();
    ASSERT_TRUE(dictionaryVector->isConstant(rows));
  }

  // Test nulls added by DictionaryVector.
  auto dictNulls = AlignedBuffer::allocate<bool>(
      dictionarySize, pool_.get(), bits::kNotNull);
  auto rawNulls = dictNulls->asMutable<uint64_t>();
  bits::setNull(rawNulls, 0);
  bits::setNull(rawNulls, 2);
  dictionaryVector = BaseVector::wrapInDictionary(
      dictNulls, indices, dictionarySize, flatVector);
  // Elements 0 and 1 are not equal because the dictionary adds a null at 0.
  {
    SelectivityVector rows(2);
    ASSERT_FALSE(dictionaryVector->isConstant(rows));
  }
  // The vector is constant for 0, 2 because both add a null.
  {
    SelectivityVector rows(dictionarySize, false);
    rows.setValid(0, true);
    rows.setValid(2, true);
    rows.updateBounds();
    ASSERT_TRUE(dictionaryVector->isConstant(rows));
  }
}

TEST_F(VectorTest, clearNulls) {
  auto vectorSize = 100;
  auto vector = BaseVector::create(INTEGER(), vectorSize, pool_.get());
  ASSERT_FALSE(vector->mayHaveNulls());

  // No op if doesn't have nulls
  SelectivityVector selection{vectorSize};
  vector->clearNulls(selection);
  ASSERT_FALSE(vector->mayHaveNulls());

  // De-allocate nulls if all selected
  auto rawNulls = vector->mutableRawNulls();
  ASSERT_EQ(bits::countNulls(rawNulls, 0, vectorSize), 0);
  bits::setNull(rawNulls, 50);
  ASSERT_TRUE(vector->isNullAt(50));
  ASSERT_EQ(bits::countNulls(rawNulls, 0, vectorSize), 1);
  vector->clearNulls(selection);
  ASSERT_FALSE(vector->mayHaveNulls());

  // Clear within vectorSize
  rawNulls = vector->mutableRawNulls();
  bits::setNull(rawNulls, 50);
  bits::setNull(rawNulls, 70);
  selection.clearAll();
  selection.setValidRange(40, 60, true);
  selection.updateBounds();
  vector->clearNulls(selection);
  ASSERT_TRUE(!vector->isNullAt(50));
  ASSERT_TRUE(vector->isNullAt(70));

  // Clear with end > vector size
  selection.resize(120);
  selection.clearAll();
  selection.setValidRange(60, 120, true);
  selection.updateBounds();
  vector->clearNulls(selection);
  ASSERT_TRUE(!vector->isNullAt(70));
  ASSERT_TRUE(vector->mayHaveNulls());

  // Clear with begin > vector size
  rawNulls = vector->mutableRawNulls();
  bits::setNull(rawNulls, 70);
  selection.clearAll();
  selection.setValidRange(100, 120, true);
  selection.updateBounds();
  vector->clearNulls(selection);
  ASSERT_TRUE(vector->isNullAt(70));
}

TEST_F(VectorTest, setStringToNull) {
  constexpr int32_t kSize = 100;
  auto vectorMaker = std::make_unique<test::VectorMaker>(pool_.get());
  auto target = vectorMaker->flatVector<StringView>(
      kSize, [](auto /*row*/) { return StringView("Non-inlined string"); });
  target->setNull(kSize - 1, true);
  auto unknownNull = std::make_shared<ConstantVector<UnknownValue>>(
      pool_.get(), kSize, true, UNKNOWN(), UnknownValue());

  auto stringNull = BaseVector::wrapInConstant(kSize, kSize - 1, target);
  SelectivityVector rows(kSize, false);
  rows.setValid(2, true);
  rows.updateBounds();
  target->copy(unknownNull.get(), rows, nullptr);
  EXPECT_TRUE(target->isNullAt(2));

  rows.setValid(4, true);
  rows.updateBounds();
  target->copy(stringNull.get(), rows, nullptr);
  EXPECT_TRUE(target->isNullAt(4));
  auto nulls = AlignedBuffer::allocate<uint64_t>(
      bits::nwords(kSize), pool_.get(), bits::kNull64);
  auto flatNulls = std::make_shared<FlatVector<UnknownValue>>(
      pool_.get(), nulls, kSize, BufferPtr(nullptr), std::vector<BufferPtr>());
  rows.setValid(6, true);
  rows.updateBounds();
  target->copy(flatNulls.get(), rows, nullptr);
  EXPECT_TRUE(target->isNullAt(6));
  EXPECT_EQ(4, bits::countNulls(target->rawNulls(), 0, kSize));
}

TEST_F(VectorTest, clearAllNulls) {
  auto vectorSize = 100;
  auto vector = BaseVector::create(INTEGER(), vectorSize, pool_.get());
  ASSERT_FALSE(vector->mayHaveNulls());

  auto rawNulls = vector->mutableRawNulls();
  ASSERT_EQ(bits::countNulls(rawNulls, 0, vectorSize), 0);
  bits::setNull(rawNulls, 50);
  ASSERT_TRUE(vector->isNullAt(50));
  ASSERT_EQ(bits::countNulls(rawNulls, 0, vectorSize), 1);
  vector->clearAllNulls();
  ASSERT_FALSE(vector->mayHaveNulls());
  ASSERT_FALSE(vector->isNullAt(50));
}
