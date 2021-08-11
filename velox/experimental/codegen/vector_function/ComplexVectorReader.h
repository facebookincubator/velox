/*
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

#include <math.h>
#include <sys/socket.h>
#include <array>
#include <type_traits>
#include "velox/buffer/Buffer.h"
#include "velox/common/base/Nulls.h"
#include "velox/experimental/codegen/vector_function/StringTypes.h"
#include "velox/experimental/codegen/vector_function/VectorReader-inl.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook {
namespace velox {
namespace codegen {

template <TypeKind SQLType>
constexpr bool arrayReadableType = TypeTraits<SQLType>::isPrimitiveType;

template <typename T, typename Config>
struct ComplexVectorReader {};

// Reader for ArrayVector(FlatVector<NativeVal>)
// Use the templated Array<> type here instead of ARRAY typekind for
// static template expansion
template <typename SQLElementType, typename Config>
struct ComplexVectorReader<Array<SQLElementType>, Config> {
  static_assert(arrayReadableType<SQLElementType::NativeType::typeKind>);
  using ElementType = typename SQLElementType::NativeType::NativeType;
  using ElementValueType =
      typename VectorReader<SQLElementType, Config>::ValueType;
  using ElementInputType =
      typename VectorReader<SQLElementType, Config>::InputType;
  using ValueType = std::optional<std::vector<ElementValueType>>;
  using InputType = std::optional<std::vector<ElementInputType>>;

  explicit ComplexVectorReader(VectorPtr& vector) {
    VELOX_CHECK(vector->type()->kind() == TypeKind::ARRAY);
    arrayVectorPtr_ = std::dynamic_pointer_cast<ArrayVector>(vector);
    VELOX_CHECK_NOT_NULL(arrayVectorPtr_);
  }

  struct PointerType {
    explicit PointerType(
        ArrayVectorPtr& arrayVectorPtr,
        size_t rowIndex,
        vector_size_t offset)
        : arrayVectorPtr_(arrayVectorPtr),
          rowIndex_(rowIndex),
          vectorReader_(
              const_cast<VectorPtr&>(arrayVectorPtr->elements()),
              offset) {}

    inline bool has_value() {
      return !arrayVectorPtr_->isNullAt(rowIndex_);
    }

    inline std::optional<ElementType&> value() {
      return *(vectorReader_[0]);
    }

    inline size_t size() {
      if (!has_value()) {
        throw std::logic_error("element has no value");
      }
      return arrayVectorPtr_->sizeAt(rowIndex_);
    }

    inline void setNullAndSize() {
      static_assert(Config::isWriter_);
      auto mutableNulls = arrayVectorPtr_->mutableRawNulls();
      auto oldNullCount = arrayVectorPtr_->getNullCount();
      if (oldNullCount.hasValue()) {
        arrayVectorPtr_->setNullCount(oldNullCount.value() + 1);
      }
      bits::setBit(mutableNulls, rowIndex_, bits::kNull);
      setSize(0);
    }

    inline void setNotNullAndSize(vector_size_t size) {
      static_assert(Config::isWriter_);
      auto mutableNulls = arrayVectorPtr_->mutableRawNulls();
      auto oldNullCount = arrayVectorPtr_->getNullCount();
      if (oldNullCount.hasValue()) {
        arrayVectorPtr_->setNullCount(oldNullCount.value() - 1);
      }
      bits::setBit(mutableNulls, rowIndex_, bits::kNotNull);
      setSize(size);
    }

    inline ElementType& operator*() {
      return *(vectorReader_[0]);
    }

    inline const ElementType& operator*() const {
      return *(vectorReader_[0]);
    }

    inline typename VectorReader<SQLElementType, Config>::PointerType
    operator[](size_t elementIndex) {
      return vectorReader_[elementIndex];
    }

    inline PointerType& operator=(InputType& other) {
      if (!other.has_value()) {
        setNullAndSize();
        return *this;
      } else {
        auto val = other.value();
        setNotNullAndSize(val.size());
        for (size_t i = 0; i < val.size(); i++) {
          vectorReader_[i] = val[i];
        }
        return *this;
      }
    }

    operator ValueType const() {
      if (!has_value()) {
        return {};
      } else {
        std::vector<ElementValueType> val(size());
        for (size_t i = 0; i < size(); i++) {
          val.push_back(this[i]);
        }
        return val;
      }
    }

   private:
    ArrayVectorPtr& arrayVectorPtr_;
    size_t rowIndex_;
    VectorReader<SQLElementType, Config> vectorReader_;

    inline void setSize(vector_size_t size) {
      // reserve metadata vectors and resize if needed
      auto mutableSizes =
          arrayVectorPtr_->mutableSizes(size)->asMutable<vector_size_t>();
      auto mutableOffsets =
          arrayVectorPtr_->mutableOffsets(size)->asMutable<vector_size_t>();

      // FIXME: this assumes that setSize() is called in sequential
      // order
      mutableSizes[rowIndex_] = size;
      if (rowIndex_ == 0) {
        mutableOffsets[rowIndex_] = 0;
      } else {
        mutableOffsets[rowIndex_] =
            mutableOffsets[rowIndex_ - 1] + mutableSizes[rowIndex_ - 1];
      }
      return;
    }
  };

  inline PointerType operator[](size_t rowIndex) {
    // We only support simple arrays for now
    VELOX_CHECK_NOT_NULL(
        arrayVectorPtr_->elements()->asFlatVector<ElementType>());
    vector_size_t offset;
    if constexpr (Config::isWriter_) {
      if (rowIndex == 0) {
        offset = 0;
      } else {
        offset = arrayVectorPtr_->offsetAt(rowIndex - 1) +
            arrayVectorPtr_->sizeAt(rowIndex - 1);
      }
    } else {
      offset = arrayVectorPtr_->offsetAt(rowIndex);
    }
    return PointerType{arrayVectorPtr_, rowIndex, offset};
  }

 private:
  ArrayVectorPtr arrayVectorPtr_;
};

} // namespace codegen
} // namespace velox
} // namespace facebook
