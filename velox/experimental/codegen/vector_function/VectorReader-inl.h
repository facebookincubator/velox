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

#include <stdexcept>
#include "velox/buffer/Buffer.h"
#include "velox/common/base/Nulls.h"
#include "velox/experimental/codegen/vector_function/StringTypes.h"
#include "velox/type/Type.h"
#include "velox/vector/FlatVector.h"

namespace facebook {
namespace velox {
namespace codegen {

// Expected members of the vector reader config.
// struct Config {
//   static constexpr bool isWriter_
//   // when true, the reader will never read a nullvalue
//   static constexpr bool mayReadNull_
//
//   static constexpr bool isWriter_
//   // true means set to null, false means not null
//   static constexpr bool intializedWithNullSet_
//
//   // when true, the reader will never receive a null value to write
//   static constexpr bool mayWriteNull_
//
//
//  String writer config
//  constexpr static bool inputStringBuffersShared = false;
//  constexpr static bool constantStringBuffersShared = false;
// };

// TODO: add bounds check everywhere

// TODO: move readers to different directory
/// Only support scalarType for now
template <TypeKind SQLType>
constexpr bool readableType = TypeTraits<SQLType>::isFixedWidth ||
    TypeTraits<SQLType>::typeKind == TypeKind::VARCHAR;

/// Reads from velox vector and produce native values.
/// This class is meant to behave as much as possible as a std::countainer
/// We introduce a ReferenceType<T> which approximate T &
/// This Reader doesn't own the underlying vector.
/// \tparam SQLType
/// \tparam nullable true iff the underlying vector is nullable
/// \tparam presetNullBits set all the nulls bits to bits::kNotNull if true
/// \tpaarm assumeNullBitSet assume all nulls bits are set to bits::kNotNull if
/// true
template <
    typename SQLType,
    typename Config,
    class T = typename std::enable_if<
        readableType<SQLType::NativeType::typeKind>,
        typename TypeTraits<SQLType::NativeType::typeKind>::NativeType>::type>
struct VectorReader {
  using NativeType = typename SQLType::NativeType::NativeType;

  // The type used for codegen expression inputs
  using ValueType = std::optional<NativeType>;
  using InputType = ValueType;
  explicit VectorReader(VectorPtr& vector, vector_size_t offset = 0)
      : offset_(offset) {
    VELOX_CHECK_NOT_NULL(
        std::dynamic_pointer_cast<FlatVector<NativeType>>(vector));
    VELOX_CHECK_EQ(vector->typeKind(), SQLType::NativeType::typeKind);

    auto flatVector = vector->asFlatVector<NativeType>();

    if constexpr (Config::isWriter_) {
      mutableRawNulls_ = flatVector->mutableRawNulls();
      mutableRawValues_ = flatVector->mutableRawValues();
    } else {
      if constexpr (Config::mayReadNull_) {
        // TODO when read only vector does not have nulls we dont need to
        // allocate nulls
        mutableRawNulls_ = flatVector->mutableRawNulls();
      }
      mutableRawValues_ = const_cast<NativeType*>(flatVector->rawValues());
    }
  }

  NativeType* mutableRawValues_;
  uint64_t* mutableRawNulls_;
  vector_size_t offset_ = 0;

  struct PointerType {
    size_t rowIndex_;
    NativeType* mutableValues_;
    uint64_t* mutableNulls_;
    vector_size_t offset_;

    inline size_t index() const {
      return rowIndex_ + offset_;
    }

    inline bool has_value() {
      // FIXME: generated code should avoid calling on this on the writer and
      // rather reads the on stack value
      // static_assert(!Config::isWriter_);
      if constexpr (!Config::mayReadNull_) {
        return true;

      } else {
        // read nullability
        return !bits::isBitNull(mutableNulls_, index());
      }
    }

    inline NativeType& value() {
      if (!has_value()) {
        throw std::logic_error("element has no value");
      }
      return mutableValues_[index()];
    }

    inline NativeType& operator*() {
      return mutableValues_[index()];
    }

    inline const NativeType& operator*() const {
      return mutableValues_[index()];
    }

    inline PointerType& operator=(const NativeType& other) {
      static_assert(Config::isWriter_);

      if constexpr (Config::intializedWithNullSet_) {
        bits::setBit(mutableNulls_, index(), bits::kNotNull);
      }

      mutableValues_[index()] = other;
      return *this;
    }

    inline PointerType& operator=(const ValueType& other) {
      static_assert(Config::isWriter_);

      if constexpr (!Config::mayWriteNull_) {
        *this = *other;
        return *this;
      } else {
        if (other.has_value()) {
          *this = *other;
        } else {
          if constexpr (!Config::intializedWithNullSet_) {
            bits::setBit(mutableNulls_, index(), bits::kNull);
          }
        }
        return *this;
      }
    }

    operator ValueType const() {
      if (!this->has_value()) {
        return {};
      }
      return {*(*this)};
    }
  };

  inline PointerType operator[](size_t rowIndex) {
    return {
        rowIndex,
        this->mutableRawValues_,
        this->mutableRawNulls_,
        this->offset_};
  }
};

// Reader for flatVector<bool>
// PointerType is the type that is consumed
// by the expression for outputs.
// Note: when rawValues is intitialized the initial value for all the values is
// by default 0 and hence we on
// TOOD : there is shared code acorss readers, see if we can in away combine it
template <typename SQLType, typename Config>
struct VectorReader<
    SQLType,
    Config,
    std::
        enable_if_t<SQLType::NativeType::typeKind == TypeKind::BOOLEAN, bool>> {
  explicit VectorReader(VectorPtr& vector, vector_size_t offset = 0)
      : offset_(offset) {
    VELOX_CHECK(vector->type()->kind() == TypeKind::BOOLEAN);
    auto flatVector = vector->asFlatVector<bool>();
    VELOX_CHECK_NOT_NULL(flatVector);

    if constexpr (Config::isWriter_) {
      mutableRawNulls_ = flatVector->mutableRawNulls();
      mutableRawValues_ = flatVector->template mutableRawValues<uint64_t>();
    } else {
      // TODO when read only vector does not have nulls we dont need to allocate
      // nulls
      if constexpr (Config::mayReadNull_) {
        mutableRawNulls_ = flatVector->mutableRawNulls();
      }
      mutableRawValues_ =
          const_cast<uint64_t*>(flatVector->template rawValues<uint64_t>());
    }
  }

  // The type used for codegen expression inputs
  using ValueType = std::optional<bool>;
  using InputType = ValueType;
  struct ReferenceType {
    size_t index_;
    uint64_t* mutableValues_;
    uint64_t* mutableNulls_;

    inline ReferenceType& operator=(const bool& other) {
      static_assert(Config::isWriter_);

      if constexpr (Config::intializedWithNullSet_) {
        bits::setBit(mutableNulls_, index_, bits::kNotNull);
      }

      bits::setBit(mutableValues_, index_, other);
      return *this;
    }

    operator bool() const {
      return bits::isBitSet(mutableValues_, index_);
    }
  };

  struct PointerType {
    size_t rowIndex_;
    uint64_t* mutableValues_;
    uint64_t* mutableNulls_;
    vector_size_t offset_;

    inline size_t index() {
      return offset_ + rowIndex_;
    }

    inline bool has_value() {
      static_assert(!Config::isWriter_);
      if constexpr (!Config::mayReadNull_) {
        return true;
      } else {
        // read nullability
        return !bits::isBitNull(mutableNulls_, index());
      }
    }

    inline ReferenceType value() {
      if (!has_value()) {
        throw std::logic_error("element has no value");
      }

      return ReferenceType{index(), mutableValues_, mutableNulls_};
    }

    inline ReferenceType operator*() {
      return ReferenceType{index(), mutableValues_, mutableNulls_};
    }

    inline const ReferenceType operator*() const {
      return ReferenceType{index(), mutableValues_, mutableNulls_};
    }

    inline PointerType& operator=(const bool& other) {
      static_assert(Config::isWriter_);
      if constexpr (Config::intializedWithNullSet_) {
        bits::setBit(mutableNulls_, index(), bits::kNotNull);
      }
      bits::setBit(mutableValues_, index(), other);
      return *this;
    }

    inline PointerType& operator=(const ValueType& other) {
      static_assert(Config::isWriter_);

      if constexpr (!Config::mayWriteNull_) {
        *this = *other;
        return *this;
      } else {
        if (other.has_value()) {
          *this = *other;
        } else {
          if constexpr (!Config::intializedWithNullSet_) {
            bits::setBit(mutableNulls_, index(), bits::kNull);
          }
        }
        return *this;
      }
    }

    operator ValueType const() {
      if (!this->has_value()) {
        return {};
      }
      return {*(*this)};
    }
  };

  inline PointerType operator[](size_t rowIndex) {
    return PointerType{
        rowIndex,
        this->mutableRawValues_,
        this->mutableRawNulls_,
        this->offset_};
  }

 private:
  uint64_t* mutableRawValues_;
  uint64_t* mutableRawNulls_;
  vector_size_t offset_ = 0;
};

//****************************************************************************

// Reader for flatVector<StringView>
// TODO: avoid copying constant string by allocating them on Velox buffer
// TODO: specialize for default nulls
template <typename SQLType, typename Config>
struct VectorReader<
    SQLType,
    Config,
    std::enable_if_t<
        SQLType::NativeType::typeKind == TypeKind::VARCHAR,
        StringView>> {
  explicit VectorReader(VectorPtr& vector, vector_size_t offset = 0)
      : vector_(vector->asFlatVector<StringView>()), offset_(offset) {
    VELOX_CHECK(vector->type()->kind() == TypeKind::VARCHAR);
    auto flatVector = vector->asFlatVector<StringView>();
    VELOX_CHECK_NOT_NULL(flatVector);

    if constexpr (Config::isWriter_) {
      mutableRawNulls_ = flatVector->mutableRawNulls();
      mutableRawValues_ = flatVector->template mutableRawValues<StringView>();
    } else {
      // TODO when read only vector does not have nulls we dont need to allocate
      // nulls
      if constexpr (Config::mayReadNull_) {
        mutableRawNulls_ = flatVector->mutableRawNulls();
      }
      mutableRawValues_ =
          const_cast<StringView*>(flatVector->template rawValues<StringView>());
    }
  }

  // The type used for codegen expression inputs
  using ValueType = std::optional<std::reference_wrapper<const StringView>>;
  using InputType = codegen::InputReferenceStringNullable;

  // This is used for the output types
  struct StringProxy {
   public:
    StringProxy(
        FlatVector<StringView>* vector,
        vector_size_t rowIndex,
        StringView* mutableValues,
        vector_size_t offset)
        : vector_(vector),
          rowIndex_(rowIndex),
          mutableValues_(mutableValues),
          offset_(offset) {
      // We need to get a reference to the StringView to avoid inlined
      // prefix
      auto& string = vector->rawValues()[index()];
      setData(const_cast<char*>(string.data()));
      setSize(string.size());
    }

    inline size_t index() const {
      return offset_ + rowIndex_;
    }

    void operator=(const InputReferenceString& other_) {
      static_assert(Config::isWriter_);

      auto& other = other_.get();
      if constexpr (Config::inputStringBuffersShared) {
        mutableValues_[index()] = other_;
      } else {
        reserve(other.size());
        if (other.size() != 0) {
          std::memcpy(data(), other.data(), other.size());
        }
        resize(other.size());
        finalize();
      }
    }

    void operator=(const ConstantString& other_) {
      static_assert(Config::isWriter_);

      auto& other = other_.get();
      if constexpr (Config::constantStringBuffersShared) {
        mutableValues_[index()] = other_;
      } else {
        reserve(other.size());
        if (other.size() != 0) {
          std::memcpy(data(), other.data(), other.size());
        }
        resize(other.size());
        finalize();
      }
    }

    size_t size() const {
      return size_;
    }

    size_t capacity() const {
      return capacity_;
    }

    char* data() const {
      return data_;
    }

    /// Reserve a space for the output string with size of at least newCapacity
    void reserve(size_t newCapacity) {
      if (newCapacity <= capacity()) {
        return;
      }

      auto* newDataBuffer = vector_->getBufferWithSpace(newCapacity);

      // If the new allocated space is on the same buffer no need to copy
      // content or reassign start address
      if (buffer_ == newDataBuffer) {
        setCapacity(newCapacity);
        return;
      }

      auto newStartAddress =
          newDataBuffer->asMutable<char>() + newDataBuffer->size();

      if (size() != 0) {
        std::memcpy(newStartAddress, data(), size());
      }

      setCapacity(newCapacity);
      setData(newStartAddress);
      buffer_ = newDataBuffer;
    }

    /// Has the semantics as std::string, except that it does not fill the
    /// space[size(), newSize] with 0 but rather leaves it as is
    void resize(size_t newSize) {
      if (newSize <= size_) {
        // shrinking
        size_ = newSize;
        return;
      }

      // newSize > size
      if (newSize <= capacity_) {
        size_ = newSize;
      } else {
        reserve(newSize);
        resize(newSize);
      }
    }

    /// Not called by the UDF Implementation. Should be called at the end to
    /// finalize the allocation and the string writing.
    void finalize() {
      VELOX_CHECK(size() == 0 || data());

      if (buffer_) {
        buffer_->setSize(buffer_->size() + size());
      }
      mutableValues_[index()] = StringView(data(), size());
      return;
    }

    vector_size_t rowIndex() const {
      return rowIndex_;
    }

    StringView* mutableValues() const {
      return mutableValues_;
    }

   private:
    void setData(char* address) {
      data_ = address;
    }

    void setSize(size_t newSize) {
      size_ = newSize;
    }

    void setCapacity(size_t newCapacity) {
      capacity_ = newCapacity;
    }

    /// Address to the start of the string
    char* data_ = nullptr;

    /// Size of the string in bytes
    size_t size_ = 0;

    /// The capacity of the string in bytes
    size_t capacity_ = 0;

    /// The buffer that the output string uses for its allocation set during
    /// reserve() call.
    Buffer* buffer_ = nullptr;

    FlatVector<StringView>* vector_;

    int32_t rowIndex_;

    StringView* mutableValues_;

    vector_size_t offset_ = 0;
  };

  struct PointerType {
    uint64_t* mutableNulls_;
    StringProxy proxy_;
    vector_size_t offset_;

    inline bool has_value() {
      if constexpr (Config::mayReadNull_) {
        return !bits::isBitNull(mutableNulls_, proxy_.index());
      } else {
        return true;
      }
    }

    inline StringProxy& value() {
      return proxy_;
    }

    operator codegen::InputReferenceStringNullable const() {
      static_assert(!Config::isWriter_);

      if (!this->has_value()) {
        return {};
      }

      return codegen::InputReferenceStringNullable{
          InputReferenceString{proxy_.mutableValues()[proxy_.index()]}};
    }

    inline PointerType& operator=(const InputReferenceStringNullable& other) {
      static_assert(Config::isWriter_);
      if constexpr (!Config::mayWriteNull_) {
        if constexpr (Config::intializedWithNullSet_) {
          bits::setBit(mutableNulls_, proxy_.index(), !bits::kNull);
        }
        proxy_ = *other;

      } else {
        // may have null
        if (other.has_value()) {
          if constexpr (Config::intializedWithNullSet_) {
            bits::setBit(mutableNulls_, proxy_.index(), !bits::kNull);
          }
          proxy_ = *other;

        } else {
          if constexpr (!Config::intializedWithNullSet_) {
            bits::setBit(mutableNulls_, proxy_.index(), bits::kNull);
          }
        }
      }

      return *this;
    }

    inline PointerType& operator=(const ConstantStringNullable& other) {
      static_assert(Config::isWriter_);
      if constexpr (!Config::mayWriteNull_) {
        if constexpr (Config::intializedWithNullSet_) {
          bits::setBit(mutableNulls_, proxy_.index(), bits::kNotNull);
        }
        proxy_ = *other;

      } else {
        // may have null
        if (other.has_value()) {
          if constexpr (Config::intializedWithNullSet_) {
            bits::setBit(mutableNulls_, proxy_.index(), bits::kNotNull);
          }
          proxy_ = *other;

        } else {
          if constexpr (!Config::intializedWithNullSet_) {
            bits::setBit(mutableNulls_, proxy_.index(), bits::kNull);
          }
        }
      }

      return *this;
    }

    inline PointerType& operator=(const std::nullopt_t&) {
      static_assert(Config::isWriter_);

      if constexpr (!Config::intializedWithNullSet_) {
        bits::setBit(mutableNulls_, proxy_.index(), bits::kNull);
      }
      return *this;
    }

    inline StringProxy& operator*() {
      // static_assert(Config::isWriter_);
      if constexpr (Config::intializedWithNullSet_) {
        bits::setBit(mutableNulls_, proxy_.index(), bits::kNotNull);
      }
      return proxy_;
    }

    inline const StringProxy& operator*() const {
      // static_assert(Config::isWriter_);
      if constexpr (Config::intializedWithNullSet_) {
        bits::setBit(mutableNulls_, proxy_.index(), bits::kNotNull);
      }
      return proxy_;
    }
  };

  inline PointerType operator[](size_t rowIndex) {
    return PointerType{
        mutableRawNulls_,
        StringProxy(vector_, rowIndex, mutableRawValues_, this->offset_),
        this->offset_};
  }

 private:
  StringView* mutableRawValues_;
  uint64_t* mutableRawNulls_;
  FlatVector<StringView>* vector_;
  vector_size_t offset_ = 0;
};

} // namespace codegen
} // namespace velox
} // namespace facebook
