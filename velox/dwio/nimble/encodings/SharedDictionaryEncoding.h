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
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include "velox/buffer/Buffer.h"
#include "velox/common/Casts.h"
#include "velox/common/EnumDeclare.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/DataTypeDispatch.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

/// Identifies where an alphabet is stored or resolved, and the dictionary id
/// namespace to use. The scope controls both dictionary lifetime and whether
/// the reader loads an in-stripe auxiliary stream, the file catalog, or an
/// external catalog.
enum class SharedDictionaryScope : uint8_t {
  /// Alphabet is stored as an auxiliary stream in the current stripe, and its
  /// dictionary id is local to that stripe.
  Stripe = 0,
  /// Alphabet is stored once in the file shared dictionary catalog, and its
  /// dictionary id is local to that file. Value 1 is reserved so unknown
  /// historical values remain invalid.
  File = 2,
  /// Alphabet is resolved from an external catalog, and its dictionary id is
  /// interpreted by the configured resolver.
  External = 3,
};

VELOX_DECLARE_ENUM_NAME(SharedDictionaryScope);

inline SharedDictionaryScope toSharedDictionaryScope(uint8_t value) {
  const auto scope = static_cast<SharedDictionaryScope>(value);
  switch (scope) {
    case SharedDictionaryScope::Stripe:
    case SharedDictionaryScope::File:
    case SharedDictionaryScope::External:
      return scope;
  }
  NIMBLE_UNSUPPORTED(
      "Unsupported shared dictionary scope {}.", static_cast<int>(value));
}

inline constexpr uint32_t kMaxSharedDictionarySize =
    std::numeric_limits<uint32_t>::max();

inline constexpr uint32_t kInvalidSharedDictionaryId =
    std::numeric_limits<uint32_t>::max();

namespace detail {

template <typename T>
class FixedEncodingSelectionPolicy final : public EncodingSelectionPolicy<T> {
 public:
  using physicalType = typename TypeTraits<T>::physicalType;

  explicit FixedEncodingSelectionPolicy(EncodingType encodingType)
      : encodingType_{encodingType} {}

  EncodingSelectionResult select(
      std::span<const physicalType> /* values */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) final {
    return {
        .encodingType = encodingType_,
        .encodingConfig = {},
        .estimatedSize = std::nullopt,
    };
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) final {
    NIMBLE_UNREACHABLE(
        "Fixed encoding selection does not support nullable values.");
  }

 private:
  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType parentEncodingType,
      NestedEncodingIdentifier nestedEncodingIdentifier,
      DataType nestedDataType) final {
    auto nestedEncodingReadFactors =
        ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
    nestedEncodingReadFactors.erase(
        std::remove_if(
            nestedEncodingReadFactors.begin(),
            nestedEncodingReadFactors.end(),
            [parentEncodingType](const auto& entry) {
              return entry.first == parentEncodingType;
            }),
        nestedEncodingReadFactors.end());
    UNIQUE_PTR_FACTORY(
        nestedDataType,
        ManualEncodingSelectionPolicy,
        std::move(nestedEncodingReadFactors),
        std::nullopt,
        nestedEncodingIdentifier);
  }

  const EncodingType encodingType_;
};

/// Encodes a Dictionary root and pins nested alphabet and indices encodings
/// when they are supplied.
/// Slicing uses it to rebuild a local dictionary with the encodings the sliced
/// stream already used. Nested streams without a pinned encoding fall back to
/// manual selection minus the parent encoding.
template <typename T>
class PreservedDictionaryEncodingSelectionPolicy final
    : public EncodingSelectionPolicy<T> {
 public:
  using physicalType = typename TypeTraits<T>::physicalType;

  explicit PreservedDictionaryEncodingSelectionPolicy(
      std::optional<EncodingType> alphabetEncodingType = std::nullopt,
      std::optional<EncodingType> indicesEncodingType = std::nullopt)
      : alphabetEncodingType_{alphabetEncodingType},
        indicesEncodingType_{indicesEncodingType} {}

  EncodingSelectionResult select(
      std::span<const physicalType> /* values */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) final {
    return {
        .encodingType = EncodingType::Dictionary,
        .encodingConfig = {},
        .estimatedSize = std::nullopt,
    };
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) final {
    NIMBLE_UNREACHABLE(
        "Preserved dictionary encoding selection does not support nullable "
        "values.");
  }

 private:
  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType parentEncodingType,
      NestedEncodingIdentifier nestedEncodingIdentifier,
      DataType nestedDataType) final {
    if (parentEncodingType == EncodingType::Dictionary &&
        nestedEncodingIdentifier == EncodingIdentifiers::Dictionary::Alphabet &&
        alphabetEncodingType_.has_value()) {
      UNIQUE_PTR_FACTORY(
          nestedDataType, FixedEncodingSelectionPolicy, *alphabetEncodingType_);
    }
    if (parentEncodingType == EncodingType::Dictionary &&
        nestedEncodingIdentifier == EncodingIdentifiers::Dictionary::Indices &&
        indicesEncodingType_.has_value()) {
      UNIQUE_PTR_FACTORY(
          nestedDataType, FixedEncodingSelectionPolicy, *indicesEncodingType_);
    }

    auto nestedEncodingReadFactors =
        ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
    nestedEncodingReadFactors.erase(
        std::remove_if(
            nestedEncodingReadFactors.begin(),
            nestedEncodingReadFactors.end(),
            [parentEncodingType](const auto& entry) {
              return entry.first == parentEncodingType;
            }),
        nestedEncodingReadFactors.end());
    UNIQUE_PTR_FACTORY(
        nestedDataType,
        ManualEncodingSelectionPolicy,
        std::move(nestedEncodingReadFactors),
        std::nullopt,
        nestedEncodingIdentifier);
  }

  const std::optional<EncodingType> alphabetEncodingType_;
  const std::optional<EncodingType> indicesEncodingType_;
};

} // namespace detail

/// Provides indexed access to one immutable shared dictionary alphabet.
///
/// An alphabet is always a Nimble encoded stream: writers serialize the
/// dictionary entries with encode(), and readers hand those same bytes to the
/// constructor. Encodings with an EncodingView are read by index straight from
/// the stream; the rest are decoded once into pool backed storage, so any
/// encoding can store an alphabet. Alphabets stay uncompressed either way,
/// because compression is invisible to supportsEncodingView() and would make
/// that choice unreliable.
class SharedDictionaryAlphabet {
 public:
  /// Creates an alphabet backed by bytes kept alive by encodedAlphabetOwner.
  static std::shared_ptr<const SharedDictionaryAlphabet> create(
      std::string_view encodedAlphabet,
      std::shared_ptr<const void> encodedAlphabetOwner,
      velox::memory::MemoryPool* pool);

  /// Serializes alphabet entries into the form the constructor accepts.
  ///
  /// candidateEncodings restricts the encoding selection policy. An empty list
  /// falls back to the default encoding selection policy.
  template <typename T>
  static std::string_view encode(
      std::span<const T> values,
      std::span<const EncodingType> candidateEncodings,
      Buffer& buffer,
      const Encoding::Options& options = {}) {
    return EncodingFactory::encode<T>(
        createEncodingPolicy<T>(candidateEncodings), values, buffer, options);
  }

  /// Returns the encoded size encode() would produce for values. Writers use
  /// this to size a dictionary before they commit to shared dictionary
  /// encoding. Throws when the selected encoding has no size model.
  template <typename T>
  static uint64_t estimateSize(
      std::span<const T> values,
      std::span<const EncodingType> candidateEncodings,
      const Encoding::Options& options) {
    using physicalType = typename TypeTraits<T>::physicalType;
    static_assert(sizeof(T) == sizeof(physicalType));
    const std::span<const physicalType> physicalValues{
        reinterpret_cast<const physicalType*>(values.data()), values.size()};
    const auto statistics = Statistics<physicalType>::create(physicalValues);
    auto policy = createEncodingPolicy<T>(candidateEncodings);
    const auto selection = policy->select(physicalValues, statistics, options);
    const auto estimate = selection.estimatedSize.has_value()
        ? selection.estimatedSize
        : detail::EncodingSizeEstimation<T>::estimateSize(
              selection.encodingType, physicalValues, statistics, options);
    NIMBLE_USER_CHECK(
        estimate.has_value(),
        "{} cannot size a shared dictionary alphabet of {} entries.",
        selection.encodingType,
        values.size());
    return estimate.value();
  }

  /// Returns the Nimble data type shared by all alphabet entries.
  DataType dataType() const {
    return dataType_;
  }

  /// Returns the number of entries the alphabet holds.
  uint32_t entryCount() const {
    return entryCount_;
  }

  /// Returns the Nimble encoding the serialized alphabet uses. Slicing reuses
  /// it so a rewritten local dictionary keeps the layout it came from.
  EncodingType encodingType() const {
    return encodingType_;
  }

  /// Returns the encoded alphabet stream passed to create().
  std::string_view encodedAlphabet() const {
    return encodedAlphabet_;
  }

  /// Returns the entry stored at index. T must match dataType().
  template <typename T>
  typename TypeTraits<T>::physicalType physicalValueAt(uint32_t index) const {
    checkEntryType<T>();
    checkEntryIndex(index);
    typename TypeTraits<T>::physicalType value;
    if (entryView_ != nullptr) {
      entryView_->readAt(index, &value);
      return value;
    }
    return decodedEntries<T>()[index];
  }

  /// Copies the entries selected by indices into output, which must have room
  /// for indices.size() values. T must match dataType().
  template <typename T>
  void materialize(
      std::span<const uint32_t> indices,
      typename TypeTraits<T>::physicalType* output) const {
    checkEntryType<T>();
    if (entryView_ != nullptr) {
      for (size_t i = 0; i < indices.size(); ++i) {
        checkEntryIndex(indices[i]);
        entryView_->readAt(indices[i], output + i);
      }
      return;
    }
    const auto entries = decodedEntries<T>();
    for (size_t i = 0; i < indices.size(); ++i) {
      checkEntryIndex(indices[i]);
      output[i] = entries[indices[i]];
    }
  }

  /// Copies every entry into output in alphabet order. T must match dataType().
  template <typename T>
  void materializeAll(Vector<T>& output) const {
    checkEntryType<T>();
    output.resize(entryCount_);
    if (entryCount_ == 0) {
      return;
    }
    auto physicalOutput = EncodingPhysicalType<T>::asEncodingPhysicalTypeSpan(
        std::span<T>{output.data(), output.size()});
    if (entryView_ != nullptr) {
      entryView_->read(/*offset=*/0, entryCount_, physicalOutput.data());
      return;
    }
    const auto entries = decodedEntries<T>();
    std::copy(entries.begin(), entries.end(), physicalOutput.begin());
  }

 private:
  SharedDictionaryAlphabet(
      std::string_view encoded,
      const Encoding::Options& options,
      std::shared_ptr<const void> encodedAlphabetOwner,
      velox::memory::MemoryPool* pool);

  void decodeEntries(Encoding& encoding, velox::memory::MemoryPool* pool) {
    NIMBLE_RETURN_BY_DATA_TYPE_OR(
        // The macro's default case handles Undefined.
        // @lint-ignore CLANGTIDY clang-diagnostic-switch-enum
        dataType_,
        T,
        (decodeEntriesTyped<T>(encoding, pool), void()),
        NIMBLE_UNSUPPORTED(
            "{} is not supported by shared dictionary alphabets.", dataType_));
  }

  template <typename T>
  void decodeEntriesTyped(Encoding& encoding, velox::memory::MemoryPool* pool) {
    entries_ =
        velox::AlignedBuffer::allocate<typename TypeTraits<T>::physicalType>(
            entryCount_, pool);
    encoding.materialize(entryCount_, entries_->asMutable<void>());
  }

  inline void checkEntryIndex(uint32_t index) const {
    NIMBLE_DCHECK_LT(
        index, entryCount_, "Shared dictionary index exceeds alphabet size.");
  }

  template <typename T>
  std::span<const typename TypeTraits<T>::physicalType> decodedEntries() const {
    NIMBLE_CHECK_NOT_NULL(entries_);
    return {entries_->as<typename TypeTraits<T>::physicalType>(), entryCount_};
  }

  template <typename T>
  static std::unique_ptr<EncodingSelectionPolicy<T>> createEncodingPolicy(
      std::span<const EncodingType> candidateEncodings) {
    // Compression is disabled because supportsEncodingView() only sees the
    // encoding type, so a compressed stream would be mistaken for one the
    // reader can index into.
    const ManualEncodingSelectionPolicyFactory factory{
        encodingReadFactors(candidateEncodings),
        /*compressionOptions=*/std::nullopt};
    auto policy = factory.createPolicy(TypeTraits<T>::dataType);
    return std::unique_ptr<EncodingSelectionPolicy<T>>(
        static_cast<EncodingSelectionPolicy<T>*>(policy.release()));
  }

  static std::vector<std::pair<EncodingType, float>> encodingReadFactors(
      std::span<const EncodingType> candidateEncodings) {
    if (candidateEncodings.empty()) {
      return ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
    }
    std::vector<std::pair<EncodingType, float>> readFactors;
    readFactors.reserve(candidateEncodings.size());
    for (const auto encodingType : candidateEncodings) {
      readFactors.emplace_back(encodingType, 1.0f);
    }
    return readFactors;
  }

  // Reading with the wrong T makes the view write its own physical type into a
  // buffer sized for T, so this guards memory safety rather than just
  // correctness. Every caller resolves the alphabet and checks its type once up
  // front, so this stays a debug assertion instead of a per-entry branch. The
  // view bounds checks the index itself.
  template <typename T>
  void checkEntryType() const {
    static_assert(
        !std::is_same_v<T, std::string>,
        "Use std::string_view for shared dictionary string alphabets.");
    NIMBLE_DCHECK_EQ(
        dataType(),
        TypeTraits<T>::dataType,
        "Shared dictionary alphabet has unexpected type.");
  }

  // Keeps borrowed encoded bytes alive when create() receives an owner.
  const std::shared_ptr<const void> encodedAlphabetOwner_;
  const std::string_view encodedAlphabet_;
  const DataType dataType_;
  const EncodingType encodingType_;
  uint32_t entryCount_{0};
  // entryView_ and entries_ are mutually exclusive lookup modes. entryPayload_
  // is used by decoded entry storage when variable-width values in entries_
  // need owned payload bytes.
  Buffer entryPayload_;
  // Entries decoded up front for encodings without a view.
  velox::BufferPtr entries_;
  // Reads entries straight from the encoded stream when the encoding supports
  // indexed access.
  const std::unique_ptr<EncodingView> entryView_;
};

/// The layout for a shared dictionary encoding is an encoding prefix followed
/// by encoded indices. The value stream's catalog binding supplies the
/// alphabet.
template <typename T>
class SharedDictionaryEncoding
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using physicalType = typename TypeTraits<T>::physicalType;

  SharedDictionaryEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {});

  ~SharedDictionaryEncoding() override {
    this->releaseBuffer(indicesBuffer_);
  }

  SharedDictionaryEncoding(const SharedDictionaryEncoding&) = delete;
  SharedDictionaryEncoding& operator=(const SharedDictionaryEncoding&) = delete;
  SharedDictionaryEncoding(SharedDictionaryEncoding&&) = delete;
  SharedDictionaryEncoding& operator=(SharedDictionaryEncoding&&) = delete;

  void reset() final {
    indicesEncoding_->reset();
  }

  void skip(uint32_t rowCount) final {
    indicesEncoding_->skip(rowCount);
  }

  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename V>
  void readWithVisitor(V& visitor, ReadWithVisitorParams& params);

  void materializeIndices(uint32_t rowCount, uint32_t* buffer) override {
    indicesEncoding_->materialize(rowCount, buffer);
  }

  bool dictionaryEnabled() const override {
    // TODO: Enable dictionary preservation after SharedDictionary implements
    // readIndicesWithVisitor().
    return false;
  }

  uint32_t dictionarySize() const override {
    return alphabet_->entryCount();
  }

  static std::string_view encode(
      std::span<const uint32_t> indices,
      const EncodingSelectionPolicyCreator& nestedPolicyCreator,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static std::string_view encodeNullable(
      EncodingSelection<physicalType>&& selection,
      std::span<const physicalType> values,
      std::span<const bool> nulls,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  std::string debugString(int offset) const final;

 private:
  static std::string_view encodeIndices(
      std::span<const uint32_t> indices,
      const EncodingSelectionPolicyCreator& nestedPolicyCreator,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static std::string_view encodeMaterializedDictionarySlice(
      const SharedDictionaryAlphabet& alphabet,
      std::string_view encodedIndices,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static void materializeLocalDictionaryIndices(
      std::span<const uint32_t> sortedUniqueSharedIndices,
      std::span<const uint32_t> slicedSharedIndices,
      std::span<uint32_t> localIndices);

  uint32_t* ensureIndexBuffer(uint32_t numElements) {
    const auto bytes = numElements * sizeof(uint32_t);
    if (indicesBuffer_ == nullptr || indicesBuffer_->capacity() < bytes) {
      if (indicesBuffer_ != nullptr) {
        this->releaseBuffer(indicesBuffer_);
      }
      indicesBuffer_ = this->getBuffer(bytes);
    }
    return indicesBuffer_->asMutable<uint32_t>();
  }

  std::shared_ptr<const SharedDictionaryAlphabet> alphabet_;
  std::unique_ptr<Encoding> indicesEncoding_;
  velox::BufferPtr indicesBuffer_;
};

template <typename T>
SharedDictionaryEncoding<T>::SharedDictionaryEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options} {
  static_assert(isIntegralType<T>() && !std::is_same_v<T, bool>);
  NIMBLE_CHECK_NOT_NULL(
      options.sharedDictionaryAlphabet,
      "Shared dictionary encoding requires an alphabet.");

  const char* pos = data.data() + this->dataOffset();
  const auto indicesOffset = static_cast<size_t>(pos - data.data());
  NIMBLE_CHECK_LT(
      indicesOffset,
      data.size(),
      "Shared dictionary encoding is missing its indices.");
  alphabet_ = options.sharedDictionaryAlphabet;
  NIMBLE_CHECK_NOT_NULL(alphabet_);
  NIMBLE_CHECK_EQ(
      alphabet_->dataType(),
      TypeTraits<T>::dataType,
      "Shared dictionary alphabet has unexpected type.");

  indicesEncoding_ = EncodingFactory().create(
      *this->pool_,
      {pos, data.size() - indicesOffset},
      std::move(stringBufferFactory),
      options);
  NIMBLE_CHECK_EQ(
      indicesEncoding_->dataType(),
      DataType::Uint32,
      "Shared dictionary indices have unexpected type.");
  NIMBLE_CHECK_EQ(
      indicesEncoding_->rowCount(),
      this->rowCount(),
      "Shared dictionary index count differs from row count.");
}

template <typename T>
void SharedDictionaryEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  auto* indices = ensureIndexBuffer(rowCount);
  indicesEncoding_->materialize(rowCount, indices);
  alphabet_->template materialize<T>(
      std::span<const uint32_t>{indices, rowCount},
      static_cast<physicalType*>(buffer));
}

template <typename T>
template <typename V>
void SharedDictionaryEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  if constexpr (sizeof(T) < sizeof(uint32_t)) {
    // Nested index decoders may use the reader values buffer before the hook
    // copies indices out, so it must fit uint32_t indices rather than T values.
    visitor.reader().template ensureValuesCapacity<uint32_t>(
        visitor.numRows(), /*preserveValues=*/true);
  }
  const auto startRowIndex = visitor.rowIndex();
  const auto numIndices = visitor.numRows() - startRowIndex;
  auto* indices = ensureIndexBuffer(numIndices);
  velox::common::AlwaysTrue indicesFilter;
  detail::DictionaryIndicesHook indicesHook(indices, startRowIndex);
  auto indicesVisitor = DecoderVisitor<
      int32_t,
      velox::common::AlwaysTrue,
      velox::dwio::common::ExtractToHook<detail::DictionaryIndicesHook>,
      V::dense>(
      indicesFilter,
      &visitor.reader(),
      velox::RowSet(visitor.rows(), visitor.numRows()),
      velox::dwio::common::ExtractToHook<detail::DictionaryIndicesHook>(
          &indicesHook));
  indicesVisitor.setRowIndex(startRowIndex);
  callReadWithVisitor(*indicesEncoding_, indicesVisitor, params);
  detail::readWithVisitorSlow(visitor, params, nullptr, [&] {
    return alphabet_->template physicalValueAt<T>(
        indices[visitor.rowIndex() - startRowIndex]);
  });
}

template <typename T>
std::string_view SharedDictionaryEncoding<T>::encode(
    std::span<const uint32_t> indices,
    const EncodingSelectionPolicyCreator& nestedPolicyCreator,
    Buffer& buffer,
    const Encoding::Options& options) {
  static_assert(isIntegralType<T>() && !std::is_same_v<T, bool>);

  ScopedEncodingBuffer scopedBuffer{
      &buffer.getMemoryPool(), options.encodingBufferPool};
  const auto encodedIndices =
      encodeIndices(indices, nestedPolicyCreator, scopedBuffer.get(), options);

  const auto rowCount = static_cast<uint32_t>(indices.size());
  const uint64_t encodingSize =
      static_cast<uint64_t>(
          Encoding::serializePrefixSize(rowCount, options.useVarintRowCount)) +
      encodedIndices.size();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::SharedDictionary,
      TypeTraits<T>::dataType,
      rowCount,
      options.useVarintRowCount,
      pos);
  encoding::writeBytes(encodedIndices, pos);
  NIMBLE_CHECK_EQ(
      static_cast<uint64_t>(pos - reserved),
      encodingSize,
      "Encoding size mismatch.");

  return {reserved, encodingSize};
}

template <typename T>
std::string_view SharedDictionaryEncoding<T>::encodeNullable(
    EncodingSelection<physicalType>&& selection,
    std::span<const physicalType> values,
    std::span<const bool> nulls,
    Buffer& buffer,
    const Encoding::Options& options) {
  static_assert(isIntegralType<T>() && !std::is_same_v<T, bool>);

  auto nullsPolicy = selection.template createNestedPolicy<bool>(
      EncodingType::Nullable, EncodingIdentifiers::Nullable::Nulls);
  NIMBLE_CHECK_NOT_NULL(nullsPolicy);
  auto typedNullsPolicy = std::unique_ptr<EncodingSelectionPolicy<bool>>(
      static_cast<EncodingSelectionPolicy<bool>*>(nullsPolicy.release()));
  auto* pool = &buffer.getMemoryPool();
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  const auto serializedValues = EncodingFactory::encode<T>(
      std::move(selection), values, scopedBuffer.get(), options);
  const auto serializedNulls = EncodingFactory::encode<bool>(
      std::move(typedNullsPolicy), nulls, scopedBuffer.get(), options);
  return NullableEncoding<T>::encodeNullable(
      static_cast<uint32_t>(nulls.size()),
      serializedValues,
      serializedNulls,
      buffer,
      options);
}

template <typename T>
std::string_view SharedDictionaryEncoding<T>::encodeIndices(
    std::span<const uint32_t> indices,
    const EncodingSelectionPolicyCreator& nestedPolicyCreator,
    Buffer& buffer,
    const Encoding::Options& options) {
  static_assert(isIntegralType<T>() && !std::is_same_v<T, bool>);
  NIMBLE_CHECK_LE(
      indices.size(),
      kMaxSharedDictionarySize,
      "Shared dictionary index count exceeds maximum.");
  NIMBLE_CHECK_NOT_NULL(
      nestedPolicyCreator,
      "Shared dictionary encoding requires nested policy creator.");
  auto policy = nestedPolicyCreator(DataType::Uint32);
  NIMBLE_CHECK_NOT_NULL(policy);
  return EncodingFactory::encode<uint32_t>(
      std::unique_ptr<EncodingSelectionPolicy<uint32_t>>(
          static_cast<EncodingSelectionPolicy<uint32_t>*>(policy.release())),
      indices,
      buffer,
      options);
}

template <typename T>
std::string_view SharedDictionaryEncoding<T>::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  static_assert(isIntegralType<T>() && !std::is_same_v<T, bool>);
  const auto sourceRowCount =
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
  NIMBLE_CHECK_LE(offset, sourceRowCount);
  NIMBLE_CHECK_LE(length, sourceRowCount - offset);
  NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");

  NIMBLE_CHECK_NOT_NULL(
      options.sharedDictionaryAlphabet,
      "Shared dictionary slicing requires an alphabet.");
  const char* pos = encoded.data() +
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
  const auto indicesOffset = static_cast<size_t>(pos - encoded.data());
  NIMBLE_CHECK_LT(
      indicesOffset,
      encoded.size(),
      "Shared dictionary encoding is missing its indices.");
  const std::string_view encodedIndices{pos, encoded.size() - indicesOffset};

  const auto alphabet = options.sharedDictionaryAlphabet;
  NIMBLE_CHECK_NOT_NULL(alphabet);
  NIMBLE_CHECK_EQ(
      alphabet->dataType(),
      TypeTraits<T>::dataType,
      "Shared dictionary alphabet has unexpected type.");

  return encodeMaterializedDictionarySlice(
      *alphabet, encodedIndices, offset, length, buffer, options);
}

template <typename T>
void SharedDictionaryEncoding<T>::materializeLocalDictionaryIndices(
    std::span<const uint32_t> sortedUniqueSharedIndices,
    std::span<const uint32_t> slicedSharedIndices,
    std::span<uint32_t> localIndices) {
  NIMBLE_DCHECK_GT(sortedUniqueSharedIndices.size(), 1);
  NIMBLE_DCHECK_EQ(localIndices.size(), slicedSharedIndices.size());

  for (size_t i{0}; i < slicedSharedIndices.size(); ++i) {
    const auto it = std::lower_bound(
        sortedUniqueSharedIndices.begin(),
        sortedUniqueSharedIndices.end(),
        slicedSharedIndices[i]);
    NIMBLE_CHECK(
        it != sortedUniqueSharedIndices.end() && *it == slicedSharedIndices[i],
        "Shared dictionary slice index missing from local dictionary.");
    localIndices[i] = static_cast<uint32_t>(
        std::distance(sortedUniqueSharedIndices.begin(), it));
  }
}

template <typename T>
std::string_view SharedDictionaryEncoding<T>::encodeMaterializedDictionarySlice(
    const SharedDictionaryAlphabet& alphabet,
    std::string_view encodedIndices,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  auto* pool = &buffer.getMemoryPool();
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  ScopedVector<uint32_t> slicedIndices{length, pool, options.bufferPool};
  auto indicesEncoding = EncodingFactory{options}.create(
      *pool, encodedIndices, [&scopedBuffer](uint32_t size) -> void* {
        return scopedBuffer.get().reserve(size);
      });
  indicesEncoding->skip(offset);
  indicesEncoding->materialize(length, slicedIndices.data());

  ScopedVector<uint32_t> uniqueIndices{length, pool, options.bufferPool};
  std::copy(slicedIndices.begin(), slicedIndices.end(), uniqueIndices.begin());
  std::sort(uniqueIndices.begin(), uniqueIndices.end());
  uniqueIndices.resize(
      static_cast<uint64_t>(
          std::unique(uniqueIndices.begin(), uniqueIndices.end()) -
          uniqueIndices.begin()));

  ScopedVector<physicalType> values{
      uniqueIndices.size(), pool, options.bufferPool};
  alphabet.template materialize<T>(
      std::span<const uint32_t>{uniqueIndices.data(), uniqueIndices.size()},
      values.data());

  if (uniqueIndices.size() == 1) {
    const uint64_t encodingSize =
        EncodingPrefix::serializedSize(length, options.useVarintRowCount) +
        sizeof(physicalType);
    char* reserved = buffer.reserve(encodingSize);
    char* writePos = reserved;
    EncodingPrefix::serialize(
        EncodingType::Constant,
        TypeTraits<T>::dataType,
        length,
        options.useVarintRowCount,
        writePos);
    encoding::write<physicalType>(values[0], writePos);
    NIMBLE_CHECK_EQ(
        static_cast<uint64_t>(writePos - reserved),
        encodingSize,
        "Encoding size mismatch.");
    return {reserved, encodingSize};
  }

  ScopedVector<uint32_t> localIndices{length, pool, options.bufferPool};
  materializeLocalDictionaryIndices(uniqueIndices, slicedIndices, localIndices);

  // TODO: When the slice references most of the shared alphabet, consider
  // preserving the shared dictionary form by slicing the shared alphabet or
  // carrying the full shared alphabet instead of rebuilding a local dictionary.
  auto policy =
      std::make_unique<detail::PreservedDictionaryEncodingSelectionPolicy<T>>(
          alphabet.encodingType(),
          EncodingPrefix::encodingType(encodedIndices));
  EncodingSelection<physicalType> selection{
      {.encodingType = EncodingType::Dictionary},
      Statistics<physicalType>::create(
          std::span<const physicalType>{values.data(), values.size()}),
      std::move(policy)};
  const auto serializedAlphabet = selection.template encodeNested<physicalType>(
      EncodingIdentifiers::Dictionary::Alphabet,
      std::span<const physicalType>{values.data(), values.size()},
      scopedBuffer.get(),
      options);
  const auto serializedIndices = selection.template encodeNested<uint32_t>(
      EncodingIdentifiers::Dictionary::Indices,
      std::span<const uint32_t>{localIndices.data(), localIndices.size()},
      scopedBuffer.get(),
      options);

  NIMBLE_CHECK_LE(
      serializedAlphabet.size(),
      std::numeric_limits<uint32_t>::max(),
      "Shared dictionary slice alphabet encoding exceeds maximum size.");
  const uint64_t encodingSize =
      EncodingPrefix::serializedSize(length, options.useVarintRowCount) +
      sizeof(uint32_t) + serializedAlphabet.size() + serializedIndices.size();
  char* reserved = buffer.reserve(encodingSize);
  char* writePos = reserved;
  EncodingPrefix::serialize(
      EncodingType::Dictionary,
      TypeTraits<T>::dataType,
      length,
      options.useVarintRowCount,
      writePos);
  encoding::writeUint32(
      static_cast<uint32_t>(serializedAlphabet.size()), writePos);
  encoding::writeBytes(serializedAlphabet, writePos);
  encoding::writeBytes(serializedIndices, writePos);
  NIMBLE_CHECK_EQ(
      static_cast<uint64_t>(writePos - reserved),
      encodingSize,
      "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string SharedDictionaryEncoding<T>::debugString(int offset) const {
  return fmt::format(
      "{}\n{}entries={}\n{}indices child:\n{}",
      Encoding::debugString(offset),
      std::string(offset + 2, ' '),
      dictionarySize(),
      std::string(offset, ' '),
      indicesEncoding_->debugString(offset + 2));
}

} // namespace facebook::nimble

template <>
struct fmt::formatter<facebook::nimble::SharedDictionaryScope>
    : fmt::formatter<std::string_view> {
  auto format(
      facebook::nimble::SharedDictionaryScope scope,
      format_context& ctx) const {
    switch (scope) {
      case facebook::nimble::SharedDictionaryScope::Stripe:
      case facebook::nimble::SharedDictionaryScope::File:
      case facebook::nimble::SharedDictionaryScope::External:
        return fmt::formatter<std::string_view>::format(
            facebook::nimble::SharedDictionaryScopeName::toName(scope), ctx);
    }
    return fmt::formatter<std::string_view>::format(
        fmt::format("Unknown({})", static_cast<int>(scope)), ctx);
  }
};
