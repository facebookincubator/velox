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

#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include "folly/ScopeGuard.h"
#include "velox/common/Casts.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryBuilder.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryTypes.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"
#include "velox/dwio/nimble/tablet/Chunk.h"
#include "velox/dwio/nimble/velox/StreamData.h"

namespace facebook::nimble {

/// Selects SharedDictionary at the root using writer-provided indices, while
/// delegating nested index streams to the writer's regular selection policy.
template <typename T>
class SharedDictionaryEncodingSelectionPolicy final
    : public EncodingSelectionPolicy<T> {
 public:
  using physicalType = typename TypeTraits<T>::physicalType;

  using Mapping = typename SharedDictionaryBuilder<T>::Mapping;

  SharedDictionaryEncodingSelectionPolicy(
      SharedDictionaryScope scope,
      uint32_t dictionaryId,
      Mapping mapping,
      EncodingSelectionPolicyCreator nestedPolicyCreator)
      : mapping_{std::move(mapping)},
        sharedDictionaryInput_{
            .scope = scope,
            .dictionaryId = dictionaryId,
            .indices = mapping_.indices()},
        nestedPolicyCreator_{std::move(nestedPolicyCreator)} {
    NIMBLE_CHECK_NOT_NULL(
        nestedPolicyCreator_,
        "Shared dictionary encoding selection requires nested policy creator.");
  }

  EncodingSelectionResult select(
      std::span<const physicalType> /*values*/,
      const Statistics<physicalType>& /*statistics*/,
      const Encoding::Options& /*options*/) final {
    return {
        .encodingType = EncodingType::SharedDictionary,
        .sharedDictionaryInput = sharedDictionaryInput_};
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /*values*/,
      std::span<const bool> /*nulls*/,
      const Statistics<physicalType>& /*statistics*/,
      const Encoding::Options& /*options*/) final {
    // Nullable would delegate the values to the nested policy and silently drop
    // the shared dictionary the caller already committed to, so callers wrap
    // this policy in a nullable encoding instead of routing nulls through it.
    NIMBLE_UNREACHABLE(
        "Shared dictionary encoding selection does not support nullable "
        "values.");
  }

 private:
  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType /*parentEncodingType*/,
      NestedEncodingIdentifier /*nestedEncodingIdentifier*/,
      DataType nestedDataType) final {
    auto policy = nestedPolicyCreator_(nestedDataType);
    NIMBLE_CHECK_NOT_NULL(policy);
    return policy;
  }

  // Owns the indices sharedDictionaryInput_ points at, so the policy stays
  // valid for as long as the encoding that reads them.
  const Mapping mapping_;
  const SharedDictionaryEncodingInput sharedDictionaryInput_;
  const EncodingSelectionPolicyCreator nestedPolicyCreator_;
};

template <typename T>
class SharedDictionaryWriter {
 public:
  /// Shared dictionary writer configuration and services.
  struct Options {
    SharedDictionaryScope scope{SharedDictionaryScope::Stripe};

    /// Identifies the dictionary within its scope:
    /// - Stripe: auxiliary stream id in the current stripe that stores the
    ///   alphabet.
    /// - File: entry id in the file shared dictionary catalog.
    /// - External: id passed through to the external resolver.
    /// The sentinel default catches accidental use before the writer assigns an
    /// id in the selected scope.
    uint32_t dictionaryId{kInvalidSharedDictionaryId};

    /// Uses a prebuilt logical alphabet from the configured resolver instead
    /// of growing it while encoding values. External-scope dictionaries always
    /// use a prebuilt alphabet; file-scope dictionaries use one when this is
    /// set, then serialize it through Nimble encoding for file storage.
    bool usesPrebuiltAlphabet{false};

    /// Restricts the encodings considered for the stored dictionary alphabet,
    /// as described by SharedDictionaryAlphabet::encode(). External
    /// dictionaries do not store an alphabet in the file, so this only affects
    /// stripe/file-owned alphabets.
    std::vector<EncodingType> alphabetEncodingCandidates;

    /// Creates encoding selection policies for nested index streams and stored
    /// alphabets.
    EncodingSelectionPolicyCreator encodingSelectionPolicyCreator;

    Encoding::Options encodingOptions;

    /// Resolves prebuilt logical alphabets keyed by scope, dictionary id, and
    /// data type. Required for external dictionaries, and for file
    /// dictionaries when usesPrebuiltAlphabet is set.
    std::shared_ptr<const SharedDictionaryResolver> resolver;
  };

  SharedDictionaryWriter(
      velox::memory::MemoryPool* pool,
      const Options& options)
      : pool_{velox::checkedNotNull(pool)},
        options_{options},
        materializedAlphabet_{pool_} {
    NIMBLE_CHECK_NOT_NULL(
        options_.encodingSelectionPolicyCreator,
        "Shared dictionary writer requires an encoding selection policy "
        "creator.");
  }

  /// Encodes one non-null value-stream chunk through SharedDictionaryEncoding,
  /// as opposed to encodeAlphabet(), which serializes the dictionary itself.
  ///
  /// stripeIndex selects the stripe-local dictionary state to use and resets
  /// stripe-scoped dictionaries at stripe boundaries. buffer owns any bytes
  /// appended by shared dictionary encoding. streamData must provide the raw
  /// non-null scalar values and stream metadata used for error messages.
  /// Encoding options, including the optional buffer pool, are captured in
  /// Options.
  std::string_view encodeValues(
      size_t stripeIndex,
      Buffer& buffer,
      const StreamData& streamData) {
    // Nullable encodings should wrap this non-null value stream in the caller.
    // Checked before ensureBuilder() so a rejected stream leaves the stripe
    // dictionary state untouched.
    NIMBLE_CHECK(
        !streamData.hasNullValues(),
        "Shared dictionary writer expects a non-null value stream.");
    NIMBLE_CHECK_GT(
        streamData.rowCount(),
        0,
        "{} shared dictionary {} cannot encode an empty value stream.",
        options_.scope,
        options_.dictionaryId);
    ensureBuilder(stripeIndex);

    const auto values = physicalValues(streamData);
    return EncodingFactory::encode<T>(
        createEncodingPolicy(values, options_.encodingOptions),
        logicalValues(values),
        buffer,
        options_.encodingOptions);
  }

  /// Returns where the dictionary alphabet is stored or resolved.
  SharedDictionaryScope scope() const {
    return options_.scope;
  }

  /// Returns the dictionary id within the namespace defined by scope().
  uint32_t dictionaryId() const {
    return options_.dictionaryId;
  }

  /// Returns the logical value type encoded by this shared dictionary.
  DataType dataType() const {
    return TypeTraits<T>::dataType;
  }

  /// Serializes the alphabet this writer owns, ready for the caller to store
  /// as a stripe auxiliary stream or a file catalog entry.
  ///
  /// This consumes the current builder state. Stripe-scoped callers must call
  /// this at every stripe boundary before encoding the next stripe. File-scoped
  /// callers must call this once after every stream using the dictionary is
  /// encoded.
  ///
  /// Returns nullopt when there is nothing to store: a stripe that fell back to
  /// direct encoding, a scope that has not encoded a value yet, or an external
  /// dictionary, whose alphabet the resolver owns and the file never stores.
  /// Throws when the scope committed to shared dictionary encoding but holds no
  /// entries, which would produce a stream no reader could resolve, or when the
  /// active stripe/file dictionary was already finalized.
  std::optional<Chunk> encodeAlphabet(Buffer& buffer) {
    checkAlphabetNotFinalized();

    SCOPE_EXIT {
      builder_.reset();
      useSharedDictionary_.reset();
    };

    // External alphabets are resolver owned and never stored in the file.
    if (options_.scope == SharedDictionaryScope::External) {
      return std::nullopt;
    }
    SCOPE_EXIT {
      recordAlphabetFinalized();
    };

    if (!usesDictionary()) {
      NIMBLE_CHECK_NULL(
          builder_,
          "{} shared dictionary {} has an active builder without an encoding "
          "decision.",
          options_.scope,
          options_.dictionaryId);
      if (options_.scope == SharedDictionaryScope::Stripe ||
          !stripeIndex_.has_value()) {
        return std::nullopt;
      }
      NIMBLE_FAIL(
          "{} shared dictionary {} lost its encoding decision.",
          options_.scope,
          options_.dictionaryId);
    }
    NIMBLE_CHECK_NOT_NULL(builder_);
    // Committing to shared dictionary encoding with nothing to store would
    // leave the encoded stream pointing at a dictionary the reader can never
    // resolve, so fail here rather than write an unreadable file.
    const auto alphabet = builder_->alphabet();
    NIMBLE_CHECK(
        !alphabet.empty(),
        "{} shared dictionary {} selected shared dictionary encoding without "
        "any entries.",
        options_.scope,
        options_.dictionaryId);
    const auto rowCount = static_cast<uint32_t>(alphabet.size());
    const auto encoded = SharedDictionaryAlphabet::encode<T>(
        alphabet,
        options_.alphabetEncodingCandidates,
        buffer,
        options_.encodingOptions);
    return Chunk{.rowCount = rowCount, .content = {encoded}};
  }

 private:
  using physicalType = typename TypeTraits<T>::physicalType;
  using BuilderMapping = typename SharedDictionaryBuilder<T>::Mapping;

  // Updates the active stripe and creates the builder when this chunk can use
  // one. Direct-abandoned and alphabet-emitted stripes intentionally keep no
  // builder.
  void ensureBuilder(size_t stripeIndex) {
    checkValuesNotFinalized(stripeIndex);
    if (stripeIndex_ == stripeIndex) {
      return;
    }

    if (options_.scope == SharedDictionaryScope::Stripe ||
        !stripeIndex_.has_value()) {
      NIMBLE_CHECK_NULL(
          builder_,
          "{} shared dictionary {} reached a stripe boundary before encoding "
          "its alphabet.",
          options_.scope,
          options_.dictionaryId);
      NIMBLE_CHECK(
          !useSharedDictionary_.has_value(),
          "{} shared dictionary {} reached a stripe boundary with an active "
          "encoding decision.",
          options_.scope,
          options_.dictionaryId);
    } else {
      NIMBLE_CHECK_NOT_NULL(
          builder_,
          "{} shared dictionary {} reached a stripe boundary without an active "
          "builder.",
          options_.scope,
          options_.dictionaryId);
      NIMBLE_CHECK(
          useSharedDictionary_.value_or(false),
          "{} shared dictionary {} reached a stripe boundary without an active "
          "encoding decision.",
          options_.scope,
          options_.dictionaryId);
    }
    // Recorded for every scope, including file and external, because
    // encodeAlphabet() reads it to tell "nothing encoded yet" apart from a
    // scope that lost its encoding decision.
    NIMBLE_CHECK(
        !stripeIndex_.has_value() || stripeIndex > *stripeIndex_,
        "{} shared dictionary {} cannot move from stripe {} back to stripe {}.",
        options_.scope,
        options_.dictionaryId,
        *stripeIndex_,
        stripeIndex);
    stripeIndex_ = stripeIndex;

    if (builder_ != nullptr) {
      return;
    }
    builder_ = createBuilder();
  }

  // Returns whether the active scope committed to the shared dictionary.
  bool usesDictionary() const {
    return useSharedDictionary_.value_or(false);
  }

  void checkValuesNotFinalized(size_t stripeIndex) const {
    if (!lastFinalizedStripe_.has_value()) {
      return;
    }
    if (options_.scope == SharedDictionaryScope::File) {
      NIMBLE_FAIL(
          "{} shared dictionary {} cannot encode values after its alphabet "
          "was finalized.",
          options_.scope,
          options_.dictionaryId);
    }
    if (options_.scope == SharedDictionaryScope::Stripe &&
        lastFinalizedStripe_ == stripeIndex) {
      NIMBLE_FAIL(
          "{} shared dictionary {} cannot encode values after its alphabet "
          "was finalized for stripe {}.",
          options_.scope,
          options_.dictionaryId,
          stripeIndex);
    }
  }

  // Rejects repeated alphabet finalization for the same active stripe. File
  // scope records the value stripe that finalized the one file-level alphabet.
  void checkAlphabetNotFinalized() const {
    if (options_.scope == SharedDictionaryScope::External ||
        !stripeIndex_.has_value() || lastFinalizedStripe_ != stripeIndex_) {
      return;
    }
    if (options_.scope == SharedDictionaryScope::File) {
      NIMBLE_FAIL(
          "{} shared dictionary {} already finalized its alphabet.",
          options_.scope,
          options_.dictionaryId);
    }
    NIMBLE_FAIL(
        "{} shared dictionary {} already finalized its alphabet for stripe {}.",
        options_.scope,
        options_.dictionaryId,
        *stripeIndex_);
  }

  void recordAlphabetFinalized() {
    if (!stripeIndex_.has_value()) {
      return;
    }
    lastFinalizedStripe_ = stripeIndex_;
  }

  // Picks between shared dictionary and direct encoding for one chunk. The
  // first stripe-scope chunk decides for the whole stripe; file and external
  // dictionaries always use the shared dictionary.
  std::unique_ptr<EncodingSelectionPolicy<T>> createEncodingPolicy(
      std::span<const physicalType> values,
      const Encoding::Options& options) {
    const auto lookupValues = logicalValues(values);
    // File and external dictionaries never fall back to direct encoding.
    if (options_.scope != SharedDictionaryScope::Stripe) {
      return selectSharedDictionaryPolicy(builder().lookup(lookupValues));
    }
    // A stripe decides on its first chunk and the rest of the stripe follows.
    if (useSharedDictionary_.has_value()) {
      if (!useSharedDictionary_.value()) {
        return createTypedPolicy(TypeTraits<T>::dataType);
      }
      return selectSharedDictionaryPolicy(builder().lookup(lookupValues));
    }

    auto& dictionaryBuilder = builder();
    auto mapping = dictionaryBuilder.lookup(lookupValues);
    auto directPolicy = createTypedPolicy(TypeTraits<T>::dataType);
    const auto statistics = Statistics<physicalType>::create(values);
    const auto directSelection =
        directPolicy->select(values, statistics, options);
    const auto directEstimate = detail::EncodingSizeEstimation<T>::estimateSize(
        directSelection.encodingType, values, statistics, options);
    const auto sharedEstimate =
        estimateSharedDictionarySize(dictionaryBuilder, mapping, options);

    // Ties go to direct encoding, which needs no alphabet stream. A direct
    // encoding nobody can size keeps the shared dictionary, whose cost is then
    // the only one to compare.
    if (directEstimate.has_value() &&
        directEstimate.value() <= sharedEstimate) {
      // Releasing the builder drops the entries this lookup inserted.
      abandonStripeDictionary();
      return createTypedPolicy(TypeTraits<T>::dataType);
    }
    return selectSharedDictionaryPolicy(std::move(mapping));
  }

  static std::span<const physicalType> physicalValues(
      const StreamData& streamData) {
    static_assert(sizeof(T) == sizeof(physicalType));
    NIMBLE_CHECK_EQ(
        streamData.data().size() % sizeof(T),
        0,
        "Shared dictionary writer value stream has incomplete values.");
    return {
        reinterpret_cast<const physicalType*>(streamData.data().data()),
        streamData.data().size() / sizeof(T)};
  }

  static std::span<const T> logicalValues(
      std::span<const physicalType> values) {
    static_assert(sizeof(T) == sizeof(physicalType));
    return {
        reinterpret_cast<const T*>(values.data()),
        values.size(),
    };
  }

  static std::span<const physicalType> physicalValues(
      std::span<const T> values) {
    static_assert(sizeof(T) == sizeof(physicalType));
    return {
        reinterpret_cast<const physicalType*>(values.data()),
        values.size(),
    };
  }

  // Marks the active scope as committed to shared dictionary encoding.
  void selectDictionary() {
    useSharedDictionary_ = true;
  }

  // Drops lookup-inserted stripe entries when the first chunk chooses direct
  // encoding; the rest of the stripe stays direct.
  void abandonStripeDictionary() {
    NIMBLE_CHECK_EQ(options_.scope, SharedDictionaryScope::Stripe);
    NIMBLE_CHECK_NOT_NULL(builder_);
    builder_.reset();
    useSharedDictionary_ = false;
  }

  uint64_t estimateSharedDictionarySize(
      const SharedDictionaryBuilder<T>& builder,
      const BuilderMapping& mapping,
      const Encoding::Options& options) const {
    // Indices address the alphabet, so their bit width follows the entry
    // count. This mirrors how DictionaryEncoding sizes its own index stream,
    // and avoids building Statistics over every index just to size them.
    const auto indices = mapping.indices();
    const auto entryCount = builder.alphabet().size();
    NIMBLE_CHECK_GT(
        entryCount,
        0,
        "{} shared dictionary {} has no entries to size.",
        options_.scope,
        options_.dictionaryId);
    const auto indexEstimate = FixedBitWidthEncoding<uint32_t>::estimateSize(
        indices.size(),
        /*minValue=*/0,
        entryCount - 1,
        options);
    return EncodingPrefix::serializedSize(
               static_cast<uint32_t>(indices.size()),
               options.useVarintRowCount) +
        sizeof(uint8_t) + varint::varintSize(options_.dictionaryId) +
        indexEstimate + estimateNewEntrySize(builder, mapping, options);
  }

  uint64_t estimateNewEntrySize(
      const SharedDictionaryBuilder<T>& builder,
      const BuilderMapping& mapping,
      const Encoding::Options& options) const {
    const auto newEntryCount = mapping.newEntryCount();
    if (newEntryCount == 0) {
      return 0;
    }
    // Streaming builders append new entries, so they are the alphabet's tail.
    return SharedDictionaryAlphabet::estimateSize<T>(
        builder.alphabet().last(newEntryCount),
        options_.alphabetEncodingCandidates,
        options);
  }

  // Creates the root policy that dispatches to SharedDictionaryEncoding.
  std::unique_ptr<EncodingSelectionPolicy<T>> selectSharedDictionaryPolicy(
      BuilderMapping mapping) {
    selectDictionary();
    return std::make_unique<SharedDictionaryEncodingSelectionPolicy<T>>(
        options_.scope,
        options_.dictionaryId,
        std::move(mapping),
        options_.encodingSelectionPolicyCreator);
  }

  // Creates a root policy for a nested stream and checks the writer supplied
  // policy creator did not return null.
  std::unique_ptr<EncodingSelectionPolicyBase> createPolicy(
      DataType dataType) const {
    auto policy = options_.encodingSelectionPolicyCreator(dataType);
    NIMBLE_CHECK_NOT_NULL(policy);
    return policy;
  }

  // Creates a typed root policy for this writer's value type.
  std::unique_ptr<EncodingSelectionPolicy<T>> createTypedPolicy(
      DataType dataType) const {
    return std::unique_ptr<EncodingSelectionPolicy<T>>(
        static_cast<EncodingSelectionPolicy<T>*>(
            createPolicy(dataType).release()));
  }

  SharedDictionaryBuilder<T>& builder() {
    NIMBLE_CHECK_NOT_NULL(
        builder_,
        "{} shared dictionary {} has no active builder.",
        options_.scope,
        options_.dictionaryId);
    return *builder_;
  }

  // Selects the builder implementation based on dictionary scope and whether
  // the file-scope alphabet is prebuilt.
  std::unique_ptr<SharedDictionaryBuilder<T>> createBuilder() {
    if (options_.scope == SharedDictionaryScope::External) {
      return createResolvedDictionaryBuilder(
          options_.resolver.get(), SharedDictionaryScope::External);
    }
    if (options_.usesPrebuiltAlphabet) {
      NIMBLE_USER_CHECK_EQ(
          options_.scope,
          SharedDictionaryScope::File,
          "Prebuilt shared dictionary alphabets require file scope.");
      return createResolvedDictionaryBuilder(
          options_.resolver.get(), SharedDictionaryScope::File);
    }
    return std::make_unique<StreamingSharedDictionaryBuilder<T>>(
        options_.scope, options_.dictionaryId, pool_);
  }

  // Resolves a prebuilt logical alphabet and dispatches to the file or external
  // builder setup.
  std::unique_ptr<SharedDictionaryBuilder<T>> createResolvedDictionaryBuilder(
      const SharedDictionaryResolver* resolver,
      SharedDictionaryScope scope) {
    NIMBLE_USER_CHECK_NOT_NULL(
        resolver,
        "{} shared dictionary {} requires a writer resolver.",
        scope,
        options_.dictionaryId);
    prebuiltAlphabet_ = resolver->resolve(
        scope, options_.dictionaryId, TypeTraits<T>::dataType);
    NIMBLE_USER_CHECK_NOT_NULL(
        prebuiltAlphabet_,
        "{} shared dictionary {} was not found.",
        scope,
        options_.dictionaryId);
    NIMBLE_USER_CHECK_EQ(
        prebuiltAlphabet_->dataType(),
        TypeTraits<T>::dataType,
        "{} shared dictionary {} has the wrong type.",
        scope,
        options_.dictionaryId);
    if (scope == SharedDictionaryScope::File) {
      return createFilePrebuiltBuilder(*prebuiltAlphabet_);
    }
    return createExternalPrebuiltBuilder(*prebuiltAlphabet_);
  }

  // Creates a file-scope fixed builder and keeps the materialized logical
  // alphabet alive for later Nimble file-catalog serialization.
  std::unique_ptr<SharedDictionaryBuilder<T>> createFilePrebuiltBuilder(
      const SharedDictionaryAlphabet& alphabet) {
    materializeAlphabet(alphabet, materializedAlphabet_);
    const std::span<const T> materializedAlphabet{
        materializedAlphabet_.data(), materializedAlphabet_.size()};
    auto dictionaryIndex = buildDictionaryIndex(materializedAlphabet);
    return std::make_unique<FixedSharedDictionaryBuilder<T>>(
        materializedAlphabet,
        std::move(dictionaryIndex),
        SharedDictionaryScope::File,
        options_.dictionaryId,
        pool_);
  }

  // Creates an external builder from a value-to-index map without keeping
  // serialized alphabet entries in the writer.
  std::unique_ptr<SharedDictionaryBuilder<T>> createExternalPrebuiltBuilder(
      const SharedDictionaryAlphabet& alphabet) const {
    Vector<T> externalValues{pool_};
    materializeAlphabet(alphabet, externalValues);
    auto dictionaryIndex = buildDictionaryIndex(
        std::span<const T>{externalValues.data(), externalValues.size()});
    return std::make_unique<ExternalSharedDictionaryBuilder<T>>(
        std::move(dictionaryIndex), options_.dictionaryId, pool_);
  }

  // Copies alphabet entries into writer logical value storage.
  void materializeAlphabet(
      const SharedDictionaryAlphabet& alphabet,
      Vector<T>& values) const {
    // SharedDictionaryAlphabet materializes encoding physical values; builders
    // store logical writer values.
    values.resize(alphabet.entryCount());
    for (uint32_t i = 0; i < alphabet.entryCount(); ++i) {
      values[i] = EncodingPhysicalType<T>::asEncodingLogicalType(
          alphabet.template physicalValueAt<T>(i));
    }
  }

  // Allocates writer-owned vectors and materialized alphabet values.
  velox::memory::MemoryPool* const pool_;
  // Immutable writer configuration captured at construction.
  const Options options_;
  // Owns materialized prebuilt alphabets for file-scope builders. The builder
  // keeps a non-owning view into this vector to avoid another copy.
  Vector<T> materializedAlphabet_;
  // Keeps resolver-returned alphabet storage alive for prebuilt builders whose
  // values may reference its backing bytes.
  std::shared_ptr<const SharedDictionaryAlphabet> prebuiltAlphabet_;
  // Maps input values to dictionary indices for the active dictionary scope.
  std::unique_ptr<SharedDictionaryBuilder<T>> builder_;
  // Stripe currently being encoded; unset until the first encode() call.
  std::optional<size_t> stripeIndex_;
  // Active value stripe whose encodeAlphabet() call already completed. For
  // file scope, this marks the single file-level alphabet as finalized.
  std::optional<size_t> lastFinalizedStripe_;
  // Encoding the active scope committed to, unset until its first chunk picks
  // one. True keeps the shared dictionary; false stays direct for the rest of
  // the stripe and emits no stripe alphabet. Stripe-scope dictionaries clear it
  // at each stripe boundary, while file and external dictionaries decide once.
  std::optional<bool> useSharedDictionary_;
};

} // namespace facebook::nimble
