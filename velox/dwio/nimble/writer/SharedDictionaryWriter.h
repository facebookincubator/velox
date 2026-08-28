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

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "folly/ScopeGuard.h"
#include "velox/common/Casts.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/DataTypeDispatch.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/SharedDictionaryConfig.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryBuilder.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/tablet/Chunk.h"
#include "velox/dwio/nimble/tablet/SharedDictionaryReader.h"
#include "velox/dwio/nimble/writer/StreamData.h"

namespace facebook::nimble {

/// Type-erased owner API for shared dictionary writers.
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
    uint32_t dictionaryId{};

    /// Uses an externally resolved logical alphabet instead of growing one
    /// while encoding values. External-scope dictionaries always resolve their
    /// alphabet this way; file-scope dictionaries use one when this is set,
    /// then store the resolved encoded alphabet as-is in the file.
    bool useExternalAlphabet{false};

    /// Restricts the encodings considered for the stored dictionary alphabet,
    /// as described by SharedDictionaryAlphabet::encode(). External
    /// dictionaries do not store an alphabet in the file. File dictionaries
    /// with useExternalAlphabet store the resolved encoded alphabet as-is, so
    /// this only affects stripe/file-owned alphabets.
    std::vector<EncodingType> alphabetEncodings;

    /// Creates encoding selection policies for nested index streams and stored
    /// alphabets.
    EncodingSelectionPolicyCreator encodingSelectionPolicyCreator;

    Encoding::Options encodingOptions;

    /// Resolves external logical alphabets keyed by dictionary id and data
    /// type. Required for external dictionaries, and for file dictionaries when
    /// useExternalAlphabet is set.
    std::shared_ptr<const ExternalDictionaryResolver> resolver;
  };

  virtual ~SharedDictionaryWriter() = default;

  /// Returns where the dictionary alphabet is stored or resolved.
  virtual SharedDictionaryScope scope() const = 0;

  /// Returns whether the active scope committed to shared dictionary encoding.
  virtual bool usesDictionary() const = 0;

  /// Returns whether any chunk used the shared dictionary.
  virtual bool hasUsedDictionary() const = 0;

  /// Returns the alphabet bytes when the selected scope stores one.
  virtual std::optional<Chunk> encodeAlphabet(Buffer& buffer) = 0;

  /// Returns the dictionary id within scope().
  virtual uint32_t dictionaryId() const = 0;

  /// Returns the logical value type encoded by this shared dictionary.
  virtual DataType dataType() const = 0;
};

/// Typed shared dictionary writer state used after scalar value encoding.
template <typename T>
class TypedSharedDictionaryWriter final : public SharedDictionaryWriter {
 public:
  using Options = SharedDictionaryWriter::Options;

  static_assert(isSharedDictionaryType<T>());

  TypedSharedDictionaryWriter(
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

  /// Creates an encoding selection policy backed by this writer's dictionary
  /// state for one value-stream chunk.
  std::unique_ptr<::facebook::nimble::EncodingSelectionPolicy<T>>
  createEncodingPolicy(size_t stripeIndex) {
    return std::make_unique<DictionaryEncodingSelectionPolicy>(
        this, stripeIndex);
  }

  /// Returns where the dictionary alphabet is stored or resolved.
  SharedDictionaryScope scope() const final {
    return options_.scope;
  }

  /// Returns the dictionary id within the namespace defined by scope().
  uint32_t dictionaryId() const final {
    return options_.dictionaryId;
  }

  /// Returns the logical value type encoded by this shared dictionary.
  DataType dataType() const final {
    return TypeTraits<T>::dataType;
  }

  /// Returns whether the active scope committed to shared dictionary encoding.
  bool usesDictionary() const final {
    return useDictionary_.value_or(false);
  }

  /// Returns whether any chunk used the shared dictionary.
  bool hasUsedDictionary() const final {
    return hasUsedDictionary_;
  }

  /// Returns alphabet bytes, ready for the caller to store as a stripe
  /// auxiliary stream or a file catalog entry.
  ///
  /// This consumes the current builder state. Stripe-scoped callers must call
  /// this at every stripe boundary before encoding the next stripe. File-scoped
  /// callers must call this once after every stream using the dictionary is
  /// encoded.
  ///
  /// Returns nullopt when there is nothing to store: a stripe that fell back to
  /// non-shared encoding or a scope that has not encoded a value yet. Throws
  /// for external dictionaries, whose alphabets are resolver-owned and never
  /// encoded into the file.
  /// Throws when the scope committed to shared dictionary encoding but holds no
  /// entries, which would produce a stream no reader could resolve, or when the
  /// active stripe/file dictionary was already finalized.
  std::optional<Chunk> encodeAlphabet(Buffer& buffer) final {
    NIMBLE_CHECK_NE(
        options_.scope,
        SharedDictionaryScope::External,
        "External shared dictionary {} cannot encode an alphabet; its "
        "resolver owns the alphabet.",
        options_.dictionaryId);

    checkAlphabetNotFinalized();

    SCOPE_EXIT {
      builder_.reset();
      useDictionary_.reset();
    };

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
    if (options_.useExternalAlphabet) {
      NIMBLE_CHECK_EQ(options_.scope, SharedDictionaryScope::File);
      NIMBLE_CHECK_NOT_NULL(externalAlphabet_);
      return Chunk{
          .rowCount = externalAlphabet_->entryCount(),
          .content = {externalAlphabet_->encodedAlphabet()}};
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
        options_.alphabetEncodings,
        buffer,
        alphabetEncodingOptions());
    return Chunk{.rowCount = rowCount, .content = {encoded}};
  }

 private:
  using physicalType = typename TypeTraits<T>::physicalType;
  using BuilderMapping = typename SharedDictionaryBuilder<T>::Mapping;

  class DictionaryEncodingSelectionPolicy final
      : public ::facebook::nimble::EncodingSelectionPolicy<T> {
   public:
    DictionaryEncodingSelectionPolicy(
        TypedSharedDictionaryWriter* writer,
        size_t stripeIndex)
        : writer_{velox::checkedNotNull(writer)}, stripeIndex_{stripeIndex} {}

    EncodingSelectionResult select(
        std::span<const physicalType> values,
        const Statistics<physicalType>& statistics,
        const Encoding::Options& options) final {
      NIMBLE_CHECK_GT(
          values.size(),
          0,
          "{} shared dictionary {} cannot encode an empty value stream.",
          writer_->options_.scope,
          writer_->options_.dictionaryId);
      return selectValues(values, statistics, options, std::nullopt);
    }

    EncodingSelectionResult selectNullable(
        std::span<const physicalType> values,
        std::span<const bool> notNulls,
        const Statistics<physicalType>& statistics,
        const Encoding::Options& options) final {
      NIMBLE_CHECK_GT(
          notNulls.size(),
          0,
          "{} shared dictionary {} cannot encode an empty value stream.",
          writer_->options_.scope,
          writer_->options_.dictionaryId);
      return selectValues(values, statistics, options, notNulls);
    }

   private:
    EncodingSelectionResult selectValues(
        std::span<const physicalType> values,
        const Statistics<physicalType>& statistics,
        const Encoding::Options& options,
        std::optional<std::span<const bool>> notNulls) {
      if (notNulls.has_value() && values.empty()) {
        return selectNonSharedEncoding(values, statistics, options, notNulls);
      }

      const std::span<const T> logicalValues{
          reinterpret_cast<const T*>(values.data()), values.size()};
      writer_->ensureBuilder(stripeIndex_);

      if (writer_->options_.scope != SharedDictionaryScope::Stripe) {
        return selectDictionary(logicalValues);
      }
      if (writer_->useDictionary_.has_value()) {
        if (!writer_->useDictionary_.value()) {
          return selectNonSharedEncoding(values, statistics, options, notNulls);
        }
        return selectDictionary(logicalValues);
      }

      // Decide against the regular value encoding, not the nullable wrapper.
      const auto nonSharedSelection =
          selectNonSharedEncoding(values, statistics, options);
      if (nonSharedSelection.encodingType != EncodingType::Dictionary) {
        writer_->abandonStripeDictionary();
        if (notNulls.has_value()) {
          return selectNonSharedEncoding(values, statistics, options, notNulls);
        }
        return nonSharedSelection;
      }
      return selectDictionary(logicalValues);
    }

    EncodingSelectionResult selectNonSharedEncoding(
        std::span<const physicalType> values,
        const Statistics<physicalType>& statistics,
        const Encoding::Options& options,
        std::optional<std::span<const bool>> notNulls = std::nullopt) {
      auto policy =
          writer_->createNonSharedEncodingPolicy(TypeTraits<T>::dataType);
      nonSharedEncodingPolicy_ =
          std::unique_ptr<::facebook::nimble::EncodingSelectionPolicy<T>>(
              static_cast<::facebook::nimble::EncodingSelectionPolicy<T>*>(
                  policy.release()));
      writer_->validateNonSharedValuePolicy(*nonSharedEncodingPolicy_);
      auto selection = notNulls.has_value()
          ? nonSharedEncodingPolicy_->selectNullable(
                values, *notNulls, statistics, options)
          : nonSharedEncodingPolicy_->select(values, statistics, options);
      NIMBLE_CHECK_NE(
          selection.encodingType,
          EncodingType::SharedDictionary,
          "Regular encoding selection must not select SharedDictionary.");
      return selection;
    }

    EncodingSelectionResult selectDictionary(std::span<const T> logicalValues) {
      writer_->selectDictionary();
      mapping_ = writer_->builder().lookup(logicalValues);
      NIMBLE_CHECK_EQ(
          mapping_->indices().size(),
          logicalValues.size(),
          "Shared dictionary index count differs from value count.");
      return {
          .encodingType = EncodingType::SharedDictionary,
          .sharedDictionaryInput =
              SharedDictionaryEncodingInput{.indices = mapping_->indices()}};
    }

    std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
        EncodingType parentEncodingType,
        NestedEncodingIdentifier nestedEncodingIdentifier,
        DataType nestedDataType) final {
      if (parentEncodingType == EncodingType::SharedDictionary) {
        NIMBLE_CHECK_EQ(
            nestedEncodingIdentifier,
            EncodingIdentifiers::SharedDictionary::Indices);
        NIMBLE_CHECK_EQ(nestedDataType, DataType::Uint32);
        return writer_->createNonSharedEncodingPolicy(nestedDataType);
      }
      if (parentEncodingType == EncodingType::Nullable) {
        if (nestedEncodingIdentifier == EncodingIdentifiers::Nullable::Data) {
          return createNestedNonSharedEncodingPolicy(
              parentEncodingType, nestedEncodingIdentifier, nestedDataType);
        }
        NIMBLE_CHECK_EQ(
            nestedEncodingIdentifier, EncodingIdentifiers::Nullable::Nulls);
        NIMBLE_CHECK_EQ(nestedDataType, DataType::Bool);
        // Nulls are an independent bitmap stream, not a child of the selected
        // value encoding policy.
        return writer_->createNonSharedEncodingPolicy(nestedDataType);
      }
      return createNestedNonSharedEncodingPolicy(
          parentEncodingType, nestedEncodingIdentifier, nestedDataType);
    }

    std::unique_ptr<EncodingSelectionPolicyBase>
    createNestedNonSharedEncodingPolicy(
        EncodingType parentEncodingType,
        NestedEncodingIdentifier nestedEncodingIdentifier,
        DataType nestedDataType) {
      NIMBLE_CHECK_NOT_NULL(
          nonSharedEncodingPolicy_,
          "Non-shared nested encoding policy requires a selected value "
          "encoding policy.");
      NIMBLE_RETURN_BY_DATA_TYPE(
          // The macro's default case handles Undefined.
          // @lint-ignore CLANGTIDY clang-diagnostic-switch-enum
          nestedDataType,
          NestedT,
          nonSharedEncodingPolicy_->template create<NestedT>(
              parentEncodingType, nestedEncodingIdentifier));
    }

    TypedSharedDictionaryWriter* const writer_;
    const size_t stripeIndex_;
    std::unique_ptr<EncodingSelectionPolicy<T>> nonSharedEncodingPolicy_;
    std::optional<BuilderMapping> mapping_;
  };

  // Updates the active stripe and creates the builder when this chunk can use
  // one. Non-shared and alphabet-emitted stripes intentionally keep no builder.
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
          !useDictionary_.has_value(),
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
          useDictionary_.value_or(false),
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
      NIMBLE_CHECK_NE(options_.scope, SharedDictionaryScope::Stripe);
      return;
    }
    builder_ = createBuilder();
  }

  // Once an owned alphabet is emitted, more values would produce streams that
  // reference dictionary entries missing from the encoded alphabet.
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
    if (!stripeIndex_.has_value() || lastFinalizedStripe_ != stripeIndex_) {
      return;
    }
    NIMBLE_FAIL(
        "{} shared dictionary {} already finalized its alphabet{}.",
        options_.scope,
        options_.dictionaryId,
        options_.scope == SharedDictionaryScope::Stripe
            ? " for stripe " + std::to_string(*stripeIndex_)
            : "");
  }

  // Records that the active stripe/file dictionary can no longer accept value
  // chunks. Non-shared stripes record this too even though they emit no
  // alphabet.
  void recordAlphabetFinalized() {
    if (!stripeIndex_.has_value()) {
      return;
    }
    lastFinalizedStripe_ = stripeIndex_;
  }

  // Marks the active scope as committed to shared dictionary encoding.
  void selectDictionary() {
    useDictionary_ = true;
    hasUsedDictionary_ = true;
  }

  // Drops lookup-inserted stripe entries when the first chunk chooses
  // non-shared encoding; the rest of the stripe stays non-shared.
  void abandonStripeDictionary() {
    NIMBLE_CHECK_EQ(options_.scope, SharedDictionaryScope::Stripe);
    NIMBLE_CHECK_NOT_NULL(builder_);
    builder_.reset();
    useDictionary_ = false;
  }

  Encoding::Options alphabetEncodingOptions() const {
    auto options = options_.encodingOptions;
    // Stored alphabet streams do not carry the row-count encoding mode.
    // SharedDictionaryAlphabet::create() decodes them with default options.
    options.useVarintRowCount = false;
    return options;
  }

  std::unique_ptr<EncodingSelectionPolicyBase> createNonSharedEncodingPolicy(
      DataType dataType) const {
    auto policy = options_.encodingSelectionPolicyCreator(dataType);
    NIMBLE_CHECK_NOT_NULL(policy);
    return policy;
  }

  // Stripe scope can select shared dictionary only when regular value
  // selection can choose Dictionary. Manual policies expose that candidate set,
  // so fail early for manual configs that remove this path.
  template <typename PolicyT>
  void validateNonSharedValuePolicy(
      const ::facebook::nimble::EncodingSelectionPolicy<PolicyT>& policy)
      const {
    if (options_.scope != SharedDictionaryScope::Stripe) {
      return;
    }
    const auto* manualPolicy =
        dynamic_cast<const ManualEncodingSelectionPolicy<PolicyT>*>(&policy);
    if (manualPolicy == nullptr) {
      return;
    }
    for (const auto& readFactor :
         manualPolicy->candidateEncodingReadFactors()) {
      if (readFactor.first == EncodingType::Dictionary) {
        return;
      }
    }
    NIMBLE_USER_FAIL(
        "Stripe shared dictionary selection requires regular Dictionary in "
        "the non-shared value encoding candidates.");
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
  // the file-scope alphabet is resolved externally.
  std::unique_ptr<SharedDictionaryBuilder<T>> createBuilder() {
    if (options_.scope == SharedDictionaryScope::External) {
      return createExternalBuilder(SharedDictionaryScope::External);
    }
    if (options_.useExternalAlphabet) {
      NIMBLE_USER_CHECK_EQ(
          options_.scope,
          SharedDictionaryScope::File,
          "External shared dictionary alphabets require file scope.");
      return createExternalBuilder(SharedDictionaryScope::File);
    }
    return std::make_unique<StreamingSharedDictionaryBuilder<T>>(
        options_.scope, options_.dictionaryId, pool_);
  }

  // Resolves an external logical alphabet. The resolver key is
  // scope-independent; scope only controls how the writer stores or references
  // the resolved alphabet.
  std::unique_ptr<SharedDictionaryBuilder<T>> createExternalBuilder(
      SharedDictionaryScope scope) {
    NIMBLE_USER_CHECK_NOT_NULL(
        options_.resolver,
        "{} shared dictionary {} requires a dictionary resolver.",
        scope,
        options_.dictionaryId);
    externalAlphabet_ = options_.resolver->resolve(
        options_.dictionaryId, TypeTraits<T>::dataType);
    NIMBLE_USER_CHECK_NOT_NULL(
        externalAlphabet_,
        "{} shared dictionary {} was not found.",
        scope,
        options_.dictionaryId);
    NIMBLE_USER_CHECK_EQ(
        externalAlphabet_->dataType(),
        TypeTraits<T>::dataType,
        "{} shared dictionary {} has the wrong type.",
        scope,
        options_.dictionaryId);
    NIMBLE_CHECK(
        materializedAlphabet_.empty(),
        "{} shared dictionary {} already materialized its external alphabet.",
        options_.scope,
        options_.dictionaryId);
    externalAlphabet_->template materializeAll<T>(materializedAlphabet_);
    NIMBLE_USER_CHECK(
        !materializedAlphabet_.empty(),
        "{} shared dictionary {} has an empty external alphabet.",
        scope,
        options_.dictionaryId);
    return std::make_unique<ExternalSharedDictionaryBuilder<T>>(
        scope,
        std::span<const T>{
            materializedAlphabet_.data(), materializedAlphabet_.size()},
        options_.dictionaryId,
        pool_);
  }

  // Allocates writer-owned vectors and materialized alphabet values.
  velox::memory::MemoryPool* const pool_;
  // Immutable writer configuration captured at construction.
  const Options options_;
  // Keeps resolver-returned alphabet storage alive for external builders whose
  // values may reference its backing bytes.
  std::shared_ptr<const SharedDictionaryAlphabet> externalAlphabet_;
  // Owns materialized external alphabets for resolved builders. The builder
  // keeps a non-owning view into this vector to avoid another copy.
  Vector<T> materializedAlphabet_;
  // Maps input values to dictionary indices for the active dictionary scope.
  std::unique_ptr<SharedDictionaryBuilder<T>> builder_;
  // Stripe currently being encoded; unset until the first encode() call.
  std::optional<size_t> stripeIndex_;
  // Active value stripe whose alphabet was finalized. For non-shared stripes,
  // finalization emits no alphabet but still closes the stripe.
  std::optional<size_t> lastFinalizedStripe_;
  // Encoding the active scope committed to, unset until its first chunk picks
  // one. True keeps the shared dictionary; false stays non-shared for the rest
  // of the stripe and emits no stripe alphabet. Stripe-scope dictionaries clear
  // it at each stripe boundary, while file and external dictionaries decide
  // once.
  std::optional<bool> useDictionary_;
  // True once any chunk uses the dictionary. For stripe scope, this survives
  // per-stripe resets so final file metadata emits a binding when at least one
  // stripe used the dictionary and omits it when every stripe stayed
  // non-shared.
  bool hasUsedDictionary_{false};
};

} // namespace facebook::nimble
