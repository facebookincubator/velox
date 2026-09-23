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

#include "velox/dwio/nimble/tools/ExternalDictionaryBuilder.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "folly/container/F14Set.h"
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/tools/ExternalDictionaryGenerated.h"

namespace facebook::nimble {
namespace {

inline constexpr size_t kMaxEncodedAlphabetBytes =
    std::numeric_limits<uint32_t>::max();

std::string_view encodedAlphabetView(
    const flatbuffers::Vector<uint8_t>* encodedAlphabet) {
  NIMBLE_CHECK_FILE_NOT_NULL(
      encodedAlphabet, "External dictionary artifact has no alphabet.");
  NIMBLE_CHECK_FILE_GT(
      encodedAlphabet->size(),
      0,
      "External dictionary artifact has an empty alphabet.");
  return {
      reinterpret_cast<const char*>(encodedAlphabet->data()),
      encodedAlphabet->size()};
}

template <typename T>
Vector<T> makeAlphabet(
    const Vector<T>& source,
    bool sortValues,
    velox::memory::MemoryPool* pool) {
  Vector<T> alphabet{pool};
  alphabet.reserve(source.size());

  if (sortValues) {
    alphabet.resize(source.size());
    std::copy(source.begin(), source.end(), alphabet.begin());
    std::sort(alphabet.begin(), alphabet.end());
    alphabet.resize(
        std::unique(alphabet.begin(), alphabet.end()) - alphabet.begin());
    return alphabet;
  }

  folly::F14FastSet<T> seen;
  for (auto value : source) {
    if (seen.insert(value).second) {
      alphabet.push_back(value);
    }
  }
  return alphabet;
}

template <typename T>
std::unique_ptr<EncodingSelectionPolicy<T>> createAlphabetPolicy(
    const ExternalDictionaryBuilder::Options& options) {
  NIMBLE_USER_CHECK(
      !options.alphabetEncoding.has_value() || options.readFactors.empty(),
      "alphabet_encoding and read_factors cannot both be set.");

  std::vector<std::pair<EncodingType, float>> readFactors;
  if (options.alphabetEncoding.has_value()) {
    readFactors = {{*options.alphabetEncoding, 1.0f}};
  } else if (!options.readFactors.empty()) {
    readFactors = options.readFactors;
  } else {
    readFactors =
        ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
  }

  ManualEncodingSelectionPolicyFactory factory{
      std::move(readFactors), /*compressionOptions=*/std::nullopt};
  return std::unique_ptr<EncodingSelectionPolicy<T>>(
      static_cast<EncodingSelectionPolicy<T>*>(
          factory.createPolicy(TypeTraits<T>::dataType).release()));
}

template <typename T>
ExternalDictionary buildAlphabet(
    const Vector<T>& source,
    const ExternalDictionaryBuilder::Options& options,
    velox::memory::MemoryPool* pool) {
  static_assert(isIntegralType<T>() && !std::is_same_v<T, bool>);

  NIMBLE_CHECK_NOT_NULL(pool);
  auto alphabet = makeAlphabet<T>(source, options.sortValues, pool);
  NIMBLE_USER_CHECK(
      !alphabet.empty(), "External dictionary alphabet input is empty.");

  auto buffer = std::make_unique<Buffer>(*pool);
  const auto encoded = EncodingFactory::encode<T>(
      createAlphabetPolicy<T>(options),
      std::span<const T>{alphabet.data(), alphabet.size()},
      *buffer);
  std::string encodedAlphabet{encoded};
  return {
      .dataType = TypeTraits<T>::dataType,
      .alphabetEncodingType = EncodingPrefix::encodingType(encodedAlphabet),
      .sortValues = options.sortValues,
      .valueCount = alphabet.size(),
      .encodedAlphabet = std::move(encodedAlphabet),
  };
}

} // namespace

ExternalDictionaryBuilder::ExternalDictionaryBuilder(
    velox::memory::MemoryPool* pool)
    : pool_{pool} {
  NIMBLE_CHECK_NOT_NULL(pool_);
}

#define BUILD_EXTERNAL_DICTIONARY_ALPHABET(type)                  \
  ExternalDictionary ExternalDictionaryBuilder::build(            \
      const Vector<type>& source, const Options& options) const { \
    return buildAlphabet<type>(source, options, pool_);           \
  }
BUILD_EXTERNAL_DICTIONARY_ALPHABET(int8_t)
BUILD_EXTERNAL_DICTIONARY_ALPHABET(uint8_t)
BUILD_EXTERNAL_DICTIONARY_ALPHABET(int16_t)
BUILD_EXTERNAL_DICTIONARY_ALPHABET(uint16_t)
BUILD_EXTERNAL_DICTIONARY_ALPHABET(int32_t)
BUILD_EXTERNAL_DICTIONARY_ALPHABET(uint32_t)
BUILD_EXTERNAL_DICTIONARY_ALPHABET(int64_t)
BUILD_EXTERNAL_DICTIONARY_ALPHABET(uint64_t)
#undef BUILD_EXTERNAL_DICTIONARY_ALPHABET

std::string ExternalDictionaryBuilder::serialize(
    const ExternalDictionary& alphabet) const {
  NIMBLE_USER_CHECK(
      !alphabet.encodedAlphabet.empty(),
      "External dictionary alphabet is empty.");
  NIMBLE_USER_CHECK_LE(
      alphabet.encodedAlphabet.size(),
      kMaxEncodedAlphabetBytes,
      "External dictionary alphabet exceeds uint32_t size.");
  NIMBLE_USER_CHECK_EQ(
      EncodingPrefix::dataType(alphabet.encodedAlphabet),
      alphabet.dataType,
      "External dictionary data type does not match the encoded alphabet.");
  NIMBLE_USER_CHECK_EQ(
      EncodingPrefix::encodingType(alphabet.encodedAlphabet),
      alphabet.alphabetEncodingType,
      "External dictionary encoding type does not match the encoded alphabet.");
  NIMBLE_USER_CHECK_EQ(
      EncodingPrefix::readRowCount(
          alphabet.encodedAlphabet, /*useVarint=*/false),
      alphabet.valueCount,
      "External dictionary value count does not match the encoded alphabet.");

  flatbuffers::FlatBufferBuilder builder;
  const auto encodedAlphabet = builder.CreateVector(
      reinterpret_cast<const uint8_t*>(alphabet.encodedAlphabet.data()),
      alphabet.encodedAlphabet.size());
  const auto root = serialization::CreateExternalDictionary(
      builder,
      static_cast<uint8_t>(alphabet.dataType),
      static_cast<uint8_t>(alphabet.alphabetEncodingType),
      alphabet.sortValues,
      alphabet.valueCount,
      encodedAlphabet);
  serialization::FinishExternalDictionaryBuffer(builder, root);
  return {
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

ExternalDictionary ExternalDictionaryBuilder::deserialize(
    std::string_view data) const {
  return deserializeImpl(data);
}

ExternalDictionary ExternalDictionaryBuilder::deserializeImpl(
    std::string_view data) const {
  flatbuffers::Verifier verifier{
      reinterpret_cast<const uint8_t*>(data.data()), data.size()};
  NIMBLE_CHECK_FILE(
      serialization::VerifyExternalDictionaryBuffer(verifier),
      "Invalid external dictionary artifact.");

  const auto* serialized = serialization::GetExternalDictionary(data.data());
  NIMBLE_CHECK_FILE_NOT_NULL(
      serialized, "External dictionary artifact is missing its root.");
  const auto encodedAlphabet =
      encodedAlphabetView(serialized->encoded_alphabet());
  const auto dataType = static_cast<DataType>(serialized->data_type());
  const auto alphabetEncodingType =
      static_cast<EncodingType>(serialized->alphabet_encoding_type());
  NIMBLE_CHECK_FILE_EQ(
      EncodingPrefix::dataType(encodedAlphabet),
      dataType,
      "External dictionary data type does not match the encoded alphabet.");
  NIMBLE_CHECK_FILE_EQ(
      EncodingPrefix::encodingType(encodedAlphabet),
      alphabetEncodingType,
      "External dictionary encoding type does not match the encoded alphabet.");
  NIMBLE_CHECK_FILE_EQ(
      serialized->value_count(),
      EncodingPrefix::readRowCount(encodedAlphabet, /*useVarint=*/false),
      "External dictionary value count does not match the encoded alphabet.");

  return {
      .dataType = dataType,
      .alphabetEncodingType = alphabetEncodingType,
      .sortValues = serialized->sort_values(),
      .valueCount = serialized->value_count(),
      .encodedAlphabet = std::string{encodedAlphabet},
  };
}

ExternalDictionary ExternalDictionaryBuilder::deserializeFromFile(
    std::string_view path) const {
  const auto file =
      velox::filesystems::getFileSystem(path, nullptr)->openFileForRead(path);
  const auto data = file->pread(/*offset=*/0, file->size());
  return deserialize(data);
}

} // namespace facebook::nimble
