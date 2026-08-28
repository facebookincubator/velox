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

#include "velox/dwio/nimble/velox/tests/SharedDictionaryTestUtils.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <string_view>

#include "velox/common/file/File.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::nimble::test {
namespace {

std::unique_ptr<EncodingSelectionPolicyBase> sharedDictionarySelectionPolicy(
    DataType dataType,
    SharedDictionarySelectionPolicyOptions selectionOptions) {
  std::vector<std::pair<EncodingType, float>> readFactors;
  if (isSharedDictionaryType(dataType)) {
    if (selectionOptions.forceDictionaryForSharedTypes) {
      readFactors = {{EncodingType::Dictionary, 1.0}};
    } else {
      readFactors = {
          {EncodingType::Trivial, 1.0}, {EncodingType::Dictionary, 1.0}};
    }
  } else {
    readFactors = {{EncodingType::Trivial, 1.0}};
  }
  return ManualEncodingSelectionPolicyFactory{
      readFactors, /*compressionOptions=*/std::nullopt}
      .createPolicy(dataType);
}

velox::MapVectorPtr makeSharedDictionaryMap(
    velox::memory::MemoryPool* pool,
    velox::vector_size_t rowCount,
    const SharedDictionarySource& source,
    std::span<const int32_t> valueUniverse,
    bool nullableData,
    uint32_t salt) {
  velox::test::VectorMaker vectorMaker{pool};
  std::vector<velox::vector_size_t> offsets;
  std::vector<velox::vector_size_t> sizes;
  std::vector<int64_t> keys;
  std::vector<std::optional<int32_t>> values;
  std::vector<velox::vector_size_t> nullRows;
  offsets.reserve(rowCount);
  sizes.reserve(rowCount);

  for (velox::vector_size_t row{0}; row < rowCount; ++row) {
    offsets.push_back(static_cast<velox::vector_size_t>(keys.size()));
    if (nullableData && (row + salt) % 29 == 0) {
      nullRows.push_back(row);
      sizes.push_back(0);
      continue;
    }

    if ((row + salt) % 5 == 0) {
      sizes.push_back(0);
      continue;
    }

    keys.push_back(source.dictionaryKey);
    if (nullableData && (row + salt) % 7 == 0) {
      values.emplace_back(std::nullopt);
    } else {
      const auto valueIndex =
          (row * (salt + 3) + source.dictionaryKey) % valueUniverse.size();
      values.emplace_back(valueUniverse[valueIndex]);
    }
    sizes.push_back(1);
  }

  return vectorMaker.mapVector(
      offsets,
      sizes,
      vectorMaker.flatVector<int64_t>(keys),
      vectorMaker.flatVectorNullable<int32_t>(values),
      nullRows);
}

} // namespace

std::vector<int32_t> sharedDictionaryValueUniverse() {
  std::vector<int32_t> values{
      std::numeric_limits<int32_t>::min(),
      -1,
      0,
      std::numeric_limits<int32_t>::max()};
  values.reserve(256);
  for (int32_t i{0}; i < 252; ++i) {
    values.push_back(i * 13 - 57);
  }
  return values;
}

SharedDictionaryTestResolver::SharedDictionaryTestResolver(
    const std::vector<SharedDictionaryTestDictionary>& dictionaries,
    velox::memory::MemoryPool* pool) {
  alphabets_.reserve(dictionaries.size());
  for (const auto& [dictionaryId, values] : dictionaries) {
    Buffer buffer{*pool};
    const auto encoded = SharedDictionaryAlphabet::encode<int32_t>(
        values, std::span<const EncodingType>{}, buffer);
    auto encodedOwner = std::make_shared<const std::string>(encoded);
    const std::string_view encodedAlphabet{
        encodedOwner->data(), encodedOwner->size()};
    alphabets_.emplace_back(
        dictionaryId,
        SharedDictionaryAlphabet::create(
            encodedAlphabet, std::move(encodedOwner), pool));
  }
}

std::shared_ptr<const SharedDictionaryAlphabet>
SharedDictionaryTestResolver::resolve(uint32_t dictionaryId, DataType dataType)
    const {
  if (dataType != DataType::Int32) {
    return nullptr;
  }
  for (const auto& [candidateId, alphabet] : alphabets_) {
    if (candidateId == dictionaryId) {
      return alphabet;
    }
  }
  return nullptr;
}

void configureSharedDictionarySelectionPolicy(
    WriterOptions& options,
    SharedDictionarySelectionPolicyOptions selectionOptions) {
  options.encodingSelectionPolicyCreator =
      [selectionOptions](DataType dataType) {
        return sharedDictionarySelectionPolicy(dataType, selectionOptions);
      };
}

velox::RowVectorPtr makeSharedDictionaryInput(
    velox::memory::MemoryPool* pool,
    velox::vector_size_t rowCount,
    const std::vector<SharedDictionarySource>& sources,
    std::span<const int32_t> valueUniverse,
    bool nullableData) {
  std::vector<std::string> names;
  std::vector<velox::VectorPtr> children;
  names.reserve(sources.size());
  children.reserve(sources.size());
  for (size_t i{0}; i < sources.size(); ++i) {
    names.push_back(sources[i].columnName);
    children.push_back(makeSharedDictionaryMap(
        pool,
        rowCount,
        sources[i],
        valueUniverse,
        nullableData,
        static_cast<uint32_t>(i * 17 + 3)));
  }
  velox::test::VectorMaker vectorMaker{pool};
  return vectorMaker.rowVector(names, children);
}

WriterOptions sharedDictionaryWriterOptions(
    const std::vector<SharedDictionarySource>& sources,
    const SharedDictionaryWriterOptions& writerOptions) {
  WriterOptions options;
  options.maxStreamChunkRawSize = 512;
  options.minStreamChunkRawSize = 1;
  options.skipConstantFlatMapInMapStreams =
      writerOptions.skipConstantFlatMapInMapStreams;
  configureSharedDictionarySelectionPolicy(
      options, writerOptions.selectionPolicy);

  auto builder = SharedDictionaryEncodingConfig::builder();
  if (writerOptions.externalDictionaryResolver != nullptr) {
    builder.setExternalResolver(writerOptions.externalDictionaryResolver);
  }
  for (const auto& source : sources) {
    options.flatMapColumns.emplace(source.columnName, std::set<std::string>{});
    builder.addFlatmapValueDictionary(
        source.columnName,
        source.dictionaryKey,
        SharedDictionaryConfig{
            .scope = source.scope,
            .dictionaryId = source.scope == SharedDictionaryScope::Stripe
                ? 0
                : source.dictionaryId});
  }
  options.experimentalSharedDictionaryEncoding = builder.build();
  return options;
}

std::string writeWithRandomStripes(
    velox::memory::MemoryPool* pool,
    const velox::RowVectorPtr& input,
    WriterOptions options,
    uint32_t seed) {
  std::mt19937 rng{seed};
  std::uniform_int_distribution<velox::vector_size_t> batchSizeDistribution{
      1, 41};
  std::uniform_int_distribution<velox::vector_size_t> stripeSizeDistribution{
      71, 193};

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  Writer writer{input->type(), std::move(writeFile), *pool, std::move(options)};

  velox::vector_size_t nextStripeBoundary = stripeSizeDistribution(rng);
  for (velox::vector_size_t row{0}; row < input->size();) {
    const auto batchSize =
        std::min(batchSizeDistribution(rng), input->size() - row);
    writer.write(input->slice(row, batchSize));
    row += batchSize;
    if (row < input->size() && row >= nextStripeBoundary) {
      writer.flush();
      nextStripeBoundary = row + stripeSizeDistribution(rng);
    }
  }
  writer.close();
  return file;
}

} // namespace facebook::nimble::test
