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

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/velox/SharedDictionaryConfig.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::nimble::test {

using SharedDictionaryTestDictionary =
    std::pair<uint32_t, std::vector<int32_t>>;

/// Returns a deterministic int32 universe with edge values for tests.
std::vector<int32_t> sharedDictionaryValueUniverse();

/// Resolves one or more fixed int32 shared dictionaries by id.
class SharedDictionaryTestResolver final : public ExternalDictionaryResolver {
 public:
  SharedDictionaryTestResolver(
      const std::vector<SharedDictionaryTestDictionary>& dictionaries,
      velox::memory::MemoryPool* pool);

  std::shared_ptr<const SharedDictionaryAlphabet> resolve(
      uint32_t dictionaryId,
      DataType dataType) const final;

 private:
  std::vector<
      std::pair<uint32_t, std::shared_ptr<const SharedDictionaryAlphabet>>>
      alphabets_;
};

struct SharedDictionarySelectionPolicyOptions {
  // Force int32 streams toward Dictionary encoding before shared-dictionary
  // wrapping. Integration tests disable this when they need a mixed direct and
  // shared-dictionary stripe sequence.
  bool forceDictionaryForInt32{true};
};

/// Installs the shared-dictionary selection policy on writer options.
void configureSharedDictionarySelectionPolicy(
    WriterOptions& options,
    SharedDictionarySelectionPolicyOptions selectionOptions = {});

struct SharedDictionarySource {
  // Top-level map column configured as a flat map.
  std::string columnName{};

  // Flat-map key whose value stream uses shared dictionary encoding.
  int64_t dictionaryKey{};

  // Scope used for this key's shared dictionary.
  SharedDictionaryScope scope{SharedDictionaryScope::Stripe};

  // Dictionary id for file and external scopes. Stripe scope always uses zero.
  uint32_t dictionaryId{};
};

struct SharedDictionaryWriterOptions {
  // Mirrors WriterOptions::skipConstantFlatMapInMapStreams for reader tests
  // that exercise both writer modes.
  bool skipConstantFlatMapInMapStreams{false};

  // Resolver used for external shared dictionaries, when configured.
  std::shared_ptr<const ExternalDictionaryResolver> externalDictionaryResolver{
      nullptr};

  // Selection-policy tuning for the generated writer options.
  SharedDictionarySelectionPolicyOptions selectionPolicy{};
};

/// Builds a deterministic row vector containing shared-dictionary flat maps.
velox::RowVectorPtr makeSharedDictionaryInput(
    velox::memory::MemoryPool* pool,
    velox::vector_size_t rowCount,
    const std::vector<SharedDictionarySource>& sources,
    std::span<const int32_t> valueUniverse,
    bool nullableData);

/// Builds writer options for shared-dictionary flat-map sources.
WriterOptions sharedDictionaryWriterOptions(
    const std::vector<SharedDictionarySource>& sources,
    const SharedDictionaryWriterOptions& writerOptions = {});

/// Writes input in deterministic random-sized batches and stripes.
std::string writeWithRandomStripes(
    velox::memory::MemoryPool* pool,
    const velox::RowVectorPtr& input,
    WriterOptions options,
    uint32_t seed);

} // namespace facebook::nimble::test
