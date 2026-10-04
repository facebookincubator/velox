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
#include "velox/dwio/nimble/index/BloomFilter.h"

#include <cstring>

#include "folly/Synchronized.h"
#include "folly/container/F14Map.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/BlockedBloomFilter.h"

namespace facebook::nimble::index {

namespace {

using FactoryMap = folly::
    F14FastMap<BloomFilterType, std::shared_ptr<const BloomFilterFactory>>;

folly::Synchronized<FactoryMap>& factoryRegistry() {
  static folly::Synchronized<FactoryMap> registry;
  return registry;
}

// Registers the layouts this library implements on first use. A function local
// static rather than a static initializer, so registration cannot race other
// static construction and the linker cannot drop it.
void ensureBuiltInFactoriesRegistered() {
  static const bool registered = [] {
    factoryRegistry().wlock()->emplace(
        BloomFilterType::kBlocked,
        std::make_shared<const BlockedBloomFilterFactory>());
    return true;
  }();
  (void)registered;
}

} // namespace

void BloomFilterTrailer::write(char* destination, BloomFilterType type) {
  destination[0] = static_cast<char>(type);
  std::memset(destination + 1, 0, kSize - 1);
}

std::optional<BloomFilterTrailer> BloomFilterTrailer::read(
    std::string_view serialized) {
  if (serialized.size() < kSize) {
    return std::nullopt;
  }
  // Read the trailer through an unsigned type. 'char' is signed here, so a
  // type byte above 0x7f would arrive negative.
  const auto* raw = reinterpret_cast<const uint8_t*>(serialized.data()) +
      serialized.size() - kSize;
  // Reject the filter if anything sits in the reserved bytes, even when the
  // type byte is one this build knows. That is what lets a later revision
  // repurpose an existing type: it sets a reserved byte, and every older
  // reader declines the filter rather than interpreting a payload whose
  // meaning has moved.
  for (size_t i = 1; i < kSize; ++i) {
    if (raw[i] != 0) {
      return std::nullopt;
    }
  }
  return BloomFilterTrailer{.type = static_cast<BloomFilterType>(raw[0])};
}

void registerBloomFilterFactory(
    std::shared_ptr<const BloomFilterFactory> factory) {
  NIMBLE_CHECK_NOT_NULL(factory);
  ensureBuiltInFactoriesRegistered();
  const auto type = factory->type();
  auto factories = factoryRegistry().wlock();
  const auto [_, inserted] = factories->emplace(type, std::move(factory));
  NIMBLE_CHECK(
      inserted,
      "Bloom filter factory is already registered for type: {}",
      static_cast<uint32_t>(type));
}

const BloomFilterFactory* bloomFilterFactory(BloomFilterType type) {
  ensureBuiltInFactoriesRegistered();
  auto factories = factoryRegistry().rlock();
  const auto it = factories->find(type);
  return it == factories->end() ? nullptr : it->second.get();
}

void BloomFilterReader::maybeContains(
    std::span<const std::string_view> keys,
    std::span<bool> out) const {
  NIMBLE_CHECK_GE(out.size(), keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    out[i] = maybeContains(keys[i]);
  }
}

std::unique_ptr<BloomFilterBuilder> createBloomFilterBuilder(
    const BloomFilterConfig& config,
    uint64_t numKeys,
    velox::memory::MemoryPool* pool) {
  const auto* factory = bloomFilterFactory(config.type);
  NIMBLE_USER_CHECK_NOT_NULL(
      factory,
      "No bloom filter factory is registered for type: {}",
      static_cast<uint32_t>(config.type));
  return factory->createBuilder(config, numKeys, pool);
}

std::unique_ptr<BloomFilterReader> createBloomFilterReader(
    std::string_view serialized,
    velox::memory::MemoryPool* pool) {
  const auto trailer = BloomFilterTrailer::read(serialized);
  NIMBLE_CHECK_FILE(
      trailer.has_value(),
      "Bloom filter trailer is unreadable. Filter bytes: {}",
      serialized.size());
  const auto* factory = bloomFilterFactory(trailer->type);
  NIMBLE_USER_CHECK_NOT_NULL(
      factory,
      "No bloom filter factory is registered for type: {}",
      static_cast<uint32_t>(trailer->type));
  return factory->createReader(
      serialized.substr(0, serialized.size() - BloomFilterTrailer::kSize),
      pool);
}

} // namespace facebook::nimble::index
