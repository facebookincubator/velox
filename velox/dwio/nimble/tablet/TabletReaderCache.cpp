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

#include "velox/dwio/nimble/tablet/TabletReaderCache.h"

#include <fmt/format.h>
#include <glog/logging.h>

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"
#include "velox/dwio/nimble/velox/SchemaUtils.h"

namespace facebook::nimble {

TabletReaderCache::Generator::Generator(
    std::vector<std::shared_ptr<velox::memory::MemoryPool>> pools,
    std::shared_ptr<folly::Executor> executor)
    : pools_{std::move(pools)},
      executor_{std::move(executor)},
      shardMask_{static_cast<uint32_t>(pools_.size()) - 1} {
  NIMBLE_CHECK_GT(pools_.size(), 0);
  NIMBLE_CHECK(
      velox::bits::isPowerOfTwo(pools_.size()),
      fmt::format(
          "numShards must be a power of 2, but got: {}", pools_.size()));
  NIMBLE_CHECK_NOT_NULL(executor_);
}

std::unique_ptr<CachedTabletReader> TabletReaderCache::Generator::operator()(
    const std::string& filename,
    const Properties* properties,
    void* /*stats*/) {
  NIMBLE_CHECK_NOT_NULL(properties);
  const auto shardIdx = std::hash<std::string>{}(filename)&shardMask_;
  auto options = properties->tabletOptions;
  auto ioOptions = velox::io::ReaderOptions(pools_[shardIdx].get());
  ioOptions.setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  ioOptions.setIndexIoStats(std::make_shared<velox::io::IoStatistics>());
  ioOptions.setIOExecutor(executor_);
  options.ioOptions = std::move(ioOptions);
  auto tablet = TabletReader::create(
      properties->readFile, pools_[shardIdx].get(), options);
  auto section = tablet->loadOptionalSection(
      std::string(kSchemaSection), /*keepCache=*/true);
  NIMBLE_CHECK(section.has_value(), "Schema section not found in tablet");
  auto nimbleSchema =
      SchemaDeserializer::deserialize(section->content().data());
  auto veloxSchema = asRowType(convertToVeloxType(*nimbleSchema));
  return std::make_unique<CachedTabletReader>(CachedTabletReader{
      std::move(tablet), std::move(nimbleSchema), std::move(veloxSchema)});
}

TabletReaderCache::Factory TabletReaderCache::createFactory(
    const Options& opts) {
  NIMBLE_CHECK_NOT_NULL(opts.executor);
  NIMBLE_CHECK(
      velox::bits::isPowerOfTwo(opts.numShards),
      fmt::format(
          "numShards must be a power of 2, but got: {}", opts.numShards));

  std::vector<std::shared_ptr<velox::memory::MemoryPool>> pools;
  pools.reserve(opts.numShards);
  for (uint32_t i = 0; i < opts.numShards; ++i) {
    pools.push_back(
        velox::memory::memoryManager()->addLeafPool(
            fmt::format("tablet_reader_cache_{}", i)));
  }

  auto cache = std::make_unique<TabletReaderCache::LRUCache>(
      opts.maxEntries, opts.expireDurationMs);
  auto generator = std::make_unique<TabletReaderCache::Generator>(
      std::move(pools), opts.executor);
  return Factory(std::move(cache), std::move(generator));
}

TabletReaderCache::TabletReaderCache(const Options& options)
    : factory_{createFactory(options)} {
  LOG(INFO) << "TabletReaderCache created: " << options.toString();
}

CachedTabletReader TabletReaderCache::get(
    const std::shared_ptr<velox::ReadFile>& readFile,
    const TabletReader::Options& tabletOptions) {
  Properties properties{readFile, tabletOptions};
  auto cached = factory_.generate(readFile->getName(), &properties);
  return CachedTabletReader{
      cached->tablet, cached->nimbleSchema, cached->veloxSchema};
}

velox::SimpleLRUCacheStats TabletReaderCache::stats() {
  return factory_.cacheStats();
}

std::optional<CachedTabletReader> TabletReaderCache::testingGet(
    const std::string& filename) {
  auto cached = factory_.get(filename);
  if (cached.get() == nullptr) {
    return std::nullopt;
  }
  return CachedTabletReader{
      cached->tablet, cached->nimbleSchema, cached->veloxSchema};
}

namespace {
struct SingletonState {
  ~SingletonState() {
    delete instance.load(std::memory_order_acquire);
  }

  std::atomic<TabletReaderCache*> instance{nullptr};
  std::mutex mutex;
};

SingletonState& singletonState() {
  static SingletonState state;
  return state;
}
} // namespace

void TabletReaderCache::initialize(const Options& options) {
  auto& state = singletonState();
  std::lock_guard<std::mutex> lock(state.mutex);
  NIMBLE_CHECK_NULL(
      state.instance.load(std::memory_order_acquire),
      "TabletReaderCache::initialize() must only be called once");
  state.instance.store(
      new TabletReaderCache(options), std::memory_order_release);
}

TabletReaderCache& TabletReaderCache::getInstance() {
  auto* instance = singletonState().instance.load(std::memory_order_acquire);
  NIMBLE_CHECK_NOT_NULL(
      instance,
      "TabletReaderCache::initialize() must be called before getInstance()");
  return *instance;
}

void TabletReaderCache::testingReset() {
  auto& state = singletonState();
  std::lock_guard<std::mutex> lock(state.mutex);
  delete state.instance.exchange(nullptr, std::memory_order_acq_rel);
}

} // namespace facebook::nimble
