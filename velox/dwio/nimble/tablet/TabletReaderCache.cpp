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
#include "velox/dwio/nimble/common/SchemaSerialization.h"
#include "velox/dwio/nimble/common/SchemaUtils.h"
#include "velox/dwio/nimble/tablet/Constants.h"

namespace facebook::nimble {

CachedTabletReader::CachedTabletReader(
    std::shared_ptr<TabletReader> tablet,
    std::shared_ptr<const facebook::nimble::Type> nimbleSchema,
    velox::RowTypePtr veloxSchema,
    std::shared_ptr<velox::io::IoStatistics> metadataIoStats,
    std::shared_ptr<velox::io::IoStatistics> indexIoStats,
    std::function<void(const CachedTabletReader&)> onRelease)
    : tablet_{std::move(tablet)},
      nimbleSchema_{std::move(nimbleSchema)},
      veloxSchema_{std::move(veloxSchema)},
      metadataIoStats_{std::move(metadataIoStats)},
      indexIoStats_{std::move(indexIoStats)},
      onRelease_{std::move(onRelease)} {}

CachedTabletReader::~CachedTabletReader() {
  if (onRelease_ == nullptr) {
    return;
  }
  // A destructor is implicitly noexcept, so an observer that threw here would
  // terminate the process rather than propagate. Unlike onCreate, this one has
  // no choice but to swallow. It runs on a scan thread, during eviction or on
  // the last release.
  try {
    onRelease_(*this);
  } catch (const std::exception& e) {
    LOG_EVERY_N(WARNING, 100)
        << "CachedTabletReader release observer threw: " << e.what();
  }
}

TabletReaderCache::Generator::Generator(
    std::vector<std::shared_ptr<velox::memory::MemoryPool>> pools,
    std::shared_ptr<folly::Executor> executor,
    std::function<void(const CachedTabletReader&)> onCreate,
    std::function<void(const CachedTabletReader&)> onRelease)
    : pools_{std::move(pools)},
      executor_{std::move(executor)},
      onCreate_{std::move(onCreate)},
      onRelease_{std::move(onRelease)},
      shardMask_{static_cast<uint32_t>(pools_.size()) - 1} {
  NIMBLE_CHECK_GT(pools_.size(), 0);
  NIMBLE_CHECK(
      velox::bits::isPowerOfTwo(pools_.size()),
      fmt::format(
          "numShards must be a power of 2, but got: {}", pools_.size()));
  NIMBLE_CHECK_NOT_NULL(executor_);
}

std::unique_ptr<std::shared_ptr<CachedTabletReader>>
TabletReaderCache::Generator::operator()(
    const std::string& filename,
    const Properties* properties,
    void* /*stats*/) {
  NIMBLE_CHECK_NOT_NULL(properties);
  const auto shardIdx = std::hash<std::string>{}(filename)&shardMask_;
  auto options = properties->tabletOptions;
  // The caller's ioOptions are replaced here, so its metadata/index statistics
  // never see this tablet's IO. Keeping the pair on the entry lets a lifetime
  // observer reach it instead.
  auto metadataIoStats = std::make_shared<velox::io::IoStatistics>();
  auto indexIoStats = std::make_shared<velox::io::IoStatistics>();
  auto ioOptions = velox::io::ReaderOptions(pools_[shardIdx].get());
  ioOptions.setMetadataIoStats(metadataIoStats);
  ioOptions.setIndexIoStats(indexIoStats);
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
  auto entry = std::make_shared<CachedTabletReader>(
      std::move(tablet),
      std::move(nimbleSchema),
      std::move(veloxSchema),
      std::move(metadataIoStats),
      std::move(indexIoStats),
      onRelease_);
  // Deliberately unguarded: the caller can still act on a failure here, and
  // registering an entry is a set insert whose only realistic failure is
  // allocation. The entry unwinds normally if it throws, firing onRelease.
  if (onCreate_ != nullptr) {
    onCreate_(*entry);
  }
  return std::make_unique<std::shared_ptr<CachedTabletReader>>(
      std::move(entry));
}

TabletReaderCache::Factory TabletReaderCache::createFactory(
    const Options& opts) {
  NIMBLE_CHECK_NOT_NULL(opts.executor);
  NIMBLE_CHECK(
      velox::bits::isPowerOfTwo(opts.numShards),
      fmt::format(
          "numShards must be a power of 2, but got: {}", opts.numShards));
  // An observer that is told about creation but never about release has no way
  // to know an entry is gone. Anything it keyed on the entry -- a registry of
  // live tablets, say -- is left holding a pointer to freed memory.
  NIMBLE_CHECK(
      (opts.onCreate == nullptr) == (opts.onRelease == nullptr),
      "TabletReaderCache onCreate and onRelease must be set together");

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
      std::move(pools), opts.executor, opts.onCreate, opts.onRelease);
  return Factory(std::move(cache), std::move(generator));
}

TabletReaderCache::TabletReaderCache(const Options& options)
    : factory_{createFactory(options)} {
  LOG(INFO) << "TabletReaderCache created: " << options.toString();
}

std::shared_ptr<CachedTabletReader> TabletReaderCache::get(
    const std::shared_ptr<velox::ReadFile>& readFile,
    const TabletReader::Options& tabletOptions) {
  Properties properties{readFile, tabletOptions};
  // Copying the shared_ptr out and letting the CachedPtr die here keeps the
  // cache pin momentary; the entry stays alive through the returned pointer.
  auto cached = factory_.generate(readFile->getName(), &properties);
  return *cached;
}

velox::SimpleLRUCacheStats TabletReaderCache::stats() {
  return factory_.cacheStats();
}

std::shared_ptr<CachedTabletReader> TabletReaderCache::testingGet(
    const std::string& filename) {
  auto cached = factory_.get(filename);
  if (cached.get() == nullptr) {
    return nullptr;
  }
  return *cached;
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
