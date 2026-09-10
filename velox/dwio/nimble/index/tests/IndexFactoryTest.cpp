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

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

#include "folly/Synchronized.h"
#include "velox/common/config/Config.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/index/ClusterIndexFactory.h"
#include "velox/dwio/nimble/index/ClusterIndexWriter.h"
#include "velox/dwio/nimble/index/DenseIndexFactory.h"
#include "velox/dwio/nimble/index/DenseIndexRegistry.h"
#include "velox/dwio/nimble/index/HashIndexConfig.h"
#include "velox/dwio/nimble/index/HashIndexWriter.h"
#include "velox/dwio/nimble/index/IndexConstants.h"
#include "velox/dwio/nimble/index/IndexKeyEncoder.h"
#include "velox/dwio/nimble/index/SortedIndexConfig.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/IndexGenerated.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/FlatVector.h"

namespace facebook::nimble::index::test {
namespace {

constexpr std::string_view kCustomClusterIndexName{"test.cluster.v1"};
constexpr std::string_view kCustomDenseIndexName{"test.dense.v1"};
constexpr std::string_view kValidationDenseIndexName{
    "test.dense.validation.v1"};
constexpr std::string_view kFailingDenseIndexName{"test.dense.failing.v1"};
constexpr std::string_view kInvalidDescriptorIndexName{
    "test.dense.invalid_descriptor.v1"};
constexpr std::string_view kLifecycleDenseIndexName{"test.dense.lifecycle.v1"};

enum class FailureStage { kNone, kFlush, kClose };

folly::Synchronized<std::vector<std::string>> lifecycleEvents;

struct TestIndexConfig final : IndexConfig {
  TestIndexConfig(
      IndexFamily family,
      std::string name,
      std::string column = "id",
      FailureStage failureStage = FailureStage::kNone)
      : IndexConfig{family, std::move(name)},
        column{std::move(column)},
        failureStage{failureStage} {}

  const std::string column;
  const FailureStage failureStage;
};

class TestClusterIndexFactory final : public ClusterIndexFactory {
 public:
  std::string_view name() const override {
    return kCustomClusterIndexName;
  }

  std::unique_ptr<IndexWriter> createWriter(
      const IndexConfig& config,
      const velox::TypePtr& inputType,
      velox::memory::MemoryPool* pool) const override {
    const auto& options = checkedIndexConfig<TestIndexConfig>(config);
    const ClusterIndexConfig builtInConfig{
        std::string{name()},
        {options.column},
        {},
        true,
        false,
        EncodingLayout{EncodingType::Prefix, {}, CompressionType::Uncompressed},
        10'000,
        CompressionType::Uncompressed};
    return ClusterIndexWriter::create(builtInConfig, inputType, pool);
  }

  std::unique_ptr<IndexKeyEncoder> createKeyEncoder(
      const std::vector<std::string>& columns,
      const velox::RowTypePtr& inputType,
      const std::vector<SortOrder>& sortOrders,
      velox::memory::MemoryPool* pool) const override {
    return clusterIndexFactory(kClusterIndexName)
        .createKeyEncoder(columns, inputType, sortOrders, pool);
  }

  std::unique_ptr<ClusterIndex> createReader(
      Section rootSection,
      velox::memory::MemoryPool* pool,
      const IndexLookup::Options& options) const override {
    return clusterIndexFactory(kClusterIndexName)
        .createReader(std::move(rootSection), pool, options);
  }
};

class NamedIndexWriter final : public IndexWriter {
 public:
  NamedIndexWriter(std::string name, std::unique_ptr<IndexWriter> writer)
      : name_{std::move(name)}, writer_{std::move(writer)} {}

  void write(const velox::VectorPtr& input) override {
    writer_->write(input);
  }

  void flush(
      const WriteDataFn& writeDataFn,
      const CreateMetadataSectionFn& createMetadataFn) override {
    writer_->flush(writeDataFn, createMetadataFn);
  }

  std::optional<IndexDescriptor> close(
      const WriteDataFn& writeDataFn,
      const CreateMetadataSectionFn& createMetadataFn) override {
    auto descriptor = writer_->close(writeDataFn, createMetadataFn);
    if (descriptor.has_value()) {
      descriptor->name = name_;
    }
    return descriptor;
  }

 private:
  const std::string name_;
  std::unique_ptr<IndexWriter> writer_;
};

class HashDelegatingDenseIndexFactory final : public DenseIndexFactory {
 public:
  explicit HashDelegatingDenseIndexFactory(
      std::string name = std::string{kCustomDenseIndexName})
      : name_{std::move(name)} {}

  std::string_view name() const override {
    return name_;
  }

  std::unique_ptr<IndexWriter> createWriter(
      std::span<const IndexConfig*> configs,
      const velox::TypePtr& inputType,
      velox::memory::MemoryPool* pool) const override {
    std::vector<std::shared_ptr<const IndexConfig>> builtInConfigs;
    builtInConfigs.reserve(configs.size());
    for (const auto* config : configs) {
      NIMBLE_CHECK_NOT_NULL(config);
      const auto& options = checkedIndexConfig<TestIndexConfig>(*config);
      builtInConfigs.emplace_back(
          std::make_shared<const HashIndexConfig>(
              std::string{kDenseHashIndexName},
              std::vector<std::string>{options.column},
              0.7f,
              std::nullopt,
              0));
    }
    std::vector<const IndexConfig*> builtInConfigPtrs;
    builtInConfigPtrs.reserve(builtInConfigs.size());
    for (const auto& config : builtInConfigs) {
      builtInConfigPtrs.emplace_back(config.get());
    }
    return std::make_unique<NamedIndexWriter>(
        name_, HashIndexWriter::create(builtInConfigPtrs, inputType, pool));
  }

  std::unique_ptr<DenseIndexReader> createReader(
      Section rootSection,
      const IndexLookup::Options& options,
      velox::memory::MemoryPool* pool) const override {
    return denseIndexFactory(kDenseHashIndexName)
        ->createReader(std::move(rootSection), options, pool);
  }

 private:
  const std::string name_;
};

class ValidationDenseIndexReader final : public DenseIndexReader {
 public:
  const IndexLookup* findIndex(const std::vector<std::string>&) const override {
    return nullptr;
  }

  std::vector<std::vector<std::string>> indexColumns() const override {
    return {{"id"}};
  }
};

class ValidationDenseIndexFactory final : public DenseIndexFactory {
 public:
  std::string_view name() const override {
    return kValidationDenseIndexName;
  }

  std::unique_ptr<IndexWriter> createWriter(
      std::span<const IndexConfig*>,
      const velox::TypePtr&,
      velox::memory::MemoryPool*) const override {
    NIMBLE_UNREACHABLE("Validation factory does not create writers");
  }

  std::unique_ptr<DenseIndexReader> createReader(
      Section,
      const IndexLookup::Options&,
      velox::memory::MemoryPool*) const override {
    return std::make_unique<ValidationDenseIndexReader>();
  }
};

class FailingIndexWriter final : public IndexWriter {
 public:
  void write(const velox::VectorPtr&) override {
    NIMBLE_FAIL("Injected index write failure");
  }

  void flush(const WriteDataFn&, const CreateMetadataSectionFn&) override {}

  std::optional<IndexDescriptor> close(
      const WriteDataFn&,
      const CreateMetadataSectionFn&) override {
    return std::nullopt;
  }
};

class InvalidDescriptorIndexWriter final : public IndexWriter {
 public:
  void write(const velox::VectorPtr&) override {}

  void flush(const WriteDataFn&, const CreateMetadataSectionFn&) override {}

  std::optional<IndexDescriptor> close(
      const WriteDataFn&,
      const CreateMetadataSectionFn&) override {
    return IndexDescriptor{
        IndexFamily::Dense,
        "test.dense.wrong.v1",
        MetadataSection{0, 0, CompressionType::Uncompressed}};
  }
};

class LifecycleIndexWriter final : public IndexWriter {
 public:
  explicit LifecycleIndexWriter(FailureStage failureStage)
      : failureStage_{failureStage} {}

  void write(const velox::VectorPtr&) override {
    lifecycleEvents.wlock()->emplace_back("write");
  }

  void flush(const WriteDataFn&, const CreateMetadataSectionFn&) override {
    lifecycleEvents.wlock()->emplace_back("flush");
    if (failureStage_ == FailureStage::kFlush) {
      NIMBLE_FAIL("Injected index flush failure");
    }
  }

  std::optional<IndexDescriptor> close(
      const WriteDataFn&,
      const CreateMetadataSectionFn&) override {
    lifecycleEvents.wlock()->emplace_back("close");
    if (failureStage_ == FailureStage::kClose) {
      NIMBLE_FAIL("Injected index close failure");
    }
    return std::nullopt;
  }

 private:
  const FailureStage failureStage_;
};

class LifecycleIndexFactory final : public DenseIndexFactory {
 public:
  std::string_view name() const override {
    return kLifecycleDenseIndexName;
  }

  std::unique_ptr<IndexWriter> createWriter(
      std::span<const IndexConfig*> configs,
      const velox::TypePtr&,
      velox::memory::MemoryPool*) const override {
    NIMBLE_CHECK_EQ(configs.size(), 1);
    NIMBLE_CHECK_NOT_NULL(configs.front());
    const auto& config = checkedIndexConfig<TestIndexConfig>(*configs.front());
    return std::make_unique<LifecycleIndexWriter>(config.failureStage);
  }

  std::unique_ptr<DenseIndexReader> createReader(
      Section,
      const IndexLookup::Options&,
      velox::memory::MemoryPool*) const override {
    NIMBLE_UNREACHABLE("Test factory does not create readers");
  }
};

template <typename Writer>
class StubDenseIndexFactory final : public DenseIndexFactory {
 public:
  explicit StubDenseIndexFactory(std::string name) : name_{std::move(name)} {}

  std::string_view name() const override {
    return name_;
  }

  std::unique_ptr<IndexWriter> createWriter(
      std::span<const IndexConfig*>,
      const velox::TypePtr&,
      velox::memory::MemoryPool*) const override {
    return std::make_unique<Writer>();
  }

  std::unique_ptr<DenseIndexReader> createReader(
      Section,
      const IndexLookup::Options&,
      velox::memory::MemoryPool*) const override {
    NIMBLE_UNREACHABLE("Test factory does not create readers");
  }

 private:
  const std::string name_;
};

class IndexFactoryTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::filesystems::registerLocalFileSystem();
    velox::memory::MemoryManager::testingSetInstance({});
    registerClusterIndexFactory(
        std::make_shared<const TestClusterIndexFactory>());
    registerDenseIndexFactory(
        std::make_shared<const HashDelegatingDenseIndexFactory>());
    registerDenseIndexFactory(
        std::make_shared<const ValidationDenseIndexFactory>());
    registerDenseIndexFactory(
        std::make_shared<const StubDenseIndexFactory<FailingIndexWriter>>(
            std::string{kFailingDenseIndexName}));
    registerDenseIndexFactory(
        std::make_shared<
            const StubDenseIndexFactory<InvalidDescriptorIndexWriter>>(
            std::string{kInvalidDescriptorIndexName}));
    registerDenseIndexFactory(std::make_shared<const LifecycleIndexFactory>());
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool("IndexFactoryTest");
    leafPool_ = rootPool_->addLeafChild("leaf");
    tempDirectory_ = velox::common::testutil::TempDirectoryPath::create();
  }

  static velox::RowTypePtr rowType() {
    return velox::ROW({{"id", velox::INTEGER()}});
  }

  velox::RowVectorPtr makeBatch() {
    constexpr int32_t kNumRows{20};
    auto ids =
        velox::BaseVector::create(velox::INTEGER(), kNumRows, leafPool_.get());
    auto* flatIds = ids->asFlatVector<int32_t>();
    for (int32_t i = 0; i < kNumRows; ++i) {
      flatIds->set(i, i);
    }
    return std::make_shared<velox::RowVector>(
        leafPool_.get(),
        rowType(),
        nullptr,
        kNumRows,
        std::vector<velox::VectorPtr>{ids});
  }

  std::unique_ptr<Writer> makeWriter(
      std::string_view pathSuffix,
      WriterOptions options) {
    const auto path = tempDirectory_->getPath() + "/" + std::string{pathSuffix};
    auto fileSystem = velox::filesystems::getFileSystem(path, {});
    auto file = fileSystem->openFileForWrite(
        path,
        {.shouldCreateParentDirectories = true,
         .shouldThrowOnFileAlreadyExists = false});
    return std::make_unique<Writer>(
        rowType(), std::move(file), *rootPool_, std::move(options));
  }

  std::string writeFile() {
    const auto path = tempDirectory_->getPath() + "/custom_indexes";
    auto fileSystem = velox::filesystems::getFileSystem(path, {});
    auto file = fileSystem->openFileForWrite(
        path,
        {.shouldCreateParentDirectories = true,
         .shouldThrowOnFileAlreadyExists = false});
    WriterOptions options;
    options.clusterIndexConfig = std::make_shared<const TestIndexConfig>(
        IndexFamily::Cluster, std::string{kCustomClusterIndexName});
    options.denseIndexConfigs.emplace_back(
        std::make_shared<const TestIndexConfig>(
            IndexFamily::Dense, std::string{kCustomDenseIndexName}));
    Writer writer(rowType(), std::move(file), *rootPool_, options);
    writer.write(makeBatch());
    writer.close();
    return path;
  }

  std::shared_ptr<TabletReader> openFile(
      const std::string& path,
      std::string clusterIndexName) {
    auto fileSystem = velox::filesystems::getFileSystem(path, {});
    TabletReader::Options options;
    options.loadClusterIndex = true;
    options.loadDenseIndexes = true;
    options.clusterIndexName = std::move(clusterIndexName);
    options.ioOptions.emplace(leafPool_.get())
        .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>())
        .setIndexIoStats(std::make_shared<velox::io::IoStatistics>());
    return TabletReader::create(
        fileSystem->openFileForRead(path), leafPool_.get(), options);
  }

  std::string encodeKey(int32_t value) {
    auto keyVector =
        velox::BaseVector::create(velox::INTEGER(), 1, leafPool_.get());
    keyVector->asFlatVector<int32_t>()->set(0, value);
    auto input = std::make_shared<velox::RowVector>(
        leafPool_.get(),
        rowType(),
        nullptr,
        1,
        std::vector<velox::VectorPtr>{keyVector});
    auto encoder = clusterIndexFactory(kCustomClusterIndexName)
                       .createKeyEncoder(
                           {"id"},
                           rowType(),
                           {SortOrder{.ascending = true}},
                           leafPool_.get());
    Buffer buffer{*leafPool_};
    std::vector<std::string_view> keys;
    encoder->encode(
        input, keys, [&buffer](size_t size) { return buffer.reserve(size); });
    return std::string{keys.front()};
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
  std::shared_ptr<velox::common::testutil::TempDirectoryPath> tempDirectory_;
};

TEST_F(IndexFactoryTest, customFactoriesEndToEnd) {
  const auto path = writeFile();
  auto tablet = openFile(path, std::string{kCustomClusterIndexName});
  ASSERT_NE(tablet->clusterIndex(), nullptr);
  const auto key = encodeKey(7);
  auto clusterResult = tablet->clusterIndex()->lookup(
      IndexLookup::LookupRequest::pointLookup({key}));
  ASSERT_EQ(clusterResult.size(), 1);
  EXPECT_FALSE(clusterResult[0].empty());

  const auto* denseIndex = tablet->denseIndex({"id"});
  ASSERT_NE(denseIndex, nullptr);
  auto denseResult =
      denseIndex->lookup(IndexLookup::LookupRequest::pointLookup({key}));
  ASSERT_EQ(denseResult.size(), 1);
  ASSERT_EQ(denseResult[0].size(), 1);
  EXPECT_EQ(denseResult[0][0], RowRange(7, 8));

  auto wrongClusterName = openFile(path, std::string{kClusterIndexName});
  EXPECT_EQ(wrongClusterName->clusterIndex(), nullptr);
}

TEST_F(IndexFactoryTest, builtInClusterFactoryIsRegistered) {
  EXPECT_EQ(clusterIndexFactory(kClusterIndexName).name(), kClusterIndexName);
}

TEST_F(IndexFactoryTest, builtInDenseFactoriesAreRegistered) {
  EXPECT_EQ(
      denseIndexFactory(kDenseHashIndexName)->name(), kDenseHashIndexName);
  EXPECT_EQ(
      denseIndexFactory(kDenseSortedIndexName)->name(), kDenseSortedIndexName);
}

TEST_F(IndexFactoryTest, rejectDuplicateClusterFactoryRegistration) {
  NIMBLE_ASSERT_THROW(
      registerClusterIndexFactory(
          std::make_shared<const TestClusterIndexFactory>()),
      "Index factory 'test.cluster.v1' is already registered for family 'cluster'");
}

TEST_F(IndexFactoryTest, rejectDuplicateDenseFactoryRegistration) {
  NIMBLE_ASSERT_THROW(
      registerDenseIndexFactory(
          std::make_shared<const HashDelegatingDenseIndexFactory>()),
      "Index factory 'test.dense.v1' is already registered for family 'dense'");
}

TEST_F(IndexFactoryTest, rejectUnknownClusterFactory) {
  NIMBLE_ASSERT_THROW(
      clusterIndexFactory("test.missing.cluster.v1"),
      "Unknown index factory 'test.missing.cluster.v1' for family 'cluster'");
}

TEST_F(IndexFactoryTest, rejectUnknownDenseFactory) {
  EXPECT_EQ(denseIndexFactory("test.missing.dense.v1"), nullptr);
}

TEST_F(IndexFactoryTest, rejectDuplicateDenseIndex) {
  auto file = std::make_shared<velox::InMemoryReadFile>(std::string{});
  auto ioStats = std::make_shared<velox::io::IoStatistics>();
  velox::io::ReaderOptions ioOptions{leafPool_.get()};
  ioOptions.setMetadataIoStats(ioStats);
  ioOptions.setIndexIoStats(ioStats);
  IndexLookup::Options options{.file = file, .ioOptions = &ioOptions};

  auto makeSection = [&] {
    return Section{MetadataBuffer{
        velox::AlignedBuffer::allocate<char>(1, leafPool_.get())}};
  };
  std::vector<DenseIndexRegistry::Entry> entries;
  entries.emplace_back(
      DenseIndexRegistry::Entry{
          std::string{kValidationDenseIndexName}, makeSection()});
  entries.emplace_back(
      DenseIndexRegistry::Entry{
          std::string{kValidationDenseIndexName}, makeSection()});

  NIMBLE_ASSERT_THROW(
      DenseIndexRegistry::create(std::move(entries), options, leafPool_.get()),
      "Duplicate dense index 'test.dense.validation.v1' columns");
}

TEST_F(IndexFactoryTest, concurrentFactoryAccess) {
  constexpr size_t kFactoryCount = 32;
  constexpr size_t kReaderCount = 8;
  std::vector<std::shared_ptr<const HashDelegatingDenseIndexFactory>> factories;
  factories.reserve(kFactoryCount);
  for (size_t i = 0; i < kFactoryCount; ++i) {
    factories.emplace_back(
        std::make_shared<const HashDelegatingDenseIndexFactory>(
            "test.concurrent.dense." + std::to_string(i)));
  }

  std::vector<std::thread> threads;
  threads.reserve(kFactoryCount);
  for (const auto& factory : factories) {
    threads.emplace_back([factory] { registerDenseIndexFactory(factory); });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  std::atomic_bool lookupFailed{false};
  threads.clear();
  threads.reserve(kReaderCount);
  for (size_t i = 0; i < kReaderCount; ++i) {
    threads.emplace_back([&factories, &lookupFailed] {
      for (const auto& factory : factories) {
        if (denseIndexFactory(factory->name()) != factory.get()) {
          lookupFailed = true;
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_FALSE(lookupFailed);
}

TEST_F(IndexFactoryTest, indexWriteFailurePoisonsWriter) {
  WriterOptions options;
  options.denseIndexConfigs.emplace_back(
      std::make_shared<const TestIndexConfig>(
          IndexFamily::Dense, std::string{kFailingDenseIndexName}));
  auto writer = makeWriter("failing_writer", std::move(options));

  NIMBLE_ASSERT_THROW(
      writer->write(makeBatch()), "Injected index write failure");
  NIMBLE_ASSERT_THROW(writer->flush(), "Injected index write failure");
  NIMBLE_ASSERT_THROW(writer->close(), "Injected index write failure");
}

TEST_F(IndexFactoryTest, rejectInvalidWriterDescriptor) {
  WriterOptions options;
  options.denseIndexConfigs.emplace_back(
      std::make_shared<const TestIndexConfig>(
          IndexFamily::Dense, std::string{kInvalidDescriptorIndexName}));
  auto writer = makeWriter("invalid_descriptor", std::move(options));
  writer->write(makeBatch());

  NIMBLE_ASSERT_THROW(
      writer->close(),
      "Index writer returned an unexpected name for test.dense.invalid_descriptor.v1");
}

TEST_F(IndexFactoryTest, rejectMismatchedBuiltInConfigType) {
  WriterOptions options;
  options.denseIndexConfigs.emplace_back(
      std::make_shared<const TestIndexConfig>(
          IndexFamily::Dense, std::string{kDenseHashIndexName}));

  EXPECT_THROW(
      makeWriter("mismatched_options", std::move(options)),
      velox::VeloxRuntimeError);
}

TEST_F(IndexFactoryTest, rejectInvalidIndexConfigIdentity) {
  {
    WriterOptions options;
    options.denseIndexConfigs.emplace_back(nullptr);
    NIMBLE_ASSERT_USER_THROW(
        makeWriter("null_dense_config", std::move(options)),
        "Dense index config cannot be null");
  }
  {
    WriterOptions options;
    options.clusterIndexConfig = std::make_shared<const TestIndexConfig>(
        IndexFamily::Dense, std::string{kCustomClusterIndexName});
    NIMBLE_ASSERT_USER_THROW(
        makeWriter("wrong_cluster_family", std::move(options)),
        "Cluster index configuration must use the cluster family");
  }
  {
    WriterOptions options;
    options.denseIndexConfigs.emplace_back(
        std::make_shared<const TestIndexConfig>(
            IndexFamily::Cluster, std::string{kCustomDenseIndexName}));
    NIMBLE_ASSERT_USER_THROW(
        makeWriter("wrong_dense_family", std::move(options)),
        "Dense index configuration must use the dense family");
  }
}

TEST_F(IndexFactoryTest, preserveIndexDescriptorOrder) {
  WriterOptions options;
  options.denseIndexConfigs.emplace_back(
      HashIndexConfigBuilder{}.withKeyColumns({"id"}).build());
  options.denseIndexConfigs.emplace_back(
      HashIndexConfigBuilder{}.withKeyColumns({"id", "id"}).build());
  options.denseIndexConfigs.emplace_back(
      SortedIndexConfigBuilder{}.withKeyColumns({"id"}).build());
  options.clusterIndexConfig = ClusterIndexConfigBuilder{}
                                   .withKeyColumns({"id"})
                                   .withEnforceKeyOrder(true)
                                   .build();
  auto writer = makeWriter("typed_descriptor_order", std::move(options));
  writer->write(makeBatch());
  writer->close();

  const auto path = tempDirectory_->getPath() + "/typed_descriptor_order";
  auto fileSystem = velox::filesystems::getFileSystem(path, {});
  TabletReader::Options readerOptions;
  readerOptions.ioOptions.emplace(leafPool_.get())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  auto tablet = TabletReader::create(
      fileSystem->openFileForRead(path), leafPool_.get(), readerOptions);
  auto section = tablet->loadOptionalSection(std::string{kIndexSection});
  ASSERT_TRUE(section.has_value());
  const auto* root =
      flatbuffers::GetRoot<serialization::IndexRoot>(section->content().data());
  ASSERT_NE(root->indexes(), nullptr);
  std::vector<std::string> names;
  for (const auto* descriptor : *root->indexes()) {
    names.emplace_back(descriptor->name()->str());
  }
  EXPECT_EQ(
      names,
      (std::vector<std::string>{
          std::string{kDenseHashIndexName},
          std::string{kDenseSortedIndexName},
          std::string{kClusterIndexName}}));
}

TEST_F(IndexFactoryTest, indexWriterLifecycle) {
  struct TestCase {
    std::string name;
    std::string failureStage;
    std::vector<std::string> expectedEvents;
  };
  const std::vector<TestCase> testCases{
      {"success", "", {"write", "flush", "close"}},
      {"flush_failure", "flush", {"write", "flush"}},
      {"close_failure", "close", {"write", "flush", "close"}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    lifecycleEvents.wlock()->clear();
    WriterOptions options;
    const auto failureStage = testCase.failureStage == "flush"
        ? FailureStage::kFlush
        : testCase.failureStage == "close" ? FailureStage::kClose
                                           : FailureStage::kNone;
    options.denseIndexConfigs.emplace_back(
        std::make_shared<const TestIndexConfig>(
            IndexFamily::Dense,
            std::string{kLifecycleDenseIndexName},
            "id",
            failureStage));
    auto writer = makeWriter(testCase.name, std::move(options));
    writer->write(makeBatch());

    if (testCase.failureStage == "flush") {
      NIMBLE_ASSERT_THROW(writer->close(), "Injected index flush failure");
      NIMBLE_ASSERT_THROW(writer->flush(), "Injected index flush failure");
      NIMBLE_ASSERT_THROW(writer->close(), "Injected index flush failure");
    } else if (testCase.failureStage == "close") {
      NIMBLE_ASSERT_THROW(writer->close(), "Injected index close failure");
      NIMBLE_ASSERT_THROW(writer->close(), "Injected index close failure");
    } else {
      writer->close();
    }
    EXPECT_EQ(lifecycleEvents.copy(), testCase.expectedEvents);
  }
}

} // namespace
} // namespace facebook::nimble::index::test
