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

#include "velox/dwio/nimble/tablet/SharedDictionaryReader.h"

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryCatalog.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"

namespace facebook::nimble {
namespace {

class TestDictionaryResolver final : public ExternalDictionaryResolver {
 public:
  explicit TestDictionaryResolver(
      std::shared_ptr<const SharedDictionaryAlphabet> alphabet)
      : alphabet_{std::move(alphabet)} {}

  std::shared_ptr<const SharedDictionaryAlphabet> resolve(
      uint32_t dictionaryId,
      DataType dataType) const final {
    ++resolveCount_;
    dictionaryId_ = dictionaryId;
    dataType_ = dataType;
    return alphabet_;
  }

  uint32_t resolveCount() const {
    return resolveCount_;
  }

  uint32_t dictionaryId() const {
    return dictionaryId_;
  }

  DataType dataType() const {
    return dataType_;
  }

 private:
  const std::shared_ptr<const SharedDictionaryAlphabet> alphabet_;
  mutable uint32_t resolveCount_{0};
  mutable uint32_t dictionaryId_{kInvalidSharedDictionaryId};
  mutable DataType dataType_{DataType::Undefined};
};

class SharedDictionaryReaderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() final {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  }

  std::string encodeAlphabet(std::span<const int32_t> values) {
    Buffer buffer{*pool_};
    return std::string{SharedDictionaryAlphabet::encode<int32_t>(
        values, std::array{EncodingType::Trivial}, buffer)};
  }

  std::shared_ptr<const SharedDictionaryAlphabet> createAlphabet(
      std::span<const int32_t> values) {
    auto owner = std::make_shared<const std::string>(encodeAlphabet(values));
    const std::string_view encodedAlphabet{*owner};
    return SharedDictionaryAlphabet::create(
        encodedAlphabet, std::move(owner), pool_.get());
  }

  std::shared_ptr<TabletReader> createTablet(
      std::optional<std::string> catalog,
      std::shared_ptr<const ExternalDictionaryResolver> resolver = nullptr) {
    auto file = std::make_shared<std::string>();
    velox::InMemoryWriteFile writeFile{file.get()};
    auto writer = TabletWriter::create(&writeFile, *pool_, {});
    if (catalog.has_value()) {
      writer->writeOptionalSection(std::string{kDictionarySection}, *catalog);
    }
    writer->close();

    auto options = test::makeTestTabletOptions(pool_.get());
    options.externalDictionaryResolver = std::move(resolver);
    files_.push_back(file);
    auto readFile =
        std::make_shared<testing::InMemoryTrackableReadFile>(*file, true);
    return TabletReader::create(readFile, pool_.get(), options);
  }

  std::shared_ptr<TabletReader> createTabletWithSharedDictionarySection(
      std::span<const SharedDictionaryReference> stripeDictionaryReferences,
      std::span<const SharedDictionaryReference> fileDictionaryReferences,
      uint32_t fileDictionaryId,
      std::string_view fileEncodedAlphabet) {
    return createTabletWithSharedDictionarySectionAndReadFile(
               stripeDictionaryReferences,
               fileDictionaryReferences,
               fileDictionaryId,
               fileEncodedAlphabet)
        .first;
  }

  std::pair<
      std::shared_ptr<TabletReader>,
      std::shared_ptr<testing::InMemoryTrackableReadFile>>
  createTabletWithSharedDictionarySectionAndReadFile(
      std::span<const SharedDictionaryReference> stripeDictionaryReferences,
      std::span<const SharedDictionaryReference> fileDictionaryReferences,
      uint32_t fileDictionaryId,
      std::string_view fileEncodedAlphabet) {
    auto file = std::make_shared<std::string>();
    velox::InMemoryWriteFile writeFile{file.get()};
    TabletWriter::Options writerOptions;
    writerOptions.closeCallback =
        [&](const WriteDataFn& writeDataFn,
            const CreateMetadataSectionFn& /*createMetadataFn*/,
            const WriteOptionalSectionFn& writeMetadataFn) {
          const auto [offset, length] =
              writeDataFn(std::vector<std::string_view>{fileEncodedAlphabet});
          const std::array fileDictionaries{FileDictionary{
              .dictionaryId = fileDictionaryId,
              .dataType = DataType::Int32,
              .offset = offset,
              .length = length}};
          const auto catalog = SharedDictionaryCatalog::serialize(
              stripeDictionaryReferences,
              fileDictionaryReferences,
              {},
              fileDictionaries);
          writeMetadataFn(std::string{kDictionarySection}, catalog);
        };
    auto writer =
        TabletWriter::create(&writeFile, *pool_, std::move(writerOptions));
    writer->close();

    auto options = test::makeTestTabletOptions(pool_.get());
    options.maxFooterIoBytes = 1024;
    files_.push_back(file);
    auto readFile =
        std::make_shared<testing::InMemoryTrackableReadFile>(*file, true);
    return {TabletReader::create(readFile, pool_.get(), options), readFile};
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::vector<std::shared_ptr<std::string>> files_;
};

TEST_F(SharedDictionaryReaderTest, noCatalog) {
  auto tablet = createTablet(std::nullopt);

  EXPECT_FALSE(tablet->hasStripeDictionaries());
  EXPECT_FALSE(tablet->hasFileOrExternalDictionaries());
  EXPECT_FALSE(tablet->stripeDictionaryStreamId(10).has_value());
  EXPECT_TRUE(tablet->stripeDictionaryStreamIds(std::array<uint32_t, 2>{10, 20})
                  .empty());
  EXPECT_EQ(tablet->resolveDictionaryAlphabet(10), nullptr);
}

TEST_F(SharedDictionaryReaderTest, basic) {
  constexpr uint32_t kStripeValueStreamId = 10;
  constexpr uint32_t kDictionaryStreamId = 100;
  constexpr uint32_t kFileValueStreamId = 20;
  constexpr uint32_t kSecondFileValueStreamId = 21;
  constexpr uint32_t kFileDictionaryId = 7;
  const std::array<int32_t, 3> values{10, 20, 30};
  const auto fileEncodedAlphabet = encodeAlphabet(values);
  const std::array stripeDictionaryReferences{SharedDictionaryReference{
      .valueStreamId = kStripeValueStreamId,
      .dictionaryId = kDictionaryStreamId,
      .dataType = DataType::Int32}};
  const std::array fileDictionaryReferences{
      SharedDictionaryReference{
          .valueStreamId = kFileValueStreamId,
          .dictionaryId = kFileDictionaryId,
          .dataType = DataType::Int32},
      SharedDictionaryReference{
          .valueStreamId = kSecondFileValueStreamId,
          .dictionaryId = kFileDictionaryId,
          .dataType = DataType::Int32}};

  auto tablet = createTabletWithSharedDictionarySection(
      stripeDictionaryReferences,
      fileDictionaryReferences,
      kFileDictionaryId,
      fileEncodedAlphabet);

  EXPECT_TRUE(tablet->hasStripeDictionaries());
  EXPECT_TRUE(tablet->hasFileOrExternalDictionaries());
  EXPECT_EQ(
      tablet->stripeDictionaryStreamId(kStripeValueStreamId),
      kDictionaryStreamId);
  EXPECT_FALSE(
      tablet->stripeDictionaryStreamId(kFileValueStreamId).has_value());
  EXPECT_FALSE(tablet->stripeDictionaryStreamId(uint32_t{999}).has_value());
  const std::array nonStripeValueStreamIds{uint32_t{999}, kFileValueStreamId};
  EXPECT_TRUE(
      tablet->stripeDictionaryStreamIds(nonStripeValueStreamIds).empty());

  const std::array valueStreamIds{
      uint32_t{999}, kStripeValueStreamId, kFileValueStreamId};
  const auto dictionaryStreamIds =
      tablet->stripeDictionaryStreamIds(valueStreamIds);
  ASSERT_EQ(dictionaryStreamIds.size(), 1);
  EXPECT_FALSE(dictionaryStreamIds.contains(uint32_t{999}));
  EXPECT_EQ(dictionaryStreamIds.at(kStripeValueStreamId), kDictionaryStreamId);
  EXPECT_FALSE(dictionaryStreamIds.contains(kFileValueStreamId));

  EXPECT_EQ(tablet->resolveDictionaryAlphabet(kStripeValueStreamId), nullptr);
  const auto alphabet = tablet->resolveDictionaryAlphabet(kFileValueStreamId);
  ASSERT_NE(alphabet, nullptr);
  EXPECT_EQ(alphabet->entryCount(), values.size());
  EXPECT_EQ(alphabet->physicalValueAt<int32_t>(1), 20);
  EXPECT_EQ(tablet->resolveDictionaryAlphabet(kFileValueStreamId), alphabet);
  EXPECT_EQ(
      tablet->resolveDictionaryAlphabet(kSecondFileValueStreamId), alphabet);
}

TEST_F(SharedDictionaryReaderTest, rejectsFileDictionaryOutsideTablet) {
  constexpr uint32_t kValueStreamId = 20;
  constexpr uint32_t kDictionaryId = 7;
  const std::array fileReferences{SharedDictionaryReference{
      .valueStreamId = kValueStreamId,
      .dictionaryId = kDictionaryId,
      .dataType = DataType::Int32}};
  const std::array fileDictionaries{FileDictionary{
      .dictionaryId = kDictionaryId,
      .dataType = DataType::Int32,
      .offset = std::numeric_limits<uint64_t>::max(),
      .length = 1}};
  auto tablet = createTablet(
      SharedDictionaryCatalog::serialize(
          {}, fileReferences, {}, fileDictionaries));

  NIMBLE_ASSERT_FILE_THROW(
      tablet->resolveDictionaryAlphabet(kValueStreamId),
      "File shared dictionary 7 points outside the tablet.");
}

TEST_F(SharedDictionaryReaderTest, cachesFileAlphabet) {
  constexpr uint32_t kValueStreamId = 20;
  constexpr uint32_t kSecondValueStreamId = 21;
  constexpr uint32_t kDictionaryId = 7;
  const std::array<int32_t, 2> values{40, 50};
  const auto fileEncodedAlphabet = encodeAlphabet(values);
  const std::array fileDictionaryReferences{
      SharedDictionaryReference{
          .valueStreamId = kValueStreamId,
          .dictionaryId = kDictionaryId,
          .dataType = DataType::Int32},
      SharedDictionaryReference{
          .valueStreamId = kSecondValueStreamId,
          .dictionaryId = kDictionaryId,
          .dataType = DataType::Int32}};

  const auto [tablet, readFile] =
      createTabletWithSharedDictionarySectionAndReadFile(
          std::span<const SharedDictionaryReference>{},
          fileDictionaryReferences,
          kDictionaryId,
          fileEncodedAlphabet);

  EXPECT_FALSE(tablet->hasStripeDictionaries());
  EXPECT_TRUE(tablet->hasFileOrExternalDictionaries());
  readFile->resetChunks();
  const auto alphabet = tablet->resolveDictionaryAlphabet(kValueStreamId);
  ASSERT_NE(alphabet, nullptr);
  const auto readsAfterFirstResolve = readFile->chunks().size();
  EXPECT_GT(readsAfterFirstResolve, 0);
  EXPECT_EQ(alphabet->physicalValueAt<int32_t>(0), 40);
  EXPECT_EQ(tablet->resolveDictionaryAlphabet(kValueStreamId), alphabet);
  EXPECT_EQ(tablet->resolveDictionaryAlphabet(kSecondValueStreamId), alphabet);
  EXPECT_EQ(readFile->chunks().size(), readsAfterFirstResolve);
}

TEST_F(SharedDictionaryReaderTest, rejectsMissingExternalResolver) {
  constexpr uint32_t kValueStreamId = 30;
  constexpr uint32_t kDictionaryId = 11;
  const std::array externalReferences{SharedDictionaryReference{
      .valueStreamId = kValueStreamId,
      .dictionaryId = kDictionaryId,
      .dataType = DataType::Int32}};
  auto tablet = createTablet(
      SharedDictionaryCatalog::serialize({}, {}, externalReferences, {}));

  NIMBLE_ASSERT_USER_THROW(
      tablet->resolveDictionaryAlphabet(kValueStreamId),
      "External shared dictionary 11 requires an ExternalDictionaryResolver.");
}

TEST_F(SharedDictionaryReaderTest, cachesExternalAlphabet) {
  constexpr uint32_t kValueStreamId = 30;
  constexpr uint32_t kDictionaryId = 11;
  const std::array externalDictionaryReferences{SharedDictionaryReference{
      .valueStreamId = kValueStreamId,
      .dictionaryId = kDictionaryId,
      .dataType = DataType::Int32}};
  const std::array<int32_t, 2> values{40, 50};
  auto resolver =
      std::make_shared<TestDictionaryResolver>(createAlphabet(values));
  const auto catalog = SharedDictionaryCatalog::serialize(
      {}, {}, externalDictionaryReferences, {});
  auto tablet = createTablet(catalog, resolver);

  EXPECT_FALSE(tablet->hasStripeDictionaries());
  EXPECT_TRUE(tablet->hasFileOrExternalDictionaries());
  const auto alphabet = tablet->resolveDictionaryAlphabet(kValueStreamId);
  ASSERT_NE(alphabet, nullptr);
  EXPECT_EQ(alphabet->physicalValueAt<int32_t>(0), 40);
  EXPECT_EQ(tablet->resolveDictionaryAlphabet(kValueStreamId), alphabet);
  EXPECT_EQ(resolver->resolveCount(), 1);
  EXPECT_EQ(resolver->dictionaryId(), kDictionaryId);
  EXPECT_EQ(resolver->dataType(), DataType::Int32);
}

} // namespace
} // namespace facebook::nimble
