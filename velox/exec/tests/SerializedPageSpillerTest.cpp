/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "velox/exec/SerializedPageSpiller.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/SerializedPageUtil.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/exec/tests/utils/TempFilePath.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
namespace facebook::velox::exec::test {
class SerializedPageSpillerTest : public exec::test::OperatorTestBase {
 public:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    rng_.seed(0);
  }

 protected:
  std::vector<std::shared_ptr<SerializedPage>> generateData(
      uint32_t numPages,
      int64_t maxPageSize,
      bool hasVoidedNumRows,
      int64_t maxNumRows) {
    std::vector<std::shared_ptr<SerializedPage>> pages;
    pages.reserve(numPages);
    for (auto i = 0; i < numPages; ++i) {
      auto iobufBytes = folly::Random().rand64(maxPageSize, rng_);

      // Setup a chained iobuf.
      std::unique_ptr<folly::IOBuf> iobuf;
      if (iobufBytes > 1) {
        auto firstHalfBytes = iobufBytes / 2;
        iobuf = folly::IOBuf::create(firstHalfBytes);
        std::memset(iobuf->writableData(), 'x', firstHalfBytes);
        iobuf->append(firstHalfBytes);

        auto secondHalfBytes = iobufBytes - firstHalfBytes;
        auto secondHalfBuf = folly::IOBuf::create(secondHalfBytes);
        std::memset(secondHalfBuf->writableData(), 'y', secondHalfBytes);
        secondHalfBuf->append(secondHalfBytes);
        iobuf->prependChain(std::move(secondHalfBuf));
      } else {
        iobuf = folly::IOBuf::create(iobufBytes);
        std::memset(iobuf->writableData(), 'x', iobufBytes);
        iobuf->append(iobufBytes);
      }

      std::optional<int64_t> numRowsOpt;
      if (!hasVoidedNumRows || folly::Random().oneIn(2, rng_)) {
        numRowsOpt = std::optional(folly::Random().rand64(maxNumRows, rng_));
      }
      pages.push_back(std::make_shared<SerializedPage>(
          std::move(iobuf), nullptr, numRowsOpt));
    }
    return pages;
  }

  void checkIOBufsEqual(
      std::unique_ptr<folly::IOBuf>& buf1,
      std::unique_ptr<folly::IOBuf>& buf2) {
    auto coalescedBuf1 = buf1->coalesce();
    auto coalescedBuf2 = buf2->coalesce();
    ASSERT_EQ(coalescedBuf1.size(), coalescedBuf2.size());
    ASSERT_EQ(
        std::memcmp(
            coalescedBuf1.data(), coalescedBuf2.data(), coalescedBuf1.size()),
        0);
  }

  void checkSerializedPageEqual(SerializedPage& page1, SerializedPage& page2) {
    ASSERT_EQ(page1.numRows().has_value(), page2.numRows().has_value());
    if (page1.numRows().has_value()) {
      ASSERT_EQ(page1.numRows().value(), page1.numRows().value());
    }
    auto buf1 = page1.getIOBuf();
    auto buf2 = page2.getIOBuf();
    checkIOBufsEqual(buf1, buf2);
  }

  void checkSpillerConsistency(SerializedPageSpiller& spiller) {
    ASSERT_LE(spiller.spillFilePaths_.size(), spiller.nextFileId_ + 1);
    const auto spillStatsRPtr = spiller.spillStats_->rlock();
    ASSERT_EQ(spiller.totalBytes_, spillStatsRPtr->spilledBytes);
    ASSERT_EQ(spiller.totalBytes_, spillStatsRPtr->spilledInputBytes);
    ASSERT_EQ(spiller.spillFilePaths_.size(), spillStatsRPtr->spillRuns);
    ASSERT_EQ(spiller.spillFilePaths_.size(), spillStatsRPtr->spilledFiles);
    ASSERT_EQ(spiller.spillFilePaths_.size(), spillStatsRPtr->spillWrites);
    if (spiller.totalBytes_ > 0) {
      ASSERT_GT(spillStatsRPtr->spillWriteTimeNanos, 0);
    }
  }

  void checkReaderConsistency(SerializedPageSpillReader& reader) {
    ASSERT_GE(reader.remainingPages_, reader.bufferedPages_.size());
    if (reader.remainingPages_ == 0 || reader.remainingBytes_ == 0) {
      ASSERT_EQ(reader.remainingBytes_, reader.remainingPages_);
    }
    uint64_t bufferedBytes{0};
    for (int32_t i = 0; i < reader.bufferedPages_.size(); ++i) {
      const auto& page = reader.bufferedPages_[i];
      if (page == nullptr) {
        continue;
      }
      bufferedBytes += page->size();
    }
    ASSERT_GE(reader.remainingBytes_, bufferedBytes);
  }

  void assertNumBufferedPages(
      SerializedPageSpillReader& reader,
      uint64_t numPages) {
    ASSERT_EQ(reader.bufferedPages_.size(), numPages);
  }

  folly::Random::DefaultGenerator rng_;
};
} // namespace facebook::velox::exec::test

TEST_F(SerializedPageSpillerTest, pageSpillerBasic) {
  auto pool = rootPool_->addLeafChild("destinationBufferSpiller");

  struct TestValue {
    uint32_t numPages;
    int64_t maxPageSize;
    bool hasVoidedNumRows;
    int64_t maxNumRows;
    uint64_t readBufferSize;
    uint64_t writeBufferSize;

    std::string debugString() const {
      return fmt::format(
          "numPages {}, maxPageSize {}, hasVoidedNumRows {}, maxNumRows {}, "
          "readBufferSize {}, writeBufferSize {}",
          numPages,
          maxPageSize,
          hasVoidedNumRows,
          maxNumRows,
          readBufferSize,
          writeBufferSize);
    }
  };

  std::vector<TestValue> testValues{
      {10, 64, true, 20, 1024, 1024},
      {10, 64, false, 20, 1024, 0},
      {0, 64, true, 20, 1024, 256},
      {10, 64, true, 20, 1, 2048},
      {10, 0, true, 20, 128, 2048}};

  for (const auto& testValue : testValues) {
    SCOPED_TRACE(testValue.debugString());

    auto tempFile = exec::test::TempFilePath::create();
    const auto& prefixPath = tempFile->getPath();
    auto fs = filesystems::getFileSystem(prefixPath, {});
    SCOPE_EXIT {
      fs->remove(prefixPath);
    };

    auto pages = generateData(
        testValue.numPages,
        testValue.maxPageSize,
        testValue.hasVoidedNumRows,
        testValue.maxNumRows);

    folly::Synchronized<common::SpillStats> spillStats;
    SerializedPageSpiller spiller(
        prefixPath, "", testValue.writeBufferSize, pool.get(), &spillStats);
    checkSpillerConsistency(spiller);

    spiller.spill(&pages);
    checkSpillerConsistency(spiller);

    auto spillResults = spiller.finishSpill();

    SerializedPageSpillReader reader(
        std::move(spillResults),
        testValue.readBufferSize,
        pool.get(),
        &spillStats);

    ASSERT_EQ(reader.empty(), pages.empty());
    ASSERT_EQ(reader.remainingPages(), pages.size());

    VELOX_ASSERT_THROW(reader.at(pages.size()), "");
    checkReaderConsistency(reader);

    VELOX_ASSERT_THROW(reader.isEmptyAt(pages.size()), "");
    checkReaderConsistency(reader);

    VELOX_ASSERT_THROW(reader.sizeAt(pages.size()), "");
    checkReaderConsistency(reader);

    VELOX_ASSERT_THROW(reader.deleteFront(pages.size() + 1), "");
    checkReaderConsistency(reader);

    if (pages.empty()) {
      ASSERT_TRUE(reader.empty());
      continue;
    }

    ASSERT_FALSE(reader.empty());
    uint32_t i = 0;
    while (!reader.empty()) {
      ASSERT_EQ(reader.isEmptyAt(0), pages[i] == nullptr);
      if (reader.isEmptyAt(0)) {
        VELOX_ASSERT_THROW(reader.sizeAt(0), "");
      } else {
        ASSERT_EQ(reader.sizeAt(0), pages[i]->size());
      }
      checkReaderConsistency(reader);

      auto unspilledPage = reader.at(0);
      checkReaderConsistency(reader);

      ASSERT_LT(i, pages.size());
      ASSERT_EQ(unspilledPage->numRows(), pages[i]->numRows());
      ASSERT_EQ(unspilledPage->size(), pages[i]->size());
      auto originalIOBuf = pages[i]->getIOBuf();
      auto unspilledIOBuf = unspilledPage->getIOBuf();
      checkIOBufsEqual(originalIOBuf, unspilledIOBuf);
      if (testValue.maxPageSize == 0) {
        ASSERT_GE(pool->usedBytes(), 0);
      } else {
        ASSERT_GT(pool->usedBytes(), 0);
      }
      reader.deleteFront(1);
      ++i;
    }
    ASSERT_EQ(i, pages.size());
    ASSERT_TRUE(reader.empty());
    ASSERT_EQ(reader.remainingPages(), 0);

    VELOX_ASSERT_THROW(reader.at(0), "");
    checkReaderConsistency(reader);

    VELOX_ASSERT_THROW(reader.isEmptyAt(0), "");
    checkReaderConsistency(reader);

    VELOX_ASSERT_THROW(reader.sizeAt(0), "");
    checkReaderConsistency(reader);

    VELOX_ASSERT_THROW(reader.deleteFront(1), "");
    checkReaderConsistency(reader);

    ASSERT_NO_THROW(reader.deleteAll());
    checkReaderConsistency(reader);
  }
  ASSERT_EQ(pool->usedBytes(), 0);
}

TEST_F(SerializedPageSpillerTest, spillReaderAccessors) {
  auto pool = rootPool_->addLeafChild("spillReaderAccessors");
  auto pages = generateData(20, 1LL << 20, true, 1000);

  struct TestValue {
    std::string testName;
    std::function<void(
        std::vector<std::shared_ptr<SerializedPage>>&,
        SerializedPageSpillReader&,
        uint64_t)>
        accessorVerifier;
    std::string debugString() {
      return testName;
    }
  };

  std::vector<TestValue> testValues{
      {"SerializedPageSpillReader::at",
       [this](auto& originalPages, auto& reader, auto index) {
         // Accessor verifier for SerializedPageSpillReader::at()
         if (index >= originalPages.size()) {
           VELOX_ASSERT_THROW(reader.at(index), "");
           return;
         }
         auto originalPage = originalPages[index];
         auto unspilledPage = reader.at(index);
         if (originalPage == nullptr) {
           ASSERT_EQ(unspilledPage, nullptr);
           return;
         }
         ASSERT_EQ(originalPage->size(), unspilledPage->size());
         ASSERT_EQ(originalPage->numRows(), unspilledPage->numRows());
         auto originalIOBuf = originalPage->getIOBuf();
         auto unspilledIOBuf = unspilledPage->getIOBuf();
         checkIOBufsEqual(originalIOBuf, unspilledIOBuf);
       }},
      {"SerializedPageSpillReader::isEmptyAt",
       [this](auto& originalPages, auto& reader, auto index) {
         // Accessor verifier for SerializedPageSpillReader::isEmptyAt()
         if (index >= originalPages.size()) {
           VELOX_ASSERT_THROW(reader.isEmptyAt(index), "");
           return;
         }
         ASSERT_EQ(originalPages[index] == nullptr, reader.isEmptyAt(index));
       }},
      {"SerializedPageSpillReader::sizeAt",
       [this](auto& originalPages, auto& reader, auto index) {
         // Accessor verifier for SerializedPageSpillReader::sizeAt()
         if (index >= originalPages.size()) {
           VELOX_ASSERT_THROW(reader.sizeAt(index), "");
           return;
         }
         auto originalPage = originalPages[index];
         if (originalPage == nullptr) {
           VELOX_ASSERT_THROW(reader.sizeAt(index), "");
           return;
         }
         ASSERT_EQ(originalPage->size(), reader.sizeAt(index));
       }}};

  for (auto& testValue : testValues) {
    SCOPED_TRACE(testValue.debugString());
    auto tempFile = exec::test::TempFilePath::create();
    const auto& prefixPath = tempFile->getPath();
    auto fs = filesystems::getFileSystem(prefixPath, {});
    SCOPE_EXIT {
      fs->remove(prefixPath);
    };

    folly::Synchronized<common::SpillStats> spillStats;
    SerializedPageSpiller spiller(
        prefixPath, "", 1024, pool.get(), &spillStats);
    spiller.spill(&pages);
    checkSpillerConsistency(spiller);
    auto spillResults = spiller.finishSpill();

    SerializedPageSpillReader reader(
        std::move(spillResults), 1024, pool.get(), &spillStats);

    testValue.accessorVerifier(pages, reader, 1);
    checkReaderConsistency(reader);
    assertNumBufferedPages(reader, 2);

    testValue.accessorVerifier(pages, reader, 10);
    checkReaderConsistency(reader);
    assertNumBufferedPages(reader, 11);

    testValue.accessorVerifier(pages, reader, 25);
    checkReaderConsistency(reader);
    assertNumBufferedPages(reader, 11);

    testValue.accessorVerifier(pages, reader, 19);
    checkReaderConsistency(reader);
    assertNumBufferedPages(reader, 20);

    testValue.accessorVerifier(pages, reader, 5);
    checkReaderConsistency(reader);
    assertNumBufferedPages(reader, 20);
  }
  ASSERT_EQ(pool->usedBytes(), 0);
}

TEST_F(SerializedPageSpillerTest, spillReaderDelete) {
  auto pool = rootPool_->addLeafChild("spillReaderDelete");
  const auto kNumPages = 20;
  auto pages = generateData(kNumPages, 1LL << 20, true, 1000);

  struct TestValue {
    uint32_t numBufferedPages;
    uint32_t numDelete;

    std::string debugString() {
      return fmt::format(
          "numBufferedPages {}, numDelete {}", numBufferedPages, numDelete);
    }
  };

  std::vector<TestValue> testValues{
      {0, 0},
      {0, 10},
      {0, 20},
      {0, 25},
      {10, 0},
      {10, 5},
      {10, 15},
      {10, 20},
      {10, 25}};
  for (auto& testValue : testValues) {
    SCOPED_TRACE(testValue.debugString());
    // Test delete front.
    auto tempFile = exec::test::TempFilePath::create();
    const auto& prefixPath = tempFile->getPath();
    auto fs = filesystems::getFileSystem(prefixPath, {});
    SCOPE_EXIT {
      fs->remove(prefixPath);
    };

    folly::Synchronized<common::SpillStats> spillStats;
    SerializedPageSpiller spiller(
        prefixPath, "", 1024, pool.get(), &spillStats);
    spiller.spill(&pages);
    checkSpillerConsistency(spiller);
    auto spillResult = spiller.finishSpill();

    SerializedPageSpillReader reader(
        std::move(spillResult), 1024, pool.get(), &spillStats);

    // Unspill pages to buffer
    if (testValue.numBufferedPages > 0) {
      reader.at(testValue.numBufferedPages - 1);
    }
    checkReaderConsistency(reader);
    assertNumBufferedPages(reader, testValue.numBufferedPages);

    if (testValue.numDelete > kNumPages) {
      VELOX_ASSERT_THROW(reader.deleteFront(testValue.numDelete), "");
      checkReaderConsistency(reader);
      assertNumBufferedPages(reader, testValue.numBufferedPages);
      continue;
    } else {
      reader.deleteFront(testValue.numDelete);
    }
    checkReaderConsistency(reader);
    if (testValue.numDelete <= testValue.numBufferedPages) {
      assertNumBufferedPages(
          reader, (testValue.numBufferedPages - testValue.numDelete));
    } else {
      assertNumBufferedPages(reader, 0);
    }
  }

  {
    // Test delete all.
    auto tempFile = exec::test::TempFilePath::create();
    const auto& prefixPath = tempFile->getPath();
    auto fs = filesystems::getFileSystem(prefixPath, {});
    SCOPE_EXIT {
      fs->remove(prefixPath);
    };

    folly::Synchronized<common::SpillStats> spillStats;
    SerializedPageSpiller spiller(
        prefixPath, "", 1024, pool.get(), &spillStats);
    spiller.spill(&pages);
    checkSpillerConsistency(spiller);
    auto spillResult = spiller.finishSpill();

    SerializedPageSpillReader reader(
        std::move(spillResult), 1024, pool.get(), &spillStats);
    reader.at(10);

    reader.deleteAll();
    checkReaderConsistency(reader);
    assertNumBufferedPages(reader, 0);
  }
}
