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
#include "velox/exec/Spill.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/tests/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::filesystems;

class SpillTest : public testing::Test,
                  public facebook::velox::test::VectorTestBase {
 protected:
  void SetUp() override {
    mappedMemory_ = memory::MappedMemory::getInstance();
    if (!isRegisteredVectorSerde()) {
      facebook::velox::serializer::presto::PrestoVectorSerde::
          registerVectorSerde();
    }
    filesystems::registerLocalFileSystem();
  }

  void spillStateTest(
      int64_t targetFileSize,
      int numPartitions,
      int numBatches,
      int64_t expectedNumSpilledFiles) {
    auto tempDirectory = exec::test::TempDirectoryPath::create();
    const std::string kSpillPath = tempDirectory->path + "/test";
    std::shared_ptr<FileSystem> fs;
    std::vector<std::string> spilledFiles;
    const int kNumRowsPerBatch = 100'000;
    {
      // We make a state that has 'numPartitions' partitions, each with its own
      // file list. We write 'numBatches' sorted vectors in each partition. The
      // vectors have the ith element = i * 'numBatches' + batch, where batch is
      // the batch number of the vector in the partition. When read back, both
      // partitions produce an ascending sequence of integers without gaps.
      SpillState state(
          kSpillPath,
          numPartitions,
          1,
          targetFileSize,
          *pool(),
          *mappedMemory_);
      EXPECT_EQ(targetFileSize, state.targetFileSize());
      EXPECT_EQ(numPartitions, state.maxPartitions());

      for (auto partition = 0; partition < state.maxPartitions(); ++partition) {
        EXPECT_FALSE(state.isPartitionSpilled(partition));
        // Expect an exception if partition is not set to spill.
        {
          RowVectorPtr dummyInput;
          EXPECT_ANY_THROW(state.appendToPartition(partition, dummyInput));
        }
        state.setPartitionSpilled(partition);
        EXPECT_TRUE(state.isPartitionSpilled(partition));
        EXPECT_FALSE(state.hasFiles(partition));
        for (auto batch = 0; batch < numBatches; ++batch) {
          // We add a sorted run in two pieces: 1, 11, 21,,, followed by X00001
          // , X00011, X00021   etc. where the last digit is the batch number.
          // Each sorted run has 20000 rows.
          state.appendToPartition(
              partition,
              makeRowVector(
                  {makeFlatVector<int64_t>(kNumRowsPerBatch, [&](auto row) {
                    return row * numBatches + batch;
                  })}));
          EXPECT_TRUE(state.hasFiles(partition));

          state.appendToPartition(
              partition,
              makeRowVector(
                  {makeFlatVector<int64_t>(kNumRowsPerBatch, [&](auto row) {
                    return row * numBatches + batch +
                        kNumRowsPerBatch * numBatches;
                  })}));
          EXPECT_TRUE(state.hasFiles(partition));

          // Indicates that the next additions to 'partition' are not sorted
          // with respect to the values added so far.
          state.finishWrite(partition);
          EXPECT_TRUE(state.hasFiles(partition));
        }
      }
      EXPECT_LT(
          numPartitions * numBatches * sizeof(int64_t), state.spilledBytes());
      EXPECT_EQ(expectedNumSpilledFiles, state.spilledFiles());
      spilledFiles = state.TEST_spilledFiles();
      std::unordered_set<std::string> spilledFileSet(
          spilledFiles.begin(), spilledFiles.end());
      EXPECT_EQ(spilledFileSet.size(), spilledFiles.size());
      EXPECT_EQ(expectedNumSpilledFiles, spilledFileSet.size());
      // Verify the spilled file exist on file system.
      fs = filesystems::getFileSystem(tempDirectory->path, nullptr);
      for (const auto& spilledFile : spilledFileSet) {
        auto readFile = fs->openFileForRead(spilledFile);
        EXPECT_NE(readFile.get(), nullptr);
      }

      for (auto partition = 0; partition < state.maxPartitions(); ++partition) {
        auto merge = state.startMerge(partition, nullptr);
        // We expect all the rows in dense increasing order.
        for (auto i = 0; i < numPartitions * numBatches * kNumRowsPerBatch;
             ++i) {
          auto stream = merge->next();
          ASSERT_NE(nullptr, stream);
          EXPECT_EQ(
              i,
              stream->current()
                  .childAt(0)
                  ->asUnchecked<FlatVector<int64_t>>()
                  ->valueAt(stream->currentIndex()));
          EXPECT_EQ(
              i, stream->decoded(0).valueAt<int64_t>(stream->currentIndex()));

          stream->pop();
        }
        ASSERT_EQ(nullptr, merge->next());
      }
      // Both spilled bytes and files stats are cleared after merge read.
      EXPECT_EQ(0, state.spilledBytes());
      EXPECT_EQ(0, state.spilledFiles());
    }
    // Verify the spilled file has been removed from file system after spill
    // state destruction.
    for (const auto& spilledFile : spilledFiles) {
      EXPECT_ANY_THROW(fs->openFileForRead(spilledFile));
    }
  }

  memory::MappedMemory* mappedMemory_;
};

TEST_F(SpillTest, spillState) {
  // Set the target file size to a large value to avoid new file creation
  // triggered by batch write.
  spillStateTest(1'000'000'000, 2, 10, 2 * 10);
}

TEST_F(SpillTest, spillStateWithSmallTargetFileSize) {
  // Set the target file size to a small value to open a new file on each batch
  // write.
  spillStateTest(1, 2, 10, 2 * 10 * 2);
}
