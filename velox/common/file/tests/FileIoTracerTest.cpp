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

#include "velox/common/file/FileIoTracer.h"

#include <random>
#include <thread>
#include <unordered_map>

#include "gtest/gtest.h"

using namespace facebook::velox;

class FileIoTracerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Reset thread-local tag before each test
    threadIoTag() = nullptr;
  }

  void TearDown() override {
    // Ensure thread-local tag is reset after each test
    threadIoTag() = nullptr;
  }
};

TEST_F(FileIoTracerTest, ioTag) {
  {
    IoTag tag("TestTag");
    EXPECT_EQ(tag.name, "TestTag");
    EXPECT_EQ(tag.parent, nullptr);
    EXPECT_EQ(tag.toString(), "TestTag");
    EXPECT_EQ(tag.depth(), 1);
  }

  {
    IoTag parent("Parent");
    IoTag child("Child", &parent);

    EXPECT_EQ(child.name, "Child");
    EXPECT_EQ(child.parent, &parent);
    EXPECT_EQ(child.toString(), "Parent -> Child");
    EXPECT_EQ(child.depth(), 2);
  }

  {
    IoTag tag1("Level1");
    IoTag tag2("Level2", &tag1);
    IoTag tag3("Level3", &tag2);
    IoTag tag4("Level4", &tag3);

    EXPECT_EQ(tag4.toString(), "Level1 -> Level2 -> Level3 -> Level4");
    EXPECT_EQ(tag4.depth(), 4);
    EXPECT_EQ(tag3.depth(), 3);
    EXPECT_EQ(tag2.depth(), 2);
    EXPECT_EQ(tag1.depth(), 1);
  }
}

TEST_F(FileIoTracerTest, threadIoTag) {
  {
    EXPECT_EQ(threadIoTag(), nullptr);
  }

  {
    EXPECT_EQ(threadIoTag(), nullptr);

    {
      ScopedIoTag tag1("Tag1");
      EXPECT_NE(threadIoTag(), nullptr);
      EXPECT_EQ(threadIoTag()->name, "Tag1");
      EXPECT_EQ(threadIoTag()->parent, nullptr);
      EXPECT_EQ(threadIoTag()->toString(), "Tag1");
    }

    EXPECT_EQ(threadIoTag(), nullptr);
  }

  {
    EXPECT_EQ(threadIoTag(), nullptr);

    {
      ScopedIoTag tag1("TableScan");
      EXPECT_EQ(threadIoTag()->toString(), "TableScan");
      EXPECT_EQ(threadIoTag()->depth(), 1);

      {
        ScopedIoTag tag2("ColumnReader");
        EXPECT_EQ(threadIoTag()->toString(), "TableScan -> ColumnReader");
        EXPECT_EQ(threadIoTag()->depth(), 2);

        {
          ScopedIoTag tag3("PrefixEncoding");
          EXPECT_EQ(
              threadIoTag()->toString(),
              "TableScan -> ColumnReader -> PrefixEncoding");
          EXPECT_EQ(threadIoTag()->depth(), 3);
        }

        EXPECT_EQ(threadIoTag()->toString(), "TableScan -> ColumnReader");
        EXPECT_EQ(threadIoTag()->depth(), 2);
      }

      EXPECT_EQ(threadIoTag()->toString(), "TableScan");
      EXPECT_EQ(threadIoTag()->depth(), 1);
    }

    EXPECT_EQ(threadIoTag(), nullptr);
  }

  {
    ScopedIoTag tag("TestTag");
    EXPECT_EQ(tag.tag().name, "TestTag");
    EXPECT_EQ(tag.tag().parent, nullptr);
  }
}

TEST_F(FileIoTracerTest, threadLocalIsolation) {
  // Verify that thread-local tags are isolated between threads
  std::atomic_bool stop{false};

  std::thread thread1([&]() {
    ScopedIoTag tag("Thread1Tag");
    while (!stop) {
      EXPECT_EQ(threadIoTag()->name, "Thread1Tag");
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });

  std::thread thread2([&]() {
    ScopedIoTag tag("Thread2Tag");
    while (!stop) {
      EXPECT_EQ(threadIoTag()->name, "Thread2Tag");
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });

  std::this_thread::sleep_for(std::chrono::seconds(2));
  stop = true;

  thread1.join();
  thread2.join();
}

TEST_F(FileIoTracerTest, inMemoryFileIoTracer) {
  {
    std::vector<IoRecord> records;
    auto tracer = InMemoryFileIoTracer::create(records);

    tracer->record(IoType::Read, 100, 200);
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(records[0].type, IoType::Read);
    EXPECT_EQ(records[0].offset, 100);
    EXPECT_EQ(records[0].length, 200);
    EXPECT_EQ(records[0].tag, "");
    tracer->finish();
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(records[0].type, IoType::Read);
    EXPECT_EQ(records[0].offset, 100);
    EXPECT_EQ(records[0].length, 200);
    EXPECT_EQ(records[0].tag, "");
  }

  {
    std::vector<IoRecord> records;
    auto tracer = InMemoryFileIoTracer::create(records);

    {
      ScopedIoTag tag1("TableScan");
      tracer->record(IoType::Read, 0, 100);

      {
        ScopedIoTag tag2("ColumnReader");
        tracer->record(IoType::AsyncRead, 100, 200);
      }

      tracer->record(IoType::Write, 300, 50);
    }

    tracer->record(IoType::AsyncWrite, 400, 75);

    ASSERT_EQ(records.size(), 4);

    EXPECT_EQ(records[0].type, IoType::Read);
    EXPECT_EQ(records[0].offset, 0);
    EXPECT_EQ(records[0].length, 100);
    EXPECT_EQ(records[0].tag, "TableScan");

    EXPECT_EQ(records[1].type, IoType::AsyncRead);
    EXPECT_EQ(records[1].offset, 100);
    EXPECT_EQ(records[1].length, 200);
    EXPECT_EQ(records[1].tag, "TableScan -> ColumnReader");

    EXPECT_EQ(records[2].type, IoType::Write);
    EXPECT_EQ(records[2].offset, 300);
    EXPECT_EQ(records[2].length, 50);
    EXPECT_EQ(records[2].tag, "TableScan");

    EXPECT_EQ(records[3].type, IoType::AsyncWrite);
    EXPECT_EQ(records[3].offset, 400);
    EXPECT_EQ(records[3].length, 75);
    EXPECT_EQ(records[3].tag, "");
  }
}

TEST_F(FileIoTracerTest, ioType) {
  // Verify enum values exist and are distinct
  EXPECT_NE(IoType::Read, IoType::AsyncRead);
  EXPECT_NE(IoType::Read, IoType::Write);
  EXPECT_NE(IoType::Read, IoType::AsyncWrite);
  EXPECT_NE(IoType::AsyncRead, IoType::Write);
  EXPECT_NE(IoType::AsyncRead, IoType::AsyncWrite);
  EXPECT_NE(IoType::Write, IoType::AsyncWrite);
}

TEST_F(FileIoTracerTest, ioRecord) {
  {
    IoRecord record;
    record.type = IoType::Read;
    record.offset = 1024;
    record.length = 4096;
    record.tag = "TableScan -> ColumnReader";

    EXPECT_EQ(record.toString(), "Read [1024, 4096] TableScan -> ColumnReader");
  }

  {
    IoRecord record;
    record.type = IoType::AsyncRead;
    record.offset = 0;
    record.length = 512;
    record.tag = "";

    EXPECT_EQ(record.toString(), "AsyncRead [0, 512]");
  }

  {
    IoRecord record;
    record.type = IoType::Write;
    record.offset = 100;
    record.length = 200;
    record.tag = "Writer";

    EXPECT_EQ(record.toString(), "Write [100, 200] Writer");
  }

  {
    IoRecord record;
    record.type = IoType::AsyncWrite;
    record.offset = 500;
    record.length = 1000;
    record.tag = "AsyncWriter";

    EXPECT_EQ(record.toString(), "AsyncWrite [500, 1000] AsyncWriter");
  }
}

TEST_F(FileIoTracerTest, inMemoryFileIoTracerFuzz) {
  constexpr int kNumThreads = 8;
  constexpr int kMaxRecordsPerThread = 1'000;

  std::vector<IoRecord> sharedRecords;
  auto tracer = InMemoryFileIoTracer::create(sharedRecords);

  std::vector<std::thread> threads;
  std::vector<std::vector<IoRecord>> perThreadRecords(kNumThreads);
  std::vector<IoType> ioTypes = {
      IoType::Read, IoType::AsyncRead, IoType::Write, IoType::AsyncWrite};

  threads.reserve(kNumThreads);
  for (int threadIdx = 0; threadIdx < kNumThreads; ++threadIdx) {
    threads.emplace_back([&, threadIdx]() {
      std::mt19937 rng(threadIdx);
      std::uniform_int_distribution<int> countDist(1, kMaxRecordsPerThread);
      std::uniform_int_distribution<uint64_t> offsetDist(0, 1000000);
      std::uniform_int_distribution<uint64_t> lengthDist(1, 10000);
      std::uniform_int_distribution<int> typeDist(0, 3);

      const std::string tagName = "Thread" + std::to_string(threadIdx);
      ScopedIoTag tag(tagName);

      const int numRecords = countDist(rng);
      for (int i = 0; i < numRecords; ++i) {
        IoRecord localRecord;
        localRecord.type = ioTypes[typeDist(rng)];
        localRecord.offset = offsetDist(rng);
        localRecord.length = lengthDist(rng);
        localRecord.tag = tagName;

        perThreadRecords[threadIdx].push_back(localRecord);
        tracer->record(
            localRecord.type, localRecord.offset, localRecord.length);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  tracer->finish();

  size_t expectedTotalRecords = 0;
  for (const auto& threadRecords : perThreadRecords) {
    expectedTotalRecords += threadRecords.size();
  }
  EXPECT_EQ(sharedRecords.size(), expectedTotalRecords);

  std::unordered_map<std::string, std::vector<IoRecord>> recordsByTag;
  for (const auto& record : sharedRecords) {
    recordsByTag[record.tag].push_back(record);
  }

  for (int threadIdx = 0; threadIdx < kNumThreads; ++threadIdx) {
    const std::string tagName = "Thread" + std::to_string(threadIdx);
    const auto& expected = perThreadRecords[threadIdx];
    const auto& actual = recordsByTag[tagName];

    ASSERT_EQ(actual.size(), expected.size())
        << "Thread " << threadIdx << " record count mismatch";

    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(actual[i].type, expected[i].type)
          << "Thread " << threadIdx << " record " << i << " type mismatch";
      EXPECT_EQ(actual[i].offset, expected[i].offset)
          << "Thread " << threadIdx << " record " << i << " offset mismatch";
      EXPECT_EQ(actual[i].length, expected[i].length)
          << "Thread " << threadIdx << " record " << i << " length mismatch";
      EXPECT_EQ(actual[i].tag, expected[i].tag)
          << "Thread " << threadIdx << " record " << i << " tag mismatch";
    }
  }
}
