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

#include "velox/connectors/hive/storage_adapters/hdfs/HdfsReadFile.h"

#include <cstring>
#include <string>

#include "gtest/gtest.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/external/hdfs/ArrowHdfsInternal.h"

// Deterministic coverage for the transient-read-failure retry path in
// HdfsReadFile::Impl::preadInternal, without a live HDFS cluster.
//
// LibHdfsShim dispatches every libhdfs3 call through a plain function pointer
// (e.g. this->hdfsRead(...)), and HdfsReadFile takes the shim plus an opaque
// hdfsFS by pointer. So we can hand it a shim whose pointers target the stubs
// below and drive Read() to fail a fixed number of times before succeeding --
// something a real MiniCluster (which can only be up or down) cannot do
// reproducibly.
//
// Retries are opt-in via the maxReadAttempts constructor argument. The tests
// that exercise the retry loop pass a small non-default value; a fast base
// delay (kTestRetryDelayMs) keeps the backoff sleeps negligible.

namespace facebook::velox {
namespace {

using filesystems::arrow::io::internal::LibHdfsShim;

// hdfsFS / hdfsFile are opaque pointers; the stubs never dereference them, so
// any non-null value works.
hdfsFS kFakeFs = reinterpret_cast<hdfsFS>(0x1);
hdfsFile kFakeHandle = reinterpret_cast<hdfsFile>(0x2);

// Small backoff base so the retry tests don't actually sleep for long.
constexpr int kTestRetryDelayMs = 1;

// libhdfs3 uses C function pointers, which cannot capture state, so the stub
// behaviour is driven through translation-unit-local variables. The fixture
// resets all of them before every test.
tSize gFileSize;
int gReadCalls; // number of times stubRead was entered
int gFailCount; // first gFailCount reads fail (return gFailReturn)
tSize gFailReturn; // -1 (libhdfs3 error) or 0 (no progress)
tSize gMaxChunk; // >0 caps the bytes returned by a successful read (short read)
hdfsFileInfo gFileInfo;

hdfsFileInfo* stubGetPathInfo(hdfsFS, const char*) {
  gFileInfo = {};
  gFileInfo.mSize = gFileSize;
  gFileInfo.mBlockSize = gFileSize;
  return &gFileInfo;
}

// gFileInfo is a static, so there is nothing to free.
void stubFreeFileInfo(hdfsFileInfo*, int) {}

hdfsFile stubOpenFile(hdfsFS, const char*, int, int, short, tSize) { // NOLINT
  return kFakeHandle;
}

int stubCloseFile(hdfsFS, hdfsFile) {
  return 0;
}

int stubSeek(hdfsFS, hdfsFile, tOffset) {
  return 0;
}

tSize stubRead(hdfsFS, hdfsFile, void* buffer, tSize length) {
  if (gReadCalls++ < gFailCount) {
    return gFailReturn;
  }
  const tSize n = (gMaxChunk > 0 && gMaxChunk < length) ? gMaxChunk : length;
  std::memset(buffer, 'x', n);
  return n;
}

char* stubGetLastExceptionRootCause() {
  return strdup("mock transient failure");
}

class HdfsReadFileRetryTest : public testing::Test {
 protected:
  void SetUp() override {
    gFileSize = 1024;
    gReadCalls = 0;
    gFailCount = 0;
    gFailReturn = -1;
    gMaxChunk = 0;

    shim_.Initialize();
    shim_.hdfsGetPathInfo = stubGetPathInfo;
    shim_.hdfsFreeFileInfo = stubFreeFileInfo;
    shim_.hdfsOpenFile = stubOpenFile;
    shim_.hdfsCloseFile = stubCloseFile;
    shim_.hdfsSeek = stubSeek;
    shim_.hdfsRead = stubRead;
    shim_.hdfsGetLastExceptionRootCause = stubGetLastExceptionRootCause;
  }

  LibHdfsShim shim_;
};

// With retries disabled (maxReadAttempts == 1, the default), the very first
// transient failure throws immediately and Read() is attempted exactly once.
// This pins "default == original fail-fast behavior" as a regression guard: the
// default behavior is a no-op unless a caller explicitly opts in.
TEST_F(HdfsReadFileRetryTest, failFastByDefault) {
  gFailCount = 1;
  gFailReturn = -1;

  HdfsReadFile readFile(&shim_, kFakeFs, "/mock");
  VELOX_ASSERT_THROW(readFile.pread(0, gFileSize), "after 1 attempts");
  EXPECT_EQ(gReadCalls, 1);
}

// A non-positive maxReadAttempts (e.g. a misconfigured
// hive.hdfs.read-max-attempts) is rejected at construction with a user error,
// rather than silently producing a confusing "after 0 attempts" message on the
// first read failure.
TEST_F(HdfsReadFileRetryTest, rejectsNonPositiveMaxAttempts) {
  VELOX_ASSERT_THROW(
      HdfsReadFile(&shim_, kFakeFs, "/mock", /*maxReadAttempts=*/0),
      "hive.hdfs.read-max-attempts must be at least 1");
}

// Two transient errors, then success: the read recovers and returns the full
// payload. Exactly three Read() calls (2 failed + 1 success) confirms the retry
// loop re-issued the read rather than giving up or double-counting.
TEST_F(HdfsReadFileRetryTest, recoversAfterTransientNegativeFailures) {
  gFailCount = 2;
  gFailReturn = -1;

  HdfsReadFile readFile(
      &shim_, kFakeFs, "/mock", /*maxReadAttempts=*/4, kTestRetryDelayMs);
  const auto data = readFile.pread(0, gFileSize);

  EXPECT_EQ(data, std::string(gFileSize, 'x'));
  EXPECT_EQ(gReadCalls, 3);
}

// A zero return means the read made no progress. It must be retried just like a
// negative return; treating it as success would spin the while loop forever.
// This pins the `bytesRead <= 0` predicate (not `< 0`).
TEST_F(HdfsReadFileRetryTest, retriesOnZeroReturn) {
  gFailCount = 2;
  gFailReturn = 0;

  HdfsReadFile readFile(
      &shim_, kFakeFs, "/mock", /*maxReadAttempts=*/4, kTestRetryDelayMs);
  const auto data = readFile.pread(0, gFileSize);

  EXPECT_EQ(data, std::string(gFileSize, 'x'));
  EXPECT_EQ(gReadCalls, 3);
}

// A persistent failure throws after the retry budget is exhausted. With
// maxReadAttempts == 4, Read() is attempted 4 times in total (1 initial + 3
// retries) before the final throw, and the message reports it.
TEST_F(HdfsReadFileRetryTest, throwsAfterExhaustion) {
  gFailCount = 1000; // always fails
  gFailReturn = -1;

  HdfsReadFile readFile(
      &shim_, kFakeFs, "/mock", /*maxReadAttempts=*/4, kTestRetryDelayMs);
  VELOX_ASSERT_THROW(readFile.pread(0, gFileSize), "after 4 attempts");
  EXPECT_EQ(gReadCalls, 4);
}

// A successful but short read is not a failure: preadInternal must keep the
// bytes and loop until the request is filled, without consuming any retries.
// maxReadAttempts == 1 proves short reads never touch the retry budget.
TEST_F(HdfsReadFileRetryTest, shortReadsAccumulateWithoutRetry) {
  gFailCount = 0;
  gMaxChunk = 256; // 1024 / 256 = 4 short reads

  HdfsReadFile readFile(&shim_, kFakeFs, "/mock");
  const auto data = readFile.pread(0, gFileSize);

  EXPECT_EQ(data, std::string(gFileSize, 'x'));
  EXPECT_EQ(gReadCalls, 4);
}

} // namespace
} // namespace facebook::velox
