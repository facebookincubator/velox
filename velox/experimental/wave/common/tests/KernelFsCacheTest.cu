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

#include "velox/experimental/wave/common/KernelFsCache.h"

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <unistd.h>
#include <filesystem>
#include <fstream>

#include "velox/experimental/wave/common/Cuda.h"
#include "velox/experimental/wave/common/GpuArena.h"

namespace facebook::velox::wave {
namespace {

namespace fs = std::filesystem;

// Sharing one cache directory between processes is the case these tests are
// about. A second process is modelled by a second KernelFsCache over the same
// directory: the two have no state in common except the files, which is
// exactly the relationship two processes have. Where a test needs the
// in-process kernel cache out of the way as well, it clears that too.
class KernelFsCacheTest : public testing::Test {
 protected:
  void SetUp() override {
    device_ = getDevice();
    setDevice(device_);
    arena_ = std::make_unique<GpuArena>(1 << 20, getAllocator(device_));
    // One directory per test. Sharing it would let a compile still running
    // for a finished test scan a directory the next test is deleting.
    cacheDir_ = fmt::format(
        "/tmp/kernel_fs_cache_test_{}_{}",
        getpid(),
        testing::UnitTest::GetInstance()->current_test_info()->name());
    std::error_code error;
    fs::remove_all(cacheDir_, error);
  }

  void TearDown() override {
    // Drop the in-process cache first: it owns the futures the deferred
    // compiles run on, so the directory is not removed while one is still
    // scanning it.
    CompiledKernel::clearCache();
    std::error_code error;
    fs::remove_all(cacheDir_, error);
  }

  // A kernel whose text, and so whose cache entry, varies with 'tag'.
  static std::string kernelText(int32_t tag) {
    // Plain int rather than int32_t: NVRTC compiles this text on its own, with
    // no wave header in scope to declare the fixed-width names.
    return fmt::format(
        "namespace facebook::velox::wave {{\n"
        "__global__ void addKernel(int* ints) {{\n"
        "  ints[threadIdx.x] += {};\n"
        "}}\n"
        "}}\n",
        tag);
  }

  static KernelGenFunc genFor(const std::string& text) {
    return [text]() -> KernelSpec {
      KernelSpec spec;
      spec.code = text;
      spec.entryPoints = {"facebook::velox::wave::addKernel"};
      spec.filePath = "kernel_fs_cache_test.cu";
      return spec;
    };
  }

  // getKernel defers everything that touches the cache, so a test that asserts
  // on hits, misses or files has to make the kernel real first. info() forces
  // the deferred compile or load to finish.
  std::unique_ptr<CompiledKernel> getReady(
      KernelFsCache& cache,
      const std::string& text) {
    auto kernel = cache.getKernel(text, genFor(text));
    kernel->info(0);
    return kernel;
  }

  // Runs the kernel over 32 ints and returns what it added, so a test can tell
  // that the cubin it loaded is the one belonging to its own source rather
  // than another entry's.
  int32_t runAndGetAddend(CompiledKernel& kernel) {
    WaveBufferPtr ints = arena_->allocate<int32_t>(32);
    auto* raw = ints->as<int32_t>();
    for (auto i = 0; i < 32; ++i) {
      raw[i] = 0;
    }
    void* params = &raw;
    auto stream = std::make_unique<Stream>();
    kernel.launch(0, 1, 32, 0, stream.get(), &params);
    stream->wait();
    return raw[0];
  }

  int32_t countFiles(const std::string& extension) {
    int32_t count = 0;
    std::error_code error;
    if (!fs::exists(cacheDir_, error)) {
      return 0;
    }
    for (auto& entry : fs::directory_iterator(cacheDir_)) {
      if (entry.path().extension() == extension) {
        ++count;
      }
    }
    return count;
  }

  Device* device_{nullptr};
  std::unique_ptr<GpuArena> arena_;
  std::string cacheDir_;
};

// Two caches over one directory, each compiling a different kernel, must not
// land on the same file names. With entries named by an ordinal each picked
// its own next number from the same starting point, so both wrote entry 0 and
// one kernel's .cu ended up beside the other's .cubin -- a hit that returns
// the wrong machine code. Naming by content is what separates them.
TEST_F(KernelFsCacheTest, concurrentWritersDoNotCollide) {
  auto firstText = kernelText(11);
  auto secondText = kernelText(22);

  KernelFsCache first(cacheDir_);
  KernelFsCache second(cacheDir_);

  // Both scan the directory while it is empty, as two processes starting
  // together would, before either has published anything.
  auto firstKernel = getReady(first, firstText);
  auto secondKernel = getReady(second, secondText);

  EXPECT_EQ(runAndGetAddend(*firstKernel), 11);
  EXPECT_EQ(runAndGetAddend(*secondKernel), 22);

  // Two distinct kernels, so two distinct entries rather than one overwriting
  // the other.
  EXPECT_EQ(countFiles(".cu"), 2);
  EXPECT_EQ(countFiles(".cubin"), 2);

  // Read back through a third cache, which sees only the files. If the two
  // writers had collided this is where the wrong cubin would surface. The two
  // kernels are released first: while one is held the in-process cache keeps
  // serving it and the filesystem is never consulted.
  firstKernel.reset();
  secondKernel.reset();
  CompiledKernel::clearCache();
  KernelFsCache reader(cacheDir_);
  EXPECT_EQ(runAndGetAddend(*getReady(reader, firstText)), 11);
  EXPECT_EQ(runAndGetAddend(*getReady(reader, secondText)), 22);
  EXPECT_EQ(reader.hits(), 2);
  EXPECT_EQ(reader.misses(), 0);
}

// Both processes compiling the SAME kernel converge on one entry instead of
// accumulating a copy each.
TEST_F(KernelFsCacheTest, sameKernelFromTwoCachesSharesOneEntry) {
  auto text = kernelText(7);

  KernelFsCache first(cacheDir_);
  KernelFsCache second(cacheDir_);

  EXPECT_EQ(runAndGetAddend(*getReady(first, text)), 7);
  CompiledKernel::clearCache();
  EXPECT_EQ(runAndGetAddend(*getReady(second, text)), 7);

  EXPECT_EQ(countFiles(".cu"), 1);
  EXPECT_EQ(countFiles(".cubin"), 1);
}

// An entry published after this cache scanned the directory is still found.
// The scan happens once, so without a second look a long-lived process
// recompiles everything its neighbours added while it was running.
TEST_F(KernelFsCacheTest, findsEntryPublishedAfterScan) {
  auto scanned = kernelText(3);
  auto later = kernelText(4);

  KernelFsCache reader(cacheDir_);
  // Force reader to scan the directory now, while it holds only 'scanned'.
  {
    KernelFsCache writer(cacheDir_);
    getReady(writer, scanned);
  }
  CompiledKernel::clearCache();
  getReady(reader, scanned);
  EXPECT_EQ(reader.hits(), 1);

  // A second process publishes another entry afterwards.
  {
    KernelFsCache writer(cacheDir_);
    getReady(writer, later);
  }
  CompiledKernel::clearCache();

  EXPECT_EQ(runAndGetAddend(*getReady(reader, later)), 4);
  EXPECT_EQ(reader.hits(), 2);
  EXPECT_EQ(reader.misses(), 0);
}

// A cubin that is present but truncated -- what a reader saw mid-write before
// entries were published by rename -- is treated as absent and recompiled,
// rather than loaded.
TEST_F(KernelFsCacheTest, tornEntryIsRecompiledNotLoaded) {
  auto text = kernelText(5);
  {
    KernelFsCache writer(cacheDir_);
    getReady(writer, text);
  }
  CompiledKernel::clearCache();

  // Truncate the cubin, leaving the .cu and .names in place.
  for (auto& entry : fs::directory_iterator(cacheDir_)) {
    if (entry.path().extension() == ".cubin") {
      std::ofstream out(
          entry.path().string(), std::ios::trunc | std::ios::binary);
    }
  }

  KernelFsCache reader(cacheDir_);
  EXPECT_EQ(runAndGetAddend(*getReady(reader, text)), 5);
  EXPECT_EQ(reader.misses(), 1);
  EXPECT_EQ(reader.hits(), 0);
}

// A directory written before entries were named by content is still usable.
// Its entries are named by an ordinal; they must be found by the text of their
// .cu like any other, so upgrading does not throw away a warm cache.
TEST_F(KernelFsCacheTest, readsCacheWrittenWithOrdinalNames) {
  auto text = kernelText(9);

  // Build an entry, then rename it to the ordinal form an earlier version
  // wrote, which is what an existing cache directory on disk looks like.
  {
    KernelFsCache writer(cacheDir_);
    getReady(writer, text);
  }
  CompiledKernel::clearCache();
  std::string stem;
  for (auto& entry : fs::directory_iterator(cacheDir_)) {
    if (entry.path().extension() == ".cu") {
      stem = entry.path().stem().string();
    }
  }
  ASSERT_FALSE(stem.empty());
  std::error_code error;
  fs::rename(
      fmt::format("{}/{}.cu", cacheDir_, stem),
      fmt::format("{}/0.cu", cacheDir_),
      error);
  fs::rename(
      fmt::format("{}/{}.cubin", cacheDir_, stem),
      fmt::format("{}/0.cubin", cacheDir_),
      error);
  fs::rename(
      fmt::format("{}/{}.cubin.names", cacheDir_, stem),
      fmt::format("{}/0.cubin.names", cacheDir_),
      error);
  ASSERT_FALSE(error);

  KernelFsCache legacy(cacheDir_);
  EXPECT_EQ(runAndGetAddend(*getReady(legacy, text)), 9);
  EXPECT_EQ(legacy.hits(), 1);
  EXPECT_EQ(legacy.misses(), 0);
  // Served from the existing file rather than recompiled beside it.
  EXPECT_EQ(countFiles(".cu"), 1);
}

// A process that dies between writing its temporaries and renaming them
// leaves a complete-looking triple under a .tmp. stem. It must not be taken
// for a cache entry: nothing accounts for that name, and adopting it would
// make a half-finished publish permanent.
TEST_F(KernelFsCacheTest, leftoverTemporaryIsNotAdopted) {
  auto text = kernelText(13);
  {
    KernelFsCache writer(cacheDir_);
    getReady(writer, text);
  }
  CompiledKernel::clearCache();

  // Rename the published entry to the form a dead process would have left.
  std::string stem;
  for (auto& entry : fs::directory_iterator(cacheDir_)) {
    if (entry.path().extension() == ".cu") {
      stem = entry.path().stem().string();
    }
  }
  ASSERT_FALSE(stem.empty());
  const std::string temp = ".tmp.999999.0";
  std::error_code error;
  for (auto suffix : {".cu", ".cubin", ".cubin.names"}) {
    fs::rename(
        fmt::format("{}/{}{}", cacheDir_, stem, suffix),
        fmt::format("{}/{}{}", cacheDir_, temp, suffix),
        error);
  }
  ASSERT_FALSE(error);

  KernelFsCache reader(cacheDir_);
  EXPECT_EQ(runAndGetAddend(*getReady(reader, text)), 13);
  // Recompiled rather than served from the leftover.
  EXPECT_EQ(reader.hits(), 0);
  EXPECT_EQ(reader.misses(), 1);
}

// A failed compile leaves nothing behind for another process to trip over.
TEST_F(KernelFsCacheTest, failedCompileLeavesNoFiles) {
  KernelFsCache cache(cacheDir_);
  auto badText = std::string("#error intentional\n");
  bool threw = false;
  try {
    auto kernel = cache.getKernel(badText, genFor(badText));
    kernel->info(0);
  } catch (...) {
    threw = true;
  }
  EXPECT_TRUE(threw);
  EXPECT_EQ(cache.size(), 0);
  EXPECT_EQ(countFiles(".cu"), 0);
  EXPECT_EQ(countFiles(".cubin"), 0);
}

} // namespace
} // namespace facebook::velox::wave
