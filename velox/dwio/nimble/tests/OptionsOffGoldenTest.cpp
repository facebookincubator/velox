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

// With every typed-lanes option off, the writer and the reader behave exactly
// as they did before those options existed. An option is "off" when it is left
// at its default, set explicitly to its off value, or set to a value that has
// no effect while typed lanes are off (the row floor). Every off form must
// write the file the untouched options write, compared within one build. The
// comparison with files the earlier writer wrote is internal-only
// (fb/OptionsOffFbTest.cpp): the bytes of a whole file depend on library
// versions, FlatBuffers among them, so its golden digests hold only for the
// internal build. The published statistics are compared with the set it
// published then. Managed Compression output depends on configs served at run
// time, so its compress calls are checked by an internal test instead.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "folly/json/dynamic.h"
#include "velox/common/file/File.h"
#include "velox/dwio/nimble/common/MetricsLogger.h"
#include "velox/dwio/nimble/reader/BatchReader.h"
#include "velox/dwio/nimble/tests/OptionsOffFixtures.h"
#include "velox/dwio/nimble/tests/OptionsOffForms.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble {
namespace {

namespace options_off = test::options_off;
using options_off::fixedBitWidthLevelWithoutLanes;
using options_off::OffForm;
using options_off::offForms;
using options_off::sampleRateWithoutLanes;

// Captures the metrics the writer logs when it closes a file.
class CapturingLogger final : public MetricsLogger {
 public:
  void logFileClose(const FileCloseMetrics& metrics) const override {
    closeMetrics = metrics.serialize();
  }

  mutable folly::dynamic closeMetrics;
};

class OptionsOffGoldenTest : public ::testing::Test,
                             public velox::test::VectorTestBase {
 protected:
  static void SetUpTestSuite() {
    if (!velox::memory::MemoryManager::testInstance()) {
      velox::memory::MemoryManager::testingSetInstance(
          velox::memory::MemoryManager::Options{});
    }
  }

  void SetUp() override {
    batches_ = options_off::makeBatches(*pool());
  }

  options_off::WrittenFile write(
      const options_off::Fixture& fixture,
      const OffForm& form) {
    auto options = options_off::fixtureOptions(fixture);
    form.apply(options);
    return options_off::writeFile(batches_, std::move(options), *rootPool_);
  }

  // Reads every row of 'bytes' back and checks it against the batches.
  void expectReadsBack(const std::string& bytes) {
    const auto type = batches_[0]->type();
    velox::vector_size_t numRows{0};
    for (const auto& batch : batches_) {
      numRows += batch->size();
    }
    auto expected = velox::BaseVector::create(type, numRows, pool());
    velox::vector_size_t offset{0};
    for (const auto& batch : batches_) {
      expected->copy(batch.get(), offset, 0, batch->size());
      offset += batch->size();
    }

    velox::InMemoryReadFile readFile{bytes};
    BatchReader reader{&readFile, *pool()};
    auto actual = velox::BaseVector::create(type, numRows, pool());
    offset = 0;
    velox::VectorPtr result;
    while (reader.next(numRows, result)) {
      ASSERT_LE(offset + result->size(), numRows);
      actual->copy(result.get(), offset, 0, result->size());
      offset += result->size();
    }
    ASSERT_EQ(offset, numRows);
    velox::test::assertEqualVectors(expected, actual);
  }

  // The fixture batches, made once per test.
  std::vector<velox::RowVectorPtr> batches_;
};

// Every off form writes the file the untouched options write, on every
// fixture.
TEST_F(OptionsOffGoldenTest, offFormsWriteTheSameFile) {
  auto forms = offForms();
  forms.push_back(sampleRateWithoutLanes());
  for (const auto& fixture : options_off::allFixtures()) {
    const auto untouched = write(fixture, forms[0]);
    for (size_t form = 1; form < forms.size(); ++form) {
      SCOPED_TRACE(fixture.name + " / " + forms[form].name);
      EXPECT_EQ(write(fixture, forms[form]).bytes, untouched.bytes);
    }
  }
}

// With real Zstd compression, every off form writes the same file as the
// untouched options. The bytes depend on the Zstd library, so this compares
// within one build only.
TEST_F(OptionsOffGoldenTest, offFormsWriteTheSameFileWithZstd) {
  const CompressionOptions zstd{
      .compressionType = CompressionType::Zstd,
      .zstdMinCompressionSize = 0,
  };
  const auto forms = offForms();
  for (const auto& fixture : options_off::allFixtures()) {
    if (fixture.compression !=
        options_off::FixtureCompression::kZstdBelowMinSize) {
      continue;
    }
    auto untouchedOptions = options_off::fixtureOptions(fixture, zstd);
    const auto untouched = options_off::writeFile(
        batches_, std::move(untouchedOptions), *rootPool_);
    for (size_t form = 1; form < forms.size(); ++form) {
      SCOPED_TRACE(fixture.name + " / " + forms[form].name);
      auto options = options_off::fixtureOptions(fixture, zstd);
      forms[form].apply(options);
      EXPECT_EQ(
          options_off::writeFile(batches_, std::move(options), *rootPool_)
              .bytes,
          untouched.bytes);
    }
    expectReadsBack(untouched.bytes);
  }
}

// With the options off, the writer publishes the same runtime statistics and
// file-close metrics as before the typed-lanes options existed: no encoding
// counters.
TEST_F(OptionsOffGoldenTest, publishedStatisticsMatchPreStack) {
  using Keys = Writer::RuntimeStats;
  const std::vector<std::string> preStackRuntimeStats{
      std::string{Keys::kWrittenBytes},
      std::string{Keys::kInputBytes},
      std::string{Keys::kWriteCpuNanos},
      std::string{Keys::kWriteWallNanos},
      std::string{Keys::kIngestionCpuNanos},
      std::string{Keys::kIngestionWallNanos},
      std::string{Keys::kEncodingCpuNanos},
      std::string{Keys::kEncodingWallNanos},
      std::string{Keys::kEncodingSelectionCpuNanos},
      std::string{Keys::kRowsPerStripe},
      std::string{Keys::kChunkSizeBytes},
      std::string{Keys::kDuplicateStreamCount},
      std::string{Keys::kDuplicateStreamBytes},
  };
  const std::vector<std::string> preStackCloseMetrics{
      "encodingCpuNs",
      "encodingWallNs",
      "fileSize",
      "inputSize",
      "rowCount",
      "stripeCount",
  };
  const options_off::Fixture fixture{
      .name = "freshFixedBitWidth.zstdBelowMinSize",
      .compression = options_off::FixtureCompression::kZstdBelowMinSize,
      .path = options_off::FixturePath::kFreshFixedBitWidth,
      .exactBits = false,
  };
  for (const auto& form : offForms()) {
    SCOPED_TRACE(form.name);
    auto logger = std::make_shared<CapturingLogger>();
    auto options = options_off::fixtureOptions(fixture);
    options.metricsLogger = logger;
    form.apply(options);
    const auto written =
        options_off::writeFile(batches_, std::move(options), *rootPool_);
    EXPECT_THAT(
        written.runtimeStatNames,
        ::testing::UnorderedElementsAreArray(preStackRuntimeStats));
    std::vector<std::string> closeMetrics;
    for (const auto& key : logger->closeMetrics.keys()) {
      closeMetrics.push_back(key.asString());
    }
    EXPECT_THAT(
        closeMetrics,
        ::testing::UnorderedElementsAreArray(preStackCloseMetrics));
  }
}

// The off forms set no option that reports encoding counters; a sample rate,
// a typed-lanes level or typed lanes each does.
TEST(OptionsOffNeedsEncodingStatsTest, onlyWithACounterOption) {
  for (const auto& form : offForms()) {
    SCOPED_TRACE(form.name);
    WriterOptions options;
    form.apply(options);
    EXPECT_FALSE(options.needsEncodingStats());
  }
  for (const auto& form :
       {sampleRateWithoutLanes(), fixedBitWidthLevelWithoutLanes()}) {
    SCOPED_TRACE(form.name);
    WriterOptions options;
    form.apply(options);
    EXPECT_TRUE(options.needsEncodingStats());
  }
  WriterOptions lanes;
  lanes.fixedBitWidthTypedLanes = true;
  EXPECT_TRUE(lanes.needsEncodingStats());
}

} // namespace
} // namespace facebook::nimble
