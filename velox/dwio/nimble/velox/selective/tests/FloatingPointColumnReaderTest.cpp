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

#include "velox/dwio/nimble/velox/selective/SelectiveNimbleReader.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/dwio/common/TypeUtils.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <random>
#include <unordered_map>

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

class FloatingPointColumnReaderTest : public ::testing::Test,
                                      public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(velox::memory::MemoryManager::Options{});
    registerSelectiveNimbleReaderFactory();
  }

  static void TearDownTestCase() {
    unregisterSelectiveNimbleReaderFactory();
  }

  memory::MemoryPool* rootPool() {
    return rootPool_.get();
  }

  struct Readers {
    std::unique_ptr<dwio::common::Reader> reader;
    std::unique_ptr<dwio::common::RowReader> rowReader;
  };

  Readers makeReaders(
      const RowVectorPtr& expected,
      const std::string& file,
      const std::shared_ptr<common::ScanSpec>& scanSpec) {
    auto readFile = std::make_shared<InMemoryReadFile>(file);
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions options(pool());
    options.setDataIoStats(dataIoStats_);
    options.setMetadataIoStats(metadataIoStats_);
    options.setScanSpec(scanSpec);

    Readers readers;
    readers.reader = factory->createReader(
        std::make_unique<dwio::common::BufferedInput>(readFile, *pool()),
        options);

    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(scanSpec);
    rowOptions.setRequestedType(asRowType(expected->type()));
    rowOptions.setStringDecoderZeroCopy(true);
    readers.rowReader = readers.reader->createRowReader(rowOptions);
    return readers;
  }

  void validate(
      const RowVector& expected,
      dwio::common::RowReader& rowReader,
      int batchSize) {
    auto result = BaseVector::create(asRowType(expected.type()), 0, pool());
    int numScanned = 0;
    int offset = 0;
    while (numScanned < expected.size()) {
      numScanned += rowReader.next(batchSize, result);
      result->validate();
      for (int i = 0; i < result->size(); ++i) {
        ASSERT_TRUE(result->equalValueAt(&expected, i, offset + i))
            << "Mismatch at row " << (offset + i) << ": expected "
            << expected.toString(offset + i) << " got " << result->toString(i);
      }
      offset += result->size();
    }
    ASSERT_EQ(numScanned, expected.size());
    ASSERT_EQ(0, rowReader.next(1, result));
  }

  void validateWithFilter(
      const RowVector& input,
      dwio::common::RowReader& rowReader,
      int batchSize,
      const std::function<bool(int)>& filter) {
    auto result = BaseVector::create(asRowType(input.type()), 0, pool());
    int numScanned = 0;
    int inputRow = 0;
    while (numScanned < input.size()) {
      numScanned += rowReader.next(batchSize, result);
      result->validate();
      for (int i = 0; i < result->size(); ++i) {
        for (;;) {
          ASSERT_LT(inputRow, input.size());
          if (filter(inputRow)) {
            break;
          }
          ++inputRow;
        }
        ASSERT_TRUE(result->equalValueAt(&input, i, inputRow))
            << "Mismatch at input row " << inputRow;
        ++inputRow;
      }
    }
    while (inputRow < input.size()) {
      ASSERT_FALSE(filter(inputRow));
      ++inputRow;
    }
    ASSERT_EQ(numScanned, input.size());
    ASSERT_EQ(0, rowReader.next(1, result));
  }

  template <typename T>
  RowVectorPtr makeAlpInput() {
    return makeRowVector({
        makeFlatVector<T>(
            257,
            [](auto i) {
              return static_cast<T>((static_cast<int32_t>(i % 67) - 33)) /
                  static_cast<T>(4);
            }),
    });
  }

  template <typename T>
  RowVectorPtr makeInput(const std::vector<T>& data) {
    return makeRowVector({
        makeFlatVector<T>(data.size(), [&](auto i) { return data[i]; }),
    });
  }

  template <typename T>
  void testAlpRead(int batchSize) {
    SCOPED_TRACE(fmt::format("type={}", TypeTraits<T>::dataType));

    auto input = makeAlpInput<T>();
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    const auto file = test::createNimbleFile(
        *rootPool(), input, makeSingleScalarColumnWriterOptions());

    auto readers = makeReaders(input, file, scanSpec);
    validate(*input, *readers.rowReader, batchSize);
  }

  template <typename T>
  void testAlpReadWithFilter(int batchSize) {
    SCOPED_TRACE(fmt::format("type={}", TypeTraits<T>::dataType));
    constexpr T kLower = static_cast<T>(-2.5);
    constexpr T kUpper = static_cast<T>(2.5);

    auto input = makeAlpInput<T>();
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    scanSpec->childByName("c0")->setFilter(
        std::make_unique<common::FloatingPointRange<T>>(
            kLower, false, false, kUpper, false, false, false));
    const auto file = test::createNimbleFile(
        *rootPool(), input, makeSingleScalarColumnWriterOptions());

    auto readers = makeReaders(input, file, scanSpec);
    validateWithFilter(*input, *readers.rowReader, batchSize, [](auto row) {
      const auto value = static_cast<T>((static_cast<int32_t>(row % 67) - 33)) /
          static_cast<T>(4);
      return value >= kLower && value <= kUpper;
    });
  }

  template <typename T>
  void testAlpReadWithRandomFilter(uint32_t seed, int batchSize) {
    SCOPED_TRACE(fmt::format("type={} seed={}", TypeTraits<T>::dataType, seed));

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> valueDistribution{-10'000, 10'000};
    std::uniform_int_distribution<int32_t> boundDistribution{-50, 50};

    std::vector<T> data(511);
    for (auto& value : data) {
      value = static_cast<T>(valueDistribution(rng)) / static_cast<T>(100);
    }

    auto lower = static_cast<T>(boundDistribution(rng));
    auto upper = static_cast<T>(boundDistribution(rng));
    if (lower > upper) {
      std::swap(lower, upper);
    }

    auto input = makeInput(data);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    scanSpec->childByName("c0")->setFilter(
        std::make_unique<common::FloatingPointRange<T>>(
            lower, false, false, upper, false, false, false));
    const auto file = test::createNimbleFile(
        *rootPool(), input, makeSingleScalarColumnWriterOptions());

    auto readers = makeReaders(input, file, scanSpec);
    validateWithFilter(*input, *readers.rowReader, batchSize, [&](auto row) {
      return data[row] >= lower && data[row] <= upper;
    });
  }

  const std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  const std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};

 private:
  static EncodingLayout makeAlpEncodingLayout() {
    return EncodingLayout{
        EncodingType::ALP,
        {},
        CompressionType::Uncompressed,
        {EncodingLayout{
            EncodingType::FixedBitWidth, {}, CompressionType::Uncompressed}}};
  }

  static WriterOptions makeSingleScalarColumnWriterOptions() {
    using StreamLayouts = std::
        unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>;

    WriterOptions writerOptions;
    writerOptions.encodingLayoutTree.emplace(
        Kind::Row,
        StreamLayouts{},
        "",
        std::vector<EncodingLayoutTree>{EncodingLayoutTree{
            Kind::Scalar,
            StreamLayouts{
                {EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                 makeAlpEncodingLayout()}},
            "c0"}});
    return writerOptions;
  }
};

TEST_F(FloatingPointColumnReaderTest, alpFloatAndDoubleRead) {
  testAlpRead<float>(19);
  testAlpRead<double>(23);
  testAlpReadWithFilter<float>(17);
  testAlpReadWithFilter<double>(29);

  for (const auto seed : {0xA11CEu, 0xBEEFu, 0xC0FFEEu}) {
    SCOPED_TRACE(fmt::format("seed={}", seed));
    testAlpReadWithRandomFilter<float>(seed, 31);
    testAlpReadWithRandomFilter<double>(seed + 1, 37);
  }
}

} // namespace
} // namespace facebook::nimble
