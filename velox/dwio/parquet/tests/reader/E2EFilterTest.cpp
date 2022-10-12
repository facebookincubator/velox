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

#include "velox/dwio/common/tests/E2EFilterTestBase.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"

using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::parquet;

using dwio::common::MemoryInputStream;
using dwio::common::MemorySink;

class E2EFilterTest : public E2EFilterTestBase {
 protected:
  void SetUp() override {
    E2EFilterTestBase::SetUp();
    writerProperties_ = ::parquet::WriterProperties::Builder().build();
  }

  void writeToMemory(
      const TypePtr&,
      const std::vector<RowVectorPtr>& batches,
      bool /*forRowGroupSkip*/) override {
    auto sink = std::make_unique<MemorySink>(*pool_, 200 * 1024 * 1024);
    sinkPtr_ = sink.get();

    writer_ = std::make_unique<facebook::velox::parquet::Writer>(
        std::move(sink), *pool_, 10000, writerProperties_);
    for (auto& batch : batches) {
      writer_->write(batch);
    }
    writer_->close();
  }

  virtual void testWithTypes(
      const std::string& columns,
      std::function<void()> customize,
      bool wrapInStruct,
      const std::vector<std::string>& filterable,
      int32_t numCombinations,
      bool /* tryNoNulls */ = true,
      bool /* tryNoVInts */ = true) override {
    testSenario(columns, customize, wrapInStruct, filterable, numCombinations);

    // Always test no null case.
    auto newCustomize = [&]() {
      customize();
      makeNotNull(0);
    };
    testSenario(
        columns, newCustomize, wrapInStruct, filterable, numCombinations);
  }

  std::unique_ptr<dwio::common::Reader> makeReader(
      const dwio::common::ReaderOptions& opts,
      std::unique_ptr<dwio::common::InputStream> input) override {
    return std::make_unique<ParquetReader>(std::move(input), opts);
  }

  std::unique_ptr<facebook::velox::parquet::Writer> writer_;
  std::shared_ptr<::parquet::WriterProperties> writerProperties_;
};

TEST_F(E2EFilterTest, writerMagic) {
  rowType_ = ROW({INTEGER()});
  std::vector<RowVectorPtr> batches;
  batches.push_back(std::static_pointer_cast<RowVector>(
      test::BatchMaker::createBatch(rowType_, 20000, *pool_, nullptr, 0)));
  writeToMemory(rowType_, batches, false);
  auto data = sinkPtr_->getData();
  auto size = sinkPtr_->size();
  EXPECT_EQ("PAR1", std::string(data, 4));
  EXPECT_EQ("PAR1", std::string(data + size - 4, 4));
}

TEST_F(E2EFilterTest, integerDirect) {
  writerProperties_ = ::parquet::WriterProperties::Builder()
                          .disable_dictionary()
                          ->data_pagesize(4 * 1024)
                          ->build();
  testWithTypes(
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint,"
      "long_null:bigint",
      [&]() { makeAllNulls("long_null"); },
      false,
      {"short_val", "int_val", "long_val"},
      20,
      true);
}

TEST_F(E2EFilterTest, integerDictionary) {
  for (const auto compression :
       {::parquet::Compression::SNAPPY,
        ::parquet::Compression::ZSTD,
        ::parquet::Compression::GZIP,
        ::parquet::Compression::UNCOMPRESSED}) {
    if (!arrow::util::Codec::IsAvailable(compression)) {
      continue;
    }

    writerProperties_ = ::parquet::WriterProperties::Builder()
                            .data_pagesize(4 * 1024)
                            ->compression(compression)
                            ->build();

    testWithTypes(
        "short_val:smallint,"
        "int_val:int,"
        "long_val:bigint",
        [&]() {
          makeIntDistribution<int64_t>(
              "long_val",
              10, // min
              100, // max
              22, // repeats
              19, // rareFrequency
              -9999, // rareMin
              10000000000, // rareMax
              true); // keepNulls

          makeIntDistribution<int32_t>(
              "int_val",
              10, // min
              100, // max
              22, // repeats
              19, // rareFrequency
              -9999, // rareMin
              100000000, // rareMax
              false); // keepNulls

          makeIntDistribution<int16_t>(
              "short_val",
              10, // min
              100, // max
              22, // repeats
              19, // rareFrequency
              -999, // rareMin
              30000, // rareMax
              true); // keepNulls
        },
        false,
        {"short_val", "int_val", "long_val"},
        20,
        true);
  }
}

TEST_F(E2EFilterTest, floatAndDoubleDirect) {
  writerProperties_ = ::parquet::WriterProperties::Builder()
                          .disable_dictionary()
                          ->data_pagesize(4 * 1024)
                          ->build();

  testWithTypes(
      "float_val:float,"
      "double_val:double,"
      "float_val2:float,"
      "double_val2:double,"
      "long_val:bigint,"
      "float_null:float",
      [&]() {
        makeAllNulls("float_null");
        makeQuantizedFloat<float>("float_val2", 200, true);
        makeQuantizedFloat<double>("double_val2", 522, true);
      },
      false,
      {"float_val", "double_val", "float_val2", "double_val2", "float_null"},
      20,
      true,
      false);
}

TEST_F(E2EFilterTest, floatAndDouble) {
  // float_val and double_val may be direct since the
  // values are random.float_val2 and double_val2 are expected to be
  // dictionaries since the values are quantized.
  testWithTypes(
      "float_val:float,"
      "double_val:double,"
      "float_val2:float,"
      "double_val2:double,"

      "long_val:bigint,"
      "float_null:float",
      [&]() {
        makeAllNulls("float_null");
        makeQuantizedFloat<float>("float_val2", 200, true);
        makeQuantizedFloat<double>("double_val2", 522, true);
        // Make sure there are RLE's.
        makeReapeatingValues<float>("float_val2", 0, 100, 200);
        makeReapeatingValues<double>("double_val2", 0, 100, 200);
      },
      false,
      {"float_val", "double_val", "float_val2", "double_val2", "float_null"},
      20,
      true,
      false);
}

TEST_F(E2EFilterTest, stringDirect) {
  writerProperties_ = ::parquet::WriterProperties::Builder()
                          .disable_dictionary()
                          ->data_pagesize(4 * 1024)
                          ->build();

  testWithTypes(
      "string_val:string,"
      "string_val_2:string",
      [&]() {
        makeStringUnique("string_val");
        makeStringUnique("string_val_2");
      },
      false,
      {"string_val", "string_val_2"},
      20,
      true);
}

TEST_F(E2EFilterTest, stringDictionary) {
  testWithTypes(
      "string_val:string,"
      "string_val_2:string",
      [&]() {
        makeStringDistribution("string_val", 100, true, false);
        makeStringDistribution("string_val_2", 170, false, true);
      },
      false,
      {"string_val", "string_val_2"},
      20,
      true);
}
