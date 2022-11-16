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
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/dwrf/writer/FlushPolicy.h"
#include "velox/dwio/dwrf/writer/Writer.h"

#include <folly/init/Init.h>

using namespace facebook::velox::dwio::common;
using namespace facebook::velox::dwrf;
using namespace facebook::velox;
using namespace facebook::velox::common;

using dwio::common::MemorySink;

class E2EFilterTest : public E2EFilterTestBase {
 protected:
  void testWithTypes(
      const std::string& columns,
      std::function<void()> customize,
      bool wrapInStruct,
      const std::vector<std::string>& filterable,
      int32_t numCombinations,
      bool tryNoNulls = false,
      bool tryNoVInts = false) {
    for (int32_t noVInts = 0; noVInts < (tryNoVInts ? 2 : 1); ++noVInts) {
      useVInts_ = !noVInts;
      for (int32_t noNulls = 0; noNulls < (tryNoNulls ? 2 : 1); ++noNulls) {
        LOG(INFO) << "Running with " << (noNulls ? " no nulls " : "nulls")
                  << " and " << (noVInts ? " no VInts " : " VInts ")
                  << std::endl;
        auto newCustomize = customize;
        if (noNulls) {
          newCustomize = [&]() {
            customize();
            makeNotNull();
          };
        }

        testSenario(
            columns, newCustomize, wrapInStruct, filterable, numCombinations);
      }
    }
  }

  void writeToMemory(
      const TypePtr& type,
      const std::vector<RowVectorPtr>& batches,
      bool forRowGroupSkip = false) override {
    auto config = std::make_shared<dwrf::Config>();
    config->set(dwrf::Config::COMPRESSION, CompressionKind_NONE);
    config->set(dwrf::Config::USE_VINTS, useVInts_);
    WriterOptions options;
    options.config = config;
    options.schema = type;
    int32_t flushCounter = 0;
    // If we test row group skip, we have all the data in one stripe. For
    // scan, we start  a stripe every 'flushEveryNBatches_' batches.
    options.flushPolicyFactory = [&]() {
      return std::make_unique<LambdaFlushPolicy>([&]() {
        return forRowGroupSkip ? false
                               : (++flushCounter % flushEveryNBatches_ == 0);
      });
    };
    auto sink = std::make_unique<MemorySink>(*pool_, 200 * 1024 * 1024);
    sinkPtr_ = sink.get();
    writer_ = std::make_unique<Writer>(options, std::move(sink), *pool_);
    for (auto& batch : batches) {
      writer_->write(batch);
    }
    writer_->close();
  }

  std::unique_ptr<dwio::common::Reader> makeReader(
      const dwio::common::ReaderOptions& opts,
      std::unique_ptr<dwio::common::InputStream> input) override {
    return std::make_unique<DwrfReader>(opts, std::move(input));
  }

  std::unique_ptr<Writer> writer_;
};

TEST_F(E2EFilterTest, integerDirect) {
  testWithTypes(
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint,"
      "long_null:bigint",
      [&]() { makeAllNulls("long_null"); },
      true,
      {"short_val", "int_val", "long_val"},
      20,
      true,
      true);
}

TEST_F(E2EFilterTest, integerDictionary) {
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
      true,
      {"short_val", "int_val", "long_val"},
      20,
      true,
      true);
}

TEST_F(E2EFilterTest, byteRle) {
  testWithTypes(
      "tiny_val:tinyint,"
      "bool_val:boolean,"
      "long_val:bigint,"
      "tiny_null:bigint",
      [&]() { makeAllNulls("tiny_null"); },
      true,
      {"tiny_val", "bool_val", "tiny_null"},
      20);
}

TEST_F(E2EFilterTest, floatAndDouble) {
  testWithTypes(
      "float_val:float,"
      "double_val:double,"
      "long_val:bigint,"
      "float_null:float",
      [&]() { makeAllNulls("float_null"); },
      true,
      {"float_val", "double_val", "float_null"},
      20,
      true,
      false);
}

TEST_F(E2EFilterTest, stringDirect) {
  flushEveryNBatches_ = 1;
  testWithTypes(
      "string_val:string,"
      "string_val_2:string",
      [&]() {
        makeStringUnique("string_val");
        makeStringUnique("string_val_2");
      },

      true,
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
      true,
      {"string_val", "string_val_2"},
      20,
      true,
      true);
}

TEST_F(E2EFilterTest, timestamp) {
  testWithTypes(
      "timestamp_val:timestamp,"
      "long_val:bigint",
      [&]() {},
      false,
      {"long_val"},
      20,
      true,
      true);
}

TEST_F(E2EFilterTest, listAndMap) {
  testWithTypes(
      "long_val:bigint,"
      "long_val_2:bigint,"
      "int_val:int,"
      "array_val:array<struct<array_member: array<int>>>,"
      "map_val:map<bigint,struct<nested_map: map<int, int>>>",
      [&]() {},
      true,
      {"long_val", "long_val_2", "int_val"},
      10);
}

TEST_F(E2EFilterTest, nullCompactRanges) {
  // Makes a dataset with nulls at the beginning. Tries different
  // filter ombinations on progressively larger batches. tests for a
  // bug in null compaction where null bits past end of nulls buffer
  // were compacted while there actually were no nulls.

  readSizes_ = {10, 100, 1000, 10000, 10000, 10000};
  testWithTypes(
      "tiny_val:tinyint,"
      "bool_val:boolean,"
      "long_val:bigint,"
      "tiny_null:bigint",

      [&]() { makeNotNull(500); },

      true,
      {"tiny_val", "bool_val", "long_val", "tiny_null"},
      20,
      false,
      false);
}

TEST_F(E2EFilterTest, lazyStruct) {
  testWithTypes(
      "long_val:bigint,"
      "outer_struct: struct<nested1:bigint, "
      "inner_struct: struct<nested2: bigint>>",
      [&]() {},
      true,
      {"long_val"},
      10,
      true,
      false);
}

TEST_F(E2EFilterTest, filterStruct) {
  // The data has a struct member with one second level struct
  // column. Both structs have a column that gets filtered 'nestedxxx'
  // and one that does not 'dataxxx'.
  testWithTypes(
      "long_val:bigint,"
      "outer_struct: struct<nested1:bigint, "
      "  data1: string, "
      "  inner_struct: struct<nested2: bigint, data2: smallint>>",
      [&]() {},
      true,
      {"long_val",
       "outer_struct.inner_struct",
       "outer_struct.nested1",
       "outer_struct.inner_struct.nested2"},
      40,
      true,
      false);
}

// Define main so that gflags get processed.
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
