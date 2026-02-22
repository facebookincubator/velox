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

// Adapted from Apache Arrow.

#include "velox/dwio/parquet/writer/arrow/tests/TestUtil.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "velox/dwio/parquet/writer/arrow/ColumnPage.h"
#include "velox/dwio/parquet/writer/arrow/ColumnWriter.h"
#include "velox/dwio/parquet/writer/arrow/Encoding.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/tests/ColumnReader.h"

namespace facebook::velox::parquet::arrow {
namespace test {

const char* getDataDir() {
  const auto result = std::getenv("PARQUET_TEST_DATA");
  if (!result || !result[0]) {
    throw ParquetTestException(
        "Please point the PARQUET_TEST_DATA environment "
        "variable to the test data directory");
  }
  return result;
}

std::string getBadDataDir() {
  // PARQUET_TEST_DATA should point to.
  // ARROW_HOME/cpp/submodules/parquet-testing/data so need to reach one folder.
  // Up to access the "bad_data" folder.
  std::string dataDir(getDataDir());
  std::stringstream ss;
  ss << dataDir << "/../bad_data";
  return ss.str();
}

std::string getDataFile(const std::string& filename, bool isGood) {
  std::stringstream ss;

  if (isGood) {
    ss << getDataDir();
  } else {
    ss << getBadDataDir();
  }

  ss << "/" << filename;
  return ss.str();
}

void randomBytes(int n, uint32_t seed, std::vector<uint8_t>* out) {
  std::default_random_engine gen(seed);
  std::uniform_int_distribution<int> d(0, 255);

  out->resize(n);
  for (int i = 0; i < n; ++i) {
    (*out)[i] = static_cast<uint8_t>(d(gen));
  }
}

void randomBools(int n, double p, uint32_t seed, bool* out) {
  std::default_random_engine gen(seed);
  std::bernoulli_distribution d(p);
  for (int i = 0; i < n; ++i) {
    out[i] = d(gen);
  }
}

void randomInt96Numbers(
    int n,
    uint32_t seed,
    int32_t minValue,
    int32_t maxValue,
    Int96* out) {
  std::default_random_engine gen(seed);
  std::uniform_int_distribution<int32_t> d(minValue, maxValue);
  for (int i = 0; i < n; ++i) {
    out[i].value[0] = d(gen);
    out[i].value[1] = d(gen);
    out[i].value[2] = d(gen);
  }
}

void randomFixedByteArray(
    int n,
    uint32_t seed,
    uint8_t* buf,
    int len,
    FLBA* out) {
  std::default_random_engine gen(seed);
  std::uniform_int_distribution<int> d(0, 255);
  for (int i = 0; i < n; ++i) {
    out[i].ptr = buf;
    for (int j = 0; j < len; ++j) {
      buf[j] = static_cast<uint8_t>(d(gen));
    }
    buf += len;
  }
}

void randomByteArray(
    int n,
    uint32_t seed,
    uint8_t* buf,
    ByteArray* out,
    int minSize,
    int maxSize) {
  std::default_random_engine gen(seed);
  std::uniform_int_distribution<int> d1(minSize, maxSize);
  std::uniform_int_distribution<int> d2(0, 255);
  for (int i = 0; i < n; ++i) {
    int len = d1(gen);
    out[i].len = len;
    out[i].ptr = buf;
    for (int j = 0; j < len; ++j) {
      buf[j] = static_cast<uint8_t>(d2(gen));
    }
    buf += len;
  }
}

void randomByteArray(
    int n,
    uint32_t seed,
    uint8_t* buf,
    ByteArray* out,
    int maxSize) {
  randomByteArray(n, seed, buf, out, 0, maxSize);
}

} // namespace test
} // namespace facebook::velox::parquet::arrow
