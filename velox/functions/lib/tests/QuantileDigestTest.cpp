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

#include "velox/functions/lib/QuantileDigest.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/RandomSeed.h"
#include "velox/functions/lib/tests/QuantileDigestTestBase.h"

using namespace facebook::velox::functions::qdigest;

namespace facebook::velox::functions {

class QuantileDigestTest : public QuantileDigestTestBase {
 protected:
  memory::MemoryPool* pool() {
    return pool_.get();
  }

  HashStringAllocator* allocator() {
    return &allocator_;
  }

  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  std::string encodeBase64(std::string_view input) {
    return folly::base64Encode(input);
  }

  template <typename T>
  void testQuantiles(
      const std::vector<T>& values,
      const std::vector<double>& quantiles,
      const std::vector<T>& expected,
      const std::vector<double>& weight = {}) {
    ASSERT_EQ(quantiles.size(), expected.size());
    if (!weight.empty()) {
      ASSERT_EQ(values.size(), weight.size());
    }

    QuantileDigest<T> digest{StlAllocator<T>(allocator()), 0.99};
    for (auto i = 0; i < values.size(); ++i) {
      if (weight.empty()) {
        digest.add(values[i], 1.0);
      } else {
        digest.add(values[i], weight[i]);
      }
    }

    std::vector<T> results(quantiles.size());
    digest.estimateQuantiles(quantiles, results.data());
    for (auto i = 0; i < quantiles.size(); ++i) {
      EXPECT_EQ(results[i], expected[i]);
    }
  }

  template <typename T>
  void testHugeWeight() {
    constexpr int N = 10;
    constexpr double kAccuracy = 0.99;
    constexpr T kMaxValue = std::numeric_limits<T>::max();
    QuantileDigest<T> digest{StlAllocator<T>(allocator()), kAccuracy};
    for (auto i = 0; i < N; ++i) {
      digest.add(T(i), kMaxValue);
    }
  }

  template <typename T>
  void testLargeInputSize() {
    constexpr int N = 1e5;
    constexpr double kAccuracy = 0.8;
    std::vector<double> values;
    QuantileDigest<T> digest{StlAllocator<T>(allocator()), kAccuracy};
    std::default_random_engine gen(common::testutil::getRandomSeed(42));
    std::uniform_real_distribution<T> dist{-10.0, 10.0};
    for (int i = 0; i < N; ++i) {
      auto v = dist(gen);
      digest.add(v, 1.0);
      values.push_back(v);
    }
    std::sort(std::begin(values), std::end(values));
    checkQuantiles<QuantileDigest<T>, false>(
        values, digest, 0.0, values.size() * kAccuracy);

    values.clear();
    QuantileDigest<T> digestWeighted{StlAllocator<T>(allocator()), 0.8};
    for (int i = 0; i < N; ++i) {
      auto v = dist(gen);
      digest.add(v, i % 7 + 1);
      values.insert(values.end(), i % 7 + 1, v);
    }
    std::sort(std::begin(values), std::end(values));
    checkQuantiles<QuantileDigest<T>, false>(
        values, digest, 0.0, values.size() * kAccuracy);
  }

  template <typename T>
  void testEquivalentMerge() {
    constexpr double kAccuracy = 0.5;
    QuantileDigest<T> digest1{allocator(), kAccuracy};

    std::default_random_engine gen(common::testutil::getRandomSeed(42));
    std::uniform_real_distribution<> dist;
    for (auto i = 0; i < 100; ++i) {
      auto v = T(dist(gen));
      auto w = (i + 2) % 3 + 1;
      digest1.add(v, w);
    }

    QuantileDigest<T> mergeResult{allocator(), kAccuracy};
    digest1.compress();
    mergeResult.testingMerge(digest1);

    QuantileDigest<T> mergeSerializedResult{allocator(), kAccuracy};
    std::string buf(digest1.serializedByteSize(), '\0');
    digest1.serialize(buf.data());
    mergeSerializedResult.mergeSerialized(buf.data());

    EXPECT_EQ(
        mergeResult.serializedByteSize(),
        mergeSerializedResult.serializedByteSize());
    std::string mergeResultBuf(mergeResult.serializedByteSize(), '\0');
    mergeResult.serialize(mergeResultBuf.data());
    std::string mergeSerializedResultBuf(
        mergeSerializedResult.serializedByteSize(), '\0');
    mergeSerializedResult.serialize(mergeSerializedResultBuf.data());
    EXPECT_EQ(mergeResultBuf, mergeSerializedResultBuf);
  }

  template <typename T>
  void testMergeEmpty(bool mergeSerialized) {
    constexpr double kAccuracy = 0.7;
    QuantileDigest<T> digest{StlAllocator<T>(allocator()), kAccuracy};
    QuantileDigest<T> digestEmpty{StlAllocator<T>(allocator()), kAccuracy};

    std::vector<double> values;
    std::default_random_engine gen(common::testutil::getRandomSeed(42));
    std::uniform_real_distribution<> dist;
    for (auto i = 0; i < 100; ++i) {
      auto v = T(dist(gen));
      auto w = (i + 3) % 7 + 1;
      digest.add(v, w);
      values.insert(values.end(), w, v);
    }
    digest.compress();
    std::string original(digest.serializedByteSize(), '\0');
    digest.serialize(original.data());

    if (mergeSerialized) {
      std::string buf(digestEmpty.serializedByteSize(), '\0');
      digestEmpty.serialize(buf.data());
      digest.mergeSerialized(buf.data());
    } else {
      digest.testingMerge(digestEmpty);
    }

    std::string result(digest.serializedByteSize(), '\0');
    digest.serialize(result.data());
    EXPECT_EQ(original.size(), result.size());
    EXPECT_EQ(original, result);
  }

  template <typename T>
  void testMerge(bool mergeSerialized) {
    constexpr double kAccuracy = 0.5;
    QuantileDigest<T> digest1{StlAllocator<T>(allocator()), kAccuracy};
    QuantileDigest<T> digest2{StlAllocator<T>(allocator()), kAccuracy};
    QuantileDigest<T> digestEmpty{StlAllocator<T>(allocator()), kAccuracy};

    std::vector<double> allValues;
    std::vector<double> digest1Values;
    std::default_random_engine gen(common::testutil::getRandomSeed(42));
    std::uniform_real_distribution<> dist;
    for (auto i = 0; i < 10000; ++i) {
      auto v = T(dist(gen));
      auto w = (i + 3) % 7 + 1;
      digest1.add(v, w);
      allValues.insert(allValues.end(), w, v);
      digest1Values.insert(digest1Values.end(), w, v);

      v = T(dist(gen));
      w = (i + 2) % 3 + 1;
      digest2.add(v, w);
      allValues.insert(allValues.end(), w, v);
    }
    if (mergeSerialized) {
      std::string buf(digest1.serializedByteSize(), '\0');
      digest1.serialize(buf.data());
      digestEmpty.mergeSerialized(buf.data());
    } else {
      digestEmpty.testingMerge(digest1);
    }
    std::sort(std::begin(digest1Values), std::end(digest1Values));
    checkQuantiles<QuantileDigest<T>, false>(
        digest1Values, digestEmpty, 0.0, digest1Values.size() * kAccuracy);

    if (mergeSerialized) {
      std::string buf(digest2.serializedByteSize(), '\0');
      digest2.serialize(buf.data());
      digest1.mergeSerialized(buf.data());
    } else {
      digest1.testingMerge(digest2);
    }
    std::sort(std::begin(allValues), std::end(allValues));
    checkQuantiles<QuantileDigest<T>, false>(
        allValues, digest1, 0.0, allValues.size() * kAccuracy);
  }

  template <typename T>
  void testMergeWithJava(bool mergeSerialized) {
    constexpr double kAccuracy = 0.99;
    QuantileDigest<T> digest{StlAllocator<T>(allocator()), 0.99};
    std::vector<double> values;
    for (auto i = 0; i < 10000; ++i) {
      digest.add(static_cast<T>(i), 1.0);
      values.push_back(T(i));
    }

    std::string data;
    if constexpr (std::is_same_v<T, double>) {
      // Presto Query: SELECT QDIGEST_AGG(CAST(c0 AS DOUBLE), 1, 0.99) FROM
      //               UNNEST(SEQUENCE(5000, 8000, 2)) AS t(c0)
      data = decodeBase64(
          "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AAAAN0AAAAAAAAC4wKQAAAAAAAAwQAAAAAAAALzApAAAAAAAADBAAAAAAAAgvMC3AAAAAAAAAAAAAAAAAAC8wKQAAAAAAAAwQAAAAAAAQLzApAAAAAAAADBAAAAAAABgvMC3AAAAAAAAAAAAAAAAAEC8wLsAAAAAAAAAAAAAAAAAALzApAAAAAAAADBAAAAAAADAvMCkAAAAAAAAMEAAAAAAAOC8wLcAAAAAAAAAAAAAAAAAwLzAvwAAAAAAACxAAAAAAAAAvMCkAAAAAAAAMEAAAAAAAAC9wKQAAAAAAAAwQAAAAAAAIL3AtwAAAAAAAAAAAAAAAAAAvcCkAAAAAAAAMEAAAAAAAEC9wKQAAAAAAAAwQAAAAAAAYL3AtwAAAAAAAAAAAAAAAABAvcC7AAAAAAAAAAAAAAAAAAC9wKQAAAAAAAAwQAAAAAAAgL3ApAAAAAAAADBAAAAAAACgvcC3AAAAAAAAAAAAAAAAAIC9wLkAAAAAAAAuQAAAAAAAgL3AvwAAAAAAADFAAAAAAAAAvcDDAAAAAAAAAAAAAAAAAAC8wKQAAAAAAAAwQAAAAAAAAL7ApAAAAAAAADBAAAAAAAAgvsC3AAAAAAAAAAAAAAAAAAC+wKQAAAAAAAAwQAAAAAAAgL7ApAAAAAAAADBAAAAAAACgvsC3AAAAAAAAAAAAAAAAAIC+wKQAAAAAAAAwQAAAAAAAwL7ApAAAAAAAADBAAAAAAADgvsC3AAAAAAAAAAAAAAAAAMC+wLsAAAAAAAAAAAAAAAAAgL7AvwAAAAAAAAAAAAAAAAAAvsCkAAAAAAAAMEAAAAAAAAC/wKQAAAAAAAAwQAAAAAAAIL/AtwAAAAAAAAAAAAAAAAAAv8DDAAAAAAAAO0AAAAAAAAC+wMcAAAAAAAA4QAAAAAAAALzAywAAAAAAADFAAAAAAAAAuMDPAAAAAAAAKkAAAAAAAIizwA==");
    } else {
      // Presto Query: SELECT QDIGEST_AGG(CAST(c0 AS REAL), 1, 0.99)
      //               FROM UNNEST(SEQUENCE(5000, 8000, 2)) AS t(c0)
      data = decodeBase64(
          "AK5H4XoUru8/AAAAAAAAAAAAAAAAAAAAAABAnEUAAAAAAAD6RQAAAAAwAAAAOAAAAAAAAEhAAECcRQAAAIA4AAAAAAAASEAAAKBFAAAAgDgAAAAAAABGQAAApEUAAACASwAAAAAAAAAAAACgRQAAAIA0AAAAAAAAQEAAAKpFAAAAgDQAAAAAAABAQAAArkUAAACASwAAAAAAAD9AAACoRQAAAIBPAAAAAAAARkAAAKBFAAAAgDgAAAAAAABLQABgsEUAAACANAAAAAAAAD5AACC2RQAAAIBLAAAAAAAAAAAAYLBFAAAAgDQAAAAAAABAQAAAuEUAAACAOAAAAAAAgEVAAAC8RQAAAIBLAAAAAAAAAAAAALhFAAAAgE8AAAAAAABHQABgsEUAAACAUwAAAAAAADxAAACgRQAAAIBXAAAAAACAQUAAQJxFAAAAgDgAAAAAAABOQABAwEUAAACASQAAAAAAADxAAEDARQAAAIA4AAAAAAAATkAAQMxFAAAAgEoAAAAAAABGQAAAyEUAAACATwAAAAAAAEhAAEDARQAAAIA0AAAAAAAAQEAAANRFAAAAgDQAAAAAAABAQAAA1kUAAACARwAAAAAAAAAAAADURQAAAIBKAAAAAAAAPEAAwNBFAAAAgDgAAAAAAABOQABA2EUAAACANAAAAAAAADxAAEDeRQAAAIBLAAAAAAAAQkAAQNhFAAAAgE8AAAAAAAAAAADA0EUAAACAUwAAAAAAADxAAEDARQAAAIA0AAAAAAAAQEAAAORFAAAAgDQAAAAAAABAQAAA5kUAAACARwAAAAAAAAAAAADkRQAAAIA0AAAAAAAAPEAAQOpFAAAAgEYAAAAAAABCQAAA6EUAAACANAAAAAAAADRAAMDuRQAAAIBGAAAAAAAARkAAAOxFAAAAgEsAAAAAAAAAAAAA6EUAAACATwAAAAAAAAAAAEDhRQAAAIA4AAAAAAAATkAAQPBFAAAAgCgAAAAAAAAQQADA90UAAACARgAAAAAAAE5AAAD0RQAAAIBLAAAAAAAAAAAAQPBFAAAAgE0AAAAAAIBAQABA8EUAAACAUwAAAAAAAEhAAEDhRQAAAIBXAAAAAAAARkAAQMBFAAAAgFsAAAAAAAA9QABAnEUAAACA");
    }
    for (auto i = 5000; i <= 8000; i += 2) {
      values.push_back(T(i));
    }

    if (mergeSerialized) {
      digest.mergeSerialized(data.data());
    } else {
      QuantileDigest<T> digestJava{StlAllocator<T>(allocator()), data.data()};
      digest.testingMerge(digestJava);
    }
    std::sort(std::begin(values), std::end(values));
    checkQuantiles<QuantileDigest<T>, false>(
        values, digest, 0.0, values.size() * kAccuracy);
  }

  template <typename T>
  void testScale() {
    constexpr double kAccuracy = 0.75;
    QuantileDigest<T> digest{StlAllocator<T>(allocator()), kAccuracy};
    std::vector<double> values;
    for (auto i = 0; i < 10000; ++i) {
      digest.add(static_cast<T>(i), 1.0);
      values.push_back(T(i));
    }

    std::sort(std::begin(values), std::end(values));
    checkQuantiles<QuantileDigest<T>, false>(
        values, digest, 0.0, values.size() * kAccuracy);

    digest.scale(10.0);
    std::vector<double> scaledValues;
    for (const auto value : values) {
      scaledValues.insert(scaledValues.end(), 10, value);
    }
    checkQuantiles<QuantileDigest<T>, false>(
        scaledValues, digest, 0.0, scaledValues.size() * kAccuracy);
  }

 private:
  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::memoryManager()->addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool_{rootPool_->addLeafChild("leaf")};
  HashStringAllocator allocator_{pool_.get()};
};

TEST_F(QuantileDigestTest, basic) {
  std::vector<double> quantiles{
      0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  testQuantiles<int64_t>(
      {-5, -4, 4, -3, 3, -2, 2, -1, 1, 0},
      quantiles,
      {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4});
  testQuantiles<double>(
      {-5.0, -4.0, 4.0, -3.0, 3.0, -2.0, 2.0, -1.0, 1.0, 0.0},
      quantiles,
      {-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0});
  testQuantiles<float>(
      {-5.0, -4.0, 4.0, -3.0, 3.0, -2.0, 2.0, -1.0, 1.0, 0.0},
      quantiles,
      {-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0});
}

TEST_F(QuantileDigestTest, weighted) {
  std::vector<double> quantiles{
      0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

  testQuantiles<int64_t>(
      {0, 2, 4, 5}, quantiles, {0, 0, 0, 2, 4, 4, 4, 4, 4, 5, 5}, {3, 1, 5, 1});
  testQuantiles<double>(
      {0.0, 2.0, 4.0, 5.0},
      quantiles,
      {0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0},
      {3, 1, 5, 1});
  testQuantiles<float>(
      {0.0, 2.0, 4.0, 5.0},
      quantiles,
      {0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0},
      {3, 1, 5, 1});
}

TEST_F(QuantileDigestTest, largeInputSize) {
  testLargeInputSize<double>();
  testLargeInputSize<float>();
}

TEST_F(QuantileDigestTest, checkQuantilesAfterSerDe) {
  QuantileDigest<int64_t> digestBigint{StlAllocator<int64_t>(allocator()), 0.8};
  QuantileDigest<double> digestDouble{StlAllocator<double>(allocator()), 0.8};
  for (auto i = 0; i < 100; ++i) {
    digestBigint.add(static_cast<int64_t>(i), 1.0);
    digestDouble.add(static_cast<double>(i), 1.0);
  }
  // Repeat SerDe three times to match Presto result of a partial ->
  // intermediate -> intermediate -> final aggregation plan.
  // Query: SELECT
  //          VALUES_AT_QUANTILES(
  //            QDIGEST_AGG(CAST(c0 AS BIGINT/DOUBLE), 1, 0.8),
  //            ARRAY[0.01, 0.1, 0.15, 0.3, 0.55, 0.7, 0.85, 0.9, 0.99]
  //        )
  //        FROM UNNEST(SEQUENCE(0, 99)) AS t(c0)
  for (auto i = 1; i <= 3; ++i) {
    std::string buf(1 + digestBigint.serializedByteSize(), '\0');
    digestBigint.serialize(buf.data());
    digestBigint =
        QuantileDigest<int64_t>{StlAllocator<int64_t>(allocator()), buf.data()};

    buf.resize(1 + digestDouble.serializedByteSize(), '\0');
    digestDouble.serialize(buf.data());
    digestDouble =
        QuantileDigest<double>{StlAllocator<double>(allocator()), buf.data()};
  }

  std::vector<double> quantiles{
      0.01, 0.1, 0.15, 0.3, 0.55, 0.7, 0.85, 0.9, 0.99};
  std::vector<int64_t> expectedBigint{0, 8, 8, 40, 63, 80, 95, 96, 99};
  std::vector<int64_t> resultsBigint(quantiles.size());
  digestBigint.estimateQuantiles(quantiles, resultsBigint.data());
  for (auto i = 0; i < quantiles.size(); ++i) {
    EXPECT_EQ(resultsBigint[i], expectedBigint[i]);
  }

  std::vector<double> expectedDouble{
      1.0, 10.0, 15.0, 30.0, 55.0, 70.0, 85.0, 90.0, 99.0};
  std::vector<double> resultsDouble(quantiles.size());
  digestDouble.estimateQuantiles(quantiles, resultsDouble.data());
  for (auto i = 0; i < quantiles.size(); ++i) {
    EXPECT_EQ(resultsDouble[i], expectedDouble[i]);
  }
}

TEST_F(QuantileDigestTest, serializedMatchJava) {
  // Test small input size.
  // Query: SELECT QDIGEST_AGG(CAST(c0 AS BIGINT/DOUBLE), 1, 0.99) FROM
  //        UNNEST(SEQUENCE(-5, 4)) AS t(c0)
  {
    QuantileDigest<int64_t> digestBigint{
        StlAllocator<int64_t>(allocator()), 0.99};
    QuantileDigest<double> digestDouble{
        StlAllocator<double>(allocator()), 0.99};
    QuantileDigest<float> digestReal{StlAllocator<float>(allocator()), 0.99};
    for (auto i = -5; i < 5; ++i) {
      digestBigint.add(static_cast<int64_t>(i), 1.0);
      digestDouble.add(static_cast<double>(i), 1.0);
      digestReal.add(static_cast<float>(i), 1.0);
    }
    std::string buf(1 + digestBigint.serializedByteSize(), '\0');
    auto length = digestBigint.serialize(buf.data());
    buf.resize(length);
    auto encodedBuf = encodeBase64(buf);
    ASSERT_EQ(
        encodedBuf,
        "AK5H4XoUru8/AAAAAAAAAAAAAAAAAAAAAPv/////////BAAAAAAAAAATAAAAAAAAAAAAAPA/+////////38AAAAAAAAA8D/8////////fwAAAAAAAADwP/3///////9/AwAAAAAAAAAA/P///////38AAAAAAAAA8D/+////////fwAAAAAAAADwP/////////9/AwAAAAAAAAAA/v///////38HAAAAAAAAAAD8////////fwsAAAAAAAAAAPv///////9/AAAAAAAAAPA/AAAAAAAAAIAAAAAAAAAA8D8BAAAAAAAAgAMAAAAAAAAAAAAAAAAAAACAAAAAAAAAAPA/AgAAAAAAAIAAAAAAAAAA8D8DAAAAAAAAgAMAAAAAAAAAAAIAAAAAAACABwAAAAAAAAAAAAAAAAAAAIAAAAAAAAAA8D8EAAAAAAAAgAsAAAAAAAAAAAAAAAAAAACA/wAAAAAAAAAA+////////38=");

    buf.clear();
    buf.resize(1 + digestDouble.serializedByteSize(), '\0');
    length = digestDouble.serialize(buf.data());
    buf.resize(length);
    encodedBuf = encodeBase64(buf);
    ASSERT_EQ(
        encodedBuf,
        "AK5H4XoUru8/AAAAAAAAAAAAAAAAAAAAAP///////+u/AAAAAAAAEEATAAAAAAAAAAAAAPA/////////6z8AAAAAAAAA8D/////////vP8sAAAAAAAAAAP///////+s/AAAAAAAAAPA/////////9z8AAAAAAAAA8D//////////P88AAAAAAAAAAP////////c/0wAAAAAAAAAA////////6z8AAAAAAAAA8D////////8PQPsAAAAAAAAAAP///////+s/AAAAAAAAAPA/AAAAAAAAAIAAAAAAAAAA8D8AAAAAAADwv/cAAAAAAAAAAAAAAAAAAACAAAAAAAAAAPA/AAAAAAAAAMAAAAAAAAAA8D8AAAAAAAAIwM8AAAAAAAAAAAAAAAAAAADAAAAAAAAAAPA/AAAAAAAAEMDTAAAAAAAAAAAAAAAAAAAAwPsAAAAAAAAAAAAAAAAAAACA/wAAAAAAAAAA////////6z8=");

    buf.clear();
    buf.resize(1 + digestReal.serializedByteSize(), '\0');
    length = digestReal.serialize(buf.data());
    buf.resize(length);
    encodedBuf = encodeBase64(buf);
    ASSERT_EQ(
        encodedBuf,
        "AK5H4XoUru8/AAAAAAAAAAAAAAAAAAAAAP//X7//////AACAQAAAAAATAAAAAAAAAAAAAPA///9fv////38AAAAAAAAA8D///3+/////f1cAAAAAAAAAAP//X7////9/AAAAAAAAAPA///+/v////38AAAAAAAAA8D////+/////f1sAAAAAAAAAAP//v7////9/XwAAAAAAAAAA//9fv////38AAAAAAAAA8D///3/A////f3sAAAAAAAAAAP//X7////9/AAAAAAAAAPA/AAAAAAAAAIAAAAAAAAAA8D8AAIA/AAAAgHcAAAAAAAAAAAAAAAAAAACAAAAAAAAAAPA/AAAAQAAAAIAAAAAAAAAA8D8AAEBAAAAAgFsAAAAAAAAAAAAAAEAAAACAAAAAAAAAAPA/AACAQAAAAIBfAAAAAAAAAAAAAABAAAAAgHsAAAAAAAAAAAAAAAAAAACA/wAAAAAAAAAA//9fv////38=");
  }

  // Test large input size.
  // Query: SELECT QDIGEST_AGG(CAST(c0 AS BIGINT), 1, 0.99) FROM
  //        UNNEST(SEQUENCE(-5000, 4999)) AS t(c0)
  {
    QuantileDigest<int64_t> digestBigint{
        StlAllocator<int64_t>(allocator()), 0.99};
    QuantileDigest<double> digestDouble{
        StlAllocator<double>(allocator()), 0.99};
    QuantileDigest<float> digestReal{StlAllocator<float>(allocator()), 0.99};
    for (auto i = -5000; i < 5000; ++i) {
      digestBigint.add(static_cast<int64_t>(i), 1.0);
      digestDouble.add(static_cast<double>(i), 1.0);
      digestReal.add(static_cast<float>(i), 1.0);
    }

    std::string bufBigint(1 + digestBigint.serializedByteSize(), '\0');
    std::string bufDouble(1 + digestDouble.serializedByteSize(), '\0');
    std::string bufReal(1 + digestReal.serializedByteSize(), '\0');
    auto length = digestBigint.serialize(bufBigint.data());
    bufBigint.resize(length);
    length = digestDouble.serialize(bufDouble.data());
    bufDouble.resize(length);
    length = digestReal.serialize(bufReal.data());
    bufReal.resize(length);
    for (auto i = 1; i <= 3; ++i) {
      digestBigint = QuantileDigest<int64_t>{
          StlAllocator<int64_t>(allocator()), bufBigint.data()};
      bufBigint.clear();
      bufBigint.resize(1 + digestBigint.serializedByteSize(), '\0');
      length = digestBigint.serialize(bufBigint.data());
      bufBigint.resize(length);

      digestDouble = QuantileDigest<double>{
          StlAllocator<double>(allocator()), bufDouble.data()};
      bufDouble.clear();
      bufDouble.resize(1 + digestDouble.serializedByteSize(), '\0');
      length = digestDouble.serialize(bufDouble.data());
      bufDouble.resize(length);

      digestReal = QuantileDigest<float>{
          StlAllocator<float>(allocator()), bufReal.data()};
      bufReal.clear();
      bufReal.resize(1 + digestReal.serializedByteSize(), '\0');
      length = digestReal.serialize(bufReal.data());
      bufReal.resize(length);
    }
    auto encodedBuf = encodeBase64(bufBigint);
    ASSERT_EQ(
        encodedBuf,
        "AK5H4XoUru8/AAAAAAAAAAAAAAAAAAAAAHjs////////hxMAAAAAAABUAAAAFAAAAAAA4HJAeOz//////38MAAAAAACAWUAa7v//////fxAAAAAAAABnQAHv//////9/IwAAAAAAwGNAGu7//////38nAAAAAAAAAAB47P//////fxAAAAAAAIBtQBTw//////9/EAAAAAAAAGFADPL//////38QAAAAAAAAb0AI8///////fyMAAAAAAAAAAAzy//////9/JwAAAAAAAHFAFPD//////38UAAAAAABgdEAg9P//////fxQAAAAAAKBzQAr2//////9/JwAAAAAAgGdAIPT//////38rAAAAAACAZEAU8P//////fxAAAAAAAMBvQAL4//////9/EAAAAAAAAHBAAPn//////38jAAAAAAAAAAAC+P//////fyUAAAAAAIBvQAL4//////9/EAAAAAAAgGlANP3//////38AAAAAAAAAAED+////////fx4AAAAAAABlQFb///////9/IgAAAAAAAHVABv7//////38nAAAAAADAckAO/P//////fysAAAAAACBxQAL4//////9/LwAAAAAAAAAAFPD//////38zAAAAAABAdEB47P//////fwgAAAAAAABNQEYAAAAAAACADAAAAAAAQF5AhwAAAAAAAIAfAAAAAAAAAAAAAAAAAAAAgAwAAAAAAABgQAABAAAAAACADAAAAAAAAGBAgAEAAAAAAIAfAAAAAAAAAAAAAQAAAAAAgCMAAAAAAAAAAAAAAAAAAACADAAAAAAAAGBAAAIAAAAAAIAMAAAAAAAAYECAAgAAAAAAgB8AAAAAAAAAAAACAAAAAACAIQAAAAAAQFtAAAIAAAAAAIAnAAAAAABgYkAAAAAAAAAAgAwAAAAAAABgQAAEAAAAAACADAAAAAAAAGBAgAQAAAAAAIAfAAAAAAAAAAAABAAAAAAAgAwAAAAAAEBZQJsFAAAAAACAIwAAAAAAQFBAAAQAAAAAAIAMAAAAAAAAYEAABgAAAAAAgAwAAAAAAABgQIAGAAAAAACAHwAAAAAAAAAAAAYAAAAAAIAMAAAAAABAV0CjBwAAAAAAgB4AAAAAAEBQQGIHAAAAAACAIwAAAAAAgFhAAAYAAAAAAIAnAAAAAACAVkAABAAAAAAAgCsAAAAAAAAAAAAAAAAAAACADAAAAAAAAGBAAAgAAAAAAIAMAAAAAAAAYECACAAAAAAAgB8AAAAAAAAAAAAIAAAAAACAIQAAAAAAwGJAAAgAAAAAAIAMAAAAAAAAYEAACgAAAAAAgAwAAAAAAABgQIAKAAAAAACAHwAAAAAAAAAAAAoAAAAAAIAQAAAAAADAYUByCwAAAAAAgCMAAAAAAIBcQAAKAAAAAACAJwAAAAAAAAAAAAgAAAAAAIAMAAAAAAAAYEAADAAAAAAAgAwAAAAAAABgQIAMAAAAAACAHwAAAAAAAAAAAAwAAAAAAIAhAAAAAADAYEAADAAAAAAAgAwAAAAAAABgQAAOAAAAAACADAAAAAAAAGBAgA4AAAAAAIAfAAAAAAAAAAAADgAAAAAAgAwAAAAAAIBfQIIPAAAAAACAIwAAAAAAQGBAAA4AAAAAAIAnAAAAAAAAAAAADAAAAAAAgCsAAAAAAIBeQAAIAAAAAACALwAAAAAAgFpAAAAAAAAAAIAMAAAAAACAXkCGEAAAAAAAgAwAAAAAAIBdQIoRAAAAAACAHgAAAAAAQGBACBEAAAAAAIAjAAAAAAAAAAAEEAAAAAAAgAwAAAAAAIBcQI4SAAAAAACAHgAAAAAAwGFAABIAAAAAAIAQAAAAAAAAYUAAEwAAAAAAgCMAAAAAAAAAAAASAAAAAACAJwAAAAAAQGFABBAAAAAAAIAzAAAAAABAVEAAAAAAAAAAgP8AAAAAAAAAAHjs//////9/");

    encodedBuf = encodeBase64(bufDouble);
    ASSERT_EQ(
        encodedBuf,
        "AK5H4XoUru8/AAAAAAAAAAAAAAAAAAAAAP//////d0y/AAAAAACHs0B0AAAAsAAAAAAAAGBA//////93TD+sAAAAAABAVkD//////wBNP6wAAAAAAABUQP//////gE0/vwAAAAAAwFRA//////8ATT/DAAAAAAAAAAD//////3dMP6wAAAAAAABUQP//////gE4/sAAAAAAAAGBA//////8ATz/DAAAAAAAAAAD//////wBOP8cAAAAAAABaQP//////d0w/sAAAAAAAAFhA//////8BUD+wAAAAAAAAUED//////wFRP8MAAAAAAAAAAP//////AVA/rAAAAAAAAFBA//////8BUj+wAAAAAAAAWED//////wFTP8MAAAAAAAAAAP//////AVI/xwAAAAAA4GBA//////8BUD+0AAAAAAAAYED//////wFUP7AAAAAAAMBfQP//////A1Y/rAAAAAAAAFBA//////8BVz/DAAAAAAAAAAD//////wNWP8cAAAAAAIBcQP//////AVQ/ywAAAAAAQFVA//////8BUD+0AAAAAAAAYED//////wFYP6wAAAAAAABQQP//////AVo/rAAAAAAAAFBA//////+BWz/DAAAAAAAAWUD//////wFaP8cAAAAAAABVQP//////AVg/tAAAAAAAoGJA//////9XXD+wAAAAAADAVkD//////wFfP8IAAAAAAMBcQP//////B14/xwAAAAAAAFZA//////9XXD/LAAAAAABAV0D//////wFYP88AAAAAAAAAAP//////AVA/0wAAAAAAgGJA//////93TD+0AAAAAAAAV0D//////wNkP7QAAAAAAIBbQP//////S2Y/xwAAAAAAAAAA//////8DZD/KAAAAAAAAYED//////0dgP7QAAAAAAABgQP//////A2g/xQAAAAAAwFZA//////8DaD+0AAAAAADAX0D//////wduP8YAAAAAAIBbQP//////T2w/ywAAAAAAAAAA//////8DaD/PAAAAAACAXUD//////0dgP7gAAAAAAABgQP//////B3A/uAAAAAAAAGBA//////8HeD+4AAAAAABAVkD//////z99P8sAAAAAAAAAAP//////B3g/zwAAAAAAAFdA//////8HcD/TAAAAAABgYED//////0dgP9cAAAAAAIBYQP//////d0w/vAAAAAAAAGBA//////8PiD/AAAAAAAAAYED//////x+QP9MAAAAAAIBcQP//////74A/xAAAAAAAAFhA//////8/oD/XAAAAAAAAAAD//////++AP98AAAAAAABZQP//////d0w/vAAAAAAAAExAAAAAAAAAacC4AAAAAAAAS0AAAAAAAKB0wLwAAAAAAEBdQAAAAAAAsHjAzwAAAAAAAFNAAAAAAACQcMDTAAAAAACAUkAAAAAAAOBgwLQAAAAAAABJQAAAAAAAcILAuAAAAAAAQFxAAAAAAAB4hMDLAAAAAAAAAAAAAAAAAGiAwLgAAAAAAABgQAAAAAAAAIjAuAAAAAAAAGBAAAAAAAAAjMDLAAAAAAAAAAAAAAAAAACIwM8AAAAAAAAAAAAAAAAAaIDAtAAAAAAAAGBAAAAAAAAAkMC0AAAAAAAAYEAAAAAAAACSwMcAAAAAAAAAAAAAAAAAAJDAtAAAAAAAQFlAAAAAAABslsDLAAAAAABAUEAAAAAAAACQwLQAAAAAAABgQAAAAAAAAJjAtAAAAAAAAGBAAAAAAAAAmsDHAAAAAAAAAAAAAAAAAACYwLQAAAAAAEBXQAAAAAAAjJ7AxgAAAAAAQFBAAAAAAACIncDLAAAAAACAWEAAAAAAAACYwM8AAAAAAAAAAAAAAAAAAJDA0wAAAAAAgFZAAAAAAABogMCwAAAAAAAAYEAAAAAAAACgwLAAAAAAAABgQAAAAAAAAKHAwwAAAAAAAAAAAAAAAAAAoMDFAAAAAADAYkAAAAAAAACgwLAAAAAAAABgQAAAAAAAAKTAsAAAAAAAAGBAAAAAAAAApcDDAAAAAAAAAAAAAAAAAACkwLQAAAAAAMBhQAAAAAAA5KbAxwAAAAAAgFxAAAAAAAAApMDLAAAAAAAAAAAAAAAAAACgwLAAAAAAAABgQAAAAAAAAKjAsAAAAAAAAGBAAAAAAAAAqcDDAAAAAAAAAAAAAAAAAACowMUAAAAAAMBgQAAAAAAAAKjAsAAAAAAAAGBAAAAAAAAArMCwAAAAAAAAYEAAAAAAAACtwMMAAAAAAAAAAAAAAAAAAKzAsAAAAAAAgF9AAAAAAAAEr8DHAAAAAABAYEAAAAAAAACswMsAAAAAAAAAAAAAAAAAAKjAzwAAAAAAgF5AAAAAAAAAoMCsAAAAAACAXkAAAAAAAIawwKwAAAAAAIBdQAAAAAAAirHAvgAAAAAAQGBAAAAAAAAIscDDAAAAAAAAAAAAAAAAAASwwKwAAAAAAIBcQAAAAAAAjrLAvgAAAAAAwGFAAAAAAAAAssCwAAAAAAAAYUAAAAAAAACzwMMAAAAAAAAAAAAAAAAAALLAxwAAAAAAQGFAAAAAAAAEsMDTAAAAAACAW0AAAAAAAACgwNcAAAAAAABUQAAAAAAAaIDA3wAAAAAAgFNAAAAAAACAUcD/AAAAAAAgYkD//////3dMPw==");

    encodedBuf = encodeBase64(bufReal);
    ASSERT_EQ(
        encodedBuf,
        "AK5H4XoUru8/AAAAAAAAAAAAAAAAAAAAAP+/Y7r/////ADicRQAAAABuAAAAPAAAAAAA4GJA/wdouv///39OAAAAAAAAZUD/v2O6////fzwAAAAAAEBiQP8fcLr///9/PAAAAAAAIGNA/wd4uv///39PAAAAAACAYkD/H3C6////f1MAAAAAAMBbQP+/Y7r///9/QAAAAAAAwGJA/w+Auv///388AAAAAACAXED/75C6////f1MAAAAAAMBhQP8PgLr///9/PAAAAAAAAGBA/w+guv///388AAAAAACAX0D/L7C6////fzgAAAAAAABLQP+vvLr///9/TwAAAAAAAAAA/y+wuv///39TAAAAAABgYkD/D6C6////f1cAAAAAAMBaQP8PgLr///9/PAAAAAAAwFpA/1/Buv///388AAAAAAAAYED/D8i6////f08AAAAAAAAAAP9fwbr///9/UQAAAAAAwGRA/1/Buv///388AAAAAADAXUD/n+C6////fzwAAAAAAABgQP8P6Lr///9/TwAAAAAAAAAA/5/guv///39AAAAAAAAgY0D/f/a6////f1MAAAAAAMBZQP+f4Lr///9/VwAAAAAAAFVA/1/Buv///39bAAAAAACAVUD/D4C6////f18AAAAAAAAAAP+/Y7r///9/QAAAAAAAQFxA//8hu////39AAAAAAAAAYED/HzC7////f1MAAAAAAAAAAP//Ibv///9/VgAAAAAAQFZA//8Hu////39AAAAAAAAAYED/H0C7////f0AAAAAAAABgQP8fULv///9/UwAAAAAAAAAA/x9Au////39EAAAAAAAgYED//2+7////f1cAAAAAAMBfQP8fQLv///9/WwAAAAAAAAAA//8Hu////39EAAAAAABAUkD//627////f0QAAAAAAMBXQP9/6Lv///9/VgAAAAAAgGNA/3/Bu////39bAAAAAAAAAAD//4a7////f18AAAAAACBkQP//B7v///9/YwAAAAAAQGJA/79juv///39IAAAAAABAWED//0+8////f0wAAAAAAEBcQP//j7z///9/XwAAAAAAYGNA//8JvP///39QAAAAAAAAWED//wG9////f2MAAAAAAAAAAP//Cbz///9/ZQAAAAAAAD5A//8JvP///39rAAAAAAAAAAD/v2O6////f3kAAAAAAEBdQP+/Y7r///9/SAAAAAAAAExAAABIQwAAAIBEAAAAAAAAS0AAAKVDAAAAgEgAAAAAAEBdQACAxUMAAACAWwAAAAAAAFNAAICEQwAAAIBfAAAAAACAUkAAAAdDAAAAgEAAAAAAAABJQACAE0QAAACARAAAAAAAQFxAAMAjRAAAAIBXAAAAAAAAAAAAQANEAAAAgEQAAAAAAABgQAAAQEQAAACARAAAAAAAAGBAAABgRAAAAIBXAAAAAAAAAAAAAEBEAAAAgFsAAAAAAAAAAABAA0QAAACAQAAAAAAAAGBAAACARAAAAIBAAAAAAAAAYEAAAJBEAAAAgFMAAAAAAAAAAAAAgEQAAACAQAAAAAAAQFlAAGCzRAAAAIBXAAAAAABAUEAAAIBEAAAAgEAAAAAAAABgQAAAwEQAAACAQAAAAAAAAGBAAADQRAAAAIBTAAAAAAAAAAAAAMBEAAAAgEAAAAAAAEBXQABg9EQAAACAUgAAAAAAQFBAAEDsRAAAAIBXAAAAAACAWEAAAMBEAAAAgFsAAAAAAAAAAAAAgEQAAACAXwAAAAAAgFZAAEADRAAAAIA8AAAAAAAAYEAAAABFAAAAgDwAAAAAAABgQAAACEUAAACATwAAAAAAAAAAAAAARQAAAIBRAAAAAADAYkAAAABFAAAAgDwAAAAAAABgQAAAIEUAAACAPAAAAAAAAGBAAAAoRQAAAIBPAAAAAAAAAAAAACBFAAAAgEAAAAAAAMBhQAAgN0UAAACAUwAAAAAAgFxAAAAgRQAAAIBXAAAAAAAAAAAAAABFAAAAgDwAAAAAAABgQAAAQEUAAACAPAAAAAAAAGBAAABIRQAAAIBPAAAAAAAAAAAAAEBFAAAAgFEAAAAAAMBgQAAAQEUAAACAPAAAAAAAAGBAAABgRQAAAIA8AAAAAAAAYEAAAGhFAAAAgE8AAAAAAAAAAAAAYEUAAACAPAAAAAAAgF9AACB4RQAAAIBTAAAAAABAYEAAAGBFAAAAgFcAAAAAAAAAAAAAQEUAAACAWwAAAAAAgF5AAAAARQAAAIA4AAAAAACAXkAAMIRFAAAAgDgAAAAAAIBdQABQjEUAAACASgAAAAAAQGBAAECIRQAAAIBPAAAAAAAAAAAAIIBFAAAAgDgAAAAAAIBcQABwlEUAAACASgAAAAAAwGFAAACQRQAAAIA8AAAAAAAAYUAAAJhFAAAAgE8AAAAAAAAAAAAAkEUAAACAUwAAAAAAQGFAACCARQAAAIBfAAAAAACAW0AAAABFAAAAgGMAAAAAAABUQABAA0QAAACAawAAAAAAgFNAAACMQgAAAID/AAAAAACAUUD/v2O6////fw==");
  }
}

TEST_F(QuantileDigestTest, merge) {
  testMerge<double>(true);
  testMerge<float>(true);

  testMerge<double>(false);
  testMerge<float>(false);
}

TEST_F(QuantileDigestTest, mergeWithJava) {
  testMergeWithJava<double>(true);
  testMergeWithJava<float>(true);

  testMergeWithJava<double>(false);
  testMergeWithJava<float>(false);
}

TEST_F(QuantileDigestTest, mergeWithEmpty) {
  testMergeEmpty<double>(true);
  testMergeEmpty<float>(true);

  testMergeEmpty<double>(false);
  testMergeEmpty<float>(false);
}

TEST_F(QuantileDigestTest, infinity) {
  const double kInf = std::numeric_limits<double>::infinity();
  const float kFInf = std::numeric_limits<float>::infinity();
  std::vector<double> quantiles{0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0};
  testQuantiles<double>(
      {0.0, kInf, -kInf}, quantiles, {-kInf, -kInf, 0.0, 0.0, 0.0, kInf, kInf});
  testQuantiles<float>(
      {0.0, kFInf, -kFInf},
      quantiles,
      {-kFInf, -kFInf, 0.0, 0.0, 0.0, kFInf, kFInf});
}

TEST_F(QuantileDigestTest, minMax) {
  const double kAccuracy = 0.05;
  QuantileDigest<int64_t> digestBigint{
      StlAllocator<int64_t>(allocator()), kAccuracy};
  QuantileDigest<double> digestDouble{
      StlAllocator<double>(allocator()), kAccuracy};
  QuantileDigest<float> digestReal{StlAllocator<float>(allocator()), kAccuracy};

  int64_t from = -12345;
  int64_t to = 54321;
  for (auto i = from; i <= to; ++i) {
    digestBigint.add(i, 1);
    digestDouble.add(static_cast<double>(i), 1);
    digestReal.add(static_cast<float>(i), 1);
  }

  auto rankError = (to - from + 1) * kAccuracy;
  ASSERT_LE(std::abs(from - digestBigint.getMin()), rankError);
  ASSERT_LE(std::abs(to - digestBigint.getMax()), rankError);
  ASSERT_LE(std::abs(from - digestDouble.getMin()), rankError);
  ASSERT_LE(std::abs(to - digestDouble.getMax()), rankError);
  ASSERT_LE(std::abs(from - digestReal.getMin()), rankError);
  ASSERT_LE(std::abs(to - digestReal.getMax()), rankError);
}

TEST_F(QuantileDigestTest, scale) {
  testScale<int64_t>();
  testScale<double>();
  testScale<float>();
}

TEST_F(QuantileDigestTest, hugeWeight) {
  VELOX_ASSERT_THROW(
      testHugeWeight<int64_t>(), "Weighted count in digest is too large");
  VELOX_ASSERT_THROW(
      testHugeWeight<double>(), "Weighted count in digest is too large");
  VELOX_ASSERT_THROW(
      testHugeWeight<float>(), "Weighted count in digest is too large");
}

} // namespace facebook::velox::functions
