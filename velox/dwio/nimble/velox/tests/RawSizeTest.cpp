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
#include <gtest/gtest.h>

#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/nimble/velox/RawSizeUtils.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook;

constexpr velox::vector_size_t VECTOR_SIZE = 100;

std::function<bool(velox::vector_size_t)> randomNulls(int32_t n) {
  return [n](velox::vector_size_t /*index*/) {
    return folly::Random::rand32() % n == 0;
  };
}

std::function<bool(velox::vector_size_t)> noNulls() {
  return [](velox::vector_size_t /*index*/) { return false; };
}

std::string generateRandomString() {
  auto length = folly::Random::rand32() % VECTOR_SIZE + 1;
  return std::string(length, 'A');
}

// Fixed width types
template <typename T>
T getValue(velox::vector_size_t i) {
  return i;
}

// Fixed width types
template <typename T>
uint64_t getSize(T /*value*/) {
  return sizeof(T);
}

// Specialization for StringView type
template <>
velox::StringView getValue<velox::StringView>(velox::vector_size_t /*i*/) {
  static std::shared_ptr<std::string> str =
      std::make_shared<std::string>(generateRandomString());
  return velox::StringView(str->data(), str->size());
}

// Specialization for StringView type
template <>
uint64_t getSize<velox::StringView>(velox::StringView value) {
  return value.size();
}

template <>
velox::Timestamp getValue<velox::Timestamp>(velox::vector_size_t /*i*/) {
  int64_t seconds = folly::Random::rand32() % 1'000'000'000;
  uint64_t nanos = folly::Random::rand32() % 1'000'000'000;

  return velox::Timestamp(seconds, nanos);
}

template <>
uint64_t getSize<velox::Timestamp>(velox::Timestamp /*value*/) {
  // Timestamp logical size in bytes: 8 bytes for seconds (int64) + 4 bytes for
  // nanos (int32). This matches DWRF behavior and Nimble FieldWriter's
  // kTimestampLogicalSize, not sizeof(velox::Timestamp) which is 16 bytes.
  return 12;
}

class RawSizeBaseTestFixture : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::initialize(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool();
    vectorMaker_ =
        std::make_unique<velox::test::VectorMaker>(this->pool_.get());
    ranges_.clear();
  }

  void reset() {
    ranges_.clear();
  }

  velox::BufferPtr randomIndices(velox::vector_size_t size) {
    velox::BufferPtr indices =
        velox::AlignedBuffer::allocate<velox::vector_size_t>(size, pool_.get());
    auto rawIndices = indices->asMutable<velox::vector_size_t>();
    for (int32_t i = 0; i < size; i++) {
      rawIndices[i] = folly::Random::rand32(size);
    }
    return indices;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  velox::common::Ranges ranges_;
  nimble::RawSizeContext context_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
};

class RawSizeTestFixture : public RawSizeBaseTestFixture {
 protected:
  template <typename T>
  void testFlat(const std::function<bool(velox::vector_size_t)>& isNullAt) {
    reset();
    std::vector<std::optional<T>> vec;
    vec.reserve(VECTOR_SIZE);
    uint64_t expectedRawSize = 0;
    for (velox::vector_size_t i = 1; i <= VECTOR_SIZE; ++i) {
      if (isNullAt(i)) {
        expectedRawSize += nimble::kNullSize;
        vec.emplace_back(std::nullopt);
      } else {
        auto value = getValue<T>(i);
        expectedRawSize += getSize<T>(value);
        vec.emplace_back(value);
      }
    }
    auto flatVector = vectorMaker_->flatVectorNullable<T>(vec);
    ranges_.add(0, flatVector->size());
    auto rawSize = nimble::getRawSizeFromVector(flatVector, ranges_);
    ASSERT_EQ(expectedRawSize, rawSize);
  }

  template <typename T>
  void testConstant(const std::function<bool(velox::vector_size_t)>& isNullAt) {
    reset();
    std::vector<std::optional<T>> vec;
    vec.reserve(VECTOR_SIZE);
    uint64_t expectedRawSize = 0;
    if (isNullAt(1)) {
      vec.assign(VECTOR_SIZE, std::nullopt);
      expectedRawSize += nimble::kNullSize * VECTOR_SIZE;
    } else {
      auto valueAt = getValue<T>(1);
      expectedRawSize += getSize<T>(valueAt) * VECTOR_SIZE;
      vec.assign(VECTOR_SIZE, valueAt);
    }
    auto flatVector = vectorMaker_->constantVector<T>(vec);
    ranges_.add(0, flatVector->size());
    auto rawSize = nimble::getRawSizeFromVector(flatVector, ranges_);
    ASSERT_EQ(expectedRawSize, rawSize);
  }

  template <typename T>
  void testDictionary(
      const std::function<bool(velox::vector_size_t)>& isNullAt) {
    reset();
    std::vector<std::optional<T>> vec;
    vec.reserve(VECTOR_SIZE);
    uint64_t expectedRawSize = 0;
    for (velox::vector_size_t i = 1; i <= VECTOR_SIZE; ++i) {
      if (isNullAt(i)) {
        vec.emplace_back(std::nullopt);
      } else {
        auto value = getValue<T>(i);
        vec.emplace_back(value);
      }
    }
    auto flatVector = vectorMaker_->flatVectorNullable<T>(vec);
    auto indices = randomIndices(VECTOR_SIZE);
    const velox::vector_size_t* data = indices->as<velox::vector_size_t>();
    for (auto i = 0; i < VECTOR_SIZE; ++i) {
      if (vec[data[i]] == std::nullopt) {
        expectedRawSize += nimble::kNullSize;
      } else {
        expectedRawSize += getSize<T>(vec[data[i]].value());
      }
    }
    auto wrappedVector = velox::BaseVector::wrapInDictionary(
        velox::BufferPtr(nullptr), indices, VECTOR_SIZE, flatVector);
    ranges_.add(0, wrappedVector->size());
    auto rawSize = nimble::getRawSizeFromVector(wrappedVector, ranges_);
    ASSERT_EQ(expectedRawSize, rawSize);
  }

  template <typename T>
  void testArray(const std::function<bool(velox::vector_size_t)>& isNullAt) {
    reset();
    std::vector<std::optional<std::vector<std::optional<T>>>> vec;
    vec.reserve(VECTOR_SIZE);
    uint64_t expectedRawSize = 0;
    for (velox::vector_size_t i = 1; i <= VECTOR_SIZE; ++i) {
      if (isNullAt(i)) {
        vec.emplace_back(std::nullopt);
        expectedRawSize += nimble::kNullSize;
      } else {
        std::vector<std::optional<T>> innerVec;
        velox::vector_size_t innerSize =
            folly::Random::rand32() % VECTOR_SIZE + 1;
        for (velox::vector_size_t j = 0; j < innerSize; ++j) {
          if (isNullAt(j)) {
            innerVec.emplace_back(std::nullopt);
            expectedRawSize += nimble::kNullSize;
          } else {
            auto value = getValue<T>(j);
            expectedRawSize += getSize<T>(value);
            innerVec.emplace_back(value);
          }
        }
        vec.emplace_back(innerVec);
      }
    }
    auto arrayVector = vectorMaker_->arrayVectorNullable<T>(vec);
    ranges_.add(0, arrayVector->size());
    auto rawSize = nimble::getRawSizeFromVector(arrayVector, ranges_);
    ASSERT_EQ(expectedRawSize, rawSize);
  }

  template <typename T>
  void testConstantArray(
      const std::function<bool(velox::vector_size_t)>& isNullAt) {
    reset();
    std::vector<std::optional<std::vector<std::optional<T>>>> vec;
    uint64_t expectedRawSize = 0;
    velox::vector_size_t innerSize = folly::Random::rand32() % VECTOR_SIZE + 1;
    std::vector<std::optional<T>> innerVec;
    for (velox::vector_size_t j = 0; j < innerSize; ++j) {
      if (isNullAt(j)) {
        innerVec.emplace_back(std::nullopt);
        expectedRawSize += nimble::kNullSize;
      } else {
        auto value = getValue<T>(j);
        expectedRawSize += getSize<T>(value);
        innerVec.emplace_back(value);
      }
    }
    expectedRawSize *= VECTOR_SIZE;
    vec.emplace_back(innerVec);
    auto arrayVector = vectorMaker_->arrayVectorNullable<T>(vec);
    auto constVector =
        velox::BaseVector::wrapInConstant(VECTOR_SIZE, 0, arrayVector);
    ranges_.add(0, constVector->size());
    auto rawSize = nimble::getRawSizeFromVector(constVector, ranges_);
    ASSERT_EQ(expectedRawSize, rawSize);
  }

  template <typename T>
  void testDictionaryArray(
      const std::function<bool(velox::vector_size_t)>& isNullAt) {
    reset();
    std::vector<std::optional<std::vector<std::optional<T>>>> vec;
    vec.reserve(VECTOR_SIZE);
    uint64_t expectedRawSize = 0;
    std::unordered_map<velox::vector_size_t, uint64_t> indexToSize;
    for (velox::vector_size_t i = 1; i <= VECTOR_SIZE; ++i) {
      if (isNullAt(i)) {
        vec.emplace_back(std::nullopt);
      } else {
        std::vector<std::optional<T>> innerVec;
        uint64_t size = 0;
        velox::vector_size_t innerSize =
            folly::Random::rand32() % VECTOR_SIZE + 1;
        for (velox::vector_size_t j = 0; j < innerSize; ++j) {
          if (isNullAt(j)) {
            size += nimble::kNullSize;
            innerVec.emplace_back(std::nullopt);
          } else {
            auto value = getValue<T>(j);
            size += getSize<T>(value);
            innerVec.emplace_back(value);
          }
        }
        indexToSize[i - 1] = size;
        vec.emplace_back(innerVec);
      }
    }
    auto arrayVector = vectorMaker_->arrayVectorNullable<T>(vec);
    auto indices = randomIndices(VECTOR_SIZE);
    const velox::vector_size_t* data = indices->as<velox::vector_size_t>();
    for (auto i = 0; i < VECTOR_SIZE; ++i) {
      if (vec[data[i]] == std::nullopt) {
        expectedRawSize += nimble::kNullSize;
      } else {
        expectedRawSize += indexToSize[data[i]];
      }
    }
    auto wrappedVector = velox::BaseVector::wrapInDictionary(
        velox::BufferPtr(nullptr), indices, VECTOR_SIZE, arrayVector);
    ranges_.add(0, wrappedVector->size());
    auto rawSize = nimble::getRawSizeFromVector(wrappedVector, ranges_);
    ASSERT_EQ(expectedRawSize, rawSize);
  }
};

class RawSizeMapTestFixture
    : public RawSizeBaseTestFixture,
      public testing::WithParamInterface<velox::VectorEncoding::Simple> {
 public:
  RawSizeMapTestFixture() : mapEncoding_(GetParam()) {}

  // Instantiate test for both maps and flat maps.
  static std::vector<velox::VectorEncoding::Simple> getTestParams() {
    static std::vector<velox::VectorEncoding::Simple> mapEncodings = {
        velox::VectorEncoding::Simple::MAP,
        velox::VectorEncoding::Simple::FLAT_MAP,
    };
    return mapEncodings;
  }

 protected:
  template <typename TKey, typename TValue>
  void testMap(const std::function<bool(velox::vector_size_t)>& isNullAt) {
    reset();
    std::vector<
        std::optional<std::vector<std::pair<TKey, std::optional<TValue>>>>>
        vec;
    vec.reserve(VECTOR_SIZE);
    uint64_t expectedRawSize = 0;
    for (velox::vector_size_t i = 1; i <= VECTOR_SIZE; ++i) {
      if (isNullAt(i)) {
        vec.emplace_back(std::nullopt);
        expectedRawSize += nimble::kNullSize;
      } else {
        std::vector<std::pair<TKey, std::optional<TValue>>> innerVec;
        std::unordered_set<TKey> keysSeen;

        velox::vector_size_t innerSize =
            folly::Random::rand32() % VECTOR_SIZE + 1;
        innerVec.reserve(innerSize);

        for (velox::vector_size_t j = 0; j < innerSize; ++j) {
          TKey key = getValue<TKey>(j);

          // Make sure we don't generate maps with duplicated keys.
          if (!keysSeen.contains(key)) {
            keysSeen.insert(key);

            if (isNullAt(j)) {
              innerVec.emplace_back(key, std::nullopt);
              expectedRawSize += nimble::kNullSize;
            } else {
              auto value = getValue<TValue>(j);
              expectedRawSize += getSize<TValue>(value);
              innerVec.emplace_back(key, value);
            }
            expectedRawSize += getSize<TKey>(key);
          }
        }
        vec.emplace_back(std::move(innerVec));
      }
    }
    auto vector = makeVector<TKey, TValue>(vec);
    ranges_.add(0, vector->size());
    auto rawSize = nimble::getRawSizeFromVector(vector, ranges_);
    ASSERT_EQ(expectedRawSize, rawSize);
  }

  template <typename TKey, typename TValue>
  void testConstantMap(
      const std::function<bool(velox::vector_size_t)>& isNullAt) {
    reset();
    std::vector<
        std::optional<std::vector<std::pair<TKey, std::optional<TValue>>>>>
        vec;
    uint64_t expectedRawSize = 0;
    std::vector<std::pair<TKey, std::optional<TValue>>> innerVec;
    std::unordered_set<TKey> keysSeen;

    velox::vector_size_t innerSize = folly::Random::rand32() % VECTOR_SIZE + 1;
    innerVec.reserve(innerSize);

    for (velox::vector_size_t j = 0; j < innerSize; ++j) {
      TKey key = getValue<TKey>(j);

      // Make sure we don't generate maps with duplicated keys.
      if (!keysSeen.contains(key)) {
        keysSeen.insert(key);

        if (isNullAt(j)) {
          innerVec.emplace_back(key, std::nullopt);
          expectedRawSize += nimble::kNullSize;
        } else {
          auto value = getValue<TValue>(j);
          expectedRawSize += getSize<TValue>(value);
          innerVec.emplace_back(key, value);
        }
        expectedRawSize += getSize<TKey>(key);
      }
    }
    expectedRawSize *= VECTOR_SIZE;
    vec.emplace_back(std::move(innerVec));

    auto vector = makeVector<TKey, TValue>(vec);
    auto constVector =
        velox::BaseVector::wrapInConstant(VECTOR_SIZE, 0, vector);
    ranges_.add(0, constVector->size());
    auto rawSize = nimble::getRawSizeFromVector(constVector, ranges_);
    ASSERT_EQ(expectedRawSize, rawSize);
  }

  template <typename TKey, typename TValue>
  void testDictionaryMap(
      const std::function<bool(velox::vector_size_t)>& isNullAt) {
    reset();
    std::vector<
        std::optional<std::vector<std::pair<TKey, std::optional<TValue>>>>>
        vec;
    vec.reserve(VECTOR_SIZE);
    uint64_t expectedRawSize = 0;
    std::unordered_map<velox::vector_size_t, uint64_t> indexToSize;

    for (velox::vector_size_t i = 1; i <= VECTOR_SIZE; ++i) {
      if (isNullAt(i)) {
        vec.emplace_back(std::nullopt);
      } else {
        std::vector<std::pair<TKey, std::optional<TValue>>> innerVec;
        std::unordered_set<TKey> keysSeen;

        uint64_t size = 0;
        velox::vector_size_t innerSize =
            folly::Random::rand32() % VECTOR_SIZE + 1;
        innerVec.reserve(innerSize);

        for (velox::vector_size_t j = 0; j < innerSize; ++j) {
          TKey key = getValue<TKey>(j);

          // Make sure we don't generate maps with duplicated keys.
          if (!keysSeen.contains(key)) {
            keysSeen.insert(key);

            if (isNullAt(j)) {
              size += nimble::kNullSize;
              innerVec.emplace_back(key, std::nullopt);
            } else {
              auto value = getValue<TValue>(j);
              size += getSize<TValue>(value);
              innerVec.emplace_back(key, value);
            }
            size += getSize<TKey>(key);
          }
        }
        indexToSize[i - 1] = size;
        vec.emplace_back(std::move(innerVec));
      }
    }

    auto vector = makeVector<TKey, TValue>(vec);
    auto indices = randomIndices(VECTOR_SIZE);
    const velox::vector_size_t* data = indices->as<velox::vector_size_t>();

    for (auto i = 0; i < VECTOR_SIZE; ++i) {
      if (vec[data[i]] == std::nullopt) {
        expectedRawSize += nimble::kNullSize;
      } else {
        expectedRawSize += indexToSize[data[i]];
      }
    }
    auto wrappedVector = velox::BaseVector::wrapInDictionary(
        velox::BufferPtr{}, indices, VECTOR_SIZE, vector);
    ranges_.add(0, wrappedVector->size());
    auto rawSize = nimble::getRawSizeFromVector(wrappedVector, ranges_);
    ASSERT_EQ(expectedRawSize, rawSize);
  }

  template <typename TKey, typename TValue, typename TInput>
  velox::VectorPtr makeVector(const TInput& data) const {
    switch (mapEncoding_) {
      case velox::VectorEncoding::Simple::FLAT_MAP:
        return vectorMaker_->flatMapVectorNullable<TKey, TValue>(data);

      case velox::VectorEncoding::Simple::MAP:
        return vectorMaker_->mapVector<TKey, TValue>(data);

      default:
        // We don't instantiate the test using any other encodings.
        throw std::runtime_error(
            fmt::format("Only map encodings expected, got {}", mapEncoding_));
    }
  }

 private:
  velox::VectorEncoding::Simple mapEncoding_;
};

/*
 * The following tests are considered Fuzz tests. The data inside the vectors,
 * as well as the null count and positions, are randomized. The expected raw
 * size is calculated from this random data and is used to assert against the
 * raw size returned by the function under test.
 */
TEST_F(RawSizeTestFixture, Flat) {
  testFlat<bool>(noNulls());
  testFlat<int8_t>(noNulls());
  testFlat<int16_t>(noNulls());
  testFlat<int32_t>(noNulls());
  testFlat<int64_t>(noNulls());
  testFlat<float>(noNulls());
  testFlat<double>(noNulls());
  testFlat<velox::StringView>(noNulls());
  testFlat<velox::Timestamp>(noNulls());
}

TEST_F(RawSizeTestFixture, FlatSomeNull) {
  testFlat<bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testFlat<int8_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testFlat<int16_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testFlat<int32_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testFlat<int64_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testFlat<float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testFlat<double>(randomNulls(folly::Random::rand32() % 10 + 1));
  testFlat<velox::StringView>(randomNulls(folly::Random::rand32() % 10 + 1));
  testFlat<velox::Timestamp>(randomNulls(folly::Random::rand32() % 10 + 1));
}

TEST_F(RawSizeTestFixture, Constant) {
  testConstant<bool>(noNulls());
  testConstant<int8_t>(noNulls());
  testConstant<int16_t>(noNulls());
  testConstant<int32_t>(noNulls());
  testConstant<int64_t>(noNulls());
  testConstant<float>(noNulls());
  testConstant<double>(noNulls());
  testConstant<velox::StringView>(noNulls());
  testConstant<velox::Timestamp>(noNulls());
}

TEST_F(RawSizeTestFixture, ConstantSomeNull) {
  testConstant<bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstant<int8_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstant<int16_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstant<int32_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstant<int64_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstant<float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstant<double>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstant<velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstant<velox::Timestamp>(randomNulls(folly::Random::rand32() % 10 + 1));
}

TEST_F(RawSizeTestFixture, Dictionary) {
  testDictionary<bool>(noNulls());
  testDictionary<int8_t>(noNulls());
  testDictionary<int16_t>(noNulls());
  testDictionary<int32_t>(noNulls());
  testDictionary<int64_t>(noNulls());
  testDictionary<float>(noNulls());
  testDictionary<double>(noNulls());
  testDictionary<velox::StringView>(noNulls());
  testDictionary<velox::Timestamp>(noNulls());
}

TEST_F(RawSizeTestFixture, DictionarySomeNull) {
  testDictionary<bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionary<int8_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionary<int16_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionary<int32_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionary<int64_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionary<float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionary<double>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionary<velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionary<velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));
}

TEST_F(RawSizeTestFixture, Array) {
  testArray<bool>(noNulls());
  testArray<int8_t>(noNulls());
  testArray<int16_t>(noNulls());
  testArray<int32_t>(noNulls());
  testArray<int64_t>(noNulls());
  testArray<float>(noNulls());
  testArray<double>(noNulls());
  testArray<velox::StringView>(noNulls());
  testArray<velox::Timestamp>(noNulls());
}

TEST_F(RawSizeTestFixture, ArraySomNull) {
  testArray<bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testArray<int8_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testArray<int16_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testArray<int32_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testArray<int64_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testArray<float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testArray<double>(randomNulls(folly::Random::rand32() % 10 + 1));
  testArray<velox::StringView>(randomNulls(folly::Random::rand32() % 10 + 1));
  testArray<velox::Timestamp>(randomNulls(folly::Random::rand32() % 10 + 1));
}

TEST_F(RawSizeTestFixture, ConstantArray) {
  testConstantArray<bool>(noNulls());
  testConstantArray<int8_t>(noNulls());
  testConstantArray<int16_t>(noNulls());
  testConstantArray<int32_t>(noNulls());
  testConstantArray<int64_t>(noNulls());
  testConstantArray<float>(noNulls());
  testConstantArray<double>(noNulls());
  testConstantArray<velox::StringView>(noNulls());
  testConstantArray<velox::Timestamp>(noNulls());
}

TEST_F(RawSizeTestFixture, ConstantArraySomeNull) {
  testConstantArray<bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantArray<int8_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantArray<int16_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantArray<int32_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantArray<int64_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantArray<float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantArray<double>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantArray<velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantArray<velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));
}

TEST_F(RawSizeTestFixture, DictionaryArray) {
  testDictionaryArray<bool>(noNulls());
  testDictionaryArray<int8_t>(noNulls());
  testDictionaryArray<int16_t>(noNulls());
  testDictionaryArray<int32_t>(noNulls());
  testDictionaryArray<int64_t>(noNulls());
  testDictionaryArray<float>(noNulls());
  testDictionaryArray<double>(noNulls());
  testDictionaryArray<velox::StringView>(noNulls());
  testDictionaryArray<velox::Timestamp>(noNulls());
}

TEST_F(RawSizeTestFixture, DictionaryArraySomeNull) {
  testDictionaryArray<bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryArray<int8_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryArray<int16_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryArray<int32_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryArray<int64_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryArray<float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryArray<double>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryArray<velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryArray<velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));
}

TEST_P(RawSizeMapTestFixture, Map) {
  testMap<bool, bool>(noNulls());
  testMap<bool, int8_t>(noNulls());
  testMap<bool, int16_t>(noNulls());
  testMap<bool, int32_t>(noNulls());
  testMap<bool, int64_t>(noNulls());
  testMap<bool, float>(noNulls());
  testMap<bool, double>(noNulls());
  testMap<bool, velox::StringView>(noNulls());
  testMap<bool, velox::Timestamp>(noNulls());

  testMap<int8_t, bool>(noNulls());
  testMap<int8_t, int8_t>(noNulls());
  testMap<int8_t, int16_t>(noNulls());
  testMap<int8_t, int32_t>(noNulls());
  testMap<int8_t, int64_t>(noNulls());
  testMap<int8_t, float>(noNulls());
  testMap<int8_t, double>(noNulls());
  testMap<int8_t, velox::StringView>(noNulls());
  testMap<int8_t, velox::Timestamp>(noNulls());

  testMap<int16_t, bool>(noNulls());
  testMap<int16_t, int8_t>(noNulls());
  testMap<int16_t, int16_t>(noNulls());
  testMap<int16_t, int32_t>(noNulls());
  testMap<int16_t, int64_t>(noNulls());
  testMap<int16_t, float>(noNulls());
  testMap<int16_t, double>(noNulls());
  testMap<int16_t, velox::StringView>(noNulls());
  testMap<int16_t, velox::Timestamp>(noNulls());

  testMap<int32_t, bool>(noNulls());
  testMap<int32_t, int8_t>(noNulls());
  testMap<int32_t, int16_t>(noNulls());
  testMap<int32_t, int32_t>(noNulls());
  testMap<int32_t, int64_t>(noNulls());
  testMap<int32_t, float>(noNulls());
  testMap<int32_t, double>(noNulls());
  testMap<int32_t, velox::StringView>(noNulls());
  testMap<int32_t, velox::Timestamp>(noNulls());

  testMap<int64_t, bool>(noNulls());
  testMap<int64_t, int8_t>(noNulls());
  testMap<int64_t, int16_t>(noNulls());
  testMap<int64_t, int32_t>(noNulls());
  testMap<int64_t, int64_t>(noNulls());
  testMap<int64_t, float>(noNulls());
  testMap<int64_t, double>(noNulls());
  testMap<int64_t, velox::StringView>(noNulls());
  testMap<int64_t, velox::Timestamp>(noNulls());

  testMap<float, bool>(noNulls());
  testMap<float, int8_t>(noNulls());
  testMap<float, int16_t>(noNulls());
  testMap<float, int32_t>(noNulls());
  testMap<float, int64_t>(noNulls());
  testMap<float, float>(noNulls());
  testMap<float, double>(noNulls());
  testMap<float, velox::StringView>(noNulls());
  testMap<float, velox::Timestamp>(noNulls());

  testMap<double, bool>(noNulls());
  testMap<double, int8_t>(noNulls());
  testMap<double, int16_t>(noNulls());
  testMap<double, int32_t>(noNulls());
  testMap<double, int64_t>(noNulls());
  testMap<double, float>(noNulls());
  testMap<double, double>(noNulls());
  testMap<double, velox::StringView>(noNulls());
  testMap<double, velox::Timestamp>(noNulls());

  testMap<velox::StringView, bool>(noNulls());
  testMap<velox::StringView, int8_t>(noNulls());
  testMap<velox::StringView, int16_t>(noNulls());
  testMap<velox::StringView, int32_t>(noNulls());
  testMap<velox::StringView, int64_t>(noNulls());
  testMap<velox::StringView, float>(noNulls());
  testMap<velox::StringView, double>(noNulls());
  testMap<velox::StringView, velox::StringView>(noNulls());
  testMap<velox::StringView, velox::Timestamp>(noNulls());

  testMap<velox::Timestamp, bool>(noNulls());
  testMap<velox::Timestamp, int8_t>(noNulls());
  testMap<velox::Timestamp, int16_t>(noNulls());
  testMap<velox::Timestamp, int32_t>(noNulls());
  testMap<velox::Timestamp, int64_t>(noNulls());
  testMap<velox::Timestamp, float>(noNulls());
  testMap<velox::Timestamp, double>(noNulls());
  testMap<velox::Timestamp, velox::StringView>(noNulls());
  testMap<velox::Timestamp, velox::Timestamp>(noNulls());
}

TEST_P(RawSizeMapTestFixture, MapSomeNull) {
  testMap<bool, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<bool, int8_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<bool, int16_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<bool, int32_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<bool, int64_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<bool, float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<bool, double>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<bool, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<bool, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testMap<int8_t, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int8_t, int8_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int8_t, int16_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int8_t, int32_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int8_t, int64_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int8_t, float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int8_t, double>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int8_t, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int8_t, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testMap<int16_t, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int16_t, int8_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int16_t, int16_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int16_t, int32_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int16_t, int64_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int16_t, float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int16_t, double>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int16_t, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int16_t, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testMap<int32_t, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int32_t, int8_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int32_t, int16_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int32_t, int32_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int32_t, int64_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int32_t, float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int32_t, double>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int32_t, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int32_t, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testMap<int64_t, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int64_t, int8_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int64_t, int16_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int64_t, int32_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int64_t, int64_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int64_t, float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int64_t, double>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int64_t, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<int64_t, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testMap<float, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<float, int8_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<float, int16_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<float, int32_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<float, int64_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<float, float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<float, double>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<float, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<float, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testMap<double, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<double, int8_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<double, int16_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<double, int32_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<double, int64_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<double, float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<double, double>(randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<double, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<double, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testMap<velox::StringView, bool>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::StringView, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::StringView, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::StringView, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::StringView, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::StringView, float>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::StringView, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::StringView, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::StringView, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testMap<velox::Timestamp, bool>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::Timestamp, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::Timestamp, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::Timestamp, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::Timestamp, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::Timestamp, float>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::Timestamp, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::Timestamp, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testMap<velox::Timestamp, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));
}

TEST_P(RawSizeMapTestFixture, ConstantMap) {
  testConstantMap<bool, bool>(noNulls());
  testConstantMap<bool, int8_t>(noNulls());
  testConstantMap<bool, int16_t>(noNulls());
  testConstantMap<bool, int32_t>(noNulls());
  testConstantMap<bool, int64_t>(noNulls());
  testConstantMap<bool, float>(noNulls());
  testConstantMap<bool, double>(noNulls());
  testConstantMap<bool, velox::StringView>(noNulls());
  testConstantMap<bool, velox::Timestamp>(noNulls());

  testConstantMap<int8_t, bool>(noNulls());
  testConstantMap<int8_t, int8_t>(noNulls());
  testConstantMap<int8_t, int16_t>(noNulls());
  testConstantMap<int8_t, int32_t>(noNulls());
  testConstantMap<int8_t, int64_t>(noNulls());
  testConstantMap<int8_t, float>(noNulls());
  testConstantMap<int8_t, double>(noNulls());
  testConstantMap<int8_t, velox::StringView>(noNulls());
  testConstantMap<int8_t, velox::Timestamp>(noNulls());

  testConstantMap<int16_t, bool>(noNulls());
  testConstantMap<int16_t, int8_t>(noNulls());
  testConstantMap<int16_t, int16_t>(noNulls());
  testConstantMap<int16_t, int32_t>(noNulls());
  testConstantMap<int16_t, int64_t>(noNulls());
  testConstantMap<int16_t, float>(noNulls());
  testConstantMap<int16_t, double>(noNulls());
  testConstantMap<int16_t, velox::StringView>(noNulls());
  testConstantMap<int16_t, velox::Timestamp>(noNulls());

  testConstantMap<int32_t, bool>(noNulls());
  testConstantMap<int32_t, int8_t>(noNulls());
  testConstantMap<int32_t, int16_t>(noNulls());
  testConstantMap<int32_t, int32_t>(noNulls());
  testConstantMap<int32_t, int64_t>(noNulls());
  testConstantMap<int32_t, float>(noNulls());
  testConstantMap<int32_t, double>(noNulls());
  testConstantMap<int32_t, velox::StringView>(noNulls());
  testConstantMap<int32_t, velox::Timestamp>(noNulls());

  testConstantMap<int64_t, bool>(noNulls());
  testConstantMap<int64_t, int8_t>(noNulls());
  testConstantMap<int64_t, int16_t>(noNulls());
  testConstantMap<int64_t, int32_t>(noNulls());
  testConstantMap<int64_t, int64_t>(noNulls());
  testConstantMap<int64_t, float>(noNulls());
  testConstantMap<int64_t, double>(noNulls());
  testConstantMap<int64_t, velox::StringView>(noNulls());
  testConstantMap<int64_t, velox::Timestamp>(noNulls());

  testConstantMap<float, bool>(noNulls());
  testConstantMap<float, int8_t>(noNulls());
  testConstantMap<float, int16_t>(noNulls());
  testConstantMap<float, int32_t>(noNulls());
  testConstantMap<float, int64_t>(noNulls());
  testConstantMap<float, float>(noNulls());
  testConstantMap<float, double>(noNulls());
  testConstantMap<float, velox::StringView>(noNulls());
  testConstantMap<float, velox::Timestamp>(noNulls());

  testConstantMap<double, bool>(noNulls());
  testConstantMap<double, int8_t>(noNulls());
  testConstantMap<double, int16_t>(noNulls());
  testConstantMap<double, int32_t>(noNulls());
  testConstantMap<double, int64_t>(noNulls());
  testConstantMap<double, float>(noNulls());
  testConstantMap<double, double>(noNulls());
  testConstantMap<double, velox::StringView>(noNulls());
  testConstantMap<double, velox::Timestamp>(noNulls());

  testConstantMap<velox::StringView, bool>(noNulls());
  testConstantMap<velox::StringView, int8_t>(noNulls());
  testConstantMap<velox::StringView, int16_t>(noNulls());
  testConstantMap<velox::StringView, int32_t>(noNulls());
  testConstantMap<velox::StringView, int64_t>(noNulls());
  testConstantMap<velox::StringView, float>(noNulls());
  testConstantMap<velox::StringView, double>(noNulls());
  testConstantMap<velox::StringView, velox::StringView>(noNulls());
  testConstantMap<velox::StringView, velox::Timestamp>(noNulls());

  testConstantMap<velox::Timestamp, bool>(noNulls());
  testConstantMap<velox::Timestamp, int8_t>(noNulls());
  testConstantMap<velox::Timestamp, int16_t>(noNulls());
  testConstantMap<velox::Timestamp, int32_t>(noNulls());
  testConstantMap<velox::Timestamp, int64_t>(noNulls());
  testConstantMap<velox::Timestamp, float>(noNulls());
  testConstantMap<velox::Timestamp, double>(noNulls());
  testConstantMap<velox::Timestamp, velox::StringView>(noNulls());
  testConstantMap<velox::Timestamp, velox::Timestamp>(noNulls());
}

TEST_P(RawSizeMapTestFixture, ConstantMapSomeNull) {
  testConstantMap<bool, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<bool, int8_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<bool, int16_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<bool, int32_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<bool, int64_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<bool, float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<bool, double>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<bool, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<bool, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testConstantMap<int8_t, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int8_t, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int8_t, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int8_t, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int8_t, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int8_t, float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int8_t, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int8_t, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int8_t, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testConstantMap<int16_t, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int16_t, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int16_t, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int16_t, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int16_t, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int16_t, float>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int16_t, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int16_t, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int16_t, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testConstantMap<int32_t, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int32_t, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int32_t, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int32_t, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int32_t, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int32_t, float>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int32_t, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int32_t, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int32_t, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testConstantMap<int64_t, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int64_t, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int64_t, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int64_t, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int64_t, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int64_t, float>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int64_t, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int64_t, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<int64_t, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testConstantMap<float, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<float, int8_t>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<float, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<float, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<float, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<float, float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<float, double>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<float, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<float, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testConstantMap<double, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<double, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<double, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<double, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<double, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<double, float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<double, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<double, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<double, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testConstantMap<velox::StringView, bool>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::StringView, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::StringView, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::StringView, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::StringView, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::StringView, float>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::StringView, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::StringView, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::StringView, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testConstantMap<velox::Timestamp, bool>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::Timestamp, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::Timestamp, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::Timestamp, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::Timestamp, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::Timestamp, float>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::Timestamp, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::Timestamp, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testConstantMap<velox::Timestamp, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));
}

TEST_P(RawSizeMapTestFixture, DictionaryMap) {
  testDictionaryMap<bool, bool>(noNulls());
  testDictionaryMap<bool, int8_t>(noNulls());
  testDictionaryMap<bool, int16_t>(noNulls());
  testDictionaryMap<bool, int32_t>(noNulls());
  testDictionaryMap<bool, int64_t>(noNulls());
  testDictionaryMap<bool, float>(noNulls());
  testDictionaryMap<bool, double>(noNulls());
  testDictionaryMap<bool, velox::StringView>(noNulls());
  testDictionaryMap<bool, velox::Timestamp>(noNulls());

  testDictionaryMap<int8_t, bool>(noNulls());
  testDictionaryMap<int8_t, int8_t>(noNulls());
  testDictionaryMap<int8_t, int16_t>(noNulls());
  testDictionaryMap<int8_t, int32_t>(noNulls());
  testDictionaryMap<int8_t, int64_t>(noNulls());
  testDictionaryMap<int8_t, float>(noNulls());
  testDictionaryMap<int8_t, double>(noNulls());
  testDictionaryMap<int8_t, velox::StringView>(noNulls());
  testDictionaryMap<int8_t, velox::Timestamp>(noNulls());

  testDictionaryMap<int16_t, bool>(noNulls());
  testDictionaryMap<int16_t, int8_t>(noNulls());
  testDictionaryMap<int16_t, int16_t>(noNulls());
  testDictionaryMap<int16_t, int32_t>(noNulls());
  testDictionaryMap<int16_t, int64_t>(noNulls());
  testDictionaryMap<int16_t, float>(noNulls());
  testDictionaryMap<int16_t, double>(noNulls());
  testDictionaryMap<int16_t, velox::StringView>(noNulls());
  testDictionaryMap<int16_t, velox::Timestamp>(noNulls());

  testDictionaryMap<int32_t, bool>(noNulls());
  testDictionaryMap<int32_t, int8_t>(noNulls());
  testDictionaryMap<int32_t, int16_t>(noNulls());
  testDictionaryMap<int32_t, int32_t>(noNulls());
  testDictionaryMap<int32_t, int64_t>(noNulls());
  testDictionaryMap<int32_t, float>(noNulls());
  testDictionaryMap<int32_t, double>(noNulls());
  testDictionaryMap<int32_t, velox::StringView>(noNulls());
  testDictionaryMap<int32_t, velox::Timestamp>(noNulls());

  testDictionaryMap<int64_t, bool>(noNulls());
  testDictionaryMap<int64_t, int8_t>(noNulls());
  testDictionaryMap<int64_t, int16_t>(noNulls());
  testDictionaryMap<int64_t, int32_t>(noNulls());
  testDictionaryMap<int64_t, int64_t>(noNulls());
  testDictionaryMap<int64_t, float>(noNulls());
  testDictionaryMap<int64_t, double>(noNulls());
  testDictionaryMap<int64_t, velox::StringView>(noNulls());
  testDictionaryMap<int64_t, velox::Timestamp>(noNulls());

  testDictionaryMap<float, bool>(noNulls());
  testDictionaryMap<float, int8_t>(noNulls());
  testDictionaryMap<float, int16_t>(noNulls());
  testDictionaryMap<float, int32_t>(noNulls());
  testDictionaryMap<float, int64_t>(noNulls());
  testDictionaryMap<float, float>(noNulls());
  testDictionaryMap<float, double>(noNulls());
  testDictionaryMap<float, velox::StringView>(noNulls());
  testDictionaryMap<float, velox::Timestamp>(noNulls());

  testDictionaryMap<double, bool>(noNulls());
  testDictionaryMap<double, int8_t>(noNulls());
  testDictionaryMap<double, int16_t>(noNulls());
  testDictionaryMap<double, int32_t>(noNulls());
  testDictionaryMap<double, int64_t>(noNulls());
  testDictionaryMap<double, float>(noNulls());
  testDictionaryMap<double, double>(noNulls());
  testDictionaryMap<double, velox::StringView>(noNulls());
  testDictionaryMap<double, velox::Timestamp>(noNulls());

  testDictionaryMap<velox::StringView, bool>(noNulls());
  testDictionaryMap<velox::StringView, int8_t>(noNulls());
  testDictionaryMap<velox::StringView, int16_t>(noNulls());
  testDictionaryMap<velox::StringView, int32_t>(noNulls());
  testDictionaryMap<velox::StringView, int64_t>(noNulls());
  testDictionaryMap<velox::StringView, float>(noNulls());
  testDictionaryMap<velox::StringView, double>(noNulls());
  testDictionaryMap<velox::StringView, velox::StringView>(noNulls());
  testDictionaryMap<velox::StringView, velox::Timestamp>(noNulls());

  testDictionaryMap<velox::Timestamp, bool>(noNulls());
  testDictionaryMap<velox::Timestamp, int8_t>(noNulls());
  testDictionaryMap<velox::Timestamp, int16_t>(noNulls());
  testDictionaryMap<velox::Timestamp, int32_t>(noNulls());
  testDictionaryMap<velox::Timestamp, int64_t>(noNulls());
  testDictionaryMap<velox::Timestamp, float>(noNulls());
  testDictionaryMap<velox::Timestamp, double>(noNulls());
  testDictionaryMap<velox::Timestamp, velox::StringView>(noNulls());
  testDictionaryMap<velox::Timestamp, velox::Timestamp>(noNulls());
}

TEST_P(RawSizeMapTestFixture, DictionaryMapSomeNull) {
  testDictionaryMap<bool, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<bool, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<bool, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<bool, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<bool, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<bool, float>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<bool, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<bool, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<bool, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testDictionaryMap<int8_t, bool>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int8_t, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int8_t, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int8_t, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int8_t, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int8_t, float>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int8_t, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int8_t, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int8_t, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testDictionaryMap<int16_t, bool>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int16_t, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int16_t, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int16_t, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int16_t, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int16_t, float>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int16_t, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int16_t, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int16_t, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testDictionaryMap<int32_t, bool>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int32_t, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int32_t, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int32_t, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int32_t, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int32_t, float>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int32_t, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int32_t, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int32_t, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testDictionaryMap<int64_t, bool>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int64_t, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int64_t, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int64_t, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int64_t, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int64_t, float>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int64_t, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int64_t, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<int64_t, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testDictionaryMap<float, bool>(randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<float, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<float, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<float, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<float, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<float, float>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<float, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<float, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<float, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testDictionaryMap<double, bool>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<double, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<double, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<double, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<double, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<double, float>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<double, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<double, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<double, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testDictionaryMap<velox::StringView, bool>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::StringView, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::StringView, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::StringView, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::StringView, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::StringView, float>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::StringView, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::StringView, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::StringView, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));

  testDictionaryMap<velox::Timestamp, bool>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::Timestamp, int8_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::Timestamp, int16_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::Timestamp, int32_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::Timestamp, int64_t>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::Timestamp, float>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::Timestamp, double>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::StringView, velox::StringView>(
      randomNulls(folly::Random::rand32() % 10 + 1));
  testDictionaryMap<velox::Timestamp, velox::Timestamp>(
      randomNulls(folly::Random::rand32() % 10 + 1));
}

INSTANTIATE_TEST_SUITE_P(
    RawSizeMapTest,
    RawSizeMapTestFixture,
    testing::ValuesIn(RawSizeMapTestFixture::getTestParams()));

/*
 * The following tests are considered handcrafted tests for different nested
 * vector cases. These tests cases have specified expected sizes.
 */
TEST_F(RawSizeTestFixture, ArrayNested) {
  auto arrayVector =
      vectorMaker_->arrayVector<int64_t>({{0, 1}, {1, 0, 1}, {0}});
  auto arrayVector2 = vectorMaker_->arrayVector<int64_t>({{1}, {1, 0}, {0}});
  auto nestedArrayVector = vectorMaker_->arrayVector({0, 1}, arrayVector);
  auto nestedArrayVector2 = vectorMaker_->arrayVector({0, 2}, arrayVector2);
  nestedArrayVector->append(nestedArrayVector2.get());
  this->ranges_.add(0, nestedArrayVector->size());
  auto rawSize = nimble::getRawSizeFromVector(nestedArrayVector, this->ranges_);
  constexpr auto expectedRawSize = sizeof(int64_t) * 10;

  ASSERT_EQ(expectedRawSize, rawSize);
}

TEST_F(RawSizeTestFixture, MapNested) {
  auto mapKeys = vectorMaker_->flatVector<velox::StringView>(
      {"a", "bb", "ccc", "dddd", "eeeee"}); // 15
  auto mapValues = vectorMaker_->flatVector<int64_t>({0, 1, 2, 3, 4}); // 40

  auto mapKeys2 = vectorMaker_->mapVector({0, 1, 3}, mapKeys, mapValues);
  auto mapValues2 = vectorMaker_->flatVector<int64_t>({0, 1, 2}); //  24

  auto mapVector = vectorMaker_->mapVector({0, 1}, mapKeys2, mapValues2);

  auto mapKeys3 = vectorMaker_->flatVector<velox::StringView>(
      {"a", "bb", "ccc", "dddd"}); // 10
  auto mapValues3 = vectorMaker_->flatVector<int64_t>({0, 1, 2, 3}); // 32

  auto mapKeys4 = vectorMaker_->mapVector({0, 1, 3}, mapKeys3, mapValues3);
  auto mapValues4 = vectorMaker_->flatVector<int64_t>({0, 1, 2}); // 24

  auto mapVector2 = vectorMaker_->mapVector({0, 1}, mapKeys4, mapValues4);

  mapVector->append(mapVector2.get());
  this->ranges_.add(0, mapVector->size());
  auto rawSize = nimble::getRawSizeFromVector(mapVector, this->ranges_);
  constexpr auto expectedSize = sizeof(int64_t) * 15 + 25;

  ASSERT_EQ(expectedSize, rawSize);
}

TEST_F(RawSizeTestFixture, MapArrayNested) {
  auto mapKeys = vectorMaker_->arrayVector<int64_t>(
      {{0, 1, 0}, {1, 0, 1}, {0, 1}, {0}, {1, 0, 1, 0}});
  auto mapValues = vectorMaker_->flatVector<velox::StringView>(
      {"a", "bb", "ccc", "dddd", "eeeee"});

  auto mapKeys2 = vectorMaker_->mapVector({0, 1, 3}, mapKeys, mapValues);
  auto mapValues2 = vectorMaker_->flatVector<int64_t>({0, 1, 2});

  auto mapVector = vectorMaker_->mapVector({0, 1}, mapKeys2, mapValues2);
  this->ranges_.add(0, mapVector->size());
  auto rawSize = nimble::getRawSizeFromVector(mapVector, this->ranges_);
  constexpr auto expectedSize = sizeof(int64_t) * 16 + 15;

  ASSERT_EQ(expectedSize, rawSize);
}

TEST_F(RawSizeTestFixture, ArrayMapNested) {
  auto mapKeys = vectorMaker_->flatVector<velox::StringView>(
      {"a", "bb", "ccc", "dddd", "eeeee"});
  auto mapValues = vectorMaker_->flatVector<int64_t>({0, 1, 2, 3, 4});

  auto mapVector = vectorMaker_->mapVector({0, 1, 3}, mapKeys, mapValues);
  auto nestedArrayVector = vectorMaker_->arrayVector({0, 1}, mapVector);

  this->ranges_.add(0, nestedArrayVector->size());
  auto rawSize = nimble::getRawSizeFromVector(nestedArrayVector, this->ranges_);
  constexpr auto expectedSize = sizeof(int64_t) * 5 + 15;

  ASSERT_EQ(expectedSize, rawSize);
}

TEST_F(RawSizeTestFixture, RowSameTypes) {
  auto childVector1 = vectorMaker_->flatVector<int64_t>({0, 0, 0, 1, 1, 1});
  auto childVector2 = vectorMaker_->flatVector<int64_t>({0, 1, 0, 1, 0, 1});
  auto childVector3 = vectorMaker_->flatVector<int64_t>({0, 1, 0, 1, 0, 1});
  auto rowVector = vectorMaker_->rowVector(
      {"1", "2", "3"}, {childVector1, childVector2, childVector3});
  this->ranges_.add(0, rowVector->size());
  auto rawSize = nimble::getRawSizeFromRowVector(
      rowVector, this->ranges_, context_, /*topLevel=*/true);

  ASSERT_EQ(sizeof(int64_t) * 18, rawSize);
  size_t expectedChildCount = 3;
  ASSERT_EQ(expectedChildCount, context_.columnCount());
  for (size_t i = 0; i < expectedChildCount; ++i) {
    ASSERT_EQ(sizeof(int64_t) * 6, context_.sizeAt(i));
  }
}

TEST_F(RawSizeTestFixture, RowDifferentTypes) {
  auto childVector1 = vectorMaker_->flatVector<int64_t>({0, 0, 0, 1, 1, 1});
  auto childVector2 = vectorMaker_->flatVector<bool>({0, 1, 0, 1, 0, 1});
  auto childVector3 = vectorMaker_->flatVector<int16_t>({0, 1, 0, 1, 0, 1});
  auto rowVector = vectorMaker_->rowVector(
      {"1", "2", "3"}, {childVector1, childVector2, childVector3});
  this->ranges_.add(0, rowVector->size());
  auto rawSize = nimble::getRawSizeFromRowVector(
      rowVector, this->ranges_, context_, /*topLevel=*/true);

  constexpr auto expectedRawSize =
      sizeof(int64_t) * 6 + sizeof(bool) * 6 + sizeof(int16_t) * 6;
  ASSERT_EQ(expectedRawSize, rawSize);
  ASSERT_EQ(3, context_.columnCount());
  ASSERT_EQ(sizeof(int64_t) * 6, context_.sizeAt(0));
  ASSERT_EQ(sizeof(bool) * 6, context_.sizeAt(1));
  ASSERT_EQ(sizeof(int16_t) * 6, context_.sizeAt(2));
}

TEST_F(RawSizeTestFixture, RowDifferentTypes2) {
  auto childVector1 = vectorMaker_->flatVector<int64_t>({0, 0, 0, 1, 1, 1});
  auto childVector2 = vectorMaker_->flatVector<velox::StringView>(
      {"a", "bb", "ccc", "dddd", "eeeee", "ffffff"});
  auto childVector3 = vectorMaker_->flatVector<int16_t>({0, 1, 0, 1, 0, 1});
  auto rowVector = vectorMaker_->rowVector(
      {"1", "2", "3"}, {childVector1, childVector2, childVector3});
  this->ranges_.add(0, rowVector->size());
  auto rawSize = nimble::getRawSizeFromRowVector(
      rowVector, this->ranges_, context_, /*topLevel=*/true);

  constexpr auto expectedRawSize =
      sizeof(int64_t) * 6 + sizeof(int16_t) * 6 + 21;
  ASSERT_EQ(expectedRawSize, rawSize);
  ASSERT_EQ(3, context_.columnCount());
  ASSERT_EQ(sizeof(int64_t) * 6, context_.sizeAt(0));
  ASSERT_EQ(21, context_.sizeAt(1));
  ASSERT_EQ(sizeof(int16_t) * 6, context_.sizeAt(2));
}

TEST_F(RawSizeTestFixture, RowNulls) {
  auto childVector1 = vectorMaker_->flatVector<int64_t>({0, 0, 0, 1, 1, 1});
  auto childVector2 = vectorMaker_->flatVector<bool>({0, 1, 0, 1, 0, 1});
  auto childVector3 = vectorMaker_->flatVector<int16_t>({0, 1, 0, 1, 0, 1});
  velox::BufferPtr nulls = velox::AlignedBuffer::allocate<bool>(
      6, this->pool_.get(), velox::bits::kNotNull);
  auto* rawNulls = nulls->asMutable<uint64_t>();
  velox::bits::setNull(rawNulls, 2);
  const std::vector<velox::VectorPtr>& children = {
      childVector1, childVector2, childVector3};
  auto rowVector = std::make_shared<velox::RowVector>(
      pool_.get(),
      velox::ROW({velox::BIGINT(), velox::BOOLEAN(), velox::SMALLINT()}),
      nulls,
      6,
      children);
  this->ranges_.add(0, rowVector->size());
  auto rawSize = nimble::getRawSizeFromRowVector(
      rowVector, this->ranges_, context_, /*topLevel=*/true);

  constexpr auto expectedRawSize = sizeof(int64_t) * 5 + sizeof(bool) * 5 +
      sizeof(int16_t) * 5 + nimble::kNullSize * 1;
  ASSERT_EQ(expectedRawSize, rawSize);
  ASSERT_EQ(3, context_.columnCount());
  ASSERT_EQ(sizeof(int64_t) * 5, context_.sizeAt(0));
  ASSERT_EQ(sizeof(bool) * 5, context_.sizeAt(1));
  ASSERT_EQ(sizeof(int16_t) * 5, context_.sizeAt(2));
}

TEST_F(RawSizeTestFixture, RowAllNulls) {
  constexpr velox::vector_size_t VECTOR_TEST_SIZE = 6;
  auto childVector1 = vectorMaker_->flatVector<int64_t>({0, 0, 0, 1, 1, 1});
  auto childVector2 = vectorMaker_->flatVector<bool>({0, 1, 0, 1, 0, 1});
  auto childVector3 = vectorMaker_->flatVector<int16_t>({0, 1, 0, 1, 0, 1});
  velox::BufferPtr nulls = velox::AlignedBuffer::allocate<bool>(
      VECTOR_TEST_SIZE, this->pool_.get(), velox::bits::kNull);
  const std::vector<velox::VectorPtr>& children = {
      childVector1, childVector2, childVector3};
  auto rowVector = std::make_shared<velox::RowVector>(
      pool_.get(),
      velox::ROW({velox::BIGINT(), velox::BOOLEAN(), velox::SMALLINT()}),
      nulls,
      VECTOR_TEST_SIZE,
      children);
  this->ranges_.add(0, rowVector->size());
  auto rawSize = nimble::getRawSizeFromRowVector(
      rowVector, this->ranges_, context_, /*topLevel=*/true);

  constexpr auto expectedRawSize = nimble::kNullSize * VECTOR_TEST_SIZE;
  ASSERT_EQ(expectedRawSize, rawSize);
  ASSERT_EQ(3, context_.columnCount());
  for (size_t i = 0; i < 3; ++i) {
    ASSERT_EQ(0, context_.sizeAt(i));
  }
}

TEST_F(RawSizeTestFixture, RowNestedNull) {
  auto childVector1 =
      vectorMaker_->flatVectorNullable<int64_t>({0, 0, std::nullopt, 1, 1, 1});
  auto childVector2 = vectorMaker_->flatVectorNullable<velox::StringView>(
      {"a", "bb", "ccc", "dddd", "eeeee", std::nullopt});
  auto childVector3 =
      vectorMaker_->flatVectorNullable<int16_t>({0, 1, 0, 1, 0, std::nullopt});
  auto rowVector = vectorMaker_->rowVector(
      {"1", "2", "3"}, {childVector1, childVector2, childVector3});
  this->ranges_.add(0, rowVector->size());
  auto rawSize = nimble::getRawSizeFromRowVector(
      rowVector, this->ranges_, context_, /*topLevel=*/true);

  constexpr auto expectedRawSize = sizeof(int64_t) * 5 + (1 + 2 + 3 + 4 + 5) +
      sizeof(int16_t) * 5 + nimble::kNullSize * 3;
  ASSERT_EQ(expectedRawSize, rawSize);
  ASSERT_EQ(3, context_.columnCount());
  ASSERT_EQ(sizeof(int64_t) * 5 + nimble::kNullSize, context_.sizeAt(0));
  ASSERT_EQ(15 + nimble::kNullSize, context_.sizeAt(1));
  ASSERT_EQ(sizeof(int16_t) * 5 + nimble::kNullSize, context_.sizeAt(2));
}

TEST_F(RawSizeTestFixture, RowDictionaryChildren) {
  auto arrayVector =
      vectorMaker_->arrayVector<int64_t>({{0, 1, 2}, {3, 4}, {5}});
  std::unordered_map<velox::vector_size_t, uint64_t> indexToSizeArray;
  indexToSizeArray[0] = sizeof(int64_t) * 3; // Size for the first row
  indexToSizeArray[1] = sizeof(int64_t) * 2; // Size for the second row
  indexToSizeArray[2] = sizeof(int64_t) * 1; // Size for the third row

  uint64_t expectedArrayRawSize = 0;
  velox::BufferPtr indices =
      velox::AlignedBuffer::allocate<velox::vector_size_t>(
          VECTOR_SIZE, this->pool_.get());
  auto* rawIndices = indices->asMutable<velox::vector_size_t>();
  for (int i = 0; i < VECTOR_SIZE; i++) {
    rawIndices[i] = i % 3;
    expectedArrayRawSize += indexToSizeArray[i % 3];
  }

  // Create a dictionary vector of size VECTOR_SIZE containing no nulls
  auto dictArrayVector = velox::BaseVector::wrapInDictionary(
      velox::BufferPtr(nullptr), indices, VECTOR_SIZE, arrayVector);

  auto mapKeys = vectorMaker_->flatVector<velox::StringView>(
      {"a", "bb", "ccc", "dddd", "eeeee"});
  auto mapValues = vectorMaker_->flatVector<velox::StringView>(
      {"eeeee", "dddd", "ccc", "bb", "a"});

  auto mapVector = vectorMaker_->mapVector({0, 1, 3}, mapKeys, mapValues);
  std::unordered_map<velox::vector_size_t, uint64_t> indexToSizeMap;
  indexToSizeMap[0] = 6; // Size for the first row
  indexToSizeMap[1] = 12; // Size for the second row
  indexToSizeMap[2] = 12; // Size for the third row

  uint64_t expectedMapRawSize = 0;
  velox::BufferPtr indices2 =
      velox::AlignedBuffer::allocate<velox::vector_size_t>(
          VECTOR_SIZE, this->pool_.get());
  auto* rawIndices2 = indices2->asMutable<velox::vector_size_t>();
  for (int i = 0; i < VECTOR_SIZE; i++) {
    rawIndices2[i] = i % 3;
    expectedMapRawSize += indexToSizeMap[i % 3];
  }

  auto dictMapVector = velox::BaseVector::wrapInDictionary(
      velox::BufferPtr(nullptr), indices, VECTOR_SIZE, mapVector);
  auto rowVector =
      vectorMaker_->rowVector({"1", "2"}, {dictArrayVector, dictMapVector});

  this->ranges_.add(0, rowVector->size());
  auto rawSize = nimble::getRawSizeFromRowVector(
      rowVector, this->ranges_, context_, /*topLevel=*/true);
  const uint64_t expectedSize = expectedArrayRawSize + expectedMapRawSize;

  ASSERT_EQ(expectedSize, rawSize);
  ASSERT_EQ(2, context_.columnCount());
  ASSERT_EQ(expectedArrayRawSize, context_.sizeAt(0));
  ASSERT_EQ(expectedMapRawSize, context_.sizeAt(1));
}

TEST_F(RawSizeTestFixture, ConstRow) {
  auto childVector1 = vectorMaker_->flatVector<int64_t>({0, 0, 0, 1, 1, 1});
  auto childVector2 = vectorMaker_->flatVector<velox::StringView>(
      {"a", "bb", "ccc", "dddd", "eeeee", "ffffff"});
  auto childVector3 = vectorMaker_->flatVector<int16_t>({0, 1, 0, 1, 0, 1});
  auto rowVector = vectorMaker_->rowVector(
      {"1", "2", "3"}, {childVector1, childVector2, childVector3});
  const velox::vector_size_t CONST_VECTOR_SIZE = 10;
  auto constVector =
      velox::BaseVector::wrapInConstant(CONST_VECTOR_SIZE, 5, rowVector);
  this->ranges_.add(0, CONST_VECTOR_SIZE);
  auto rawSize = nimble::getRawSizeFromRowVector(
      constVector, this->ranges_, context_, /*topLevel=*/true);
  constexpr auto expectedRawSize =
      (sizeof(int64_t) + 6 + sizeof(int16_t)) * CONST_VECTOR_SIZE;

  ASSERT_EQ(expectedRawSize, rawSize);
  ASSERT_EQ(3, context_.columnCount());
  ASSERT_EQ(sizeof(int64_t) * CONST_VECTOR_SIZE, context_.sizeAt(0));
  ASSERT_EQ(6 * CONST_VECTOR_SIZE, context_.sizeAt(1));
  ASSERT_EQ(sizeof(int16_t) * CONST_VECTOR_SIZE, context_.sizeAt(2));
}

TEST_F(RawSizeTestFixture, ConstRowNestedNull) {
  auto childVector1 =
      vectorMaker_->flatVectorNullable<int64_t>({0, 0, std::nullopt, 1, 1, 1});
  auto childVector2 = vectorMaker_->flatVectorNullable<velox::StringView>(
      {"a", "bb", "ccc", "dddd", "eeeee", std::nullopt});
  auto childVector3 =
      vectorMaker_->flatVectorNullable<int16_t>({0, 1, 0, 1, 0, std::nullopt});
  auto rowVector = vectorMaker_->rowVector(
      {"1", "2", "3"}, {childVector1, childVector2, childVector3});
  const velox::vector_size_t CONST_VECTOR_SIZE = 10;
  auto constVector =
      velox::BaseVector::wrapInConstant(CONST_VECTOR_SIZE, 5, rowVector);
  this->ranges_.add(0, constVector->size());
  auto rawSize = nimble::getRawSizeFromRowVector(
      constVector, this->ranges_, context_, /*topLevel=*/true);
  constexpr auto expectedRawSize =
      (sizeof(int64_t) + nimble::kNullSize * 2) * CONST_VECTOR_SIZE;

  ASSERT_EQ(expectedRawSize, rawSize);
  ASSERT_EQ(3, context_.columnCount());
  ASSERT_EQ(sizeof(int64_t) * CONST_VECTOR_SIZE, context_.sizeAt(0));
  ASSERT_EQ(nimble::kNullSize * CONST_VECTOR_SIZE, context_.sizeAt(1));
  ASSERT_EQ(nimble::kNullSize * CONST_VECTOR_SIZE, context_.sizeAt(2));
}

TEST_F(RawSizeTestFixture, DictRow) {
  constexpr velox::vector_size_t VECTOR_TEST_SIZE = 5;
  auto childVector1 = vectorMaker_->flatVector<int64_t>({0, 0, 0, 1, 1, 1});
  auto childVector2 = vectorMaker_->flatVector<velox::StringView>(
      {"a", "bb", "ccc", "dddd", "eeeee", "ffffff"});
  auto childVector3 = vectorMaker_->flatVector<int16_t>({0, 1, 0, 1, 0, 1});
  auto rowVector = vectorMaker_->rowVector(
      {"1", "2", "3"}, {childVector1, childVector2, childVector3});

  velox::BufferPtr indices =
      velox::AlignedBuffer::allocate<velox::vector_size_t>(
          VECTOR_TEST_SIZE, this->pool_.get());
  auto* rawIndices = indices->asMutable<velox::vector_size_t>();
  rawIndices[0] = 0;
  rawIndices[1] = 0;
  rawIndices[2] = 1;
  rawIndices[3] = 2;
  rawIndices[4] = 3;

  // Create a dictionary vector of size VECTOR_TEST_SIZE containing no nulls
  auto dictVector = velox::BaseVector::wrapInDictionary(
      velox::BufferPtr(nullptr), indices, VECTOR_TEST_SIZE, rowVector);
  this->ranges_.add(0, dictVector->size());
  auto rawSize = nimble::getRawSizeFromRowVector(
      dictVector, this->ranges_, context_, /*topLevel=*/true);
  constexpr auto expectedRawSize =
      sizeof(int64_t) * 5 + sizeof(int16_t) * 5 + 11;

  ASSERT_EQ(expectedRawSize, rawSize);
  ASSERT_EQ(3, context_.columnCount());
  ASSERT_EQ(sizeof(int64_t) * VECTOR_TEST_SIZE, context_.sizeAt(0));
  ASSERT_EQ(11, context_.sizeAt(1));
  ASSERT_EQ(sizeof(int16_t) * VECTOR_TEST_SIZE, context_.sizeAt(2));
}

TEST_F(RawSizeTestFixture, DictRowNull) {
  constexpr velox::vector_size_t VECTOR_TEST_SIZE = 5;
  auto childVector1 =
      vectorMaker_->flatVectorNullable<int64_t>({std::nullopt, 0, 0, 1, 1, 0});
  auto childVector2 = vectorMaker_->flatVector<velox::StringView>(
      {"a", "bb", "ccc", "dddd", "eeeee", "ffffff"});
  auto childVector3 = vectorMaker_->flatVector<int16_t>({0, 1, 0, 1, 0, 1});
  auto rowVector = vectorMaker_->rowVector(
      {"1", "2", "3"}, {childVector1, childVector2, childVector3});

  velox::BufferPtr indices =
      velox::AlignedBuffer::allocate<velox::vector_size_t>(
          VECTOR_TEST_SIZE, this->pool_.get());
  auto* rawIndices = indices->asMutable<velox::vector_size_t>();
  rawIndices[0] = 0;
  rawIndices[1] = 0;
  rawIndices[2] = 1;
  rawIndices[3] = 2;
  rawIndices[4] = 3;

  // Create a dictionary vector of size VECTOR_TEST_SIZE containing no nulls
  auto dictVector = velox::BaseVector::wrapInDictionary(
      velox::BufferPtr(nullptr), indices, VECTOR_TEST_SIZE, rowVector);
  this->ranges_.add(0, dictVector->size());
  auto rawSize = nimble::getRawSizeFromRowVector(
      dictVector, this->ranges_, context_, /*topLevel=*/true);
  constexpr auto expectedRawSize =
      sizeof(int64_t) * 3 + sizeof(int16_t) * 5 + 11 + nimble::kNullSize * 2;

  ASSERT_EQ(expectedRawSize, rawSize);
  ASSERT_EQ(3, context_.columnCount());
  ASSERT_EQ(nimble::kNullSize * 2 + sizeof(int64_t) * 3, context_.sizeAt(0));
  ASSERT_EQ(11, context_.sizeAt(1));
  ASSERT_EQ(sizeof(int16_t) * VECTOR_TEST_SIZE, context_.sizeAt(2));
}

TEST_F(RawSizeTestFixture, DictRowNullTopLevel) {
  constexpr velox::vector_size_t VECTOR_TEST_SIZE = 5;
  auto childVector1 = vectorMaker_->flatVector<int64_t>({0, 0, 0, 1, 1, 1});
  auto childVector2 = vectorMaker_->flatVector<velox::StringView>(
      {"a", "bb", "ccc", "dddd", "eeeee", "ffffff"});
  auto childVector3 = vectorMaker_->flatVector<int16_t>({0, 1, 0, 1, 0, 1});
  auto rowVector = vectorMaker_->rowVector(
      {"1", "2", "3"}, {childVector1, childVector2, childVector3});

  velox::BufferPtr indices =
      velox::AlignedBuffer::allocate<velox::vector_size_t>(
          VECTOR_TEST_SIZE, this->pool_.get());
  auto* rawIndices = indices->asMutable<velox::vector_size_t>();
  rawIndices[0] = 0;
  rawIndices[1] = 0;
  rawIndices[2] = 1; // null
  rawIndices[3] = 2;
  rawIndices[4] = 3;

  velox::BufferPtr nulls = velox::AlignedBuffer::allocate<bool>(
      VECTOR_TEST_SIZE, this->pool_.get(), velox::bits::kNotNull);
  auto* rawNulls = nulls->asMutable<uint64_t>();
  velox::bits::setNull(rawNulls, 2);

  // Create a dictionary vector of size VECTOR_TEST_SIZE containing no nulls
  auto dictVector = velox::BaseVector::wrapInDictionary(
      nulls, indices, VECTOR_TEST_SIZE, rowVector);
  this->ranges_.add(0, dictVector->size());
  auto rawSize = nimble::getRawSizeFromRowVector(
      dictVector, this->ranges_, context_, /*topLevel=*/true);
  constexpr auto expectedRawSize =
      sizeof(int64_t) * 4 + sizeof(int16_t) * 4 + 9 + nimble::kNullSize * 1;

  ASSERT_EQ(expectedRawSize, rawSize);
  ASSERT_EQ(sizeof(int64_t) * (VECTOR_TEST_SIZE - 1), context_.sizeAt(0));
  ASSERT_EQ(9, context_.sizeAt(1));
  ASSERT_EQ(sizeof(int16_t) * (VECTOR_TEST_SIZE - 1), context_.sizeAt(2));
}

TEST_F(RawSizeTestFixture, ThrowOnDefaultType) {
  auto unknownVector = facebook::velox::BaseVector::create(
      facebook::velox::UNKNOWN(), 10, pool_.get());
  this->ranges_.add(0, unknownVector->size());

  EXPECT_THROW(
      nimble::getRawSizeFromVector(unknownVector, this->ranges_),
      velox::VeloxRuntimeError);
}

TEST_F(RawSizeTestFixture, ThrowOnDefaultEncodingFixedWidth) {
  auto sequenceVector =
      vectorMaker_->sequenceVector<int8_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  this->ranges_.add(0, sequenceVector->size());

  EXPECT_THROW(
      nimble::getRawSizeFromVector(sequenceVector, this->ranges_),
      velox::VeloxRuntimeError);
}

TEST_F(RawSizeTestFixture, ThrowOnDefaultEncodingVariableWidth) {
  auto sequenceVector = vectorMaker_->sequenceVector<velox::StringView>(
      {"a", "bbbb", "ccccccccc", "dddddddddddddddd"});
  this->ranges_.add(0, sequenceVector->size());

  EXPECT_THROW(
      nimble::getRawSizeFromVector(sequenceVector, this->ranges_),
      velox::VeloxRuntimeError);
}

// Test type compatibility helper functions
TEST_F(RawSizeTestFixture, TypeSizeFromKind) {
  // Fixed-width types should return their sizes via getTypeSize
  EXPECT_EQ(nimble::getTypeSize(*velox::BOOLEAN()), sizeof(bool));
  EXPECT_EQ(nimble::getTypeSize(*velox::TINYINT()), sizeof(int8_t));
  EXPECT_EQ(nimble::getTypeSize(*velox::SMALLINT()), sizeof(int16_t));
  EXPECT_EQ(nimble::getTypeSize(*velox::INTEGER()), sizeof(int32_t));
  EXPECT_EQ(nimble::getTypeSize(*velox::BIGINT()), sizeof(int64_t));
  EXPECT_EQ(nimble::getTypeSize(*velox::REAL()), sizeof(float));
  EXPECT_EQ(nimble::getTypeSize(*velox::DOUBLE()), sizeof(double));
  // Timestamp uses 12 bytes (8 for seconds + 4 for nanos)
  EXPECT_EQ(nimble::getTypeSize(*velox::TIMESTAMP()), 12);

  // Variable-width and complex types should return std::nullopt
  EXPECT_FALSE(nimble::getTypeSize(*velox::VARCHAR()).has_value());
  EXPECT_FALSE(nimble::getTypeSize(*velox::VARBINARY()).has_value());
  EXPECT_FALSE(
      nimble::getTypeSize(*velox::ARRAY(velox::INTEGER())).has_value());
  EXPECT_FALSE(
      nimble::getTypeSize(*velox::MAP(velox::INTEGER(), velox::VARCHAR()))
          .has_value());
  EXPECT_FALSE(
      nimble::getTypeSize(*velox::ROW({velox::INTEGER()})).has_value());
}

// Tests for schema-aware raw size calculation with type mismatches
class RawSizeTypeCompatibilityTestFixture : public RawSizeBaseTestFixture {
 protected:
  std::shared_ptr<velox::dwio::common::TypeWithId> makeTypeWithId(
      const velox::TypePtr& type) {
    return velox::dwio::common::TypeWithId::create(type);
  }
};

// Test scalar type mismatch: int32_t vector with BIGINT schema
TEST_F(RawSizeTypeCompatibilityTestFixture, ScalarIntegerToBigint) {
  constexpr velox::vector_size_t SIZE = 10;
  auto vector =
      vectorMaker_->flatVector<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto schemaType = makeTypeWithId(velox::BIGINT());
  folly::F14FastSet<uint32_t> flatMapNodeIds;

  ranges_.add(0, SIZE);

  // Without schema: int32_t * 10 = 40 bytes
  auto rawSizeWithoutSchema =
      nimble::getRawSizeFromVector(vector, ranges_, context_);
  EXPECT_EQ(rawSizeWithoutSchema, sizeof(int32_t) * SIZE);

  // With schema (BIGINT): int64_t * 10 = 80 bytes
  context_ = nimble::RawSizeContext();
  auto rawSizeWithSchema = nimble::getRawSizeFromVector(
      vector, ranges_, context_, schemaType.get(), flatMapNodeIds);
  EXPECT_EQ(rawSizeWithSchema, sizeof(int64_t) * SIZE);
}

// Test scalar type mismatch with nulls: int32_t vector with BIGINT schema
TEST_F(RawSizeTypeCompatibilityTestFixture, ScalarIntegerToBigintWithNulls) {
  auto vector = vectorMaker_->flatVectorNullable<int32_t>(
      {1, std::nullopt, 3, 4, std::nullopt, 6, 7, 8, 9, 10});
  auto schemaType = makeTypeWithId(velox::BIGINT());
  folly::F14FastSet<uint32_t> flatMapNodeIds;

  ranges_.add(0, 10);

  // Without schema: (int32_t * 8) + (kNullSize * 2) = 32 + 2 = 34 bytes
  auto rawSizeWithoutSchema =
      nimble::getRawSizeFromVector(vector, ranges_, context_);
  EXPECT_EQ(rawSizeWithoutSchema, sizeof(int32_t) * 8 + nimble::kNullSize * 2);

  // With schema (BIGINT): (int64_t * 8) + (kNullSize * 2) = 64 + 2 = 66 bytes
  context_ = nimble::RawSizeContext();
  auto rawSizeWithSchema = nimble::getRawSizeFromVector(
      vector, ranges_, context_, schemaType.get(), flatMapNodeIds);
  EXPECT_EQ(rawSizeWithSchema, sizeof(int64_t) * 8 + nimble::kNullSize * 2);
}

// Test REAL to DOUBLE promotion
TEST_F(RawSizeTypeCompatibilityTestFixture, ScalarRealToDouble) {
  constexpr velox::vector_size_t SIZE = 5;
  auto vector = vectorMaker_->flatVector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  auto schemaType = makeTypeWithId(velox::DOUBLE());
  folly::F14FastSet<uint32_t> flatMapNodeIds;

  ranges_.add(0, SIZE);

  // Without schema: float * 5 = 20 bytes
  auto rawSizeWithoutSchema =
      nimble::getRawSizeFromVector(vector, ranges_, context_);
  EXPECT_EQ(rawSizeWithoutSchema, sizeof(float) * SIZE);

  // With schema (DOUBLE): double * 5 = 40 bytes
  context_ = nimble::RawSizeContext();
  auto rawSizeWithSchema = nimble::getRawSizeFromVector(
      vector, ranges_, context_, schemaType.get(), flatMapNodeIds);
  EXPECT_EQ(rawSizeWithSchema, sizeof(double) * SIZE);
}

// Test SMALLINT to INTEGER promotion
TEST_F(RawSizeTypeCompatibilityTestFixture, ScalarSmallintToInteger) {
  constexpr velox::vector_size_t SIZE = 4;
  auto vector = vectorMaker_->flatVector<int16_t>({1, 2, 3, 4});
  auto schemaType = makeTypeWithId(velox::INTEGER());
  folly::F14FastSet<uint32_t> flatMapNodeIds;

  ranges_.add(0, SIZE);

  // Without schema: int16_t * 4 = 8 bytes
  auto rawSizeWithoutSchema =
      nimble::getRawSizeFromVector(vector, ranges_, context_);
  EXPECT_EQ(rawSizeWithoutSchema, sizeof(int16_t) * SIZE);

  // With schema (INTEGER): int32_t * 4 = 16 bytes
  context_ = nimble::RawSizeContext();
  auto rawSizeWithSchema = nimble::getRawSizeFromVector(
      vector, ranges_, context_, schemaType.get(), flatMapNodeIds);
  EXPECT_EQ(rawSizeWithSchema, sizeof(int32_t) * SIZE);
}

// Test TINYINT to BIGINT promotion (largest gap)
TEST_F(RawSizeTypeCompatibilityTestFixture, ScalarTinyintToBigint) {
  constexpr velox::vector_size_t SIZE = 8;
  auto vector = vectorMaker_->flatVector<int8_t>({1, 2, 3, 4, 5, 6, 7, 8});
  auto schemaType = makeTypeWithId(velox::BIGINT());
  folly::F14FastSet<uint32_t> flatMapNodeIds;

  ranges_.add(0, SIZE);

  // Without schema: int8_t * 8 = 8 bytes
  auto rawSizeWithoutSchema =
      nimble::getRawSizeFromVector(vector, ranges_, context_);
  EXPECT_EQ(rawSizeWithoutSchema, sizeof(int8_t) * SIZE);

  // With schema (BIGINT): int64_t * 8 = 64 bytes
  context_ = nimble::RawSizeContext();
  auto rawSizeWithSchema = nimble::getRawSizeFromVector(
      vector, ranges_, context_, schemaType.get(), flatMapNodeIds);
  EXPECT_EQ(rawSizeWithSchema, sizeof(int64_t) * SIZE);
}

// Test BOOLEAN to INTEGER promotion (boolean is now in integer family)
TEST_F(RawSizeTypeCompatibilityTestFixture, ScalarBooleanToInteger) {
  constexpr velox::vector_size_t SIZE = 6;
  auto vector =
      vectorMaker_->flatVector<bool>({true, false, true, true, false, true});
  auto schemaType = makeTypeWithId(velox::INTEGER());
  folly::F14FastSet<uint32_t> flatMapNodeIds;

  ranges_.add(0, SIZE);

  // Without schema: bool * 6 = 6 bytes
  auto rawSizeWithoutSchema =
      nimble::getRawSizeFromVector(vector, ranges_, context_);
  EXPECT_EQ(rawSizeWithoutSchema, sizeof(bool) * SIZE);

  // With schema (INTEGER): int32_t * 6 = 24 bytes
  context_ = nimble::RawSizeContext();
  auto rawSizeWithSchema = nimble::getRawSizeFromVector(
      vector, ranges_, context_, schemaType.get(), flatMapNodeIds);
  EXPECT_EQ(rawSizeWithSchema, sizeof(int32_t) * SIZE);
}

// Test BOOLEAN to BIGINT promotion
TEST_F(RawSizeTypeCompatibilityTestFixture, ScalarBooleanToBigint) {
  constexpr velox::vector_size_t SIZE = 4;
  auto vector = vectorMaker_->flatVector<bool>({true, false, true, false});
  auto schemaType = makeTypeWithId(velox::BIGINT());
  folly::F14FastSet<uint32_t> flatMapNodeIds;

  ranges_.add(0, SIZE);

  // Without schema: bool * 4 = 4 bytes
  auto rawSizeWithoutSchema =
      nimble::getRawSizeFromVector(vector, ranges_, context_);
  EXPECT_EQ(rawSizeWithoutSchema, sizeof(bool) * SIZE);

  // With schema (BIGINT): int64_t * 4 = 32 bytes
  context_ = nimble::RawSizeContext();
  auto rawSizeWithSchema = nimble::getRawSizeFromVector(
      vector, ranges_, context_, schemaType.get(), flatMapNodeIds);
  EXPECT_EQ(rawSizeWithSchema, sizeof(int64_t) * SIZE);
}

// Test BOOLEAN to INTEGER with nulls
TEST_F(RawSizeTypeCompatibilityTestFixture, ScalarBooleanToIntegerWithNulls) {
  auto vector = vectorMaker_->flatVectorNullable<bool>(
      {true, std::nullopt, false, true, std::nullopt});
  auto schemaType = makeTypeWithId(velox::INTEGER());
  folly::F14FastSet<uint32_t> flatMapNodeIds;

  ranges_.add(0, 5);

  // Without schema: bool * 3 + null * 2 = 3 + 2 = 5 bytes
  auto rawSizeWithoutSchema =
      nimble::getRawSizeFromVector(vector, ranges_, context_);
  EXPECT_EQ(rawSizeWithoutSchema, sizeof(bool) * 3 + nimble::kNullSize * 2);

  // With schema (INTEGER): int32_t * 3 + null * 2 = 12 + 2 = 14 bytes
  context_ = nimble::RawSizeContext();
  auto rawSizeWithSchema = nimble::getRawSizeFromVector(
      vector, ranges_, context_, schemaType.get(), flatMapNodeIds);
  EXPECT_EQ(rawSizeWithSchema, sizeof(int32_t) * 3 + nimble::kNullSize * 2);
}

// Test array with element type mismatch
TEST_F(RawSizeTypeCompatibilityTestFixture, ArrayWithTypeMismatch) {
  // Create ARRAY<int32_t> vector with 2 rows:
  // Row 0: [1, 2, 3] (3 elements)
  // Row 1: [4, 5] (2 elements)
  // Total: 5 int32_t elements = 20 bytes without schema
  auto arrayVector = vectorMaker_->arrayVector<int32_t>({{1, 2, 3}, {4, 5}});
  auto schemaType = makeTypeWithId(velox::ARRAY(velox::BIGINT()));
  folly::F14FastSet<uint32_t> flatMapNodeIds;

  ranges_.add(0, 2);

  // Without schema: int32_t * 5 = 20 bytes
  auto rawSizeWithoutSchema =
      nimble::getRawSizeFromVector(arrayVector, ranges_, context_);
  EXPECT_EQ(rawSizeWithoutSchema, sizeof(int32_t) * 5);

  // With schema (ARRAY<BIGINT>): int64_t * 5 = 40 bytes
  context_ = nimble::RawSizeContext();
  auto rawSizeWithSchema = nimble::getRawSizeFromVector(
      arrayVector, ranges_, context_, schemaType.get(), flatMapNodeIds);
  EXPECT_EQ(rawSizeWithSchema, sizeof(int64_t) * 5);
}

// Test map with value type mismatch
TEST_F(RawSizeTypeCompatibilityTestFixture, MapWithValueTypeMismatch) {
  // Create MAP<VARCHAR, int32_t> vector with 2 entries:
  // Entry 0: {"a" -> 1}
  // Entry 1: {"bb" -> 2}
  // Key sizes: 1 + 2 = 3 bytes
  // Value sizes: int32_t * 2 = 8 bytes
  // Total without schema: 11 bytes
  auto mapVector = vectorMaker_->mapVector<velox::StringView, int32_t>(
      {{{velox::StringView("a"), 1}}, {{velox::StringView("bb"), 2}}});
  auto schemaType =
      makeTypeWithId(velox::MAP(velox::VARCHAR(), velox::BIGINT()));
  folly::F14FastSet<uint32_t> flatMapNodeIds;

  ranges_.add(0, 2);

  // Without schema: 3 (keys) + 8 (int32_t values) = 11 bytes
  auto rawSizeWithoutSchema =
      nimble::getRawSizeFromVector(mapVector, ranges_, context_);
  EXPECT_EQ(rawSizeWithoutSchema, 3 + sizeof(int32_t) * 2);

  // With schema (MAP<VARCHAR, BIGINT>): 3 (keys) + 16 (int64_t values) = 19
  // bytes
  context_ = nimble::RawSizeContext();
  auto rawSizeWithSchema = nimble::getRawSizeFromVector(
      mapVector, ranges_, context_, schemaType.get(), flatMapNodeIds);
  EXPECT_EQ(rawSizeWithSchema, 3 + sizeof(int64_t) * 2);
}

// Test map with key type mismatch
TEST_F(RawSizeTypeCompatibilityTestFixture, MapWithKeyTypeMismatch) {
  // Create MAP<int32_t, VARCHAR> vector with 2 entries:
  // Entry 0: {1 -> "a"}
  // Entry 1: {2 -> "bb"}
  // Key sizes (int32_t): 4 * 2 = 8 bytes
  // Value sizes: 1 + 2 = 3 bytes
  // Total without schema: 11 bytes
  auto mapVector = vectorMaker_->mapVector<int32_t, velox::StringView>(
      {{{1, velox::StringView("a")}}, {{2, velox::StringView("bb")}}});
  auto schemaType =
      makeTypeWithId(velox::MAP(velox::BIGINT(), velox::VARCHAR()));
  folly::F14FastSet<uint32_t> flatMapNodeIds;

  ranges_.add(0, 2);

  // Without schema: 8 (int32_t keys) + 3 (varchar values) = 11 bytes
  auto rawSizeWithoutSchema =
      nimble::getRawSizeFromVector(mapVector, ranges_, context_);
  EXPECT_EQ(rawSizeWithoutSchema, sizeof(int32_t) * 2 + 3);

  // With schema (MAP<BIGINT, VARCHAR>): 16 (int64_t keys) + 3 (varchar values)
  // = 19 bytes
  context_ = nimble::RawSizeContext();
  auto rawSizeWithSchema = nimble::getRawSizeFromVector(
      mapVector, ranges_, context_, schemaType.get(), flatMapNodeIds);
  EXPECT_EQ(rawSizeWithSchema, sizeof(int64_t) * 2 + 3);
}

// Test constant vector with type mismatch
TEST_F(RawSizeTypeCompatibilityTestFixture, ConstantVectorWithTypeMismatch) {
  constexpr velox::vector_size_t SIZE = 100;
  std::vector<std::optional<int32_t>> vec(SIZE, 42);
  auto constVector = vectorMaker_->constantVector<int32_t>(vec);
  auto schemaType = makeTypeWithId(velox::BIGINT());
  folly::F14FastSet<uint32_t> flatMapNodeIds;

  ranges_.add(0, SIZE);

  // Without schema: int32_t * 100 = 400 bytes
  auto rawSizeWithoutSchema =
      nimble::getRawSizeFromVector(constVector, ranges_, context_);
  EXPECT_EQ(rawSizeWithoutSchema, sizeof(int32_t) * SIZE);

  // With schema (BIGINT): int64_t * 100 = 800 bytes
  context_ = nimble::RawSizeContext();
  auto rawSizeWithSchema = nimble::getRawSizeFromVector(
      constVector, ranges_, context_, schemaType.get(), flatMapNodeIds);
  EXPECT_EQ(rawSizeWithSchema, sizeof(int64_t) * SIZE);
}

// Test dictionary vector with type mismatch
TEST_F(RawSizeTypeCompatibilityTestFixture, DictionaryVectorWithTypeMismatch) {
  constexpr velox::vector_size_t SIZE = 10;
  auto flatVector =
      vectorMaker_->flatVector<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto indices = randomIndices(SIZE);
  auto dictVector = velox::BaseVector::wrapInDictionary(
      velox::BufferPtr(nullptr), indices, SIZE, flatVector);

  auto schemaType = makeTypeWithId(velox::BIGINT());
  folly::F14FastSet<uint32_t> flatMapNodeIds;

  ranges_.add(0, SIZE);

  // Without schema: int32_t * 10 = 40 bytes
  auto rawSizeWithoutSchema =
      nimble::getRawSizeFromVector(dictVector, ranges_, context_);
  EXPECT_EQ(rawSizeWithoutSchema, sizeof(int32_t) * SIZE);

  // With schema (BIGINT): int64_t * 10 = 80 bytes
  context_ = nimble::RawSizeContext();
  auto rawSizeWithSchema = nimble::getRawSizeFromVector(
      dictVector, ranges_, context_, schemaType.get(), flatMapNodeIds);
  EXPECT_EQ(rawSizeWithSchema, sizeof(int64_t) * SIZE);
}

// Test row vector with nested type mismatches
TEST_F(RawSizeTypeCompatibilityTestFixture, RowVectorWithTypeMismatch) {
  // Create ROW with:
  // - col0: int32_t values [1, 2]
  // - col1: float values [1.0, 2.0]
  auto childVector1 = vectorMaker_->flatVector<int32_t>({1, 2});
  auto childVector2 = vectorMaker_->flatVector<float>({1.0f, 2.0f});
  auto rowVector =
      vectorMaker_->rowVector({"col0", "col1"}, {childVector1, childVector2});

  // Schema declares BIGINT and DOUBLE
  auto schemaType = makeTypeWithId(
      velox::ROW({{"col0", velox::BIGINT()}, {"col1", velox::DOUBLE()}}));
  folly::F14FastSet<uint32_t> flatMapNodeIds;

  ranges_.add(0, 2);

  // Without schema: (int32_t * 2) + (float * 2) = 8 + 8 = 16 bytes
  auto rawSizeWithoutSchema = nimble::getRawSizeFromRowVector(
      rowVector, ranges_, context_, nullptr, {}, true);
  EXPECT_EQ(rawSizeWithoutSchema, sizeof(int32_t) * 2 + sizeof(float) * 2);

  // With schema: (int64_t * 2) + (double * 2) = 16 + 16 = 32 bytes
  context_ = nimble::RawSizeContext();
  auto rawSizeWithSchema = nimble::getRawSizeFromRowVector(
      rowVector, ranges_, context_, schemaType.get(), flatMapNodeIds, true);
  EXPECT_EQ(rawSizeWithSchema, sizeof(int64_t) * 2 + sizeof(double) * 2);
}

TEST_F(RawSizeTestFixture, LocalDecodedVectorMoveConstructor) {
  auto localDecodedVector1 =
      facebook::nimble::DecodedVectorManager::LocalDecodedVector(
          context_.getDecodedVectorManager());

  // Constuct LocalDecodedVector by LocalDecodedVector ctr
  auto localDecodedVector2 = std::move(localDecodedVector1);

  // Access the DecodedVector to ensure it's not null
  velox::DecodedVector& decodedVector = localDecodedVector2.get();
  ASSERT_NE(&decodedVector, nullptr);

  // No checks on localDecodedVector1 as it's been moved
}

// Regression test: when a ROW TypeWithId node has its ID in flatMapNodeIds
// but the schema type is ROW (not MAP), getRawSizeFromVector should treat it
// as a regular ROW vector instead of crashing in passthrough flatmap path.
// This happens during flatmap-as-struct rewrite where MAP is converted to ROW
// in the readSchema.
TEST_F(RawSizeTestFixture, RowTypeWithIdInFlatMapNodeIdsDoesNotCrash) {
  // Create a ROW vector with a single child (simulates a flatmap column
  // with 1 feature read as struct).
  auto child = vectorMaker_->flatVector<int32_t>({1, 2, 3});
  auto rowVector = vectorMaker_->rowVector({"feature0"}, {child});

  // Create a ROW TypeWithId (as readSchema would produce).
  // ID assignment: 0=root ROW, 1=child ROW (the "flatmap" column), 2=INT
  auto schema =
      velox::ROW({{"c0", velox::ROW({{"feature0", velox::INTEGER()}})}});
  auto typeWithId = velox::dwio::common::TypeWithId::create(schema);
  const auto* childTypeWithId = typeWithId->childAt(0).get();

  // Mark the ROW child's node ID as a flatmap node.
  // This simulates the rewrite flow where the ROW TypeWithId node from
  // readSchema has its ID added to flatMapNodeIds.
  folly::F14FastSet<uint32_t> flatMapNodeIds{childTypeWithId->id()};

  ranges_.add(0, rowVector->size());

  // Should not crash — the fix ensures passthrough flatmap detection
  // requires the schema type to be MAP.
  auto rawSize = nimble::getRawSizeFromVector(
      rowVector, ranges_, context_, childTypeWithId, flatMapNodeIds);

  // Should compute regular ROW vector raw size (3 int32 values).
  ASSERT_EQ(rawSize, sizeof(int32_t) * 3);
}
