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
// #include <glog/logging.h>
// #include <gtest/gtest.h>
// #include "velox/dwio/nimble/common/Buffer.h"
// #include "velox/dwio/nimble/common/Vector.h"
// #include "velox/dwio/nimble/encodings/SentinelEncoding.h"

// #include <vector>

// using namespace facebook;

// class SentinelEncodingTest : public ::testing::Test {
//  protected:
//   void SetUp() override {
//     pool_ = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
//     buffer_ = std::make_unique<nimble::Buffer>(*pool_);
//   }

//   std::shared_ptr<velox::memory::MemoryPool> pool_;
//   std::unique_ptr<nimble::Buffer> buffer_;
// };

// namespace {
// template <typename T>
// void serializeAndDeserializeNullableValues(
//     velox::memory::MemoryPool* memoryPool,
//     nimble::Buffer* buffer,
//     uint32_t count,
//     uint32_t minNonNullCount,
//     std::function<T(uint32_t, uint32_t)> valueFunc,
//     std::function<bool(uint32_t, uint32_t)> nullFunc,
//     bool sentinelFound) {
//   nimble::Vector<T> values(memoryPool);
//   nimble::Vector<T> valuesNullRemoved(memoryPool);
//   nimble::Vector<bool> nulls(memoryPool);

//   for (uint32_t i = 0; i < count; ++i) {
//     auto notNull = nullFunc(count, i);
//     nulls.push_back(notNull);
//     auto v = valueFunc(count, i);
//     values.push_back(v);
//     if (notNull) {
//       valuesNullRemoved.push_back(v);
//     }
//   }

//   std::string sentinelString;
//   auto sentinelValue =
//       nimble::findSentinelValue<T>(valuesNullRemoved, &sentinelString);
//   EXPECT_EQ(sentinelValue.has_value(), sentinelFound)
//       << (sentinelFound ? "Expect to find sentinel value"
//                         : "Expect not able to find sentinel value");

//   ASSERT_GE(valuesNullRemoved.size(), minNonNullCount)
//       << "Too few non-null values";

//   auto serializedData =
//       nimble::SentinelEncoding<T>::serialize(valuesNullRemoved, nulls,
//       buffer);
//   auto sentinelEncoding =
//       std::make_unique<nimble::SentinelEncoding<T>>(*memoryPool,
//       serializedData);
//   nimble::Vector<T> valuesResult(memoryPool, count);
//   auto requiredBytes = nimble::bits::bytesRequired(count);
//   nimble::Buffer nullsBuffer{*memoryPool, requiredBytes + 7};
//   auto nullsPtr = nullsBuffer.reserve(requiredBytes);
//   sentinelEncoding->materializeNullable(count, valuesResult.data(),
//   nullsPtr); EXPECT_EQ(sentinelEncoding->encodingType(),
//   nimble::EncodingType::Sentinel); EXPECT_EQ(sentinelEncoding->dataType(),
//   nimble::TypeTraits<T>::dataType); EXPECT_EQ(sentinelEncoding->rowCount(),
//   count); for (uint32_t i = 0; i < count; ++i) {
//     EXPECT_EQ(
//         nulls[i],
//         nimble::bits::getBit(i, reinterpret_cast<const char*>(nullsPtr)))
//         << "Wrong null value at index " << i;
//     if (nulls[i]) {
//       EXPECT_EQ(values[i], valuesResult[i]) << "Wrong value at i " << i;
//     }
//   }
// }
// } // namespace

// TEST_F(SentinelEncodingTest, SerializeAndDeserialize) {
//   uint32_t count = 300;
//   uint32_t minNonNullCount = 256;
//   auto valueFunc = [](uint32_t /* count */, uint32_t i) -> uint8_t {
//     uint8_t v = i % 256;
//     return v == 100 ? 99 : v;
//   };

//   auto nullsFunc = [](uint32_t /* count */, uint32_t i) -> bool {
//     if (i >= 256) {
//       return i % 5 != 0;
//     }

//     return true;
//   };

//   serializeAndDeserializeNullableValues<uint8_t>(
//       pool_.get(),
//       this->buffer_.get(),
//       count,
//       minNonNullCount,
//       valueFunc,
//       nullsFunc,
//       true);
// }

// TEST_F(SentinelEncodingTest, CannotFoundSentinel) {
//   uint32_t count = 300;
//   uint32_t minNonNullCount = 256;
//   auto valueFunc = [](uint32_t /* count */, uint32_t i) -> uint8_t {
//     return i % 256;
//   };

//   auto nullsFunc = [](uint32_t /* count */, uint32_t i) -> bool {
//     if (i >= 256) {
//       return i % 5 != 0;
//     }

//     return true;
//   };

//   try {
//     serializeAndDeserializeNullableValues<uint8_t>(
//         pool_.get(),
//         this->buffer_.get(),
//         count,
//         minNonNullCount,
//         valueFunc,
//         nullsFunc,
//         false);
//   } catch (const nimble::NimbleUserError& e) {
//     EXPECT_EQ(
//         "Cannot use SentinelEncoding when no value is left for sentinel.",
//         e.errorMessage());
//     return;
//   }

//   FAIL() << "Expect nimble::NimbleUserError";
// }
