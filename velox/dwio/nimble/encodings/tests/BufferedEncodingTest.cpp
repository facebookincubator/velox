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
#include "velox/dwio/nimble/encodings/common/BufferedEncoding.h"
#include <gtest/gtest.h>
#include <span>
#include <vector>
#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

using namespace facebook;

namespace {

template <typename T>
class NoOpEncodingSelectionPolicy : public nimble::EncodingSelectionPolicy<T> {
  using physicalType = typename nimble::TypeTraits<T>::physicalType;

 public:
  nimble::EncodingSelectionResult select(
      std::span<const physicalType>,
      const nimble::Statistics<physicalType>&,
      const nimble::Encoding::Options& /* options */) override {
    return {.encodingType = nimble::EncodingType::Trivial};
  }

  nimble::EncodingSelectionResult selectNullable(
      std::span<const physicalType>,
      std::span<const bool>,
      const nimble::Statistics<physicalType>&,
      const nimble::Encoding::Options& /* options */) override {
    return {.encodingType = nimble::EncodingType::Nullable};
  }

  std::unique_ptr<nimble::EncodingSelectionPolicyBase> createImpl(
      nimble::EncodingType,
      nimble::NestedEncodingIdentifier,
      nimble::DataType type) override {
    UNIQUE_PTR_FACTORY(type, NoOpEncodingSelectionPolicy);
  }
};

class BufferedEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  std::unique_ptr<nimble::Encoding> createDictEncoding(
      const std::vector<uint32_t>& data) {
    auto span = std::span<const uint32_t>(data.data(), data.size());
    nimble::EncodingSelection<uint32_t> sel{
        {.encodingType = nimble::EncodingType::Dictionary},
        nimble::Statistics<uint32_t>::create(span),
        std::make_unique<NoOpEncodingSelectionPolicy<uint32_t>>()};
    auto serialized =
        nimble::DictionaryEncoding<uint32_t>::encode(sel, span, *buffer_);
    return nimble::EncodingFactory().create(
        *pool_, serialized, [](uint32_t) -> void* { return nullptr; });
  }

  std::unique_ptr<nimble::Encoding> createTrivialEncoding(
      const std::vector<uint32_t>& data) {
    auto span = std::span<const uint32_t>(data.data(), data.size());
    nimble::EncodingSelection<uint32_t> sel{
        {.encodingType = nimble::EncodingType::Trivial},
        nimble::Statistics<uint32_t>::create(span),
        std::make_unique<NoOpEncodingSelectionPolicy<uint32_t>>()};
    auto serialized =
        nimble::TrivialEncoding<uint32_t>::encode(sel, span, *buffer_);
    return nimble::EncodingFactory().create(
        *pool_, serialized, [](uint32_t) -> void* { return nullptr; });
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

TEST_F(BufferedEncodingTest, valueModeRead) {
  std::vector<uint32_t> data = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1};
  auto encoding = createDictEncoding(data);

  // Small page size (4) to trigger multiple refills across 10 values.
  nimble::detail::BufferedEncoding<uint32_t, 4> buf(std::move(encoding));
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(buf.nextValue(), data[i]) << "row " << i;
  }
}

TEST_F(BufferedEncodingTest, valueModeNotDictionaryEnabled) {
  // Trivial encoding is not dictionary-enabled.
  std::vector<uint32_t> data = {1, 2, 3, 4, 5};
  auto encoding = createTrivialEncoding(data);

  nimble::detail::BufferedEncoding<uint32_t, 128> buf(std::move(encoding));
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(buf.nextValue(), data[i]) << "row " << i;
  }
}

TEST_F(BufferedEncodingTest, valueModeRowCount) {
  std::vector<uint32_t> data = {11, 22, 33, 44, 55};
  auto encoding = createTrivialEncoding(data);
  const auto expectedRowCount = encoding->rowCount();

  nimble::detail::BufferedEncoding<uint32_t, 2> buf(std::move(encoding));
  EXPECT_EQ(buf.rowCount(), expectedRowCount);

  EXPECT_EQ(buf.nextValue(), data[0]);
  EXPECT_EQ(buf.rowCount(), expectedRowCount);

  buf.reset();
  EXPECT_EQ(buf.rowCount(), expectedRowCount);
  EXPECT_EQ(buf.nextValue(), data[0]);
}

TEST_F(BufferedEncodingTest, dictionaryModeRead) {
  std::vector<uint32_t> data = {10, 20, 30, 10, 20, 30, 10, 20, 30, 10};
  auto encoding = createDictEncoding(data);
  ASSERT_TRUE(encoding->dictionaryEnabled());
  const auto dictSize = encoding->dictionarySize();

  // Small page size to test multi-page index loading.
  nimble::detail::BufferedDictEncoding<uint32_t, 4> buf(std::move(encoding));
  const auto* entries = static_cast<const uint32_t*>(buf.dictionaryEntries());
  for (size_t i = 0; i < data.size(); ++i) {
    auto idx = buf.nextIndex();
    EXPECT_LT(idx, dictSize) << "row " << i;
    EXPECT_EQ(entries[idx], data[i]) << "row " << i;
  }
}

TEST_F(BufferedEncodingTest, dictionaryModeDictionaryAPI) {
  std::vector<uint32_t> data = {100, 200, 100, 200, 100};
  auto encoding = createDictEncoding(data);
  ASSERT_TRUE(encoding->dictionaryEnabled());
  const auto* entries =
      static_cast<const uint32_t*>(encoding->dictionaryEntries());
  const auto dictSize = encoding->dictionarySize();

  nimble::detail::BufferedDictEncoding<uint32_t, 128> buf(std::move(encoding));
  for (size_t i = 0; i < data.size(); ++i) {
    auto idx = buf.nextIndex();
    ASSERT_LT(idx, dictSize);
    EXPECT_EQ(entries[idx], data[i]) << "row " << i;
  }
}

TEST_F(BufferedEncodingTest, dictionaryModeReset) {
  std::vector<uint32_t> data = {5, 10, 15, 5, 10};
  auto encoding = createDictEncoding(data);

  nimble::detail::BufferedDictEncoding<uint32_t, 128> buf(std::move(encoding));
  const auto* entries = static_cast<const uint32_t*>(buf.dictionaryEntries());

  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(entries[buf.nextIndex()], data[i]);
  }

  buf.reset();
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(entries[buf.nextIndex()], data[i]) << "after reset, row " << i;
  }
}

TEST_F(BufferedEncodingTest, dictionaryModeStringValues) {
  std::vector<std::string> owned = {"alpha", "bravo", "charlie"};
  std::vector<std::string_view> data;
  for (const auto& s : owned) {
    for (int i = 0; i < 3; ++i) {
      data.push_back(s);
    }
  }
  auto span = std::span<const std::string_view>(data.data(), data.size());
  nimble::EncodingSelection<std::string_view> sel{
      {.encodingType = nimble::EncodingType::Dictionary},
      nimble::Statistics<std::string_view>::create(span),
      std::make_unique<NoOpEncodingSelectionPolicy<std::string_view>>()};
  nimble::Buffer encBuf{*pool_};
  auto serialized =
      nimble::DictionaryEncoding<std::string_view>::encode(sel, span, encBuf);
  std::vector<velox::BufferPtr> stringBufs;
  auto encoding =
      nimble::EncodingFactory().create(*pool_, serialized, [&](uint32_t size) {
        auto& buf = stringBufs.emplace_back(
            velox::AlignedBuffer::allocate<char>(size, pool_.get()));
        return buf->asMutable<void>();
      });

  ASSERT_TRUE(encoding->dictionaryEnabled());
  const auto* entries =
      static_cast<const std::string_view*>(encoding->dictionaryEntries());
  const auto dictSize = encoding->dictionarySize();

  nimble::detail::BufferedDictEncoding<std::string_view, 4> buf(
      std::move(encoding));

  for (size_t i = 0; i < data.size(); ++i) {
    auto idx = buf.nextIndex();
    ASSERT_LT(idx, dictSize) << "row " << i;
    EXPECT_EQ(entries[idx], data[i]) << "row " << i;
  }
}

TEST_F(BufferedEncodingTest, dictionaryModeValueAndIndexInterleaved) {
  std::vector<uint32_t> data = {10, 20, 30, 10, 20, 30, 10, 20};
  auto encoding = createDictEncoding(data);
  ASSERT_TRUE(encoding->dictionaryEnabled());
  const auto* entries =
      static_cast<const uint32_t*>(encoding->dictionaryEntries());
  const auto dictSize = encoding->dictionarySize();

  // Page size 3 forces multiple refills across 8 values, exercising
  // index access across page boundaries.
  nimble::detail::BufferedDictEncoding<uint32_t, 3> buf(std::move(encoding));
  for (size_t i = 0; i < data.size(); ++i) {
    auto idx = buf.nextIndex();
    ASSERT_LT(idx, dictSize) << "row " << i;
    EXPECT_EQ(entries[idx], data[i]) << "row " << i;
  }
}

TEST_F(BufferedEncodingTest, dictionaryModeDictionaryMetadata) {
  std::vector<uint32_t> data = {10, 20, 30, 10, 20, 30};
  auto encoding = createDictEncoding(data);
  ASSERT_TRUE(encoding->dictionaryEnabled());
  const auto expectedSize = encoding->dictionarySize();
  const auto* expectedEntries = encoding->dictionaryEntries();

  nimble::detail::BufferedDictEncoding<uint32_t, 128> buf(std::move(encoding));
  EXPECT_EQ(buf.dictionarySize(), expectedSize);
  EXPECT_EQ(buf.dictionaryEntries(), expectedEntries);
}

} // namespace
