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

#include "velox/type/StringViewIdMap.h"
#include <folly/container/F14Set.h>
#include <folly/hash/Hash.h>
#include "velox/common/base/RawVector.h"
#include "velox/common/base/SelectivityInfo.h"

#include <gtest/gtest.h>

using namespace facebook::velox;

struct IdMapHasher {
  size_t operator()(const std::pair<StringView, int32_t>& item) const {
    return folly::hasher<StringView>()(item.first);
  }
};

struct IdMapComparer {
  bool operator()(
      const std::pair<StringView, int32_t>& left,
      const std::pair<StringView, int32_t>& right) const {
    return left.first == right.first;
  }
};

class F14IdMap {
 public:
  F14IdMap(int32_t initial) : set_(initial) {}

  int32_t id(StringView value) {
    std::pair<StringView, int32_t> item(
        value, static_cast<int32_t>(set_.size()));
    return set_.insert(item).first->second;
  }

  int64_t findId(StringView value) {
    std::pair<StringView, int32_t> item(
        value, static_cast<int32_t>(set_.size() + 1));
    auto it = set_.find(item);
    if (it == set_.end()) {
      return StringViewIdMap::kNotFound;
    }
    return it->second;
  }

  auto size() const {
    return set_.size();
  }

  void reserve(size_t size) {
    set_.reserve(size);
  }

  void clear() {
    set_.clear();
  }

 private:
  folly::F14FastSet<std::pair<StringView, int32_t>, IdMapHasher, IdMapComparer>
      set_;
};

class StringViewIdMapTest : public testing::Test {
 protected:
  static constexpr int32_t kBatchSize = 8;

  static void SetUpTestCase() {}

  void SetUp() override {}

  void testCase(int64_t size, int64_t range, int32_t maxLength) {
    std::vector<StringView> data;
    testData(size, range, maxLength, data);
    auto result = test(data);
    std::cout
        << fmt::format(
               "Size={} range={} 4-{} byte key clocks IdMap={} F14={} ({}%)",
               size,
               range,
               maxLength,
               result.mapTime,
               result.f14Time,
               100 * result.f14Time / result.mapTime)
        << std::endl;
  }

  void testData(
      int64_t size,
      int64_t range,
      int32_t maxLength,
      std::vector<StringView>& data) {
    size = bits::roundUp(size, kBatchSize);
    data.reserve(size);
    std::string string;
    string.resize(maxLength);
    for (auto i = 0; i < size; ++i) {
      auto string = fmt::format("{}", (1 + (i % range)) * 123456789);
      int32_t targetSize = std::max(4, i % maxLength);
      while (string.size() < targetSize) {
        string = string + string;
      }
      string.resize(targetSize);
      data.push_back(
          StringView(stringBody(string.data(), string.size()), string.size()));
    }
  }

  struct Times {
    float mapTime;
    float f14Time;
  };

  static void
  fillOutOfLine(const StringView* view, int32_t id, void* extra, char** place) {
    *place = reinterpret_cast<char* const*>(view)[1];
  }

  // Feeds 'data' into a StringViewIdMap  and
  // the F14IdMap reference implementation and checks that the outcome is the
  // same. returns the total clocks for both cases.
  Times test(const std::vector<StringView>& data) {
    StringViewIdMap map(1024);
    F14IdMap f14(1024);
    constexpr int32_t kNumRepeats = 1;
    SelectivityInfo mapInfo;
    SelectivityInfo f14Info;
    int32_t startIndices[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    for (auto counter = 0; counter < kNumRepeats; ++counter) {
      {
        SelectivityTimer t(mapInfo, data.size());
        char* copyPtr[8] = {};
        for (auto i = 0; i + kBatchSize <= data.size(); i += kBatchSize) {
          int32_t ids[kBatchSize];
          map.findIds8<true>(
              data.data() + i, startIndices, ids, nullptr, fillOutOfLine);
        }
      }
      if (counter < kNumRepeats - 1) {
        map.clear();
      }
    }

    for (auto counter = 0; counter < kNumRepeats; ++counter) {
      {
        SelectivityTimer t(f14Info, data.size());
        for (auto i = 0; i < data.size(); ++i) {
          f14.id(data[i]);
        }
      }

      if (counter < kNumRepeats - 1) {
        auto size = f14.size();
        f14.clear();
        f14.reserve(size);
      }
    }

    for (auto i = 0; i + kBatchSize <= data.size(); i += kBatchSize) {
      int32_t ids[kBatchSize];

      map.findIds8<false>(data.data() + i, startIndices, ids, nullptr, nullptr);

      for (auto j = 0; j < kBatchSize; ++j) {
        auto reference = f14.findId(data[i + j]);
        EXPECT_EQ(reference, ids[j]);
        if (reference != ids[j]) {
          break;
        }
      }
    }

    return {mapInfo.timeToDropValue(), f14Info.timeToDropValue()};
  }

  // If 'size' is over inline size, saves the characters in 'this' and returns
  // a pointer to the start, else returns 'string'
  char* stringBody(char* string, int32_t size) {
    if (size <= StringView::kInlineSize) {
      return string;
    }
    if (strings_.empty() ||
        strings_.back().size() + size > strings_.back().capacity()) {
      strings_.emplace_back();
      strings_.back().reserve(1'000'000);
    }
    auto start = strings_.back().size();
    strings_.back().resize(strings_.back().size() + size);
    memcpy(strings_.back().data() + start, string, size);
    return strings_.back().data() + start;
  }

  std::vector<raw_vector<char>> strings_;
};

TEST_F(StringViewIdMapTest, basic) {
  testCase(1000, 3, 12);
  testCase(1000, 3, 40);
  testCase(1000, 1000, 12);
  testCase(1000, 1000, 40);
  testCase(10000, 2500, 40);
  testCase(1000000, 1000000, 40);
  testCase(5000000, 1000000, 30);
}
enum class TestEnum : uint8_t { v1, v2, v3 };

int32_t swtest(TestEnum e, int a, int b, int c) {
  switch (e) {
    case TestEnum::v1:
      return a;
    case TestEnum::v2:
      return b;
    case TestEnum::v3:
      return c;
  }
}
