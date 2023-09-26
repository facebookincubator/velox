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

#include "velox/exec/Strings.h"
#include "velox/common/memory/HashStringAllocator.h"

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::aggregate::prestosql;

namespace {

class StringsTest : public testing::Test {
 protected:
  void SetUp() override {
    pool_ = memory::addDefaultLeafMemoryPool();
    allocator_ = std::make_unique<HashStringAllocator>(pool_.get());
  }

  // Appends a string and records the string and result for verification.
  void append(Strings& strings, const std::string& str) {
    appended_.push_back(str);
    views_.push_back(strings.append(StringView(str), *allocator_));
    check();
  }

  // Checks all appended strings are unchanged.
  void check() {
    for (auto i = 0; i < appended_.size(); ++i) {
      EXPECT_EQ(
          0,
          memcmp(appended_[i].data(), views_[i].data(), appended_[i].size()));
    }
  }

  std::vector<std::string> appended_;
  std::vector<StringView> views_;

  std::shared_ptr<memory::MemoryPool> pool_;
  std::unique_ptr<HashStringAllocator> allocator_;
};

TEST_F(StringsTest, basic) {
  constexpr int32_t kHeader = sizeof(HashStringAllocator::Header);
  // next pointer, 8 leading bytes, and header.
  constexpr int32_t kOverhead = 16 + kHeader;
  Strings strings;
  std::string s13 = "0123456789abc";
  std::string s16 = "0123456789abcdef";
  std::string large;
  large.resize(1000);
  // Initialize allocator. Start counting cumulative from post-init baseline.
  allocator_->allocate(24);
  auto sizes = HashStringAllocator::freeListSizes();

  int32_t expectedBytes =
      allocator_->cumulativeBytes() + sizes[0] + kHeader - 8;
  append(strings, StringView(s13));
  EXPECT_EQ(expectedBytes, allocator_->cumulativeBytes());

  append(strings, StringView(s16));
  append(strings, StringView(s16));
  // 13 + 2 * 16 should fit in the first allocation of 72.
  EXPECT_EQ(expectedBytes, allocator_->cumulativeBytes());

  // Now we allocate one more and expect to have 8x16 bytes of payload without
  // more allocation.
  expectedBytes += sizes[1] + kHeader;
  for (auto i = 0; i < 7; ++i) {
    append(strings, StringView(s16));
  }
  EXPECT_EQ(expectedBytes, allocator_->cumulativeBytes());

  expectedBytes += kOverhead + large.size();
  append(strings, StringView(large));
  EXPECT_EQ(expectedBytes, allocator_->cumulativeBytes());

  // If the largest size is much larger than overhead, the next
  // allocation will be a multiple of the allocated size instead. 4 *
  // 13 + 16 overhead = 68 rounds up to first size.
  expectedBytes += sizes[0] + kHeader;
  append(strings, StringView(s13));
  EXPECT_EQ(expectedBytes, allocator_->cumulativeBytes());
}

} // namespace
