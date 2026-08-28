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
#include <limits>

#include "velox/dwio/nimble/index/IndexConstants.h"

namespace facebook::nimble::index::test {

TEST(IndexConstantsTest, kKeyStreamId) {
  // kKeyStreamId should be UINT32_MAX to ensure it doesn't conflict with
  // any real stream IDs.
  EXPECT_EQ(kKeyStreamId, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(kKeyStreamId, UINT32_MAX);
}

} // namespace facebook::nimble::index::test
