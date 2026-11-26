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

#include "velox/dwio/parquet/reader/RleBpDecoder.h"

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

using facebook::velox::parquet::RleBpDecoder;

TEST(RleBpDecoderTest, unalignedRepeatingValue) {
  constexpr int32_t kCount = 3;
  constexpr int32_t kValue = 0x12345678;

  std::vector<uint8_t> encoded = {
      static_cast<uint8_t>(kCount << 1),
      0x78,
      0x56,
      0x34,
      0x12,
      0,
      0,
      0,
      0,
  };

  std::vector<uint8_t> unaligned(1 + encoded.size(), 0);
  std::copy(encoded.begin(), encoded.end(), unaligned.begin() + 1);

  RleBpDecoder decoder(
      reinterpret_cast<const char*>(unaligned.data() + 1),
      reinterpret_cast<const char*>(unaligned.data() + unaligned.size()),
      32);

  std::vector<int32_t> output(kCount);
  auto* outputPtr = output.data();
  decoder.next(outputPtr, output.size());

  for (auto value : output) {
    ASSERT_EQ(value, kValue);
  }
}
