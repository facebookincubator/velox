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

#include <gtest/gtest.h>

#include <cstring>

#include "velox/experimental/torchwave/StepProgram.h"

namespace torch::wave {
namespace {

using Count = GridLaunchDesc::Count;

// A parameter region with two tensors, [3] and [4,5], and an integer 7.
class GridCountTest : public ::testing::Test {
 protected:
  void SetUp() override {
    params_.resize(kScalar + sizeof(int64_t));
    setTensor(kVector, {3});
    setTensor(kMatrix, {4, 5});
    const int64_t scalar = 7;
    std::memcpy(params_.data() + kScalar, &scalar, sizeof(scalar));
  }

  void setTensor(int32_t offset, std::vector<int32_t> dims) {
    Tensor tensor;
    tensor.rank = static_cast<int8_t>(dims.size());
    tensor.numEl = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
      tensor.dims[i] = dims[i];
      tensor.numEl *= dims[i];
    }
    std::memcpy(params_.data() + offset, &tensor, sizeof(tensor));
  }

  int64_t count(Count kind, const std::vector<int32_t>& offsets) {
    GridLaunchDesc desc;
    desc.count = kind;
    for (auto offset : offsets) {
      desc.countOffsets.push_back(offset);
      desc.countIsScalar.push_back(offset == kScalar);
    }
    return gridCount(desc, params_.data());
  }

  static constexpr int32_t kVector = 0;
  static constexpr int32_t kMatrix = sizeof(Tensor);
  static constexpr int32_t kScalar = 2 * sizeof(Tensor);

  std::vector<uint8_t> params_;
};

TEST_F(GridCountTest, kinds) {
  EXPECT_EQ(count(Count::kMax, {kVector, kMatrix}), 20);
  EXPECT_EQ(count(Count::kSum, {kVector, kMatrix}), 23);
  EXPECT_EQ(count(Count::kProduct, {kVector, kMatrix}), 60);
  EXPECT_EQ(count(Count::kProduct, {kVector, kScalar}), 21);
  EXPECT_EQ(count(Count::kMax, {kVector, kScalar}), 7);
  EXPECT_EQ(count(Count::kHost, {kVector}), -1);
}

TEST_F(GridCountTest, emptyProduct) {
  setTensor(kVector, {0});
  EXPECT_EQ(count(Count::kProduct, {kVector, kMatrix}), 0);
}

TEST_F(GridCountTest, broadcast) {
  setTensor(kVector, {4, 1});
  setTensor(kMatrix, {1, 5});
  EXPECT_EQ(count(Count::kBroadcast, {kVector, kMatrix}), 20);
}

} // namespace
} // namespace torch::wave
