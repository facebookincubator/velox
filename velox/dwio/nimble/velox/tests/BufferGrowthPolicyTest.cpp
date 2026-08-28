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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/dwio/nimble/velox/BufferGrowthPolicy.h"
#include "velox/dwio/nimble/writer/WriterOptions.h"

namespace facebook::nimble {

struct DefaultInputBufferGrowthPolicyTestCase {
  std::map<uint64_t, float> rangedConfigs;
  uint64_t size;
  uint64_t capacity;
  uint64_t expectedNewCapacity;
};

class DefaultInputBufferGrowthPolicyTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<
          DefaultInputBufferGrowthPolicyTestCase> {};

TEST_P(DefaultInputBufferGrowthPolicyTest, getExtendedCapacity) {
  const auto& testCase = GetParam();
  DefaultInputBufferGrowthPolicy policy{testCase.rangedConfigs};

  ASSERT_EQ(
      policy.getExtendedCapacity(testCase.size, testCase.capacity),
      testCase.expectedNewCapacity);
}

INSTANTIATE_TEST_CASE_P(
    DefaultInputBufferGrowthPolicyMinCapacityTestSuite,
    DefaultInputBufferGrowthPolicyTest,
    ::testing::Values(
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{16, 2.0f}},
            .size = 8,
            .capacity = 0,
            .expectedNewCapacity = 16},
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{16, 2.0f}},
            .size = 16,
            .capacity = 0,
            .expectedNewCapacity = 16},
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{16, 2.0f}},
            .size = 8,
            .capacity = 4,
            .expectedNewCapacity = 16},
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{16, 2.0f}},
            .size = 16,
            .capacity = 10,
            .expectedNewCapacity = 16},
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{4, 4.0f}, {16, 2.0f}, {1024, 1.5f}},
            .size = 2,
            .capacity = 0,
            .expectedNewCapacity = 4}));

INSTANTIATE_TEST_CASE_P(
    DefaultInputBufferGrowthPolicyInRangeGrowthTestSuite,
    DefaultInputBufferGrowthPolicyTest,
    ::testing::Values(
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{16, 2.0f}},
            .size = 8,
            .capacity = 8,
            .expectedNewCapacity = 8},
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{16, 2.0f}},
            .size = 20,
            .capacity = 16,
            .expectedNewCapacity = 32},
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{16, 2.0f}},
            .size = 24,
            .capacity = 20,
            .expectedNewCapacity = 40},
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{16, 2.0f}},
            .size = 60,
            .capacity = 20,
            .expectedNewCapacity = 80},
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{16, 2.0f}, {1024, 1.5f}},
            .size = 24,
            .capacity = 20,
            .expectedNewCapacity = 40},
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{16, 2.0f}, {1024, 1.5f}},
            .size = 1028,
            .capacity = 1024,
            .expectedNewCapacity = 1536},
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{16, 2.0f}, {1024, 1.5f}},
            .size = 1028,
            .capacity = 1000,
            .expectedNewCapacity = 1500},
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{16, 2.0f}, {1024, 1.5f}},
            .size = 512,
            .capacity = 16,
            .expectedNewCapacity = 512}));

// We determine the growth factor only once and grow the
// capacity until it suffices.
INSTANTIATE_TEST_CASE_P(
    DefaultInputBufferGrowthPolicyCrossRangeTestSuite,
    DefaultInputBufferGrowthPolicyTest,
    ::testing::Values(
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{16, 2.0f}},
            .size = 20,
            .capacity = 0,
            .expectedNewCapacity = 32},
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{16, 2.0f}, {1024, 1.5f}},
            .size = 2048,
            .capacity = 1000,
            .expectedNewCapacity = 2250},
        DefaultInputBufferGrowthPolicyTestCase{
            .rangedConfigs = {{16, 2.0f}, {1024, 1.5f}, {2048, 1.2f}},
            .size = 2048,
            .capacity = 1000,
            .expectedNewCapacity = 2073}));

// Regression guard (D111195999): the growth policy stays amortized by default;
// only lowMemoryMode switches to ExactGrowthPolicy. Spilling writers no longer
// force lowMemoryMode, so they keep the amortized (O(N)) policy.
TEST(WriterOptionsGrowthPolicyTest, defaultUsesAmortizedGrowth) {
  WriterOptions options; // lowMemoryMode defaults to false

  // Amortized policy leaves headroom above the requested size.
  EXPECT_GT(
      options.makeInputBufferGrowthPolicy()->getExtendedCapacity(
          /*newSize=*/100, /*capacity=*/0),
      100u);
  EXPECT_GT(
      options.makeStringBufferGrowthPolicy()->getExtendedCapacity(100, 0),
      100u);
}

TEST(WriterOptionsGrowthPolicyTest, lowMemoryModeForcesExactGrowth) {
  WriterOptions options;
  options.lowMemoryMode = true;

  // Grow-to-exact: no headroom.
  EXPECT_EQ(
      options.makeInputBufferGrowthPolicy()->getExtendedCapacity(100, 0), 100u);
  EXPECT_EQ(
      options.makeStringBufferGrowthPolicy()->getExtendedCapacity(100, 0),
      100u);
}
} // namespace facebook::nimble
