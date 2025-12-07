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

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <optional>

#include "velox/functions/lib/MakeRowFromStruct.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/SelectivityVector.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::functions {
namespace {

class MakeRowFromStructTest : public velox::test::VectorTestBase,
                              public testing::Test {
 public:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void verify(
      RowVectorPtr& input,
      RowVectorPtr& expected,
      const std::vector<MakeRowFromStruct::KeyOptions>& keysToProject) {
    MakeRowFromStruct makeRowFromStruct(keysToProject);
    SelectivityVector rows(input->size());
    auto result = makeRowFromStruct.apply(*input, rows, nullptr);
    test::assertEqualVectors(expected, result);
  }
};

TEST_F(MakeRowFromStructTest, real) {
  // Oringinal values
  auto input = makeRowVector(
      {"a", "b", "c", "d"},
      {
          makeFlatVector<float>({1.1, 0.0, 1.2, 0.0, 0.0}),
          makeFlatVector<float>({0.0, 0.0, 0.0, 0.0, 0.0}),
          makeNullableFlatVector<float>(
              {1.0, 0.0, std::nullopt, std::nullopt, 1.0}),
          makeNullableFlatVector<float>(
              {0.0, 0.0, std::nullopt, std::nullopt, 0.0}),
      });

  auto expected = makeRowVector(
      {"_fd", "_fc"},
      {
          makeNullableFlatVector<float>(
              {0.0, 0.0, std::nullopt, std::nullopt, 0.0}),
          makeNullableFlatVector<float>(
              {1.0, 0.0, std::nullopt, std::nullopt, 1.0}),
      });

  std::vector<MakeRowFromStruct::KeyOptions> keysToProject = {
      {"d", "_fd", false}, {"c", "_fc", false}};

  verify(input, expected, keysToProject);

  // With null padding
  keysToProject = {{"d", "_fd", true}, {"c", "_fc", false}};

  expected = makeRowVector(
      {"_fd", "_fc"},
      {
          makeFlatVector<float>({0.0, 0.0, 0.0, 0.0, 0.0}),
          makeNullableFlatVector<float>(
              {1.0, 0.0, std::nullopt, std::nullopt, 1.0}),
      });

  verify(input, expected, keysToProject);

  // Project non-existing key
  keysToProject = {{"e", "_fe", false}, {"c", "_fc", false}};

  expected = makeRowVector(
      {"_fe", "_fc"},
      {
          makeAllNullFlatVector<float>(5),
          makeNullableFlatVector<float>(
              {1.0, 0.0, std::nullopt, std::nullopt, 1.0}),
      });

  verify(input, expected, keysToProject);

  // Project non-existing key and padding null
  keysToProject = {{"e", "_fe", true}, {"c", "_fc", true}};

  expected = makeRowVector(
      {"_fe", "_fc"},
      {
          makeFlatVector<float>({0.0, 0.0, 0.0, 0.0, 0.0}),
          makeFlatVector<float>({1.0, 0.0, 0.0, 0.0, 1.0}),
      });

  verify(input, expected, keysToProject);

  // Project the same column twice
  keysToProject = {{"c", "c_padded", true}, {"c", "c_original", false}};

  expected = makeRowVector(
      {"c_padded", "c_original"},
      {
          makeFlatVector<float>({1.0, 0.0, 0.0, 0.0, 1.0}),
          makeNullableFlatVector<float>(
              {1.0, 0.0, std::nullopt, std::nullopt, 1.0}),
      });

  verify(input, expected, keysToProject);
}

TEST_F(MakeRowFromStructTest, array) {
  // Oringinal values
  auto input = makeRowVector(
      {"a", "b", "c", "d"},
      {
          makeArrayVector<int64_t>({{4, 5, 6}, {40}}),
          makeNullableArrayVector<int64_t>({{}, {50}}),
          makeArrayVector<int64_t>({{10, 11, 12}, {60, 70}}),
          makeNullableArrayVector<int64_t>({{13, 14, 15}, {}}),
      });

  auto expected = makeRowVector(
      {"_fd", "_fc"},
      {
          makeNullableArrayVector<int64_t>({{13, 14, 15}, {}}),
          makeArrayVector<int64_t>({{10, 11, 12}, {60, 70}}),
      });

  std::vector<MakeRowFromStruct::KeyOptions> keysToProject = {
      {"d", "_fd", false}, {"c", "_fc", false}};

  verify(input, expected, keysToProject);

  // With null padding
  keysToProject = {{"d", "_fd", true}, {"c", "_fc", false}};

  expected = makeRowVector(
      {"_fd", "_fc"},
      {
          makeArrayVector<int64_t>({{13, 14, 15}, {}}),
          makeArrayVector<int64_t>({{10, 11, 12}, {60, 70}}),
      });

  verify(input, expected, keysToProject);

  // Project non-existing key
  keysToProject = {{"e", "_fe", false}, {"c", "_fc", false}};

  expected = makeRowVector(
      {"_fe", "_fc"},
      {
          makeAllNullArrayVector(2, BIGINT()),
          makeArrayVector<int64_t>({{10, 11, 12}, {60, 70}}),
      });

  verify(input, expected, keysToProject);

  // Project non-existing key and padding null
  keysToProject = {{"e", "_fe", true}, {"c", "_fc", false}};

  expected = makeRowVector(
      {"_fe", "_fc"},
      {
          makeArrayVector<int64_t>({{}, {}}),
          makeArrayVector<int64_t>({{10, 11, 12}, {60, 70}}),
      });

  verify(input, expected, keysToProject);
}

TEST_F(MakeRowFromStructTest, map) {
  // Oringinal values
  auto input = makeRowVector(
      {"a", "c", "d"},
      {
          makeMapVectorFromJson<int64_t, int64_t>(
              {"{1:1, 2:2, 3:3}", "{1:100, 2:null, 6:600}", "{}", "null"}),
          makeMapVectorFromJson<int64_t, int64_t>(
              {"{1:1, 2:2, 3:3}", "{1:100, 6:600}", "{}", "{}"}),
          makeMapVectorFromJson<int64_t, int64_t>(
              {"{1:1, 2:2}", "{1:100, 2:null, 6:600}", "{}", "null"}),
      });

  auto expected = makeRowVector(
      {"_fd", "_fc"},
      {
          makeMapVectorFromJson<int64_t, int64_t>(
              {"{1:1, 2:2}", "{1:100, 2:null, 6:600}", "{}", "null"}),
          makeMapVectorFromJson<int64_t, int64_t>(
              {"{1:1, 2:2, 3:3}", "{1:100, 6:600}", "{}", "{}"}),
      });

  std::vector<MakeRowFromStruct::KeyOptions> keysToProject = {
      {"d", "_fd", false}, {"c", "_fc", false}};

  verify(input, expected, keysToProject);

  // With null padding
  keysToProject = {{"d", "_fd", true}, {"c", "_fc", false}};

  expected = makeRowVector(
      {"_fd", "_fc"},
      {
          makeMapVectorFromJson<int64_t, int64_t>(
              {"{1:1, 2:2}", "{1:100, 2:null, 6:600}", "{}", "{}"}),
          makeMapVectorFromJson<int64_t, int64_t>(
              {"{1:1, 2:2, 3:3}", "{1:100, 6:600}", "{}", "{}"}),
      });

  verify(input, expected, keysToProject);

  // Project non-existing key
  keysToProject = {{"e", "_fe", false}, {"c", "_fc", false}};

  expected = makeRowVector(
      {"_fe", "_fc"},
      {
          makeAllNullMapVector(4, BIGINT(), BIGINT()),
          makeMapVectorFromJson<int64_t, int64_t>(
              {"{1:1, 2:2, 3:3}", "{1:100, 6:600}", "{}", "{}"}),
      });

  verify(input, expected, keysToProject);

  // Project non-existing key and padding null
  keysToProject = {{"e", "_fe", true}, {"c", "_fc", false}};

  expected = makeRowVector(
      {"_fe", "_fc"},
      {
          makeMapVectorFromJson<int64_t, int64_t>({"{}", "{}", "{}", "{}"}),
          makeMapVectorFromJson<int64_t, int64_t>(
              {"{1:1, 2:2, 3:3}", "{1:100, 6:600}", "{}", "{}"}),
      });

  verify(input, expected, keysToProject);
}

TEST_F(MakeRowFromStructTest, exception) {
  // Oringinal values
  auto input = makeRowVector(
      {"c", "d"},
      {
          makeFlatVector<float>({1.1, 0.0, 1.2, 0.0, 0.0}),
          makeFlatVector<int64_t>({0, 0, 0, 0, 0}),
      });

  std::vector<MakeRowFromStruct::KeyOptions> keysToProject = {
      {"d", "_fd", false}, {"c", "_fc", false}};

  MakeRowFromStruct makeRowFromStruct(keysToProject);
  SelectivityVector rows(input->size());

  EXPECT_THAT(
      [&]() { makeRowFromStruct.apply(*input, rows, nullptr); },
      Throws<velox::VeloxUserError>(Property(
          &velox::VeloxUserError::what,
          testing::HasSubstr("Child type mismatch"))));

  // Duplicate key name to project
  input = makeRowVector(
      {"c", "d"},
      {
          makeFlatVector<float>({1.1, 0.0, 1.2, 0.0, 0.0}),
          makeFlatVector<float>({0.0, 0.1, 0.45, 0.12, 0.14}),
      });

  keysToProject = {{"d", "_fd", false}, {"c", "_fd", false}};

  EXPECT_THAT(
      [&]() { MakeRowFromStruct makeRowFromStruct(keysToProject); },
      Throws<velox::VeloxUserError>(Property(
          &velox::VeloxUserError::what,
          testing::HasSubstr("Duplicate field names are not allowed: _fd"))));
}

} // namespace
} // namespace facebook::velox::functions
