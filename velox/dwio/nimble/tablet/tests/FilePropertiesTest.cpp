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
#include "velox/dwio/nimble/tablet/FileProperties.h"

#include <gtest/gtest.h>
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/tablet/FilePropertiesGenerated.h"

#include "flatbuffers/flatbuffers.h"

#include <vector>

namespace facebook::nimble {
namespace {

TEST(FilePropertiesTest, basic) {
  FileProperties features{/*compactRowCountEncoding=*/false,
                          /*clusterIndexKeyColumnStorageOmitted=*/false,
                          /*clusterIndexKeyColumnsWithOmittedStorage=*/{}};

  const auto decoded = FileProperties::deserialize(features.serialize());
  EXPECT_FALSE(decoded.compactRowCountEncoding());
  EXPECT_FALSE(decoded.clusterIndexKeyColumnStorageOmitted());
  EXPECT_TRUE(decoded.clusterIndexKeyColumnsWithOmittedStorage().empty());
}

TEST(FilePropertiesTest, roundTripKeyColumnsWithOmittedStorage) {
  FileProperties features{
      /*compactRowCountEncoding=*/false,
      /*clusterIndexKeyColumnStorageOmitted=*/true,
      /*clusterIndexKeyColumnsWithOmittedStorage=*/{"key0", "key1"}};

  const auto decoded = FileProperties::deserialize(features.serialize());
  EXPECT_FALSE(decoded.compactRowCountEncoding());
  EXPECT_TRUE(decoded.clusterIndexKeyColumnStorageOmitted());
  EXPECT_EQ(
      decoded.clusterIndexKeyColumnsWithOmittedStorage(),
      (std::vector<std::string>{"key0", "key1"}));
}

TEST(FilePropertiesTest, roundTripCompactRowCountEncoding) {
  FileProperties enabled{/*compactRowCountEncoding=*/true,
                         /*clusterIndexKeyColumnStorageOmitted=*/false,
                         /*clusterIndexKeyColumnsWithOmittedStorage=*/{}};

  auto decoded = FileProperties::deserialize(enabled.serialize());
  EXPECT_TRUE(decoded.compactRowCountEncoding());

  FileProperties disabled{/*compactRowCountEncoding=*/false,
                          /*clusterIndexKeyColumnStorageOmitted=*/false,
                          /*clusterIndexKeyColumnsWithOmittedStorage=*/{}};

  decoded = FileProperties::deserialize(disabled.serialize());
  EXPECT_FALSE(decoded.compactRowCountEncoding());
}

TEST(FilePropertiesTest, rejectsConstructedOmittedStorageWithoutColumns) {
  NIMBLE_ASSERT_THROW(
      FileProperties(
          /*compactRowCountEncoding=*/false,
          /*clusterIndexKeyColumnStorageOmitted=*/true,
          /*clusterIndexKeyColumnsWithOmittedStorage=*/{}),
      "clusterIndexKeyColumnStorageOmitted must match clusterIndexKeyColumnsWithOmittedStorage presence");

  NIMBLE_ASSERT_THROW(
      FileProperties(
          /*compactRowCountEncoding=*/false,
          /*clusterIndexKeyColumnStorageOmitted=*/false,
          /*clusterIndexKeyColumnsWithOmittedStorage=*/{"key0"}),
      "clusterIndexKeyColumnStorageOmitted must match clusterIndexKeyColumnsWithOmittedStorage presence");
}

TEST(FilePropertiesTest, rejectsOmittedStorageWithoutColumns) {
  auto serialized =
      [](bool clusterIndexKeyColumnStorageOmitted,
         std::vector<std::string> clusterIndexKeyColumnsWithOmittedStorage) {
        flatbuffers::FlatBufferBuilder builder;
        auto columns =
            builder.CreateVector<flatbuffers::Offset<flatbuffers::String>>(
                clusterIndexKeyColumnsWithOmittedStorage.size(),
                [&builder,
                 &clusterIndexKeyColumnsWithOmittedStorage](size_t i) {
                  return builder.CreateString(
                      clusterIndexKeyColumnsWithOmittedStorage[i]);
                });
        builder.Finish(
            serialization::CreateFileProperties(
                builder,
                clusterIndexKeyColumnStorageOmitted,
                columns,
                /*compact_encoding=*/0));

        return std::string{
            reinterpret_cast<const char*>(builder.GetBufferPointer()),
            builder.GetSize()};
      };

  struct Case {
    bool clusterIndexKeyColumnStorageOmitted;
    std::vector<std::string> clusterIndexKeyColumnsWithOmittedStorage;
  };
  const std::vector<Case> cases{{true, {}}, {false, {"key0"}}};
  for (const auto& testCase : cases) {
    SCOPED_TRACE(
        testing::Message() << "clusterIndexKeyColumnStorageOmitted="
                           << testCase.clusterIndexKeyColumnStorageOmitted);
    NIMBLE_ASSERT_THROW(
        FileProperties::deserialize(serialized(
            testCase.clusterIndexKeyColumnStorageOmitted,
            testCase.clusterIndexKeyColumnsWithOmittedStorage)),
        "cluster_index_key_column_storage_omitted must match cluster_index_key_columns_with_omitted_storage presence");
  }
}

} // namespace
} // namespace facebook::nimble
