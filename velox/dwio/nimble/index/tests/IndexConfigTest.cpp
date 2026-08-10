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

#include "velox/dwio/nimble/common/NimbleException.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/index/HashIndexConfig.h"
#include "velox/dwio/nimble/index/IndexConstants.h"
#include "velox/dwio/nimble/index/IndexSerialization.h"
#include "velox/dwio/nimble/index/SortedIndexConfig.h"
#include "velox/dwio/nimble/tablet/IndexGenerated.h"

namespace facebook::nimble::index::test {
namespace {

TEST(IndexConfigTest, buildersSetFactoryIdentity) {
  struct TestCase {
    std::shared_ptr<const IndexConfig> config;
    IndexFamily family;
    std::string_view name;
  };

  const std::vector<TestCase> testCases{
      {ClusterIndexConfigBuilder{}.build(),
       IndexFamily::Cluster,
       kClusterIndexName},
      {HashIndexConfigBuilder{}.build(),
       IndexFamily::Dense,
       kDenseHashIndexName},
      {SortedIndexConfigBuilder{}.build(),
       IndexFamily::Dense,
       kDenseSortedIndexName},
  };

  for (const auto& testCase : testCases) {
    EXPECT_EQ(testCase.config->family, testCase.family);
    EXPECT_EQ(testCase.config->name, testCase.name);
  }
}

TEST(IndexConfigTest, buildersPreserveDefaults) {
  const auto cluster = ClusterIndexConfigBuilder{}.build();
  const auto& clusterOptions = checkedIndexConfig<ClusterIndexConfig>(*cluster);
  EXPECT_TRUE(clusterOptions.columns.empty());
  EXPECT_TRUE(clusterOptions.sortOrders.empty());
  EXPECT_FALSE(clusterOptions.enforceKeyOrder);
  EXPECT_FALSE(clusterOptions.noDuplicateKey);
  EXPECT_EQ(clusterOptions.encodingLayout.encodingType(), EncodingType::Prefix);
  EXPECT_EQ(
      clusterOptions.encodingLayout.compressionType(),
      CompressionType::Uncompressed);
  EXPECT_EQ(clusterOptions.maxRowsPerKeyChunk, 10'000);
  EXPECT_EQ(
      clusterOptions.keyChunkCompressionType, CompressionType::Uncompressed);

  const auto hash = HashIndexConfigBuilder{}.build();
  const auto& hashOptions = checkedIndexConfig<HashIndexConfig>(*hash);
  EXPECT_TRUE(hashOptions.columns.empty());
  EXPECT_FLOAT_EQ(hashOptions.loadFactor, 0.7f);
  EXPECT_FALSE(hashOptions.bloomFilter.has_value());
  EXPECT_EQ(hashOptions.maxPartitionSizeBytes, 0);
  EXPECT_FLOAT_EQ(BloomFilterConfig{}.bitsPerKey, 10.0f);

  const auto sorted = SortedIndexConfigBuilder{}.build();
  const auto& sortedOptions = checkedIndexConfig<SortedIndexConfig>(*sorted);
  EXPECT_TRUE(sortedOptions.columns.empty());
  EXPECT_EQ(sortedOptions.encodingLayout.encodingType(), EncodingType::Prefix);
  EXPECT_EQ(
      sortedOptions.encodingLayout.compressionType(),
      CompressionType::Uncompressed);
  EXPECT_EQ(sortedOptions.maxRowsPerKeyChunk, 0);
}

TEST(IndexConfigTest, buildersPreserveConfiguredValues) {
  const auto cluster =
      ClusterIndexConfigBuilder{}
          .withKeyColumns({"key"})
          .withSortOrders({SortOrder{.ascending = false}})
          .withEnforceKeyOrder(true)
          .withNoDuplicateKey(true)
          .withEncodingLayout(
              EncodingLayout{EncodingType::Trivial, {}, CompressionType::Zstd})
          .withMaxRowsPerKeyChunk(123)
          .withKeyChunkCompressionType(CompressionType::Lz4)
          .build();
  const auto& clusterOptions = checkedIndexConfig<ClusterIndexConfig>(*cluster);
  EXPECT_EQ(clusterOptions.columns, std::vector<std::string>{"key"});
  EXPECT_EQ(
      clusterOptions.sortOrders,
      std::vector<SortOrder>{SortOrder{.ascending = false}});
  EXPECT_TRUE(clusterOptions.enforceKeyOrder);
  EXPECT_TRUE(clusterOptions.noDuplicateKey);
  EXPECT_EQ(
      clusterOptions.encodingLayout.encodingType(), EncodingType::Trivial);
  EXPECT_EQ(
      clusterOptions.encodingLayout.compressionType(), CompressionType::Zstd);
  EXPECT_EQ(clusterOptions.maxRowsPerKeyChunk, 123);
  EXPECT_EQ(clusterOptions.keyChunkCompressionType, CompressionType::Lz4);

  const auto hash = HashIndexConfigBuilder{}
                        .withKeyColumns({"key"})
                        .withLoadFactor(0.5f)
                        .withBloomFilter(7.0f)
                        .withMaxPartitionSizeBytes(456)
                        .build();
  const auto& hashOptions = checkedIndexConfig<HashIndexConfig>(*hash);
  EXPECT_EQ(hashOptions.columns, std::vector<std::string>{"key"});
  EXPECT_FLOAT_EQ(hashOptions.loadFactor, 0.5f);
  ASSERT_TRUE(hashOptions.bloomFilter.has_value());
  EXPECT_FLOAT_EQ(hashOptions.bloomFilter->bitsPerKey, 7.0f);
  EXPECT_EQ(hashOptions.maxPartitionSizeBytes, 456);

  const auto sorted =
      SortedIndexConfigBuilder{}
          .withKeyColumns({"key"})
          .withEncodingLayout(
              EncodingLayout{EncodingType::Trivial, {}, CompressionType::Zstd})
          .withMaxRowsPerKeyChunk(789)
          .build();
  const auto& sortedOptions = checkedIndexConfig<SortedIndexConfig>(*sorted);
  EXPECT_EQ(sortedOptions.columns, std::vector<std::string>{"key"});
  EXPECT_EQ(sortedOptions.encodingLayout.encodingType(), EncodingType::Trivial);
  EXPECT_EQ(
      sortedOptions.encodingLayout.compressionType(), CompressionType::Zstd);
  EXPECT_EQ(sortedOptions.maxRowsPerKeyChunk, 789);
}

TEST(IndexConfigTest, rejectsMismatchedConfigType) {
  const auto config = HashIndexConfigBuilder{}.build();
  EXPECT_THROW(
      checkedIndexConfig<SortedIndexConfig>(*config), velox::VeloxRuntimeError);
}

TEST(IndexConfigTest, indexFamilySerializationRoundTrip) {
  for (const auto [family, serializedFamily] :
       {std::pair{IndexFamily::Cluster, serialization::IndexFamily_Cluster},
        std::pair{IndexFamily::Dense, serialization::IndexFamily_Dense}}) {
    SCOPED_TRACE(static_cast<int>(family));
    EXPECT_EQ(toIndexFamily(family), serializedFamily);
    EXPECT_EQ(toIndexFamily(serializedFamily), family);
  }
}

TEST(IndexConfigTest, indexFamilyFormatting) {
  EXPECT_EQ(fmt::format("{}", IndexFamily::Cluster), "Cluster");
  EXPECT_EQ(fmt::format("{}", IndexFamily::Dense), "Dense");
}

} // namespace
} // namespace facebook::nimble::index::test
