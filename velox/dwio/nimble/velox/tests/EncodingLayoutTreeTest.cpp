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

#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"

using namespace facebook;

namespace {

std::optional<nimble::EncodingLayout> cloneAsOptional(
    const nimble::EncodingLayout* encodingLayout) {
  if (!encodingLayout) {
    return std::nullopt;
  }

  std::string output;
  output.resize(1024);
  auto size = encodingLayout->serialize(output);
  return {
      nimble::EncodingLayout::create({output.data(), static_cast<size_t>(size)})
          .first};
}

void verifyEncodingLayout(
    const std::optional<nimble::EncodingLayout>& expected,
    const std::optional<nimble::EncodingLayout>& actual) {
  ASSERT_EQ(expected.has_value(), actual.has_value());
  if (!expected.has_value()) {
    return;
  }

  ASSERT_EQ(expected->encodingType(), actual->encodingType());
  ASSERT_EQ(expected->compressionType(), actual->compressionType());
  ASSERT_EQ(expected->childrenCount(), actual->childrenCount());

  for (auto i = 0; i < expected->childrenCount(); ++i) {
    verifyEncodingLayout(expected->child(i), actual->child(i));
  }
}

void verifyEncodingLayoutTree(
    const nimble::EncodingLayoutTree& expected,
    const nimble::EncodingLayoutTree& actual) {
  ASSERT_EQ(expected.schemaKind(), actual.schemaKind());
  ASSERT_EQ(expected.name(), actual.name());
  ASSERT_EQ(expected.childrenCount(), actual.childrenCount());

  for (uint8_t i = 0; i < std::numeric_limits<uint8_t>::max(); ++i) {
    verifyEncodingLayout(
        cloneAsOptional(expected.encodingLayout(i)),
        cloneAsOptional(actual.encodingLayout(i)));
  }

  for (auto i = 0; i < expected.childrenCount(); ++i) {
    verifyEncodingLayoutTree(expected.child(i), actual.child(i));
  }
}

void test(const nimble::EncodingLayoutTree& expected) {
  std::string output;
  output.resize(2048);
  auto size = expected.serialize(output);

  auto actual = nimble::EncodingLayoutTree::create({output.data(), size});

  verifyEncodingLayoutTree(expected, actual);
}

} // namespace

TEST(EncodingLayoutTreeTests, SingleNode) {
  nimble::EncodingLayoutTree expected{
      nimble::Kind::Row,
      {{
          nimble::EncodingLayoutTree::StreamIdentifiers::Row::NullsStream,
          nimble::EncodingLayout{
              nimble::EncodingType::SparseBool,
              {},
              nimble::CompressionType::MetaInternal,
              {
                  nimble::EncodingLayout{
                      nimble::EncodingType::FixedBitWidth,
                      {},
                      nimble::CompressionType::Uncompressed},
              }},
      }},
      "  abc  ",
  };

  test(expected);
}

TEST(EncodingLayoutTreeTests, SingleNodeMultipleStreams) {
  nimble::EncodingLayoutTree expected{
      nimble::Kind::Row,
      {
          {
              2,
              nimble::EncodingLayout{
                  nimble::EncodingType::SparseBool,
                  {},
                  nimble::CompressionType::MetaInternal,
                  {
                      nimble::EncodingLayout{
                          nimble::EncodingType::FixedBitWidth,
                          {},
                          nimble::CompressionType::Uncompressed},
                  }},
          },
          {
              4,
              nimble::EncodingLayout{
                  nimble::EncodingType::Dictionary,
                  {},
                  nimble::CompressionType::Zstd,
                  {
                      nimble::EncodingLayout{
                          nimble::EncodingType::Constant,
                          {},
                          nimble::CompressionType::Uncompressed},
                  }},
          },
      },
      "  abcd  ",
  };

  test(expected);
}

TEST(EncodingLayoutTreeTests, WithChildren) {
  nimble::EncodingLayoutTree expected{
      nimble::Kind::Row,
      {
          {
              1,
              nimble::EncodingLayout{
                  nimble::EncodingType::SparseBool,
                  {},
                  nimble::CompressionType::MetaInternal,
                  {
                      nimble::EncodingLayout{
                          nimble::EncodingType::FixedBitWidth,
                          {},
                          nimble::CompressionType::Uncompressed},
                  }},
          },
      },
      "  abc  ",
      {
          {
              nimble::Kind::Scalar,
              {
                  {
                      0,
                      nimble::EncodingLayout{
                          nimble::EncodingType::Trivial,
                          {},
                          nimble::CompressionType::Zstd,
                          {
                              nimble::EncodingLayout{
                                  nimble::EncodingType::Constant,
                                  {},
                                  nimble::CompressionType::Uncompressed},
                          }},
                  },
              },
              "  abc1  ",
          },
          {
              nimble::Kind::Array,
              {},
              "",
          },
      },
  };

  test(expected);
}

TEST(EncodingLayoutTreeTests, SingleNodeNoEncoding) {
  nimble::EncodingLayoutTree expected{
      nimble::Kind::Row,
      {},
      "  abc  ",
  };

  test(expected);
}

TEST(EncodingLayoutTreeTests, SingleNodeEmptyName) {
  nimble::EncodingLayoutTree expected{
      nimble::Kind::Row,
      {
          {
              9,
              nimble::EncodingLayout{
                  nimble::EncodingType::SparseBool,
                  {},
                  nimble::CompressionType::MetaInternal,
                  {
                      nimble::EncodingLayout{
                          nimble::EncodingType::FixedBitWidth,
                          {},
                          nimble::CompressionType::Uncompressed},
                  }},
          },
      },
      "",
  };

  test(expected);
}
