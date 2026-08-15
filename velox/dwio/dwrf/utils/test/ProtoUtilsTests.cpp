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
#include "velox/dwio/dwrf/utils/ProtoUtils.h"
#include "velox/type/fbhive/HiveTypeParser.h"
#include "velox/type/fbhive/HiveTypeSerializer.h"

using namespace facebook::velox::dwrf;
using namespace facebook::velox::type::fbhive;

TEST(ProtoUtilsTests, AllTypes) {
  std::vector<std::string> types{
      "struct<a:boolean,b:tinyint,c:smallint,d:int,e:bigint,f:float,g:double,f:string,g:binary,h:timestamp>",
      "struct<a:map<int,array<struct<a:map<string,int>,b:array<int>>>>>"};

  for (auto& type : types) {
    HiveTypeParser parser;
    auto schema = parser.parse(type);
    proto::Footer footer;
    auto footerWrapper = FooterWriteWrapper(&footer);
    ProtoUtils::writeType(*schema, footerWrapper);

    auto out = ProtoUtils::fromFooter(footer);
    auto str = HiveTypeSerializer::serialize(out);

    EXPECT_EQ(str, type);
  }
}

TEST(ProtoUtilsTests, Projection) {
  HiveTypeParser parser;
  auto schema = parser.parse(
      "struct<a:boolean,b:tinyint,c:smallint,d:struct<a:int,b:int,c:int>>");
  proto::Footer footer;
  auto footerWrapper = FooterWriteWrapper(&footer);
  ProtoUtils::writeType(*schema, footerWrapper);

  auto type = ProtoUtils::fromFooter(
      footer, [](auto id) { return id != 2 && id != 5; });
  auto res = HiveTypeSerializer::serialize(type);

  EXPECT_EQ("struct<a:boolean,c:smallint,d:struct<b:int,c:int>>", res);
}

TEST(ProtoUtilsTests, AttributesRoundTrip) {
  // iceberg.id stamped on a subset of nodes must survive footer serialization
  // and come back keyed by the same pre-order node id, leaving the schema
  // intact. Node ids: 0=root, 1=a, 2=b, 3=c, 4=c.x, 5=c.y.
  HiveTypeParser parser;
  auto schema = parser.parse("struct<a:int,b:bigint,c:struct<x:int,y:int>>");
  proto::Footer footer;
  auto footerWrapper = FooterWriteWrapper(&footer);

  const std::unordered_map<uint32_t, std::string> idByNode{
      {1, "10"}, {2, "20"}, {4, "40"}};
  ProtoUtils::writeType(
      *schema, footerWrapper, /*parent=*/nullptr, [&](uint32_t typeId) {
        std::vector<std::pair<std::string, std::string>> attributes;
        auto it = idByNode.find(typeId);
        if (it != idByNode.end()) {
          attributes.emplace_back("iceberg.id", it->second);
        }
        return attributes;
      });

  std::string serialized;
  ASSERT_TRUE(footer.SerializeToString(&serialized));
  proto::Footer parsed;
  ASSERT_TRUE(parsed.ParseFromString(serialized));

  const std::
      unordered_map<uint32_t, std::vector<std::pair<std::string, std::string>>>
          expected{
              {1, {{"iceberg.id", "10"}}},
              {2, {{"iceberg.id", "20"}}},
              {4, {{"iceberg.id", "40"}}}};
  EXPECT_EQ(ProtoUtils::readAttributes(FooterWrapper(&parsed)), expected);
  EXPECT_EQ(
      HiveTypeSerializer::serialize(ProtoUtils::fromFooter(parsed)),
      "struct<a:int,b:bigint,c:struct<x:int,y:int>>");
}

TEST(ProtoUtilsTests, AttributesRoundTripOrc) {
  // The same iceberg.id round-trip must work for ORC footers: DWRF/ORC Iceberg
  // reads resolve columns by field id from these attributes, and Iceberg
  // manifest-tags DWRF files as ORC. Node ids: 0=root, 1=a, 2=b, 3=c, 4=c.x,
  // 5=c.y.
  HiveTypeParser parser;
  auto schema = parser.parse("struct<a:int,b:bigint,c:struct<x:int,y:int>>");
  proto::orc::Footer footer;
  auto footerWrapper = FooterWriteWrapper(&footer);

  const std::unordered_map<uint32_t, std::string> idByNode{
      {1, "10"}, {2, "20"}, {4, "40"}};
  ProtoUtils::writeType(
      *schema, footerWrapper, /*parent=*/nullptr, [&](uint32_t typeId) {
        std::vector<std::pair<std::string, std::string>> attributes;
        auto it = idByNode.find(typeId);
        if (it != idByNode.end()) {
          attributes.emplace_back("iceberg.id", it->second);
        }
        return attributes;
      });

  std::string serialized;
  ASSERT_TRUE(footer.SerializeToString(&serialized));
  proto::orc::Footer parsed;
  ASSERT_TRUE(parsed.ParseFromString(serialized));

  const std::
      unordered_map<uint32_t, std::vector<std::pair<std::string, std::string>>>
          expected{
              {1, {{"iceberg.id", "10"}}},
              {2, {{"iceberg.id", "20"}}},
              {4, {{"iceberg.id", "40"}}}};
  EXPECT_EQ(ProtoUtils::readAttributes(FooterWrapper(&parsed)), expected);
}

TEST(ProtoUtilsTests, AttributesAbsentByDefault) {
  // A type written without an attribute provider -- the existing path for every
  // DWRF file today -- yields an empty attribute map.
  HiveTypeParser parser;
  auto schema = parser.parse("struct<a:int,b:bigint>");
  proto::Footer footer;
  auto footerWrapper = FooterWriteWrapper(&footer);
  ProtoUtils::writeType(*schema, footerWrapper);

  EXPECT_TRUE(ProtoUtils::readAttributes(FooterWrapper(&footer)).empty());
}
