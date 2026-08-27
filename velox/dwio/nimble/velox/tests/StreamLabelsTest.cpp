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

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/dwio/nimble/velox/StreamLabels.h"
#include "velox/dwio/nimble/velox/tests/SchemaUtils.h"

using namespace facebook;

namespace {

nimble::StreamLabels buildLabels(nimble::SchemaBuilder& builder) {
  auto nodes = builder.schemaNodes();
  return nimble::StreamLabels{nimble::SchemaReader::getSchema(nodes)};
}

} // namespace

TEST(StreamLabelsTest, scalarFields) {
  nimble::SchemaBuilder builder;
  NIMBLE_SCHEMA(
      builder,
      NIMBLE_ROW({
          {"a", NIMBLE_INTEGER()},
          {"b", NIMBLE_BIGINT()},
      }));

  auto labels = buildLabels(builder);
  auto nodes = builder.schemaNodes();
  auto root = nimble::SchemaReader::getSchema(nodes);

  // Root row nulls → "/"
  EXPECT_EQ(labels.streamLabel(root->asRow().nullsDescriptor().offset()), "/");
  // First scalar child → "/0"
  EXPECT_EQ(
      labels.streamLabel(
          root->asRow().childAt(0)->asScalar().scalarDescriptor().offset()),
      "/0");
  // Second scalar child → "/1"
  EXPECT_EQ(
      labels.streamLabel(
          root->asRow().childAt(1)->asScalar().scalarDescriptor().offset()),
      "/1");
}

TEST(StreamLabelsTest, nestedRows) {
  nimble::SchemaBuilder builder;
  NIMBLE_SCHEMA(
      builder,
      NIMBLE_ROW({
          {"a", NIMBLE_ROW({{"x", NIMBLE_INTEGER()}})},
      }));

  auto labels = buildLabels(builder);
  auto nodes = builder.schemaNodes();
  auto root = nimble::SchemaReader::getSchema(nodes);

  // Root → "/"
  EXPECT_EQ(labels.streamLabel(root->asRow().nullsDescriptor().offset()), "/");
  // Nested row → "/0/"
  const auto& inner = root->asRow().childAt(0)->asRow();
  EXPECT_EQ(labels.streamLabel(inner.nullsDescriptor().offset()), "/0/");
  // Scalar inside nested row → "/0/0"
  EXPECT_EQ(
      labels.streamLabel(
          inner.childAt(0)->asScalar().scalarDescriptor().offset()),
      "/0/0");
}

TEST(StreamLabelsTest, array) {
  nimble::SchemaBuilder builder;
  NIMBLE_SCHEMA(
      builder,
      NIMBLE_ROW({
          {"arr", NIMBLE_ARRAY(NIMBLE_INTEGER())},
      }));

  auto labels = buildLabels(builder);
  auto nodes = builder.schemaNodes();
  auto root = nimble::SchemaReader::getSchema(nodes);

  const auto& arr = root->asRow().childAt(0)->asArray();
  // Array lengths and elements share label "/0"
  EXPECT_EQ(labels.streamLabel(arr.lengthsDescriptor().offset()), "/0");
  EXPECT_EQ(
      labels.streamLabel(
          arr.elements()->asScalar().scalarDescriptor().offset()),
      "/0");
}

TEST(StreamLabelsTest, map) {
  nimble::SchemaBuilder builder;
  NIMBLE_SCHEMA(
      builder,
      NIMBLE_ROW({
          {"m", NIMBLE_MAP(NIMBLE_INTEGER(), NIMBLE_BIGINT())},
      }));

  auto labels = buildLabels(builder);
  auto nodes = builder.schemaNodes();
  auto root = nimble::SchemaReader::getSchema(nodes);

  const auto& m = root->asRow().childAt(0)->asMap();
  // Map lengths, keys, values all share label "/0"
  EXPECT_EQ(labels.streamLabel(m.lengthsDescriptor().offset()), "/0");
  EXPECT_EQ(
      labels.streamLabel(m.keys()->asScalar().scalarDescriptor().offset()),
      "/0");
  EXPECT_EQ(
      labels.streamLabel(m.values()->asScalar().scalarDescriptor().offset()),
      "/0");
}

TEST(StreamLabelsTest, flatMap) {
  nimble::SchemaBuilder builder;
  nimble::test::FlatMapChildAdder adder;
  NIMBLE_SCHEMA(
      builder,
      NIMBLE_ROW({
          {"fm", NIMBLE_FLATMAP(Int32, NIMBLE_INTEGER(), adder)},
      }));
  adder.addChild("key1");
  adder.addChild("key2");

  auto labels = buildLabels(builder);
  auto nodes = builder.schemaNodes();
  auto root = nimble::SchemaReader::getSchema(nodes);

  const auto& fm = root->asRow().childAt(0)->asFlatMap();
  // FlatMap nulls → "/0"
  EXPECT_EQ(labels.streamLabel(fm.nullsDescriptor().offset()), "/0");
  // inMap descriptors and values for "key1" → "/0/key1"
  EXPECT_EQ(labels.streamLabel(fm.inMapDescriptorAt(0).offset()), "/0/key1");
  EXPECT_EQ(
      labels.streamLabel(fm.childAt(0)->asScalar().scalarDescriptor().offset()),
      "/0/key1");
  // inMap descriptors and values for "key2" → "/0/key2"
  EXPECT_EQ(labels.streamLabel(fm.inMapDescriptorAt(1).offset()), "/0/key2");
  EXPECT_EQ(
      labels.streamLabel(fm.childAt(1)->asScalar().scalarDescriptor().offset()),
      "/0/key2");
}

TEST(StreamLabelsTest, timestampMicroNano) {
  nimble::SchemaBuilder builder;
  NIMBLE_SCHEMA(
      builder,
      NIMBLE_ROW({
          {"ts", NIMBLE_TIMESTAMPMICRONANO()},
      }));

  auto labels = buildLabels(builder);
  auto nodes = builder.schemaNodes();
  auto root = nimble::SchemaReader::getSchema(nodes);

  const auto& ts = root->asRow().childAt(0)->asTimestampMicroNano();
  // Both micros and nanos map to same label "/0"
  EXPECT_EQ(labels.streamLabel(ts.microsDescriptor().offset()), "/0");
  EXPECT_EQ(labels.streamLabel(ts.nanosDescriptor().offset()), "/0");
}

TEST(StreamLabelsTest, arrayWithOffsets) {
  nimble::SchemaBuilder builder;
  NIMBLE_SCHEMA(
      builder,
      NIMBLE_ROW({
          {"oa", NIMBLE_OFFSETARRAY(NIMBLE_INTEGER())},
      }));

  auto labels = buildLabels(builder);
  auto nodes = builder.schemaNodes();
  auto root = nimble::SchemaReader::getSchema(nodes);

  const auto& awo = root->asRow().childAt(0)->asArrayWithOffsets();
  // Offsets and lengths map to same label "/0"
  EXPECT_EQ(labels.streamLabel(awo.offsetsDescriptor().offset()), "/0");
  EXPECT_EQ(labels.streamLabel(awo.lengthsDescriptor().offset()), "/0");
  EXPECT_EQ(
      labels.streamLabel(
          awo.elements()->asScalar().scalarDescriptor().offset()),
      "/0");
}

TEST(StreamLabelsTest, slidingWindowMap) {
  nimble::SchemaBuilder builder;
  NIMBLE_SCHEMA(
      builder,
      NIMBLE_ROW({
          {"swm", NIMBLE_SLIDINGWINDOWMAP(NIMBLE_INTEGER(), NIMBLE_BIGINT())},
      }));

  auto labels = buildLabels(builder);
  auto nodes = builder.schemaNodes();
  auto root = nimble::SchemaReader::getSchema(nodes);

  const auto& swm = root->asRow().childAt(0)->asSlidingWindowMap();
  // Offsets and lengths map to same label "/0"
  EXPECT_EQ(labels.streamLabel(swm.offsetsDescriptor().offset()), "/0");
  EXPECT_EQ(labels.streamLabel(swm.lengthsDescriptor().offset()), "/0");
  EXPECT_EQ(
      labels.streamLabel(swm.keys()->asScalar().scalarDescriptor().offset()),
      "/0");
  EXPECT_EQ(
      labels.streamLabel(swm.values()->asScalar().scalarDescriptor().offset()),
      "/0");
}

TEST(StreamLabelsTest, outOfRangeOffsetThrows) {
  nimble::SchemaBuilder builder;
  NIMBLE_SCHEMA(
      builder,
      NIMBLE_ROW({
          {"a", NIMBLE_INTEGER()},
      }));

  auto labels = buildLabels(builder);
  NIMBLE_ASSERT_THROW(labels.streamLabel(9999), "Stream offset out of range");
}
