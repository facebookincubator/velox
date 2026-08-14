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

#include "folly/Random.h"
#include "velox/dwio/nimble/velox/LayoutPlanner.h"
#include "velox/dwio/nimble/velox/tests/SchemaUtils.h"
#include <random>

namespace {
// Fixed so the shuffled order is reproducible across runs.
constexpr uint32_t kShuffleSeed = 20240816;
} // namespace

using namespace facebook;

void addNamedTypes(
    const nimble::TypeBuilder& node,
    std::string prefix,
    std::vector<std::tuple<uint32_t, std::string>>& result) {
  switch (node.kind()) {
    case nimble::Kind::Scalar: {
      result.emplace_back(
          node.asScalar().scalarDescriptor().offset(), prefix + "s");
      break;
    }
    case nimble::Kind::TimestampMicroNano: {
      auto& timestamp = node.asTimestampMicroNano();
      result.emplace_back(
          timestamp.microsDescriptor().offset(), prefix + "t.m");
      result.emplace_back(timestamp.nanosDescriptor().offset(), prefix + "t.n");
      break;
    }
    case nimble::Kind::Row: {
      auto& row = node.asRow();
      result.emplace_back(row.nullsDescriptor().offset(), prefix + "r");
      for (auto i = 0; i < row.childrenCount(); ++i) {
        addNamedTypes(
            row.childAt(i),
            fmt::format("{}r.{}({}).", prefix, row.nameAt(i), i),
            result);
      }
      break;
    }
    case nimble::Kind::Array: {
      auto& array = node.asArray();
      result.emplace_back(array.lengthsDescriptor().offset(), prefix + "a");
      addNamedTypes(
          array.elements(), folly::to<std::string>(prefix, "a."), result);
      break;
    }
    case nimble::Kind::ArrayWithOffsets: {
      auto& arrayWithOffsets = node.asArrayWithOffsets();
      result.emplace_back(
          arrayWithOffsets.offsetsDescriptor().offset(), prefix + "da.o");
      result.emplace_back(
          arrayWithOffsets.lengthsDescriptor().offset(), prefix + "da.l");
      addNamedTypes(
          arrayWithOffsets.elements(),
          folly::to<std::string>(prefix, "da.e:"),
          result);

      break;
    }
    case nimble::Kind::Map: {
      auto& map = node.asMap();
      result.emplace_back(map.lengthsDescriptor().offset(), prefix + "m");
      addNamedTypes(map.keys(), folly::to<std::string>(prefix, "m.k:"), result);
      addNamedTypes(
          map.values(), folly::to<std::string>(prefix, "m.v:"), result);
      break;
    }
    case nimble::Kind::SlidingWindowMap: {
      auto& map = node.asSlidingWindowMap();
      result.emplace_back(map.offsetsDescriptor().offset(), prefix + "sm.o");
      result.emplace_back(map.lengthsDescriptor().offset(), prefix + "sm.l");
      addNamedTypes(
          map.keys(), folly::to<std::string>(prefix, "sm.k:"), result);
      addNamedTypes(
          map.values(), folly::to<std::string>(prefix, "sm.v:"), result);
      break;
    }
    case nimble::Kind::FlatMap: {
      auto& flatmap = node.asFlatMap();
      result.emplace_back(flatmap.nullsDescriptor().offset(), prefix + "f");
      for (auto i = 0; i < flatmap.childrenCount(); ++i) {
        result.emplace_back(
            flatmap.inMapDescriptorAt(i).offset(),
            fmt::format("{}f.{}({}).im", prefix, flatmap.nameAt(i), i));
        addNamedTypes(
            flatmap.childAt(i),
            fmt::format("{}f.{}({}).", prefix, flatmap.nameAt(i), i),
            result);
      }
      break;
    }
  }
}

std::vector<std::tuple<uint32_t, std::string>> getNamedTypes(
    const nimble::TypeBuilder& root) {
  std::vector<std::tuple<uint32_t, std::string>> namedTypes;
  namedTypes.reserve(0); // To silence CLANGTIDY
  addNamedTypes(root, "", namedTypes);
  return namedTypes;
}

void testStreamLayout(
    std::mt19937& rng,
    nimble::DefaultLayoutPlanner& planner,
    std::vector<nimble::Stream>&& streams,
    std::vector<std::string>&& expected) {
  std::shuffle(streams.begin(), streams.end(), std::mt19937{kShuffleSeed});

  ASSERT_EQ(expected.size(), streams.size());
  streams = planner.getLayout(std::move(streams));
  ASSERT_EQ(expected.size(), streams.size());

  for (auto i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(expected[i], streams[i].chunks[0].content[0]) << "i = " << i;
  }

  // Now we test that planner can handle the case where less streams are
  // provided than the actual nodes in the schema.
  std::vector<nimble::Stream> streamSubset;
  std::vector<std::string> expectedSubset;
  streamSubset.reserve(streams.size());
  expectedSubset.reserve(streams.size());
  for (auto i = 0; i < streams.size(); ++i) {
    if (folly::Random::oneIn(2, rng)) {
      streamSubset.push_back(streams[i]);
      for (const auto& e : expected) {
        if (streamSubset.back().chunks[0].content[0] == e) {
          expectedSubset.push_back(e);
          break;
        }
      }
    }
  }

  std::shuffle(streamSubset.begin(), streamSubset.end(), std::mt19937{kShuffleSeed});

  ASSERT_EQ(expectedSubset.size(), streamSubset.size());
  streamSubset = planner.getLayout(std::move(streamSubset));
  ASSERT_EQ(expectedSubset.size(), streamSubset.size());

  for (auto i = 0; i < expectedSubset.size(); ++i) {
    EXPECT_EQ(expectedSubset[i], streamSubset[i].chunks[0].content[0]);
  }
}

TEST(DefaultLayoutPlannerTests, reorderFlatMap) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  nimble::SchemaBuilder builder;

  nimble::test::FlatMapChildAdder fm1;
  nimble::test::FlatMapChildAdder fm2;
  nimble::test::FlatMapChildAdder fm3;

  NIMBLE_SCHEMA(
      builder,
      NIMBLE_ROW({
          {"c1", NIMBLE_TINYINT()},
          {"c2", NIMBLE_FLATMAP(Int8, NIMBLE_TINYINT(), fm1)},
          {"c3", NIMBLE_ARRAY(NIMBLE_TINYINT())},
          {"c4", NIMBLE_FLATMAP(Int8, NIMBLE_TINYINT(), fm2)},
          {"c5", NIMBLE_TINYINT()},
          {"c6", NIMBLE_FLATMAP(Int8, NIMBLE_ARRAY(NIMBLE_TINYINT()), fm3)},
      }));

  fm1.addChild("2");
  fm1.addChild("5");
  fm1.addChild("42");
  fm1.addChild("7");
  fm2.addChild("2");
  fm2.addChild("5");
  fm2.addChild("42");
  fm2.addChild("7");
  fm3.addChild("2");
  fm3.addChild("5");
  fm3.addChild("42");
  fm3.addChild("7");

  auto namedTypes = getNamedTypes(*builder.root());

  nimble::DefaultLayoutPlanner planner{
      [&]() { return builder.root(); },
      {{{1, {3, 42, 9, 2, 21}}, {5, {3, 2, 42, 21}}}}};

  std::vector<nimble::Stream> streams;
  streams = planner.getLayout(std::move(streams));
  ASSERT_EQ(0, streams.size());

  streams.reserve(namedTypes.size());
  for (auto i = 0; i < namedTypes.size(); ++i) {
    streams.push_back(
        nimble::Stream{
            std::get<0>(namedTypes[i]),
            {{.content = {std::get<1>(namedTypes[i])}}}});
  }

  std::vector<std::string> expected{
      // Row should always be first
      "r",
      // Followed by feature order
      "r.c2(1).f",
      "r.c2(1).f.42(2).im",
      "r.c2(1).f.42(2).s",
      "r.c2(1).f.2(0).im",
      "r.c2(1).f.2(0).s",
      "r.c6(5).f",
      "r.c6(5).f.2(0).im",
      "r.c6(5).f.2(0).a",
      "r.c6(5).f.2(0).a.s",
      "r.c6(5).f.42(2).im",
      "r.c6(5).f.42(2).a",
      "r.c6(5).f.42(2).a.s",
      // From here, streams follow schema order
      "r.c1(0).s",
      "r.c2(1).f.5(1).im",
      "r.c2(1).f.5(1).s",
      "r.c2(1).f.7(3).im",
      "r.c2(1).f.7(3).s",
      "r.c3(2).a",
      "r.c3(2).a.s",
      "r.c4(3).f",
      "r.c4(3).f.2(0).im",
      "r.c4(3).f.2(0).s",
      "r.c4(3).f.5(1).im",
      "r.c4(3).f.5(1).s",
      "r.c4(3).f.42(2).im",
      "r.c4(3).f.42(2).s",
      "r.c4(3).f.7(3).im",
      "r.c4(3).f.7(3).s",
      "r.c5(4).s",
      "r.c6(5).f.5(1).im",
      "r.c6(5).f.5(1).a",
      "r.c6(5).f.5(1).a.s",
      "r.c6(5).f.7(3).im",
      "r.c6(5).f.7(3).a",
      "r.c6(5).f.7(3).a.s",
  };

  testStreamLayout(rng, planner, std::move(streams), std::move(expected));
}

TEST(DefaultLayoutPlannerTests, reorderFlatMapDynamicFeatures) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  nimble::SchemaBuilder builder;

  nimble::test::FlatMapChildAdder fm;

  NIMBLE_SCHEMA(
      builder,
      NIMBLE_ROW({
          {"c1", NIMBLE_TINYINT()},
          {"c2", NIMBLE_FLATMAP(Int8, NIMBLE_TINYINT(), fm)},
          {"c3", NIMBLE_ARRAY(NIMBLE_TINYINT())},
      }));

  fm.addChild("2");
  fm.addChild("5");
  fm.addChild("42");
  fm.addChild("7");

  auto namedTypes = getNamedTypes(*builder.root());

  nimble::DefaultLayoutPlanner planner{
      [&]() { return builder.root(); }, {{{1, {3, 42, 9, 2, 21}}}}};

  std::vector<nimble::Stream> streams;
  streams.reserve(namedTypes.size());
  for (auto i = 0; i < namedTypes.size(); ++i) {
    streams.push_back(
        nimble::Stream{
            std::get<0>(namedTypes[i]),
            {{.content = {std::get<1>(namedTypes[i])}}}});
  }

  std::vector<std::string> expected{
      // Row should always be first
      "r",
      // Followed by feature order
      "r.c2(1).f",
      "r.c2(1).f.42(2).im",
      "r.c2(1).f.42(2).s",
      "r.c2(1).f.2(0).im",
      "r.c2(1).f.2(0).s",
      // From here, streams follow schema order
      "r.c1(0).s",
      "r.c2(1).f.5(1).im",
      "r.c2(1).f.5(1).s",
      "r.c2(1).f.7(3).im",
      "r.c2(1).f.7(3).s",
      "r.c3(2).a",
      "r.c3(2).a.s",
  };

  testStreamLayout(rng, planner, std::move(streams), std::move(expected));

  fm.addChild("21");
  fm.addChild("3");
  fm.addChild("57");

  namedTypes = getNamedTypes(*builder.root());

  streams.clear();
  streams.reserve(namedTypes.size());
  for (auto i = 0; i < namedTypes.size(); ++i) {
    streams.push_back(
        nimble::Stream{
            std::get<0>(namedTypes[i]),
            {{.content = {std::get<1>(namedTypes[i])}}}});
  }

  expected = {
      // Row should always be first
      "r",
      // Followed by feature order
      "r.c2(1).f",
      "r.c2(1).f.3(5).im",
      "r.c2(1).f.3(5).s",
      "r.c2(1).f.42(2).im",
      "r.c2(1).f.42(2).s",
      "r.c2(1).f.2(0).im",
      "r.c2(1).f.2(0).s",
      "r.c2(1).f.21(4).im",
      "r.c2(1).f.21(4).s",
      // From here, streams follow schema order
      "r.c1(0).s",
      "r.c2(1).f.5(1).im",
      "r.c2(1).f.5(1).s",
      "r.c2(1).f.7(3).im",
      "r.c2(1).f.7(3).s",
      "r.c2(1).f.57(6).im",
      "r.c2(1).f.57(6).s",
      "r.c3(2).a",
      "r.c3(2).a.s",
  };

  testStreamLayout(rng, planner, std::move(streams), std::move(expected));
}

TEST(DefaultLayoutPlannerTests, noFeatureReordering) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  nimble::SchemaBuilder builder;

  nimble::test::FlatMapChildAdder fm;

  NIMBLE_SCHEMA(
      builder,
      NIMBLE_ROW({
          {"c1", NIMBLE_TINYINT()},
          {"c2", NIMBLE_FLATMAP(Int8, NIMBLE_TINYINT(), fm)},
          {"c3", NIMBLE_ARRAY(NIMBLE_TINYINT())},
      }));

  fm.addChild("2");
  fm.addChild("5");
  fm.addChild("7");

  auto namedTypes = getNamedTypes(*builder.root());

  nimble::DefaultLayoutPlanner planner{
      [&]() { return builder.root(); }, std::nullopt};

  std::vector<nimble::Stream> streams;
  streams.reserve(namedTypes.size());
  for (auto i = 0; i < namedTypes.size(); ++i) {
    streams.push_back(
        nimble::Stream{
            std::get<0>(namedTypes[i]),
            {{.content = {std::get<1>(namedTypes[i])}}}});
  }

  std::vector<std::string> expected{
      // Row should always be first
      "r",
      // Streams in the schema order
      "r.c1(0).s",
      "r.c2(1).f",
      "r.c2(1).f.2(0).im",
      "r.c2(1).f.2(0).s",
      "r.c2(1).f.5(1).im",
      "r.c2(1).f.5(1).s",
      "r.c2(1).f.7(2).im",
      "r.c2(1).f.7(2).s",
      "r.c3(2).a",
      "r.c3(2).a.s",
  };

  testStreamLayout(rng, planner, std::move(streams), std::move(expected));
}

TEST(DefaultLayoutPlannerTests, nonFlatMapOrdinalsAreIgnored) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  nimble::SchemaBuilder builder;

  nimble::test::FlatMapChildAdder fm1;
  nimble::test::FlatMapChildAdder fm2;

  NIMBLE_SCHEMA(
      builder,
      NIMBLE_ROW(
          {{"c0", NIMBLE_TINYINT()},
           {"c1", NIMBLE_FLATMAP(Int8, NIMBLE_TINYINT(), fm1)},
           {"c2", NIMBLE_ARRAY(NIMBLE_TINYINT())},
           {"c3", NIMBLE_FLATMAP(Int8, NIMBLE_TINYINT(), fm2)}}));

  fm1.addChild("2");
  fm1.addChild("5");
  fm1.addChild("7");
  fm1.addChild("42");

  fm2.addChild("2");
  fm2.addChild("5");

  nimble::DefaultLayoutPlanner planner{
      [&]() { return builder.root(); }, {{{0, {1, 2, 3}}, {1, {42, 5}}}}};

  auto namedTypes = getNamedTypes(*builder.root());
  std::vector<nimble::Stream> streams;
  streams.reserve(namedTypes.size());
  for (auto i = 0; i < namedTypes.size(); ++i) {
    streams.push_back(
        nimble::Stream{
            std::get<0>(namedTypes[i]),
            {{.content = {std::get<1>(namedTypes[i])}}}});
  }

  std::vector<std::string> expected{
      // Start with row
      "r",
      // Ordered streams
      "r.c1(1).f",
      "r.c1(1).f.42(3).im",
      "r.c1(1).f.42(3).s",
      "r.c1(1).f.5(1).im",
      "r.c1(1).f.5(1).s",
      // Non-ordered streams following the schema order
      "r.c0(0).s",
      "r.c1(1).f.2(0).im",
      "r.c1(1).f.2(0).s",
      "r.c1(1).f.7(2).im",
      "r.c1(1).f.7(2).s",
      "r.c2(2).a",
      "r.c2(2).a.s",
      "r.c3(3).f",
      "r.c3(3).f.2(0).im",
      "r.c3(3).f.2(0).s",
      "r.c3(3).f.5(1).im",
      "r.c3(3).f.5(1).s"};

  testStreamLayout(rng, planner, std::move(streams), std::move(expected));
}

TEST(DefaultLayoutPlannerTests, ordinalOutOfRangeAreIgnored) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  nimble::SchemaBuilder builder;

  nimble::test::FlatMapChildAdder fm1;
  nimble::test::FlatMapChildAdder fm2;
  nimble::test::FlatMapChildAdder fm3;

  NIMBLE_SCHEMA(
      builder,
      NIMBLE_ROW({
          {"c0", NIMBLE_TINYINT()},
          {"c1", NIMBLE_FLATMAP(Int8, NIMBLE_TINYINT(), fm1)},
          {"c2", NIMBLE_ARRAY(NIMBLE_TINYINT())},
          {"c3", NIMBLE_FLATMAP(Int8, NIMBLE_TINYINT(), fm2)},
          {"c4", NIMBLE_TINYINT()},
          {"c5", NIMBLE_FLATMAP(Int8, NIMBLE_ARRAY(NIMBLE_TINYINT()), fm3)},
      }));

  fm1.addChild("2");
  fm1.addChild("5");
  fm1.addChild("42");
  fm1.addChild("7");

  fm2.addChild("2");
  fm2.addChild("5");

  nimble::DefaultLayoutPlanner planner{
      [&]() { return builder.root(); },
      {{{6, {3, 42, 9, 2, 21}}, {3, {3, 2, 42, 21}}}}};

  auto namedTypes = getNamedTypes(*builder.root());
  std::vector<nimble::Stream> streams;
  streams.reserve(namedTypes.size());
  for (auto i = 0; i < namedTypes.size(); ++i) {
    streams.push_back(
        nimble::Stream{
            std::get<0>(namedTypes[i]),
            {{.content = {std::get<1>(namedTypes[i])}}}});
  }

  std::vector<std::string> expected{
      // Start with row
      "r",
      // Ordered streams
      "r.c3(3).f",
      "r.c3(3).f.2(0).im",
      "r.c3(3).f.2(0).s",
      // Non-ordered streams following the schema order
      "r.c0(0).s",
      "r.c1(1).f",
      "r.c1(1).f.2(0).im",
      "r.c1(1).f.2(0).s",
      "r.c1(1).f.5(1).im",
      "r.c1(1).f.5(1).s",
      "r.c1(1).f.42(2).im",
      "r.c1(1).f.42(2).s",
      "r.c1(1).f.7(3).im",
      "r.c1(1).f.7(3).s",
      "r.c2(2).a",
      "r.c2(2).a.s",
      "r.c3(3).f.5(1).im",
      "r.c3(3).f.5(1).s",
      "r.c4(4).s",
      "r.c5(5).f"};

  testStreamLayout(rng, planner, std::move(streams), std::move(expected));
}

TEST(DefaultLayoutPlannerTests, incompatibleSchema) {
  nimble::SchemaBuilder builder;

  NIMBLE_SCHEMA(builder, NIMBLE_MAP(NIMBLE_TINYINT(), NIMBLE_STRING()));

  nimble::DefaultLayoutPlanner planner{
      [&]() { return builder.root(); }, {{{3, {3, 2, 42, 21}}}}};
  try {
    planner.getLayout({});
    FAIL() << "Factory should have failed.";
  } catch (const nimble::NimbleInternalError& e) {
    EXPECT_THAT(
        e.what(),
        ::testing::HasSubstr("Layout planner requires row as the schema root"));
  }
}

TEST(DefaultLayoutPlannerTests, timestampMicros) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  nimble::SchemaBuilder builder;

  NIMBLE_SCHEMA(
      builder,
      NIMBLE_ROW(
          {{"c1", NIMBLE_TINYINT()},
           {"c2", NIMBLE_TIMESTAMPMICRONANO()},
           {"c3", NIMBLE_ARRAY(NIMBLE_TIMESTAMPMICRONANO())},
           {"c4",
            NIMBLE_MAP(
                NIMBLE_TIMESTAMPMICRONANO(), NIMBLE_TIMESTAMPMICRONANO())},
           {"c5", NIMBLE_ROW({{"c5a", NIMBLE_TIMESTAMPMICRONANO()}})}}));

  auto namedTypes = getNamedTypes(*builder.root());

  nimble::DefaultLayoutPlanner planner{
      [&]() { return builder.root(); }, std::nullopt};

  std::vector<nimble::Stream> streams;
  streams.reserve(namedTypes.size());
  for (auto i = 0; i < namedTypes.size(); ++i) {
    streams.push_back(
        nimble::Stream{
            std::get<0>(namedTypes[i]),
            {{.content = {std::get<1>(namedTypes[i])}}}});
  }

  std::vector<std::string> expected{
      // Row should always be first
      "r",
      // Streams in schema order
      "r.c1(0).s",
      "r.c2(1).t.m",
      "r.c2(1).t.n",
      "r.c3(2).a",
      "r.c3(2).a.t.m",
      "r.c3(2).a.t.n",
      "r.c4(3).m",
      "r.c4(3).m.k:t.m",
      "r.c4(3).m.k:t.n",
      "r.c4(3).m.v:t.m",
      "r.c4(3).m.v:t.n",
      "r.c5(4).r",
      "r.c5(4).r.c5a(0).t.m",
      "r.c5(4).r.c5a(0).t.n",
  };

  testStreamLayout(rng, planner, std::move(streams), std::move(expected));
}
