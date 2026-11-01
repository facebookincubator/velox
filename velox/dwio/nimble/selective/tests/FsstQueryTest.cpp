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

#include <folly/Conv.h>
#include <fstream>
#include <limits>
#include <optional>
#include <string_view>

#include "velox/connectors/hive/FileConfig.h"
#include "velox/core/QueryConfig.h"
#include "velox/dwio/nimble/common/ChunkedStream.h"
#include "velox/dwio/nimble/common/SchemaSerialization.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::nimble::test {
namespace {

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

// Selects the physical string encoding independently of the query plan.
enum class StringLayout {
  kTrivial,
  kDictionaryTrivial,
  kFsst,
  kDictionaryFsst,
};

// Selects the operators that consume the file scan or original input rows.
enum class QueryAction {
  kProject,
  kStringFunctions,
  kCount,
  kGroup,
  kOrderAscending,
  kOrderDescending,
  kInnerJoin,
  kLeftJoin,
};

// Describes SQL expressions shared by the file scan and Values reference.
struct QueryCase {
  // Identifies a failing query in parameterized test output.
  std::string name;
  // Applies simple predicates in the selective reader on the file arm.
  std::vector<std::string> pushdownFilters;
  // Applies compound expressions and string functions after reader filtering.
  std::string remainingFilter;
  // Chooses the downstream operators for both arms.
  QueryAction action{QueryAction::kProject};
};

// Compares actual Nimble file scans with queries over the original rows.
class FsstQueryTest : public HiveConnectorTestBase,
                      public ::testing::WithParamInterface<StringLayout> {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    registerSelectiveNimbleReaderFactory();
  }

  void TearDown() override {
    waitForAllTasksToBeDeleted();
    unregisterSelectiveNimbleReaderFactory();
    HiveConnectorTestBase::TearDown();
  }

  bool dictionary() const {
    return GetParam() == StringLayout::kDictionaryTrivial ||
        GetParam() == StringLayout::kDictionaryFsst;
  }

  EncodingType stringEncoding() const {
    return GetParam() == StringLayout::kTrivial ||
            GetParam() == StringLayout::kDictionaryTrivial
        ? EncodingType::Trivial
        : EncodingType::Fsst;
  }

  // Keeps the same nullable and repeated rows across all four physical layouts.
  // Uses valid UTF-8 for SQL string functions and separate arbitrary byte data.
  std::vector<RowVectorPtr> makeInput(bool arbitraryBytes) {
    const std::vector<std::optional<std::string>> values{
        "alpha",
        "beta",
        "gamma",
        "delta",
        "",
        std::nullopt,
        "alphabeta",
        "alpha repeated payload alpha repeated payload",
        "éclair",
        "猫",
        "a'b",
        std::string(4'096, 'a'),
        "beta repeated payload beta repeated payload",
        "Alpha",
        std::string("a\0b", 3),
        "alph",
        "alphabet",
    };
    std::vector<RowVectorPtr> batches;
    for (int32_t chunk = 0; chunk < 3; ++chunk) {
      std::vector<int64_t> ids;
      std::vector<std::optional<std::string>> strings;
      std::vector<int64_t> gates;
      for (int32_t row = 0; row < 96; ++row) {
        const auto id = chunk * 96 + row;
        auto value = values[id % values.size()];
        if (value && id % 13 == 0) {
          value->append(std::to_string(id));
        }
        if (arbitraryBytes && value && id % 3 == 0) {
          value->append("\0\xff\x80", 3);
        }
        ids.push_back(id);
        strings.push_back(std::move(value));
        gates.push_back(id % 5 == 0 ? 1 : 0);
      }
      batches.push_back(makeRowVector(
          {"id", "value", "gate"},
          {makeFlatVector<int64_t>(ids),
           makeNullableFlatVector(strings),
           makeFlatVector<int64_t>(gates)}));
    }
    return batches;
  }

  // Forces the selected encoding for every chunk instead of relying on writer
  // cost estimates, then writes bytes to a file consumed by the Hive connector.
  std::shared_ptr<TempFilePath> writeInput(
      const std::vector<RowVectorPtr>& batches) {
    const EncodingLayout trivial{
        EncodingType::Trivial, {}, CompressionType::Uncompressed};
    EncodingLayout strings{
        stringEncoding(), {}, CompressionType::Uncompressed, {trivial}};
    if (dictionary()) {
      strings = EncodingLayout{
          EncodingType::Dictionary,
          {},
          CompressionType::Uncompressed,
          {strings, trivial}};
    }
    WriterOptions options;
    options.fsstCompressionTargetRatio = std::numeric_limits<double>::max();
    options.encodingLayoutTree.emplace(
        Kind::Row,
        std::unordered_map<
            EncodingLayoutTree::StreamIdentifier,
            EncodingLayout>{},
        "",
        std::vector<EncodingLayoutTree>{
            {Kind::Scalar, {}, "id"},
            {Kind::Scalar,
             {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
               strings}},
             "value"},
            {Kind::Scalar, {}, "gate"},
        });
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          [](const StripeProgress&) { return false; },
          [](const StripeProgress&) { return true; });
    };
    const std::vector<VectorPtr> vectors(batches.begin(), batches.end());
    const auto data = createNimbleFile(*rootPool_, vectors, options, false);
    verifyLayout(data);
    auto file = TempFilePath::create();
    std::ofstream output(file->getPath(), std::ios::binary);
    output.write(data.data(), folly::to<std::streamsize>(data.size()));
    output.close();
    VELOX_CHECK(output.good(), "Failed to write Nimble test file.");
    return file;
  }

  // Rejects silent fallback in both FSST and baseline files by inspecting every
  // string chunk, including the values child of dictionary encodings.
  void verifyLayout(std::string_view data) {
    auto tablet = TabletReader::create(
        std::make_shared<InMemoryReadFile>(data),
        pool_.get(),
        makeTestTabletOptions(pool_.get()));
    auto section = tablet->loadOptionalSection(std::string(kSchemaSection));
    ASSERT_TRUE(section.has_value());
    const auto schema =
        SchemaDeserializer::deserialize(section->content().data());
    const auto offset =
        schema->asRow().childAt(1)->asScalar().scalarDescriptor().offset();
    uint32_t numChunks{0};
    for (uint32_t stripe = 0; stripe < tablet->stripeCount(); ++stripe) {
      auto streams =
          tablet->load(tablet->stripeIdentifier(stripe), std::span{&offset, 1});
      InMemoryChunkedStream stream{*pool_, std::move(streams[0])};
      while (stream.hasNext()) {
        const auto capture = EncodingLayoutCapture::capture(
            stream.nextChunk(), Encoding::Options{});
        const auto* values = &capture;
        if (values->encodingType() == EncodingType::Nullable) {
          ASSERT_TRUE(values->child(0).has_value());
          values = &values->child(0).value();
        }
        if (dictionary()) {
          ASSERT_EQ(values->encodingType(), EncodingType::Dictionary);
          ASSERT_TRUE(values->child(0).has_value());
          values = &values->child(0).value();
        }
        ASSERT_EQ(values->encodingType(), stringEncoding());
        ++numChunks;
      }
    }
    ASSERT_EQ(numChunks, 3);
  }

  // Builds the same consuming operators for the original-row and file arms.
  core::PlanNodePtr finishPlan(
      PlanBuilder& builder,
      QueryAction action,
      const std::shared_ptr<core::PlanNodeIdGenerator>& ids) {
    switch (action) {
      case QueryAction::kProject:
        return builder.project({"id", "value"}).planNode();
      case QueryAction::kStringFunctions:
        return builder
            .project({
                "id",
                "substr(value, 1, 4) AS prefix",
                "length(value) AS num_chars",
                "lower(value) AS folded",
            })
            .planNode();
      case QueryAction::kCount:
        return builder.singleAggregation({}, {"count(1)", "count(value)"})
            .planNode();
      case QueryAction::kGroup:
        return builder.singleAggregation({"value"}, {"count(1)"}).planNode();
      case QueryAction::kOrderAscending:
        return builder.project({"id", "value"})
            .orderBy({"value ASC NULLS FIRST", "id DESC"}, false)
            .planNode();
      case QueryAction::kOrderDescending:
        return builder.project({"id", "value"})
            .orderBy({"value DESC NULLS LAST", "id ASC"}, false)
            .planNode();
      case QueryAction::kInnerJoin:
      case QueryAction::kLeftJoin: {
        auto right = makeRowVector(
            {"key", "label"},
            {makeNullableFlatVector<std::string>(
                 {"alpha", "alpha", "beta", "", std::nullopt, "猫"}),
             makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60})});
        return builder
            .hashJoin(
                {"value"},
                {"key"},
                PlanBuilder(ids, pool_.get()).values({right}).planNode(),
                "",
                {"id", "value", "label"},
                action == QueryAction::kInnerJoin ? core::JoinType::kInner
                                                  : core::JoinType::kLeft)
            .planNode();
      }
    }
    VELOX_UNREACHABLE();
  }

  // Compares all rows with duplicate multiplicity. Ordered queries additionally
  // compare each result position against the original-row reference.
  void checkQueries(
      const std::vector<RowVectorPtr>& batches,
      const std::vector<QueryCase>& queries) {
    auto file = writeInput(batches);
    ASSERT_FALSE(HasFatalFailure());
    const auto type = asRowType(batches[0]->type());
    for (const auto& query : queries) {
      SCOPED_TRACE(query.name);
      auto referenceIds = std::make_shared<core::PlanNodeIdGenerator>();
      PlanBuilder referenceBuilder(referenceIds, pool_.get());
      referenceBuilder.values(batches);
      for (const auto& filter : query.pushdownFilters) {
        referenceBuilder.filter(filter);
      }
      if (!query.remainingFilter.empty()) {
        referenceBuilder.filter(query.remainingFilter);
      }
      const auto reference =
          AssertQueryBuilder(
              finishPlan(referenceBuilder, query.action, referenceIds))
              .copyResults(pool_.get());
      for (const bool fastPath : {false, true}) {
        for (const bool preserve : {false, true}) {
          SCOPED_TRACE(
              fmt::format("fastPath={}, preserve={}", fastPath, preserve));
          auto ids = std::make_shared<core::PlanNodeIdGenerator>();
          PlanBuilder builder(ids, pool_.get());
          builder.tableScan(type, query.pushdownFilters, query.remainingFilter);
          core::PlanNodeId scanId;
          builder.capturePlanNodeId(scanId);
          auto plan = finishPlan(builder, query.action, ids);
          auto split = HiveConnectorSplitBuilder(file->getPath())
                           .fileFormat(dwio::common::FileFormat::NIMBLE)
                           .build();
          const auto actual =
              AssertQueryBuilder(plan)
                  .split(scanId, split)
                  .config(
                      core::QueryConfig::kPreferredOutputBatchRows,
                      fastPath ? 113 : 7)
                  .connectorSessionProperty(
                      kHiveConnectorId,
                      connector::hive::FileConfig::kNimbleScanFastPathSession,
                      fastPath ? "true" : "false")
                  .connectorSessionProperty(
                      kHiveConnectorId,
                      connector::hive::FileConfig::
                          kNimblePreserveDictionaryEncodingSession,
                      preserve ? "true" : "false")
                  .copyResults(pool_.get());
          ASSERT_TRUE(assertEqualResults({reference}, {actual}));
          if (query.action == QueryAction::kOrderAscending ||
              query.action == QueryAction::kOrderDescending) {
            ASSERT_EQ(reference->size(), actual->size());
            for (vector_size_t row = 0; row < reference->size(); ++row) {
              ASSERT_TRUE(reference->equalValueAt(actual.get(), row, row));
            }
          }
        }
      }
    }
  }
};

TEST_P(FsstQueryTest, queryOperators) {
  checkQueries(
      makeInput(false),
      {
          {"projection", {}, ""},
          {"equality", {"value = 'alpha'"}, ""},
          {"inequality", {"value <> 'alpha'"}, ""},
          {"range", {"value BETWEEN 'alpha' AND 'delta'"}, "value < 'delta'"},
          {"negatedRange", {"value NOT BETWEEN 'alpha' AND 'delta'"}, ""},
          {"in", {"value IN ('alpha', 'beta', '')"}, ""},
          {"null", {"value IS NULL"}, ""},
          {"notNull", {"value IS NOT NULL"}, ""},
          {"sparse", {"gate = 1", "value >= 'alpha'"}, ""},
          {"like", {}, "value LIKE 'alpha%'"},
          {"stringFunctions", {}, "", QueryAction::kStringFunctions},
          {"globalCount", {}, "", QueryAction::kCount},
          {"groupCount", {}, "", QueryAction::kGroup},
          {"ascending", {}, "", QueryAction::kOrderAscending},
          {"descending", {}, "", QueryAction::kOrderDescending},
          {"innerJoin", {}, "", QueryAction::kInnerJoin},
          {"leftJoin", {}, "", QueryAction::kLeftJoin},
      });
}

TEST_P(FsstQueryTest, arbitraryBytes) {
  checkQueries(
      makeInput(true),
      {
          {"byteProjection", {}, ""},
          {"byteRange",
           {"value BETWEEN 'alpha' AND 'delta'"},
           "value < 'delta'"},
          {"byteNull", {"value IS NULL"}, ""},
      });
}

INSTANTIATE_TEST_SUITE_P(
    StringEncodings,
    FsstQueryTest,
    ::testing::Values(
        StringLayout::kTrivial,
        StringLayout::kDictionaryTrivial,
        StringLayout::kFsst,
        StringLayout::kDictionaryFsst));

} // namespace
} // namespace facebook::nimble::test
