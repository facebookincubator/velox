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

#include <emscripten/bind.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <mutex>
#include <numeric>
#include <sstream>

#include "velox/common/memory/Memory.h"
#include "velox/core/Expressions.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Cursor.h"
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/aggregates/AverageAggregate.h"
#include "velox/functions/prestosql/aggregates/CountAggregate.h"
#include "velox/functions/prestosql/aggregates/MinMaxAggregates.h"
#include "velox/functions/prestosql/aggregates/SumAggregate.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/type/tz/TimeZoneMap.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/VectorSaver.h"

#include <folly/Singleton.h>
#include <gflags/gflags_declare.h> // @manual

DECLARE_bool(velox_exception_user_stacktrace_enabled);
DECLARE_bool(velox_exception_system_stacktrace_enabled);

namespace {

using namespace facebook::velox;

std::string escapeJsonString(std::string_view input) {
  std::string out;
  out.reserve(input.size() + 2);
  out += '"';
  for (char c : input) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c & 0xff);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  out += '"';
  return out;
}

std::string errorJson(std::string_view message) {
  return "{\"ok\":false,\"error\":" + escapeJsonString(message) + "}";
}

void initializeGlobals(int64_t arbitratorCapacity) {
  static std::once_flag flag;
  std::call_once(flag, [arbitratorCapacity] {
    FLAGS_velox_exception_user_stacktrace_enabled = false;
    FLAGS_velox_exception_system_stacktrace_enabled = false;
    folly::SingletonVault::singleton()->registrationComplete();

    memory::MemoryManager::Options opts;
    opts.arbitratorCapacity = arbitratorCapacity;
    opts.mallocContiguousEnabled = true;
    memory::MemoryManager::initialize(opts);

    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerCountAggregate({"count"}, false, false);
    aggregate::prestosql::registerAverageAggregate({"avg"}, false, false);
    aggregate::prestosql::registerSumAggregate({"sum"}, false, false);
    aggregate::prestosql::registerMinAggregate({"min"}, false, false);
    aggregate::prestosql::registerMaxAggregate({"max"}, false, false);
    parse::registerTypeResolver();
    serializer::presto::PrestoVectorSerde::registerVectorSerde();
  });
}

// Wraps a single-process Velox runtime exposed to JavaScript via Emscripten
// bindings. Each public method is invoked from JS, runs a self-contained
// query/expression/plan exercise against an in-memory pool, and returns a
// JSON string with the result or an `errorJson(...)` payload on failure.
//
// `arbitratorCapacity` sets the global memory-arbitrator capacity in bytes.
// The first constructed engine wins; subsequent constructions inherit the
// already-initialized capacity because the global memory manager can only be
// initialized once per process. Default is 256 MiB, which is sized for
// typical browser environments; pass a larger value when embedding in
// node.js or a server-side wasm host.
class VeloxWasmEngine {
 public:
  explicit VeloxWasmEngine(int64_t arbitratorCapacity = 256LL << 20) {
    initializeGlobals(arbitratorCapacity);
    rootPool_ = memory::memoryManager()->addRootPool("velox_wasm");
    pool_ = rootPool_->addLeafChild("test");
    queryCtx_ = core::QueryCtx::create();
    execCtx_ = std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get());
  }

 private:
  FlatVectorPtr<int64_t> makeBigintFlat(const std::vector<int64_t>& values) {
    auto vec = BaseVector::create<FlatVector<int64_t>>(
        BIGINT(), values.size(), pool_.get());
    for (size_t i = 0; i < values.size(); ++i) {
      vec->set(i, values[i]);
    }
    return vec;
  }

  RowVectorPtr makeRow(
      const std::vector<std::string>& names,
      const std::vector<VectorPtr>& children) {
    std::vector<TypePtr> types;
    types.reserve(children.size());
    for (const auto& child : children) {
      types.push_back(child->type());
    }
    return std::make_shared<RowVector>(
        pool_.get(),
        ROW(std::vector<std::string>{names}, std::move(types)),
        BufferPtr(nullptr),
        children.empty() ? 0 : children[0]->size(),
        children);
  }

  std::vector<RowVectorPtr> runPlan(const core::PlanNodePtr& planNode) {
    exec::CursorParameters params;
    params.planNode = planNode;
    params.serialExecution = true;
    params.maxDrivers = 1;
    params.outputPool = pool_;
    auto cursor = exec::TaskCursor::create(params);
    std::vector<RowVectorPtr> outputs;
    while (cursor->moveNext()) {
      outputs.push_back(cursor->current());
    }
    return outputs;
  }

  static core::FieldAccessTypedExprPtr bigintField(const std::string& name) {
    return std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), name);
  }

 public:
  std::string testExpressionEval() {
    try {
      constexpr int kSize = 10;

      auto aVec =
          BaseVector::create<FlatVector<int64_t>>(BIGINT(), kSize, pool_.get());
      auto bVec =
          BaseVector::create<FlatVector<int64_t>>(BIGINT(), kSize, pool_.get());
      for (int i = 0; i < kSize; ++i) {
        aVec->set(i, static_cast<int64_t>(i));
        bVec->set(i, static_cast<int64_t>(i + 10));
      }

      auto inputType = ROW({"a", "b"}, {BIGINT(), BIGINT()});
      auto input = std::make_shared<RowVector>(
          pool_.get(),
          inputType,
          BufferPtr(nullptr),
          kSize,
          std::vector<VectorPtr>{aVec, bVec});

      // Build the expression `a + b * 2`.
      auto bField = std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "b");
      auto two = std::make_shared<core::ConstantTypedExpr>(
          BIGINT(), variant(static_cast<int64_t>(2)));
      auto bTimesTwo = std::make_shared<core::CallTypedExpr>(
          BIGINT(), std::vector<core::TypedExprPtr>{bField, two}, "multiply");

      auto aField = std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "a");
      auto expr = std::make_shared<core::CallTypedExpr>(
          BIGINT(), std::vector<core::TypedExprPtr>{aField, bTimesTwo}, "plus");

      exec::ExprSet exprSet({expr}, execCtx_.get());
      SelectivityVector rows(kSize);
      std::vector<VectorPtr> result(1);
      exec::EvalCtx evalCtx(execCtx_.get(), &exprSet, input.get());
      exprSet.eval(rows, evalCtx, result);

      auto* resultVec = result[0]->as<FlatVector<int64_t>>();
      VELOX_CHECK_NOT_NULL(resultVec);
      for (int i = 0; i < kSize; ++i) {
        int64_t a = i;
        int64_t b = i + 10;
        int64_t expected = a + b * 2;
        VELOX_CHECK_EQ(
            resultVec->valueAt(i),
            expected,
            "expression result mismatch at index {}",
            i);
      }

      return "{\"ok\":true}";
    } catch (const std::exception& exception) {
      return errorJson(exception.what());
    }
  }

  std::string testPlanExecution() {
    try {
      constexpr int kSize = 10;

      auto aVec =
          BaseVector::create<FlatVector<int64_t>>(BIGINT(), kSize, pool_.get());
      for (int i = 0; i < kSize; ++i) {
        aVec->set(i, static_cast<int64_t>(i));
      }
      auto rowType = ROW({"a"}, {BIGINT()});
      auto data = std::make_shared<RowVector>(
          pool_.get(),
          rowType,
          BufferPtr(nullptr),
          kSize,
          std::vector<VectorPtr>{aVec});

      // Build the plan `ValuesNode -> FilterNode(a > 5) -> ProjectNode(a *
      // 10)`.
      auto valuesNode = std::make_shared<core::ValuesNode>(
          "0", std::vector<RowVectorPtr>{data});

      // Filter `a > 5`.
      auto aFieldFilter =
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "a");
      auto five = std::make_shared<core::ConstantTypedExpr>(
          BIGINT(), variant(static_cast<int64_t>(5)));
      auto filterExpr = std::make_shared<core::CallTypedExpr>(
          BOOLEAN(), std::vector<core::TypedExprPtr>{aFieldFilter, five}, "gt");
      auto filterNode =
          std::make_shared<core::FilterNode>("1", filterExpr, valuesNode);

      // Project `a * 10`.
      auto aFieldProject =
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "a");
      auto ten = std::make_shared<core::ConstantTypedExpr>(
          BIGINT(), variant(static_cast<int64_t>(10)));
      auto projectExpr = std::make_shared<core::CallTypedExpr>(
          BIGINT(),
          std::vector<core::TypedExprPtr>{aFieldProject, ten},
          "multiply");
      auto projectNode = std::make_shared<core::ProjectNode>(
          "2",
          std::vector<std::string>{"result"},
          std::vector<core::TypedExprPtr>{projectExpr},
          filterNode);

      exec::CursorParameters params;
      params.planNode = projectNode;
      params.serialExecution = true;
      params.maxDrivers = 1;

      auto cursor = exec::TaskCursor::create(params);
      std::vector<int64_t> results;
      while (cursor->moveNext()) {
        auto& row = cursor->current();
        auto* col = row->childAt(0)->as<SimpleVector<int64_t>>();
        for (vector_size_t i = 0; i < row->size(); ++i) {
          results.push_back(col->valueAt(i));
        }
      }

      // Input: [0..9], filter a > 5 -> [6,7,8,9], project a*10 -> [60,70,80,90]
      std::vector<int64_t> expected = {60, 70, 80, 90};
      VELOX_CHECK_EQ(
          results.size(),
          expected.size(),
          "plan output row count mismatch: {} vs {}",
          results.size(),
          expected.size());
      for (size_t i = 0; i < expected.size(); ++i) {
        VELOX_CHECK_EQ(
            results[i],
            expected[i],
            "plan output value mismatch at index {}",
            i);
      }

      return "{\"ok\":true}";
    } catch (const std::exception& exception) {
      return errorJson(exception.what());
    }
  }

  std::string testAggregation() {
    try {
      auto input = makeRow({"a"}, {makeBigintFlat({1, 2, 3, 4, 5})});
      auto values = std::make_shared<core::ValuesNode>(
          "0", std::vector<RowVectorPtr>{input});

      auto aField = bigintField("a");
      auto makeAgg = [&](const std::string& name, const TypePtr& outType) {
        core::AggregationNode::Aggregate agg;
        agg.call = std::make_shared<core::CallTypedExpr>(
            outType, std::vector<core::TypedExprPtr>{aField}, name);
        agg.rawInputTypes = {BIGINT()};
        return agg;
      };

      auto agg = std::make_shared<core::AggregationNode>(
          "1",
          core::AggregationNode::Step::kSingle,
          std::vector<core::FieldAccessTypedExprPtr>{},
          std::vector<core::FieldAccessTypedExprPtr>{},
          std::vector<std::string>{"s", "c", "mn", "mx"},
          std::vector<core::AggregationNode::Aggregate>{
              makeAgg("sum", BIGINT()),
              makeAgg("count", BIGINT()),
              makeAgg("min", BIGINT()),
              makeAgg("max", BIGINT()),
          },
          /*ignoreNullKeys=*/false,
          /*noGroupsSpanBatches=*/false,
          values);

      const std::vector<int64_t> inputValues{1, 2, 3, 4, 5};
      const std::array<int64_t, 4> expected{
          std::accumulate(inputValues.begin(), inputValues.end(), int64_t{0}),
          static_cast<int64_t>(inputValues.size()),
          *std::min_element(inputValues.begin(), inputValues.end()),
          *std::max_element(inputValues.begin(), inputValues.end()),
      };

      auto results = runPlan(agg);
      VELOX_CHECK_EQ(results.size(), 1);
      auto& row = results[0];
      VELOX_CHECK_EQ(row->size(), 1);
      for (size_t col = 0; col < expected.size(); ++col) {
        VELOX_CHECK_EQ(
            row->childAt(col)->as<SimpleVector<int64_t>>()->valueAt(0),
            expected[col]);
      }

      return "{\"ok\":true}";
    } catch (const std::exception& exception) {
      return errorJson(exception.what());
    }
  }

  std::string testGroupBy() {
    try {
      auto input = makeRow(
          {"k", "v"},
          {makeBigintFlat({1, 2, 1, 2, 1, 3}),
           makeBigintFlat({10, 20, 30, 40, 50, 60})});
      auto values = std::make_shared<core::ValuesNode>(
          "0", std::vector<RowVectorPtr>{input});

      auto kField = bigintField("k");
      auto vField = bigintField("v");
      core::AggregationNode::Aggregate sumAgg;
      sumAgg.call = std::make_shared<core::CallTypedExpr>(
          BIGINT(), std::vector<core::TypedExprPtr>{vField}, "sum");
      sumAgg.rawInputTypes = {BIGINT()};

      auto agg = std::make_shared<core::AggregationNode>(
          "1",
          core::AggregationNode::Step::kSingle,
          std::vector<core::FieldAccessTypedExprPtr>{kField},
          std::vector<core::FieldAccessTypedExprPtr>{},
          std::vector<std::string>{"s"},
          std::vector<core::AggregationNode::Aggregate>{sumAgg},
          /*ignoreNullKeys=*/false,
          /*noGroupsSpanBatches=*/false,
          values);

      auto results = runPlan(agg);
      std::map<int64_t, int64_t> sumByKey;
      for (const auto& batch : results) {
        auto* keys = batch->childAt(0)->as<SimpleVector<int64_t>>();
        auto* sums = batch->childAt(1)->as<SimpleVector<int64_t>>();
        for (vector_size_t i = 0; i < batch->size(); ++i) {
          sumByKey[keys->valueAt(i)] = sums->valueAt(i);
        }
      }
      VELOX_CHECK_EQ(sumByKey.size(), 3);
      VELOX_CHECK_EQ(sumByKey[1], 90);
      VELOX_CHECK_EQ(sumByKey[2], 60);
      VELOX_CHECK_EQ(sumByKey[3], 60);

      return "{\"ok\":true}";
    } catch (const std::exception& exception) {
      return errorJson(exception.what());
    }
  }

  std::string testOrderBy() {
    try {
      auto input = makeRow({"a"}, {makeBigintFlat({3, 1, 4, 1, 5, 9, 2, 6})});
      auto values = std::make_shared<core::ValuesNode>(
          "0", std::vector<RowVectorPtr>{input});

      auto sorted = std::make_shared<core::OrderByNode>(
          "1",
          std::vector<core::FieldAccessTypedExprPtr>{bigintField("a")},
          std::vector<core::SortOrder>{core::SortOrder{true, true}},
          /*isPartial=*/false,
          values);

      auto results = runPlan(sorted);
      std::vector<int64_t> actual;
      for (const auto& batch : results) {
        auto* col = batch->childAt(0)->as<SimpleVector<int64_t>>();
        for (vector_size_t i = 0; i < batch->size(); ++i) {
          actual.push_back(col->valueAt(i));
        }
      }
      const std::vector<int64_t> expected = {1, 1, 2, 3, 4, 5, 6, 9};
      VELOX_CHECK_EQ(actual.size(), expected.size());
      for (size_t i = 0; i < expected.size(); ++i) {
        VELOX_CHECK_EQ(actual[i], expected[i], "sort mismatch at index {}", i);
      }

      return "{\"ok\":true}";
    } catch (const std::exception& exception) {
      return errorJson(exception.what());
    }
  }

  std::string testHashJoin() {
    try {
      auto left = makeRow(
          {"l_k", "l_v"},
          {makeBigintFlat({1, 2, 3, 4}), makeBigintFlat({10, 20, 30, 40})});
      auto right = makeRow(
          {"r_k", "r_v"},
          {makeBigintFlat({2, 3, 5}), makeBigintFlat({200, 300, 500})});

      auto leftValues = std::make_shared<core::ValuesNode>(
          "0", std::vector<RowVectorPtr>{left});
      auto rightValues = std::make_shared<core::ValuesNode>(
          "1", std::vector<RowVectorPtr>{right});

      auto outputType =
          ROW({"l_k", "l_v", "r_v"}, {BIGINT(), BIGINT(), BIGINT()});
      auto join = std::make_shared<core::HashJoinNode>(
          "2",
          core::JoinType::kInner,
          /*nullAware=*/false,
          std::vector<core::FieldAccessTypedExprPtr>{bigintField("l_k")},
          std::vector<core::FieldAccessTypedExprPtr>{bigintField("r_k")},
          /*filter=*/nullptr,
          leftValues,
          rightValues,
          outputType);

      auto results = runPlan(join);
      std::map<int64_t, std::pair<int64_t, int64_t>> rowsByKey;
      for (const auto& batch : results) {
        auto* leftKey = batch->childAt(0)->as<SimpleVector<int64_t>>();
        auto* leftValue = batch->childAt(1)->as<SimpleVector<int64_t>>();
        auto* rightValue = batch->childAt(2)->as<SimpleVector<int64_t>>();
        for (vector_size_t i = 0; i < batch->size(); ++i) {
          rowsByKey[leftKey->valueAt(i)] = {
              leftValue->valueAt(i), rightValue->valueAt(i)};
        }
      }
      VELOX_CHECK_EQ(rowsByKey.size(), 2);
      VELOX_CHECK_EQ(rowsByKey[2].first, 20);
      VELOX_CHECK_EQ(rowsByKey[2].second, 200);
      VELOX_CHECK_EQ(rowsByKey[3].first, 30);
      VELOX_CHECK_EQ(rowsByKey[3].second, 300);

      return "{\"ok\":true}";
    } catch (const std::exception& exception) {
      return errorJson(exception.what());
    }
  }

  std::string testTimezoneFallback() {
    try {
      // UTC must always resolve, even when the OS tzdata directory is absent
      // (the wasm build relies on a built-in UTC-only fallback).
      const auto* utc = tz::locateZone("UTC", false);
      VELOX_CHECK_NOT_NULL(utc, "locateZone(\"UTC\") returned null");

      // Non-UTC zones are not bundled in the wasm build, so locateZone with
      // failOnError=false must return null rather than throw.
      const auto* losAngeles = tz::locateZone("America/Los_Angeles", false);
      VELOX_CHECK_NULL(
          losAngeles,
          "locateZone(\"America/Los_Angeles\") unexpectedly returned non-null");
      return "{\"ok\":true}";
    } catch (const std::exception& exception) {
      return errorJson(exception.what());
    }
  }

  std::string testSerialization() {
    try {
      constexpr int kSize = 10;

      auto intVec =
          BaseVector::create<FlatVector<int64_t>>(BIGINT(), kSize, pool_.get());
      auto doubleVec =
          BaseVector::create<FlatVector<double>>(DOUBLE(), kSize, pool_.get());
      for (int i = 0; i < kSize; ++i) {
        intVec->set(i, static_cast<int64_t>(i * 100));
        doubleVec->set(i, i * 1.5);
      }

      auto rowType = ROW({"x", "y"}, {BIGINT(), DOUBLE()});
      auto original = std::make_shared<RowVector>(
          pool_.get(),
          rowType,
          BufferPtr(nullptr),
          kSize,
          std::vector<VectorPtr>{intVec, doubleVec});

      std::stringstream stream;
      saveVector(*original, stream);

      stream.seekg(0);
      auto restored = restoreVector(stream, pool_.get());

      VELOX_CHECK_NOT_NULL(restored);
      VELOX_CHECK_EQ(restored->size(), kSize);
      auto* restoredRow = restored->as<RowVector>();
      VELOX_CHECK_NOT_NULL(restoredRow);
      VELOX_CHECK_EQ(restoredRow->childrenSize(), 2);

      auto* restoredInts = restoredRow->childAt(0)->as<FlatVector<int64_t>>();
      auto* restoredDoubles = restoredRow->childAt(1)->as<FlatVector<double>>();
      for (int i = 0; i < kSize; ++i) {
        VELOX_CHECK_EQ(
            restoredInts->valueAt(i),
            i * 100,
            "serialization int mismatch at index {}",
            i);
        VELOX_CHECK_EQ(
            restoredDoubles->valueAt(i),
            i * 1.5,
            "serialization double mismatch at index {}",
            i);
      }

      return "{\"ok\":true}";
    } catch (const std::exception& exception) {
      return errorJson(exception.what());
    }
  }

 private:
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<core::ExecCtx> execCtx_;
};

} // namespace

EMSCRIPTEN_BINDINGS(velox_wasm) {
  emscripten::class_<VeloxWasmEngine>("VeloxWasmEngine")
      .constructor<>()
      .constructor<int64_t>()
      .function("testExpressionEval", &VeloxWasmEngine::testExpressionEval)
      .function("testPlanExecution", &VeloxWasmEngine::testPlanExecution)
      .function("testAggregation", &VeloxWasmEngine::testAggregation)
      .function("testGroupBy", &VeloxWasmEngine::testGroupBy)
      .function("testOrderBy", &VeloxWasmEngine::testOrderBy)
      .function("testHashJoin", &VeloxWasmEngine::testHashJoin)
      .function("testTimezoneFallback", &VeloxWasmEngine::testTimezoneFallback)
      .function("testSerialization", &VeloxWasmEngine::testSerialization);
}
