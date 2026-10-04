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

#pragma once

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/functions/GpuSfiExpression.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"
#include "velox/experimental/cudf/tests/utils/ExpressionTestUtil.h"
#include "velox/experimental/cudf/tests/utils/PreferGpuSfi.h"

#include "velox/core/Expressions.h"
#include "velox/external/tzdb/time_zone.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::test_utils {

/// The next value of a linear congruential generator over `state`, 53 bits
/// wide, for inputs that are spread deterministically over a range.
inline uint64_t nextRandom(uint64_t& state) {
  state = state * 6'364'136'223'846'793'005ULL + 1'442'695'040'888'963'407ULL;
  return state >> 11;
}

/// An offset change of a time zone: the instant the clocks moved and the
/// offsets, in seconds east of UTC, before and after it.
struct OffsetChange {
  int64_t utcSeconds;
  int64_t offsetBefore;
  int64_t offsetAfter;
};

/// The offset changes a zone records between the starts of `fromYear` and
/// `toYear`, in order. None for a fixed offset.
inline std::vector<OffsetChange>
offsetChanges(const std::string& timeZone, int32_t fromYear, int32_t toYear) {
  std::vector<OffsetChange> changes;
  const auto* zone = tz::locateZone(timeZone)->tz();
  if (zone == nullptr) {
    return changes;
  }
  const auto to = date::sys_days{date::year{toYear} / 1 / 1};
  auto info = zone->get_info(
      date::sys_seconds{date::sys_days{date::year{fromYear} / 1 / 1}});
  while (info.end < to) {
    const auto next = zone->get_info(info.end);
    changes.push_back(
        OffsetChange{
            info.end.time_since_epoch().count(),
            info.offset.count(),
            next.offset.count()});
    info = next;
  }
  return changes;
}

/// Compares GPU SFI with the CPU over one SQL expression and one input, with
/// the session's time zone settings under the test's control. The CPU is the
/// oracle. AST and JIT are demoted for the fixture's lifetime, so GPU SFI
/// answers whatever it registers and every comparison can require that it did.
class GpuSfiParityTestBase : public CudfFunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    CudfConfig::getInstance().allowCpuFallback = false;
    registerCudf();
  }

  static void TearDownTestCase() {
    unregisterCudf();
  }

  /// Fixed offsets, zones with and without daylight saving time, a 30-minute
  /// daylight shift (Lord_Howe), a 45-minute offset (Kathmandu), a zone that
  /// abolished daylight saving time (Sao_Paulo), and one that skipped a whole
  /// day (Apia).
  static const std::vector<std::string>& timeZones() {
    static const std::vector<std::string> kTimeZones{
        "UTC",
        "+05:30",
        "-08:00",
        "America/Los_Angeles",
        "Europe/London",
        "Asia/Kolkata",
        "Asia/Kathmandu",
        "Australia/Lord_Howe",
        "America/Sao_Paulo",
        "Pacific/Apia",
    };
    return kTimeZones;
  }

  /// Session zones the TIMESTAMP WITH TIME ZONE functions are compared under:
  /// the render zone when legacy_timestamp_with_timezone is off, and otherwise
  /// irrelevant, which the comparison also checks.
  static const std::vector<std::string>& sessionTimeZones() {
    static const std::vector<std::string> kSessionTimeZones{
        "UTC",
        "America/Los_Angeles",
        "Asia/Kathmandu",
    };
    return kSessionTimeZones;
  }

  /// Instants that stress the zone's conversion: one second and one hour
  /// either side of every offset change the zone records between `fromYear`
  /// and `toYear`, sub-second values either side of the epoch, the ends of the
  /// span, and a deterministic spread of `numSpread` values over it.
  static std::vector<Timestamp> instants(
      const std::string& timeZone,
      int32_t fromYear,
      int32_t toYear,
      int32_t numSpread) {
    const auto from = date::sys_days{date::year{fromYear} / 1 / 1};
    const auto to = date::sys_days{date::year{toYear} / 1 / 1};
    const int64_t fromSeconds =
        std::chrono::duration_cast<std::chrono::seconds>(
            from.time_since_epoch())
            .count();
    const int64_t toSeconds =
        std::chrono::duration_cast<std::chrono::seconds>(to.time_since_epoch())
            .count();

    std::vector<Timestamp> values{
        Timestamp(0, 0),
        Timestamp(0, 999'999'999),
        Timestamp(-1, 0),
        Timestamp(-1, 1),
        Timestamp(-1, 999'999'999),
        Timestamp(fromSeconds, 0),
        Timestamp(toSeconds - 1, 999'999'999),
    };

    for (const auto& change : offsetChanges(timeZone, fromYear, toYear)) {
      for (const int64_t delta : {-3'600, -1, 0, 1, 3'599}) {
        values.emplace_back(change.utcSeconds + delta, 123'000'000);
      }
    }

    uint64_t state = 0x9e37'79b9'7f4a'7c15;
    const auto span = static_cast<uint64_t>(toSeconds - fromSeconds);
    for (int32_t i = 0; i < numSpread; ++i) {
      const auto random = nextRandom(state);
      values.emplace_back(
          fromSeconds + static_cast<int64_t>(random % span),
          (state >> 7) % 1'000'000'000);
    }
    return values;
  }

  /// Sets the session time zone, whether TIMESTAMP is read in it, and whether
  /// TIMESTAMP WITH TIME ZONE renders in its embedded zone. One override
  /// replaces the whole session configuration, so the three are set together.
  void setSession(
      const std::string& sessionTimeZone,
      bool adjustTimestampToTimezone,
      bool legacyTimestampWithTimezone) {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kSessionTimezone, sessionTimeZone},
        {core::QueryConfig::kAdjustTimestampToTimezone,
         adjustTimestampToTimezone ? "true" : "false"},
        {core::QueryConfig::kLegacyTimestampWithTimezone,
         legacyTimestampWithTimezone ? "true" : "false"},
    });
  }

  /// Runs `body` under both renderings of TIMESTAMP WITH TIME ZONE and every
  /// session zone in sessionTimeZones(), with TIMESTAMP read in the session
  /// zone, naming both in the trace.
  template <typename Body>
  void forEachSessionZone(Body&& body) {
    for (const bool legacy : {true, false}) {
      for (const auto& sessionTimeZone : sessionTimeZones()) {
        SCOPED_TRACE(
            fmt::format("legacy={} session={}", legacy, sessionTimeZone));
        setSession(sessionTimeZone, /*adjustTimestampToTimezone=*/true, legacy);
        body();
      }
    }
  }

  /// Evaluates `sql` over `input` on the GPU and the CPU and compares the
  /// results, after checking that GPU SFI is the evaluator that answers.
  void assertGpuMatchesCpu(const std::string& sql, const RowVectorPtr& input) {
    const auto rowType = asRowType(input->type());
    const auto expr =
        optimizeTypedExpr(sql, rowType, queryCtx_.get(), &execCtx_);
    const auto evaluator = createCudfExpression(
        expr, rowType, pool_.get(), queryCtx_->queryConfig());
    ASSERT_NE(dynamic_cast<GpuSfiExpression*>(evaluator.get()), nullptr)
        << sql << " is not evaluated by GPU SFI";
    assertExpressionMatchesCpu(sql, input, rowType);
  }

  /// The rows of `input` the CPU answers `sql` for: the ones a TRY around it
  /// does not turn into a null, plus the rows with a null in a column the
  /// expression reads, which TRY cannot tell from an error. The GPU declines a
  /// row whose check fails rather than answer it, and the bare evaluator
  /// reports that instead of comparing; FilterProjectTest covers an operator
  /// re-evaluating such rows on the CPU.
  RowVectorPtr rowsTheCpuAnswers(
      const std::string& sql,
      const RowVectorPtr& input) {
    return rowsTheCpuAnswers(makeTypedExpr(sql, input->rowType()), input);
  }

  RowVectorPtr rowsTheCpuAnswers(
      const core::TypedExprPtr& expr,
      const RowVectorPtr& input) {
    const auto tried = std::make_shared<core::CallTypedExpr>(
        expr->type(), std::vector<core::TypedExprPtr>{expr}, "try");
    exec::ExprSet exprSet({tried}, &execCtx_);
    const auto result =
        functions::test::FunctionBaseTest::evaluate(exprSet, input);
    std::vector<VectorPtr> read;
    for (const auto& name : referencedInputFields(expr)) {
      read.push_back(input->childAt(input->rowType()->getChildIdx(name)));
    }
    std::vector<BaseVector::CopyRange> kept;
    for (vector_size_t row = 0; row < input->size(); ++row) {
      const bool nullInput =
          std::any_of(read.begin(), read.end(), [&](const auto& child) {
            return child->isNullAt(row);
          });
      if (nullInput || !result->isNullAt(row)) {
        kept.push_back({row, static_cast<vector_size_t>(kept.size()), 1});
      }
    }
    auto filtered = BaseVector::create(
        input->type(), static_cast<vector_size_t>(kept.size()), pool());
    filtered->copyRanges(input.get(), kept);
    return std::dynamic_pointer_cast<RowVector>(filtered);
  }

 private:
  PreferGpuSfi preferGpuSfi_;
};

} // namespace facebook::velox::cudf_velox::test_utils
