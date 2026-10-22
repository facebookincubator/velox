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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/prestosql/UnixtimeFunctions.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/scalar_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <array>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace facebook::velox::cudf_velox::prestosql {
namespace {

// A cuDF timestamp unit, with the spelling cudf.timestamp_unit takes for it.
struct TimestampUnit {
  cudf::type_id type;
  int64_t ticksPerSecond;
  std::string_view name;
};

constexpr std::array<TimestampUnit, 4> kTimestampUnits{{
    {cudf::type_id::TIMESTAMP_SECONDS, 1, "s"},
    {cudf::type_id::TIMESTAMP_MILLISECONDS, 1'000, "ms"},
    {cudf::type_id::TIMESTAMP_MICROSECONDS, 1'000'000, "us"},
    {cudf::type_id::TIMESTAMP_NANOSECONDS, 1'000'000'000, "ns"},
}};

// Position of a cuDF timestamp type in kTimestampUnits.
size_t unitIndex(cudf::data_type type) {
  const auto unit =
      std::ranges::find(kTimestampUnits, type.id(), &TimestampUnit::type);
  VELOX_CHECK(
      unit != kTimestampUnits.end(),
      "Unsupported cuDF timestamp type: {}",
      static_cast<int32_t>(type.id()));
  return std::ranges::distance(kTimestampUnits.begin(), unit);
}

std::unique_ptr<cudf::numeric_scalar<int64_t>> makeTicksPerSecondScalar(
    int64_t ticksPerSecond) {
  return std::unique_ptr<cudf::numeric_scalar<int64_t>>(
      static_cast<cudf::numeric_scalar<int64_t>*>(
          makeScalarFromValue<int64_t>(BIGINT(), ticksPerSecond, false)
              .release()));
}

// Runs a CUDA UDF of the form f(out, value, ticksPerSecond) over one column,
// with the tick count of one second broadcast as the second argument.
std::unique_ptr<cudf::column> transformWithTicksPerSecond(
    std::string_view udf,
    cudf::column_view input,
    const cudf::numeric_scalar<int64_t>& ticksPerSecond,
    cudf::data_type outputType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const cudf::column_view ticksPerSecondView{
      cudf::data_type{cudf::type_id::INT64},
      /*size=*/1,
      ticksPerSecond.data(),
      /*null_mask=*/nullptr,
      /*null_count=*/0};
  const cudf::transform_input transformInputs[] = {
      input,
      cudf::scalar_column_view(ticksPerSecondView),
  };
  const cudf::transform_output transformOutputs[] = {{
      outputType,
      cudf::output_nullability::PRESERVE,
  }};
  auto transformed = cudf::transform(
      std::string{udf},
      cudf::udf_source_type::CUDA,
      cudf::null_aware::NO,
      std::nullopt,
      transformInputs,
      transformOutputs,
      {},
      std::nullopt,
      stream,
      mr);
  return std::move(transformed->release().front());
}

// Repeats fromUnixtime() in velox/functions/prestosql/DateTimeImpl.h to the
// CPU's millisecond timestamp, then scales it to the unit of the result. The
// CPU's bounds, Timestamp::kMaxSeconds and kMinSeconds, keep the millisecond
// arithmetic inside int64. A millisecond outside the range of the unit is
// returned as an overflow, which cudf::transform raises once the kernel ends.
// TODO: Reuse fromUnixtime() and toUnixtime() from a compiled kernel once
// DateTimeImpl.h builds under nvcc (#17266), and drop these copies.
constexpr std::string_view kFromUnixtimeUdf = R"***(
__device__ cudf::errc velox_from_unixtime(
    int64_t* out, double unixtime, int64_t ticksPerSecond) {
  constexpr int64_t kMaxMillis = 9'223'372'036'854'775'807LL;
  constexpr int64_t kMinMillis = -kMaxMillis - 1;
  constexpr int64_t kMaxSeconds = kMaxMillis / 1'000;
  constexpr int64_t kMinSeconds = kMinMillis / 1'000 - 1;
  int64_t millis;
  if (isnan(unixtime)) {
    millis = 0;
  } else if (unixtime >= static_cast<double>(kMaxSeconds)) {
    millis = kMaxMillis;
  } else if (unixtime <= static_cast<double>(kMinSeconds)) {
    millis = kMinMillis;
  } else {
    double seconds = floor(unixtime);
    long long fraction = llround((unixtime - seconds) * 1'000.0);
    if (fraction == 1'000) {
      seconds += 1.0;
      fraction = 0;
    }
    millis = static_cast<int64_t>(seconds) * 1'000 + fraction;
  }
  if (ticksPerSecond == 1) {
    *out = millis / 1'000 - (millis % 1'000 < 0 ? 1 : 0);
    return cudf::errc::SUCCESS;
  }
  const int64_t ticksPerMilli = ticksPerSecond / 1'000;
  if (millis > kMaxMillis / ticksPerMilli ||
      millis < kMinMillis / ticksPerMilli) {
    return cudf::errc::ARITHMETIC_OVERFLOW;
  }
  *out = millis * ticksPerMilli;
  return cudf::errc::SUCCESS;
}
)***";

// Repeats toUnixtime() in velox/functions/prestosql/DateTimeImpl.h: a Velox
// timestamp holds the floor of its seconds and a non-negative nanosecond
// part, and the two are added as doubles in that order.
constexpr std::string_view kToUnixtimeUdf = R"***(
__device__ void velox_to_unixtime(
    double* out, int64_t ticks, int64_t ticksPerSecond) {
  int64_t seconds = ticks / ticksPerSecond;
  int64_t subSecondTicks = ticks % ticksPerSecond;
  if (subSecondTicks < 0) {
    subSecondTicks += ticksPerSecond;
    seconds -= 1;
  }
  const int64_t nanos = subSecondTicks * (1'000'000'000LL / ticksPerSecond);
  double result = static_cast<double>(seconds);
  result += static_cast<double>(nanos) / 1'000'000'000.0;
  *out = result;
}
)***";

} // namespace

bool FromUnixtimeFunction::canEvaluate(const core::TypedExprPtr& expr) {
  if (expr->inputs().size() != 1 || !expr->type()->isTimestamp()) {
    return false;
  }
  const auto& input = expr->inputs()[0];
  return !input->isConstantKind() && input->type()->isDouble();
}

FromUnixtimeFunction::FromUnixtimeFunction(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* /*pool*/) {
  VELOX_CHECK(
      canEvaluate(expr),
      "from_unixtime expression cannot be evaluated by "
      "prestosql::FromUnixtimeFunction");
  timestampType_ = veloxToCudfDataType(expr->type());
  ticksPerSecond_ = makeTicksPerSecondScalar(
      kTimestampUnits[unitIndex(timestampType_)].ticksPerSecond);
}

ColumnOrView FromUnixtimeFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  std::unique_ptr<cudf::column> ticks;
  try {
    ticks = transformWithTicksPerSecond(
        kFromUnixtimeUdf,
        asView(inputColumns[0]),
        *ticksPerSecond_,
        cudf::data_type{cudf::type_id::INT64},
        stream,
        mr);
  } catch (const cudf::evaluation_error&) {
    // The only error the kernel returns is a millisecond outside the range of
    // the unit, which converting a Velox timestamp to the unit rejects too.
    VELOX_USER_FAIL(
        "from_unixtime result is out of range for the cuDF timestamp unit "
        "set by {}: {}",
        CudfConfig::kCudfTimestampUnit,
        kTimestampUnits[unitIndex(timestampType_)].name);
  }
  // The kernel writes the ticks of timestampType_, so the timestamp column
  // takes over the INT64 buffers as they are.
  const auto numRows = ticks->size();
  const auto nullCount = ticks->null_count();
  auto contents = ticks->release();
  return std::make_unique<cudf::column>(
      timestampType_,
      numRows,
      std::move(*contents.data),
      std::move(*contents.null_mask),
      nullCount);
}

bool ToUnixtimeFunction::canEvaluate(const core::TypedExprPtr& expr) {
  if (expr->inputs().size() != 1 || !expr->type()->isDouble()) {
    return false;
  }
  const auto& input = expr->inputs()[0];
  return !input->isConstantKind() && input->type()->isTimestamp();
}

ToUnixtimeFunction::ToUnixtimeFunction(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* /*pool*/) {
  VELOX_CHECK(
      canEvaluate(expr),
      "to_unixtime expression cannot be evaluated by "
      "prestosql::ToUnixtimeFunction");
  for (size_t unit = 0; unit < kTimestampUnits.size(); ++unit) {
    ticksPerSecond_[unit] =
        makeTicksPerSecondScalar(kTimestampUnits[unit].ticksPerSecond);
  }
}

ColumnOrView ToUnixtimeFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  const auto input = asView(inputColumns[0]);
  // A timestamp column that a reader produced can carry a unit other than the
  // configured one, so the tick count of a second follows the column.
  return transformWithTicksPerSecond(
      kToUnixtimeUdf,
      cudf::bit_cast(input, cudf::data_type{cudf::type_id::INT64}),
      *ticksPerSecond_[unitIndex(input.type())],
      cudf::data_type{cudf::type_id::FLOAT64},
      stream,
      mr);
}

} // namespace facebook::velox::cudf_velox::prestosql
