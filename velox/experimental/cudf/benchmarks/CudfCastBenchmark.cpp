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
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SparkFunctions.h"

#include "velox/common/memory/Memory.h"
#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/type/DecimalUtil.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

#include <cuda_runtime_api.h>

#include <folly/Benchmark.h>
#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>

#include <bit>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

DEFINE_int32(cast_iterations, 100, "Conversion calls per Folly sample");
DEFINE_int32(cast_warmups, 3, "Untimed warmup calls per engine and case");
DEFINE_bool(
    cast_validate_only,
    false,
    "Validate all selected cases without timing");
DEFINE_string(
    cast_case_filter,
    "",
    "Only construct cases whose name contains this text");

namespace facebook::velox::cudf_velox {
namespace {

// Values are unscaled decimal integers. nullptr means a NULL result. These
// exact expectations use Java 17 BigDecimal.valueOf((double) input), then
// setScale(scale, HALF_UP) and a precision check, as Spark does. Numeric
// expectations are the explicit differences documented by LegacyCastTest;
// accepting one never permits another input or another output to differ.
// CPU overrides pin the native Spark implementation's known precision
// differences (DecimalUtil::rescaleFloatingPoint); they are not a Java
// reference.
struct DecimalFixture {
  uint64_t bits;
  const char* exact;
  const char* numeric;
  const char* cpu{exact};
};

struct CastCase {
  std::string name;
  TypePtr source;
  TypePtr target;
  std::vector<DecimalFixture> fixtures;
};

int128_t parseUnscaled(const char* text) {
  VELOX_CHECK_NOT_NULL(text);
  const bool negative = text[0] == '-';
  int128_t result = 0;
  for (auto p = text + negative; *p; ++p) {
    VELOX_CHECK(*p >= '0' && *p <= '9');
    result = result * 10 + (*p - '0');
  }
  return negative ? -result : result;
}

std::vector<DecimalFixture> roundingFixtures(bool real) {
  if (real) {
    return {
        {0x3fa00000, "125", "125"},
        {0xbfa00000, "-125", "-125"},
        {0x3f000000, "50", "50"},
        {0xbf000000, "-50", "-50"},
        {0, "0", "0"},
        {0x80000000, "0", "0"},
        {1, "0", "0"},
        {0x7f7fffff, nullptr, nullptr},
        {0x7f800000, nullptr, nullptr},
        {0xff800000, nullptr, nullptr},
        {0x7fc00000, nullptr, nullptr}};
  }
  return {
      {0x4042966666666666, "3718", "3718"},
      {0xc042966666666666, "-3718", "-3718"},
      {0x4029000000000000, "1250", "1250"},
      {0xc029000000000000, "-1250", "-1250"},
      {0, "0", "0"},
      {0x8000000000000000, "0", "0"},
      {1, "0", "0"},
      {0x7fefffffffffffff, nullptr, nullptr},
      {0x7ff0000000000000, nullptr, nullptr},
      {0xfff0000000000000, nullptr, nullptr},
      {0x7ff8000000000000, nullptr, nullptr}};
}

std::vector<DecimalFixture> precisionFixtures(bool real) {
  if (real) {
    return {
        {0x3dcccccd,
         "100000001490116120",
         "100000001490116119",
         "100000000000000000"},
        {0xbdcccccd,
         "-100000001490116120",
         "-100000001490116119",
         "-100000000000000000"}};
  }
  return {
      {0x3fb999999999999a, "100000000000000000", "100000000000000012"},
      {0xbfb999999999999a, "-100000000000000000", "-100000000000000012"},
      {0x3fb99999a0000000,
       "100000001490116120",
       "100000001490116126",
       "100000001490116000"},
      {0xbfb99999a0000000,
       "-100000001490116120",
       "-100000001490116126",
       "-100000001490116000"}};
}

std::vector<CastCase> castCases() {
  std::vector<CastCase> cases{
      {"real_to_integer", REAL(), INTEGER(), {}},
      {"integer_to_short_decimal", INTEGER(), DECIMAL(18, 6), {}},
      {"integer_to_long_decimal", INTEGER(), DECIMAL(38, 16), {}},
      {"bigint_to_short_decimal", BIGINT(), DECIMAL(18, 6), {}},
      {"bigint_to_long_decimal", BIGINT(), DECIMAL(38, 16), {}},
      {"bigint_to_double", BIGINT(), DOUBLE(), {}},
      {"double_to_real", DOUBLE(), REAL(), {}},
      {"short_decimal_rescale", DECIMAL(9, 2), DECIMAL(18, 6), {}},
      {"long_decimal_rescale", DECIMAL(38, 16), DECIMAL(18, 6), {}}};
  for (const auto& source : std::vector<TypePtr>{REAL(), DOUBLE()}) {
    const std::string name = source->isReal() ? "real" : "double";
    for (const auto& target :
         {DECIMAL(15, 2), DECIMAL(18, 6), DECIMAL(38, 6)}) {
      cases.push_back(
          {name + "_common_to_" + target->toString(), source, target, {}});
    }
    cases.push_back(
        {name + "_rounding_and_nonfinite",
         source,
         DECIMAL(15, 2),
         roundingFixtures(source->isReal())});
    for (const auto& target : {DECIMAL(18, 18), DECIMAL(38, 18)}) {
      cases.push_back(
          {name + "_precision_to_" + target->toString(),
           source,
           target,
           precisionFixtures(source->isReal())});
    }
  }
  cases.push_back(
      {"double_precision_overflow",
       DOUBLE(),
       DECIMAL(38, 0),
       {{0x47d2ced32a16a1b1,
         nullptr,
         "99999999999999990000000000000000000000",
         "99999999999999997748809823456034029568"},
        {0xc7d2ced32a16a1b1,
         nullptr,
         "-99999999999999990000000000000000000000",
         "-99999999999999997748809823456034029568"}}});
  cases.push_back(
      {"double_rounding_difference",
       DOUBLE(),
       DECIMAL(15, 4),
       {{0xc235b5bae25aebd0, "-932432369549211", "-932432369549212"}}});
  cases.push_back(
      {"double_extra_rounding_digit",
       DOUBLE(),
       DECIMAL(38, 3),
       {{0xc70a36e2eb1c432c,
         "-17014118346046922000000000000000000000",
         "-17014118346046921000000000000000000000",
         "-17014118346046921682671809215856640000"}}});
  cases.push_back(
      {"double_short_precision_overflow",
       DOUBLE(),
       DECIMAL(1, 0),
       {{0x4022fae147ae147b, "9", "9"},
        {0xc022fae147ae147b, "-9", "-9"},
        {0x4023fae147ae147b, nullptr, nullptr},
        {0xc023fae147ae147b, nullptr, nullptr},
        {0x4024000000000000, nullptr, nullptr},
        {0xc024000000000000, nullptr, nullptr}}});
  return cases;
}

class CastBenchmark {
 public:
  CastBenchmark(CastCase spec, vector_size_t size, bool nulls)
      : spec_(std::move(spec)),
        name_(
            fmt::format("{}_rows{}_null{}", spec_.name, size, nulls ? 20 : 0)),
        pool_(memory::memoryManager()->addLeafPool()),
        queryCtx_(core::QueryCtx::create()),
        execCtx_(pool_.get(), queryCtx_.get()),
        rows_(size),
        stream_(cudfGlobalStreamPool().get_stream()),
        mr_(get_output_mr()) {
    auto values = BaseVector::create(spec_.source, size, pool_.get());
    for (vector_size_t row = 0; row < size; ++row) {
      const auto j = row % 1000;
      if (spec_.source->isReal()) {
        const float value = !spec_.fixtures.empty()
            ? std::bit_cast<float>(static_cast<uint32_t>(fixture(row).bits))
            : spec_.target->isDecimal() ? (row % 7 - 3) / 8.0F
                                        : static_cast<float>(1.2345678 * j);
        values->asFlatVector<float>()->set(row, value);
      } else if (spec_.source->isDouble()) {
        const double value = !spec_.fixtures.empty()
            ? std::bit_cast<double>(fixture(row).bits)
            : (row % 7 - 3) / 8.0;
        values->asFlatVector<double>()->set(row, value);
      } else if (spec_.source->isInteger()) {
        values->asFlatVector<int32_t>()->set(row, 12345 * j);
      } else if (spec_.source->isBigint()) {
        values->asFlatVector<int64_t>()->set(row, 12345LL * j);
      } else {
        VELOX_CHECK(spec_.source->isLongDecimal());
        values->asFlatVector<int128_t>()->set(
            row,
            static_cast<int128_t>(12345 * j) * DecimalUtil::kPowersOfTen[10]);
      }
      // The offset leaves the first fixture visible. The companion no-NULL
      // case checks every fixture even when its period divides five.
      if (nulls && row % 5 == 4) {
        values->setNull(row, true);
      }
    }
    input_ = std::make_shared<RowVector>(
        pool_.get(),
        ROW({"c0"}, {spec_.source}),
        nullptr,
        size,
        std::vector<VectorPtr>{values});
    auto expr = std::make_shared<core::CallTypedExpr>(
        spec_.target,
        std::vector<core::TypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(spec_.source, "c0")},
        "spark_legacy_cast");
    cpu_ = std::make_unique<exec::ExprSet>(
        std::vector<core::TypedExprPtr>{expr}, &execCtx_);
    evalCtx_ =
        std::make_unique<exec::EvalCtx>(&execCtx_, cpu_.get(), input_.get());
    VELOX_CHECK(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
    auto& config = CudfConfig::getInstance();
    const auto previousMode = config.numericFloatToDecimalEnabled;
    SCOPE_EXIT {
      config.numericFloatToDecimalEnabled = previousMode;
    };
    for (int mode = 0; mode < 2; ++mode) {
      config.numericFloatToDecimalEnabled = mode != 0;
      gpu_[mode] = createCudfExpression(
          expr, input_->rowType(), pool_.get(), queryCtx_->queryConfig());
      // Direct FunctionExpression selects the production registered CUDA cast;
      // the benchmark must never silently become a CPU expression evaluator.
      VELOX_CHECK(std::dynamic_pointer_cast<FunctionExpression>(gpu_[mode]));
    }
    deviceInput_ = with_arrow::toCudfTable(input_, pool_.get(), stream_, mr_);
    views_ = tableViewToColumnViews(deviceInput_->view());
    stream_.sync();
  }

  ~CastBenchmark() {
    for (auto& expression : gpu_) {
      expression->close();
    }
  }

  const std::string& name() const {
    return name_;
  }

  void validate() {
    std::vector<VectorPtr> cpuResults(1);
    cpu_->eval(rows_, *evalCtx_, cpuResults);
    BaseVector::flattenVector(cpuResults[0]);
    auto expected = cpuResults[0];
    auto numericExpected = expected;
    if (!spec_.fixtures.empty()) {
      expected = expectedFixtures(0);
      numericExpected = expectedFixtures(1);
      differences(expectedFixtures(2), cpuResults[0], true, "cpu_expected");
    }
    const auto cpuDiff = differences(expected, cpuResults[0], false, "cpu");
    for (int mode = 0; mode < 2; ++mode) {
      auto result = gpu_[mode]->eval(views_, stream_, mr_);
      VELOX_CHECK(
          asView(result).type() == veloxToCudfDataType(spec_.target),
          "{} has an unexpected GPU storage type or scale",
          name_);
      const auto outputType = ROW({"c0"}, {spec_.target});
      auto output = with_arrow::toVeloxColumn(
          cudf::table_view({asView(result)}),
          pool_.get(),
          outputType,
          "",
          stream_,
          mr_);
      stream_.sync();
      // cuDF preserves decimal storage and scale, but its Arrow export uses
      // the storage type's maximum precision. Restore the expression's
      // declared precision after checking the physical GPU type above.
      output->setType(outputType);
      differences(
          mode ? numericExpected : expected,
          output->childAt(0),
          true,
          mode ? "numeric" : "exact");
    }
    const auto numericDiff =
        differences(expected, numericExpected, false, "numeric_documented");
    std::cout << fmt::format(
        "VALIDATED\t{}\trows={}\tcpu_value_differences={}\tcpu_validity_differences={}"
        "\tnumeric_value_differences={}\tnumeric_validity_differences={}"
        "\texact_unexpected=0\tnumeric_unexpected=0\tevaluator=FunctionExpression\n",
        name_,
        input_->size(),
        cpuDiff.first,
        cpuDiff.second,
        numericDiff.first,
        numericDiff.second);
    for (int mode = 0; mode < 3; ++mode) {
      run(mode, FLAGS_cast_warmups);
    }
  }

  unsigned run(int mode, int iterations) {
    if (mode == 0) {
      // Match ExpressionBenchmarkBuilder's CPU buffer reuse and flattening.
      // EvalCtx/SelectivityVector construction stays outside the timer; the
      // first output allocation, prepareForReuse and final destruction stay in.
      std::vector<VectorPtr> results(1);
      for (int i = 0; i < iterations; ++i) {
        cpu_->eval(rows_, *evalCtx_, results);
        BaseVector::flattenVector(results[0]);
        folly::doNotOptimizeAway(results[0]);
        results[0]->prepareForReuse();
      }
    } else {
      for (int i = 0; i < iterations; ++i) {
        {
          auto result = gpu_[mode - 1]->eval(views_, stream_, mr_);
          folly::doNotOptimizeAway(result);
        }
        // Includes materialized output allocation, destruction and completion
        // of asynchronous frees. No input/output transfer is inside this loop.
        stream_.sync();
      }
    }
    return iterations;
  }

 private:
  uint64_t inputBits(vector_size_t row) const {
    const auto& input = input_->childAt(0);
    if (spec_.source->isReal()) {
      return std::bit_cast<uint32_t>(
          input->asFlatVector<float>()->valueAt(row));
    }
    if (spec_.source->isDouble()) {
      return std::bit_cast<uint64_t>(
          input->asFlatVector<double>()->valueAt(row));
    }
    return 0;
  }

  const DecimalFixture& fixture(vector_size_t row) const {
    return spec_.fixtures[row % spec_.fixtures.size()];
  }

  VectorPtr expectedFixtures(int mode) const {
    auto result = BaseVector::create(spec_.target, input_->size(), pool_.get());
    for (vector_size_t row = 0; row < input_->size(); ++row) {
      const auto& f = fixture(row);
      const char* value = mode == 0 ? f.exact : mode == 1 ? f.numeric : f.cpu;
      if (!value || input_->childAt(0)->isNullAt(row)) {
        result->setNull(row, true);
      } else if (spec_.target->isShortDecimal()) {
        result->asFlatVector<int64_t>()->set(
            row, static_cast<int64_t>(parseUnscaled(value)));
      } else {
        result->asFlatVector<int128_t>()->set(row, parseUnscaled(value));
      }
    }
    return result;
  }

  std::pair<int64_t, int64_t> differences(
      const VectorPtr& expected,
      const VectorPtr& actual,
      bool fail,
      const std::string& mode) const {
    VELOX_CHECK_EQ(actual->size(), expected->size());
    VELOX_CHECK(
        actual->type()->equivalent(*expected->type()),
        "{} {}: expected type {}, actual type {}",
        name_,
        mode,
        expected->type()->toString(),
        actual->type()->toString());
    int64_t values = 0;
    int64_t validity = 0;
    std::vector<bool> printed(spec_.fixtures.size(), false);
    for (vector_size_t row = 0; row < input_->size(); ++row) {
      VELOX_CHECK(
          !input_->childAt(0)->isNullAt(row) || actual->isNullAt(row),
          "{} {} lost input NULL at {}",
          name_,
          mode,
          row);
      const bool nullDifference =
          expected->isNullAt(row) != actual->isNullAt(row);
      const bool valueDifference = !expected->isNullAt(row) &&
          !actual->isNullAt(row) &&
          !expected->equalValueAt(actual.get(), row, row);
      validity += nullDifference;
      values += valueDifference;
      if (nullDifference || valueDifference) {
        VELOX_CHECK(
            !fail,
            "{} {} {} -> {} row {} bits={:016x}: expected {}, actual {}",
            name_,
            mode,
            spec_.source->toString(),
            spec_.target->toString(),
            row,
            inputBits(row),
            expected->toString(row),
            actual->toString(row));
        if (!spec_.fixtures.empty() && !printed[row % spec_.fixtures.size()]) {
          printed[row % spec_.fixtures.size()] = true;
          std::cout << fmt::format(
              "DIFFERENCE\t{}\t{}\tsource={}\ttarget={}\tbits={:016x}\texpected={}\tactual={}\n",
              name_,
              mode,
              spec_.source->toString(),
              spec_.target->toString(),
              fixture(row).bits,
              expected->toString(row),
              actual->toString(row));
        }
      }
    }
    return {values, validity};
  }

  CastCase spec_;
  std::string name_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  core::ExecCtx execCtx_;
  SelectivityVector rows_;
  cuda::stream_ref stream_;
  rmm::device_async_resource_ref mr_;
  RowVectorPtr input_;
  std::unique_ptr<exec::ExprSet> cpu_;
  std::unique_ptr<exec::EvalCtx> evalCtx_;
  CudfExpressionPtr gpu_[2];
  std::unique_ptr<cudf::table> deviceInput_;
  std::vector<cudf::column_view> views_;
};

void runBenchmarks() {
  VELOX_CHECK_GT(FLAGS_cast_iterations, 0);
  VELOX_CHECK_GE(FLAGS_cast_warmups, 0);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  functions::sparksql::registerFunctions();
  auto& config = CudfConfig::getInstance();
  VELOX_CHECK(!config.numericFloatToDecimalEnabled);
  config.allowCpuFallback = false;
  registerCudf();
  registerSparkFunctions("");
  int device = -1;
  cudaDeviceProp properties{};
  VELOX_CHECK(cudaGetDevice(&device) == cudaSuccess);
  VELOX_CHECK(cudaGetDeviceProperties(&properties, device) == cudaSuccess);
  char pci[32]{};
  VELOX_CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), device) == cudaSuccess);
  std::cout << fmt::format(
      "GPU device={} name={} pci={} uuid=", device, properties.name, pci);
  for (const auto byte : properties.uuid.bytes) {
    std::cout << fmt::format("{:02x}", static_cast<unsigned char>(byte));
  }
  std::cout
      << "\nTIMING unit=per_conversion_call; GPU includes output allocation/destruction and stream synchronization; "
         "CPU reuses result buffers within each sample; transfers/construction/validation excluded\n";
  std::vector<std::shared_ptr<CastBenchmark>> benchmarks;
  for (const auto& spec : castCases()) {
    for (const auto size : {1000, 65536, 1048576}) {
      for (const bool nulls : {false, true}) {
        const auto name =
            fmt::format("{}_rows{}_null{}", spec.name, size, nulls ? 20 : 0);
        if (name.find(FLAGS_cast_case_filter) == std::string::npos) {
          continue;
        }
        auto benchmark = std::make_shared<CastBenchmark>(spec, size, nulls);
        benchmark->validate();
        for (int mode = 0; mode < 3; ++mode) {
          static constexpr const char* names[]{
              "cpu_spark", "gpu_exact", "gpu_numeric"};
          folly::addBenchmark(
              __FILE__,
              benchmark->name() + "/" + names[mode],
              [benchmark = benchmark.get(), mode]() -> unsigned {
                return benchmark->run(mode, FLAGS_cast_iterations);
              });
        }
        benchmarks.push_back(std::move(benchmark));
      }
    }
  }
  VELOX_CHECK(!benchmarks.empty(), "No cases match cast_case_filter");
  std::cout << fmt::format(
      "VALIDATION COMPLETE cases={} engines=3 warmups={} iterations={}\n",
      benchmarks.size(),
      FLAGS_cast_warmups,
      FLAGS_cast_iterations);
  if (!FLAGS_cast_validate_only) {
    folly::runBenchmarks();
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  facebook::velox::cudf_velox::runBenchmarks();
  return 0;
}
