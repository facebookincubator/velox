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
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Validation.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/DateTruncFunction.h"
#include "velox/experimental/cudf/expression/DecimalExpressionKernels.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluatorRegistry.h"
#include "velox/experimental/cudf/expression/NullMask.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/ExprConstants.h"
#include "velox/expression/ExprOptimizer.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/functions/FunctionRegistry.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/Time.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/ConstantVector.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <memory>
#include <string_view>

namespace facebook::velox::cudf_velox {

// Implementation details in anonymous namespace
namespace {

bool decimalScalarIsZero(const cudf::scalar& scalar, cuda::stream_ref stream) {
  if (!scalar.is_valid(stream)) {
    return false;
  }
  if (scalar.type().id() == cudf::type_id::DECIMAL64) {
    auto const& dec =
        static_cast<cudf::fixed_point_scalar<numeric::decimal64> const&>(
            scalar);
    return dec.value(stream) == 0;
  }
  if (scalar.type().id() == cudf::type_id::DECIMAL128) {
    auto const& dec =
        static_cast<cudf::fixed_point_scalar<numeric::decimal128> const&>(
            scalar);
    return dec.value(stream) == 0;
  }
  return false;
}

bool hasDecimalZero(
    const cudf::column_view& col,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (col.is_empty()) {
    return false;
  }
  std::unique_ptr<cudf::scalar> zero;
  auto scale = numeric::scale_type{col.type().scale()};
  if (col.type().id() == cudf::type_id::DECIMAL64) {
    zero =
        cudf::make_fixed_point_scalar<numeric::decimal64>(0, scale, stream, mr);
  } else if (col.type().id() == cudf::type_id::DECIMAL128) {
    zero = cudf::make_fixed_point_scalar<numeric::decimal128>(
        0, scale, stream, mr);
  } else {
    return false;
  }

  auto equals = cudf::binary_operation(
      col,
      *zero,
      cudf::binary_operator::EQUAL,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  auto anyAgg = cudf::make_any_aggregation<cudf::reduce_aggregation>();
  auto anyScalar = cudf::reduce(
      equals->view(),
      *anyAgg,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  auto const& boolScalar =
      static_cast<cudf::numeric_scalar<bool> const&>(*anyScalar);
  return boolScalar.is_valid(stream) && boolScalar.value(stream);
}

std::unique_ptr<cudf::scalar> castDecimalScalar(
    const cudf::scalar& src,
    cudf::data_type targetType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (!src.is_valid(stream)) {
    VELOX_CHECK(
        targetType.id() == cudf::type_id::DECIMAL64 ||
            targetType.id() == cudf::type_id::DECIMAL128,
        "castDecimalScalar: target must be DECIMAL64 or DECIMAL128");
    if (targetType.id() == cudf::type_id::DECIMAL128) {
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
          0, numeric::scale_type{targetType.scale()}, false, stream, mr);
    }
    return std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
        0, numeric::scale_type{targetType.scale()}, false, stream, mr);
  }

  __int128_t rep;
  if (src.type().id() == cudf::type_id::DECIMAL64) {
    auto const& dec =
        static_cast<cudf::fixed_point_scalar<numeric::decimal64> const&>(src);
    rep = static_cast<int64_t>(dec.value(stream));
  } else {
    auto const& dec =
        static_cast<cudf::fixed_point_scalar<numeric::decimal128> const&>(src);
    rep = static_cast<__int128_t>(dec.value(stream));
  }

  int32_t scaleDiff = src.type().scale() - targetType.scale();
  if (scaleDiff > 0) {
    for (int32_t i = 0; i < scaleDiff; ++i) {
      rep *= 10;
    }
  } else if (scaleDiff < 0) {
    for (int32_t i = 0; i < -scaleDiff; ++i) {
      rep /= 10;
    }
  }

  if (targetType.id() == cudf::type_id::DECIMAL128) {
    return cudf::make_fixed_point_scalar<numeric::decimal128>(
        rep, numeric::scale_type{targetType.scale()}, stream, mr);
  }
  return cudf::make_fixed_point_scalar<numeric::decimal64>(
      static_cast<int64_t>(rep),
      numeric::scale_type{targetType.scale()},
      stream,
      mr);
}

int32_t getDecimalPrecision(const TypePtr& type) {
  if (type->isShortDecimal()) {
    return type->asShortDecimal().precision();
  }
  if (type->isLongDecimal()) {
    return type->asLongDecimal().precision();
  }
  VELOX_FAIL("Expected decimal type, got {}", type->toString());
}

bool isCheckedDecimalArithmeticOp(cudf::binary_operator op) {
  return op == cudf::binary_operator::ADD || op == cudf::binary_operator::SUB ||
      op == cudf::binary_operator::MUL || op == cudf::binary_operator::MOD ||
      op == cudf::binary_operator::DIV;
}

/// Materialise a ConstantTypedExpr to a VectorPtr.
VectorPtr toConstantVector(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool) {
  VELOX_CHECK(expr->isConstantKind());
  const auto* c = expr->asUnchecked<core::ConstantTypedExpr>();
  return c->hasValueVector() ? c->valueVector() : c->toConstantVector(pool);
}

std::string constantToString(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool) {
  return toConstantVector(expr, pool)->toString(0);
}

std::unordered_map<std::string, std::vector<CudfFunctionSpec>>&
getCudfFunctionRegistry() {
  static std::unordered_map<std::string, std::vector<CudfFunctionSpec>>
      registry;
  return registry;
}

static bool matchCallAgainstSignatures(
    const core::TypedExprPtr& call,
    const std::vector<exec::FunctionSignaturePtr>& sigs) {
  const auto n = call->inputs().size();
  std::vector<TypePtr> argTypes;
  argTypes.reserve(n);
  for (const auto& in : call->inputs()) {
    argTypes.push_back(in->type());
  }
  for (const auto& sig : sigs) {
    exec::SignatureBinder binder(*sig, argTypes, TypeCoercer::defaults());
    if (!binder.tryBind()) {
      continue;
    }
    // binder does not confirm whether positional arguments are
    // constants(scalars) as expected. we have to check manually
    const auto& constArgs = sig->constantArguments();
    const size_t fixed = std::min(constArgs.size(), n);
    bool ok = true;
    for (size_t i = 0; i < fixed; ++i) {
      if (constArgs[i] && !call->inputs()[i]->isConstantKind()) {
        ok = false;
        break;
      }
    }
    if (!ok) {
      continue;
    }
    return true;
  }
  return false;
}

std::optional<std::string> rootFieldName(const core::TypedExprPtr& expr) {
  auto path = extractFieldPath(expr);
  if (!path.has_value() || path->empty()) {
    return std::nullopt;
  }
  return path->front();
}

bool isInputFieldReference(const core::TypedExprPtr& expr) {
  return rootFieldName(expr).has_value();
}

/// True for the special forms registered as SwitchFunction.
bool isConditionalFunction(const core::TypedExprPtr& expr) {
  if (expr->kind() != core::ExprKind::kCall) {
    return false;
  }
  const auto& name = expr->asUnchecked<core::CallTypedExpr>()->name();
  return name == "switch" || name == "if";
}

std::string_view unprefixedFunctionName(const core::CallTypedExpr& call) {
  std::string_view name = call.name();
  const auto& prefix = CudfConfig::getInstance().functionNamePrefix;
  if (name.starts_with(prefix)) {
    name.remove_prefix(prefix.size());
  }
  return name;
}

/// Whether masking every referenced input field null guarantees a null result.
/// Default-null functions preserve the activity marker when at least one of
/// their arguments does. Constants do not carry the marker.
bool preservesMaskedRowNull(const core::TypedExprPtr& expr) {
  if (isInputFieldReference(expr)) {
    return true;
  }
  if (expr->isConstantKind() || expr->isInputKind()) {
    return false;
  }
  if (expr->isCastKind()) {
    return !expr->inputs().empty() && preservesMaskedRowNull(expr->inputs()[0]);
  }
  if (expr->kind() != core::ExprKind::kCall) {
    return false;
  }

  const auto* call = expr->asUnchecked<core::CallTypedExpr>();
  const auto name = unprefixedFunctionName(*call);
  // Velox registers these as default-null, but GreatestLeastFunction uses
  // NULL_MAX / NULL_MIN, which skip null operands.
  if (name == "greatest" || name == "least") {
    return false;
  }
  auto defaultNullBehavior = isDefaultNullBehavior(call->name());
  if (!defaultNullBehavior.has_value()) {
    defaultNullBehavior = isDefaultNullBehavior(std::string{name});
  }
  if (!defaultNullBehavior.value_or(false)) {
    return false;
  }
  return std::ranges::any_of(expr->inputs(), [](const auto& input) {
    return preservesMaskedRowNull(input);
  });
}

bool isCheckedDecimalArithmetic(const core::TypedExprPtr& expr) {
  if (expr->kind() != core::ExprKind::kCall || !expr->type()->isDecimal()) {
    return false;
  }
  const auto name =
      unprefixedFunctionName(*expr->asUnchecked<core::CallTypedExpr>());
  return name == "plus" || name == "add" || name == "minus" ||
      name == "subtract" || name == "multiply" || name == "divide" ||
      name == "mod";
}

/// Calls whose cuDF implementation raises on row data but skips null rows.
/// Branch masking exists to keep the rows a conditional discards away from
/// these.
bool isFailFastCall(const core::TypedExprPtr& expr) {
  if (isCheckedDecimalArithmetic(expr)) {
    return true;
  }
  if (expr->kind() != core::ExprKind::kCall) {
    return false;
  }
  const auto name =
      unprefixedFunctionName(*expr->asUnchecked<core::CallTypedExpr>());
  return name == "subscript" || name == "element_at" || name == "get" ||
      name == "like";
}

bool containsFailFastCall(const core::TypedExprPtr& expr) {
  if (isFailFastCall(expr)) {
    return true;
  }
  if (expr->isLambdaKind()) {
    return containsFailFastCall(
        expr->asUnchecked<core::LambdaTypedExpr>()->body());
  }
  return std::ranges::any_of(expr->inputs(), [](const auto& input) {
    return containsFailFastCall(input);
  });
}

/// A fail-fast call is safe under branch masking only if at least one of its
/// inputs still carries the masked-row null to that call.
bool failFastCallsPreserveMaskedRows(const core::TypedExprPtr& expr) {
  if (isFailFastCall(expr) && !preservesMaskedRowNull(expr)) {
    return false;
  }
  return std::ranges::all_of(expr->inputs(), [](const auto& input) {
    return failFastCallsPreserveMaskedRows(input);
  });
}

void collectReferencedInputFields(
    const core::TypedExprPtr& expr,
    std::unordered_set<std::string>& fields,
    const std::unordered_set<std::string>& lambdaInputs = {}) {
  if (expr == nullptr) {
    return;
  }

  if (auto root = rootFieldName(expr);
      root.has_value() && !lambdaInputs.count(*root)) {
    fields.insert(*root);
  }

  if (expr->isLambdaKind()) {
    const auto* lambda = expr->asUnchecked<core::LambdaTypedExpr>();
    auto scopedLambdaInputs = lambdaInputs;
    for (const auto& name : lambda->signature()->names()) {
      scopedLambdaInputs.insert(name);
    }
    collectReferencedInputFields(lambda->body(), fields, scopedLambdaInputs);
    return;
  }

  for (const auto& input : expr->inputs()) {
    collectReferencedInputFields(input, fields, lambdaInputs);
  }
}

// Returns the highest-priority evaluator entry that can handle `expr`, or
// nullptr when none can.
const CudfExpressionEvaluatorEntry* findBestEvaluator(
    const core::TypedExprPtr& expr) {
  ensureBuiltinExpressionEvaluatorsRegistered();
  const auto& registry = getCudfExpressionEvaluatorRegistry();

  const CudfExpressionEvaluatorEntry* best = nullptr;
  for (const auto& [name, entry] : registry) {
    if (entry.canEvaluate && entry.canEvaluate(expr)) {
      if (best == nullptr || entry.priority > best->priority) {
        best = &entry;
      }
    }
  }
  return best;
}

// Returns true if the tree holds a CAST from VARCHAR, which fails on a string
// that spells no value of the target type; a TRY_CAST gives NULL instead.
bool containsFailingCastFromVarchar(const core::TypedExprPtr& expr) {
  if (expr->isCastKind() &&
      !expr->asUnchecked<core::CastTypedExpr>()->isTryCast() &&
      !expr->inputs().empty() && expr->inputs()[0]->type()->isVarchar()) {
    return true;
  }
  for (const auto& input : expr->inputs()) {
    if (containsFailingCastFromVarchar(input)) {
      return true;
    }
  }
  return false;
}

// Returns true if 'expr' holds a CAST from VARCHAR that the GPU would evaluate
// on rows the CPU settles without it. On the CPU a CAST error never surfaces
// on such a row: and and or hold the error and clear it on any row another
// argument decides (ConjunctExpr::finalizeErrors), coalesce evaluates a later
// argument only on the rows still null, a default-null function skips the
// rows where another argument is NULL (Expr::removeSureNulls and
// evalArgsDefaultNulls), and if and switch evaluate a branch only on the rows
// its condition selects. The GPU evaluates every argument over every row, and
// CastFunction raises on any string that spells no value, so a CAST beside
// another non-constant argument, or beside a NULL constant, keeps the tree on
// the CPU. A CAST in a branch of if or switch stays there too: the evaluator
// hands a branch its inputs with the other rows nulled, which spares a cast
// whose input stays NULL on those rows, such as a cast of a column, but not
// one whose input a branch fills in, as in if(c0 is null, 0,
// cast(coalesce(c0, 'x') as bigint)).
// TODO: Let a cast in a branch run on the GPU once the evaluator skips those
// rows instead of nulling their inputs
// (https://github.com/facebookincubator/velox/issues/19677).
bool castFromVarcharNeedsCpu(const core::TypedExprPtr& expr) {
  if (!expr->isCallKind()) {
    return false;
  }
  size_t nonConstantOrNullInputs{0};
  for (const auto& input : expr->inputs()) {
    nonConstantOrNullInputs += !input->isConstantKind() ||
        input->asUnchecked<core::ConstantTypedExpr>()->isNull();
  }
  return nonConstantOrNullInputs >= 2 && containsFailingCastFromVarchar(expr);
}

// Recursive, context-free check that some cuDF evaluator supports every node in
// the expression tree. canExprRunOnGpu wraps this with expression optimization
// and the session-dependent fallbacks.
bool canBeEvaluatedByCudf(const core::TypedExprPtr& expr) {
  ensureBuiltinExpressionEvaluatorsRegistered();

  const auto& registry = getCudfExpressionEvaluatorRegistry();

  bool supported = false;
  for (const auto& [name, entry] : registry) {
    if (entry.canEvaluate && entry.canEvaluate(expr)) {
      supported = true;
      break;
    }
  }
  if (!supported) {
    LOG_FALLBACK(expr->toString());
    return false;
  }

  if (castFromVarcharNeedsCpu(expr)) {
    LOG_FALLBACK(
        fmt::format(
            "CAST from VARCHAR beside another argument stays on the CPU: {}",
            expr->toString()));
    return false;
  }

  for (const auto& input : expr->inputs()) {
    if (input->isConstantKind() || input->isInputKind()) {
      continue;
    }
    if (!canBeEvaluatedByCudf(input)) {
      return false;
    }
  }

  return true;
}

} // namespace

void checkAllTrue(
    cudf::column_view cond,
    std::string_view userMessage,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (cond.is_empty() || cond.null_count() == cond.size()) {
    return;
  }

  const auto boolType = cudf::data_type(cudf::type_id::BOOL8);
  auto allTrue = cudf::reduce(
      cond,
      *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
      boolType,
      stream,
      mr);
  auto* result = static_cast<cudf::scalar_type_t<bool>*>(allTrue.get());
  VELOX_USER_CHECK(
      result->is_valid(stream) && result->value(stream), "{}", userMessage);
}

class SplitFunction : public CudfFunction {
 public:
  SplitFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool) {
    VELOX_CHECK(
        expr->inputs()[1]->isConstantKind(),
        "split delimiter must be a constant");
    delimiter_ = constantToString(expr->inputs()[1], pool);

    VELOX_CHECK(
        expr->inputs()[2]->isConstantKind(), "split limit must be a constant");
    maxSplitCount_ = std::stoll(constantToString(expr->inputs()[2], pool));

    // Presto specifies maxSplitCount as the maximum size of the returned array
    // while cuDF understands the parameter as how many splits can it perform.
    maxSplitCount_ -= 1;
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    cudf::string_scalar delimiterScalar(delimiter_, true, stream, mr);
    return cudf::strings::split_record(
        inputCol, delimiterScalar, maxSplitCount_, stream, mr);
  };

 private:
  std::string delimiter_;
  cudf::size_type maxSplitCount_;
};

// Lists the characters the Presto cast hooks skip around a floating-point
// literal: ASCII whitespace only (stringImpl::isAsciiWhiteSpace()).
constexpr std::string_view kFloatingPointCastTrimCharacters{
    "\t\n\v\f\r\x1c\x1d\x1e\x1f "};

// Casts a string to the integral type T, whose maximum is kMax, with the rules
// of the Presto cast (util::trimWhiteSpace() and folly::tryTo()): after ASCII
// whitespace and control characters, bytes 1 to 32, are dropped from both
// ends, an optional sign and digits whose magnitude is at most kMax, or kMax +
// 1 when negative. A non-ASCII byte anywhere rejects the string, as on the
// CPU, where PrestoCastPolicy::throwOnUnicode fails the cast before
// util::trimWhiteSpace() would drop Unicode whitespace; that is a stopgap
// (https://github.com/facebookincubator/velox/issues/10803) and this kernel
// must trim Unicode whitespace once the CPU does. Writes the value and whether
// the string is one. makePrestoIntegralUdf() defines T and kMax ahead of it.
constexpr std::string_view kPrestoIntegralUdf = R"***(
__device__ void presto_integral(T* value, bool* valid, cudf::string_view s) {
  unsigned char const* d = reinterpret_cast<unsigned char const*>(s.data());
  int n = s.size_bytes();
  int i = 0;
  while (i < n && d[i] >= 1 && d[i] <= 32) {
    ++i;
  }
  while (n > i && d[n - 1] >= 1 && d[n - 1] <= 32) {
    --n;
  }
  bool const negative = i < n && d[i] == '-';
  if (i < n && (d[i] == '+' || d[i] == '-')) {
    ++i;
  }
  unsigned long long const limit = kMax + (negative ? 1ULL : 0ULL);
  unsigned long long magnitude = 0;
  bool ok = i < n;
  for (; ok && i < n; ++i) {
    unsigned char const c = d[i];
    if (c < '0' || c > '9') {
      ok = false;
      break;
    }
    unsigned long long const digit = c - '0';
    if (magnitude > (limit - digit) / 10) {
      ok = false;
      break;
    }
    magnitude = magnitude * 10 + digit;
  }
  *valid = ok;
  *value = !ok ? T{0}
      : negative ? static_cast<T>(0ULL - magnitude)
                 : static_cast<T>(magnitude);
}
)***";

// Returns kPrestoIntegralUdf for the integral 'type', with T and kMax defined
// ahead of it.
std::string makePrestoIntegralUdf(cudf::data_type type) {
  const char* typeName = nullptr;
  uint64_t max = 0;
  switch (type.id()) {
    case cudf::type_id::INT8:
      typeName = "int8_t";
      max = std::numeric_limits<int8_t>::max();
      break;
    case cudf::type_id::INT16:
      typeName = "int16_t";
      max = std::numeric_limits<int16_t>::max();
      break;
    case cudf::type_id::INT32:
      typeName = "int32_t";
      max = std::numeric_limits<int32_t>::max();
      break;
    case cudf::type_id::INT64:
      typeName = "int64_t";
      max = std::numeric_limits<int64_t>::max();
      break;
    default:
      VELOX_FAIL(
          "Cast from VARCHAR to cuDF type {} is not supported",
          static_cast<int32_t>(type.id()));
  }
  return fmt::format(
      "using T = {};\nconstexpr unsigned long long kMax = {}ULL;\n{}",
      typeName,
      max,
      kPrestoIntegralUdf);
}

// Decides whether a string is a REAL or DOUBLE literal to the Presto cast
// (PrestoCastHooks::doCastToFloatingPoint()): after ASCII whitespace is
// dropped from both ends, an optional sign, then digits with an optional
// fraction or a fraction alone, then an optional exponent with digits; or NaN
// or Infinity in exactly that case.
constexpr const char* kPrestoFloatingPointValidUdf = R"***(
__device__ void presto_floating_point_valid(bool* valid, cudf::string_view s) {
  unsigned char const* d = reinterpret_cast<unsigned char const*>(s.data());
  int n = s.size_bytes();
  int i = 0;
  auto isSpace = [](unsigned char c) {
    return (c >= 9 && c <= 13) || (c >= 28 && c <= 32);
  };
  while (i < n && isSpace(d[i])) {
    ++i;
  }
  while (n > i && isSpace(d[n - 1])) {
    --n;
  }
  if (i < n && (d[i] == '+' || d[i] == '-')) {
    ++i;
  }
  int const rest = n - i;
  if (rest == 3 && d[i] == 'N' && d[i + 1] == 'a' && d[i + 2] == 'N') {
    *valid = true;
    return;
  }
  if (rest == 8 && d[i] == 'I' && d[i + 1] == 'n' && d[i + 2] == 'f' &&
      d[i + 3] == 'i' && d[i + 4] == 'n' && d[i + 5] == 'i' &&
      d[i + 6] == 't' && d[i + 7] == 'y') {
    *valid = true;
    return;
  }
  int digits = 0;
  for (; i < n && d[i] >= '0' && d[i] <= '9'; ++i) {
    ++digits;
  }
  if (i < n && d[i] == '.') {
    for (++i; i < n && d[i] >= '0' && d[i] <= '9'; ++i) {
      ++digits;
    }
  }
  if (digits == 0) {
    *valid = false;
    return;
  }
  if (i < n && (d[i] == 'e' || d[i] == 'E')) {
    ++i;
    if (i < n && (d[i] == '+' || d[i] == '-')) {
      ++i;
    }
    int exponentDigits = 0;
    for (; i < n && d[i] >= '0' && d[i] <= '9'; ++i) {
      ++exponentDigits;
    }
    if (exponentDigits == 0) {
      *valid = false;
      return;
    }
  }
  *valid = i == n;
}
)***";

// Casts a string to BOOLEAN with the rules of the Presto cast
// (util::castToBoolean()): t, f, 1, 0, true and false in any case, with
// nothing trimmed. Writes the value and whether the string is one.
constexpr const char* kPrestoBooleanUdf = R"***(
__device__ void presto_boolean(bool* value, bool* valid, cudf::string_view s) {
  unsigned char const* d = reinterpret_cast<unsigned char const*>(s.data());
  int const n = s.size_bytes();
  auto lower = [](unsigned char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<unsigned char>(c + ('a' - 'A'))
                                  : c;
  };
  if (n == 1) {
    unsigned char const c = lower(d[0]);
    if (c == 't' || c == '1') {
      *value = true;
      *valid = true;
      return;
    }
    if (c == 'f' || c == '0') {
      *value = false;
      *valid = true;
      return;
    }
  } else if (
      n == 4 && lower(d[0]) == 't' && lower(d[1]) == 'r' &&
      lower(d[2]) == 'u' && lower(d[3]) == 'e') {
    *value = true;
    *valid = true;
    return;
  } else if (
      n == 5 && lower(d[0]) == 'f' && lower(d[1]) == 'a' &&
      lower(d[2]) == 'l' && lower(d[3]) == 's' && lower(d[4]) == 'e') {
    *value = false;
    *valid = true;
    return;
  }
  *value = false;
  *valid = false;
}
)***";

// Returns true if CastFunction casts VARCHAR to 'target'. cudf::cast has no
// string source, so CastFunction parses the strings itself for these targets.
bool isCastFromVarcharSupported(cudf::data_type target) {
  return cudf::is_boolean(target) || cudf::is_integral_not_bool(target) ||
      cudf::is_floating_point(target);
}

// Returns true if 'type' is the plain type of its kind. A logical type under
// another name, such as JSON or INTERVAL DAY TO SECOND, has its own cast rules
// on the CPU and stays there.
bool isPlainType(const TypePtr& type) {
  return std::string_view(type->name()) == type->kindName();
}

// Whether the GPU casts VARCHAR at all: CastFunction follows Presto's rules,
// so the Presto function registration turns it on and the Spark one off.
// TODO: Register cast as a special form per dialect in place of this switch
// (https://github.com/facebookincubator/velox/issues/19748).
bool castFromVarcharEnabled = false;

// Returns true if CastFunction casts 'sourceType' to 'targetType'.
bool isCastFromVarcharEnabled(
    const TypePtr& sourceType,
    const TypePtr& targetType) {
  if (!castFromVarcharEnabled || !sourceType->isVarchar() ||
      !isPlainType(sourceType) || !isPlainType(targetType)) {
    return false;
  }
  const auto target = cudf_velox::tryVeloxToCudfDataType(targetType);
  return target.has_value() && isCastFromVarcharSupported(*target);
}

class CastFunction : public CudfFunction {
 public:
  CastFunction(const core::TypedExprPtr& expr) {
    VELOX_CHECK_EQ(expr->inputs().size(), 1, "cast expects exactly 1 input");

    const auto& sourceType = expr->inputs()[0]->type();
    targetCudfType_ = cudf_velox::veloxToCudfDataType(expr->type());
    fromVarchar_ = sourceType->isVarchar();
    if (fromVarchar_) {
      VELOX_CHECK(
          isCastFromVarcharEnabled(sourceType, expr->type()),
          "Cast from {} to {} is not supported",
          sourceType->toString(),
          expr->type()->toString());
      isTryCast_ = expr->isCastKind() &&
          expr->asUnchecked<core::CastTypedExpr>()->isTryCast();
      targetTypeName_ = expr->type()->toString();
      if (cudf::is_integral_not_bool(targetCudfType_)) {
        integralUdf_ = makePrestoIntegralUdf(targetCudfType_);
      }
      return;
    }
    const auto sourceCudfType = cudf_velox::veloxToCudfDataType(sourceType);
    VELOX_CHECK(
        cudf::is_supported_cast(sourceCudfType, targetCudfType_),
        "Cast from {} to {} is not supported",
        sourceType->toString(),
        expr->type()->toString());
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    if (fromVarchar_) {
      return castFromVarchar(inputCol, stream, mr);
    }
    return cudf::cast(inputCol, targetCudfType_, stream, mr);
  }

 private:
  // Runs 'udf', a CUDA function of one cudf::string_view that writes one value
  // per entry of 'outputTypes', over the rows of 'input', allocating every
  // output from 'mr'. A null row stays null in every output.
  static std::vector<std::unique_ptr<cudf::column>> transformStrings(
      const std::string& udf,
      cudf::column_view input,
      std::initializer_list<cudf::data_type> outputTypes,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) {
    const cudf::transform_input inputs[] = {input};
    std::vector<cudf::transform_output> outputs;
    outputs.reserve(outputTypes.size());
    for (const auto type : outputTypes) {
      outputs.push_back({type, cudf::output_nullability::PRESERVE});
    }
    return cudf::transform(
               udf,
               cudf::udf_source_type::CUDA,
               cudf::null_aware::NO,
               std::nullopt,
               inputs,
               outputs,
               {},
               std::nullopt,
               stream,
               mr)
        ->release();
  }

  // Casts strings with the rules of the CPU's Presto cast: the string, less
  // the whitespace the cast ignores, must spell a value of the target type. A
  // NULL stays NULL; any other string fails CAST and becomes NULL under
  // TRY_CAST.
  std::unique_ptr<cudf::column> castFromVarchar(
      cudf::column_view input,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const {
    auto tempMr = get_temp_mr();
    const auto boolType = cudf::data_type{cudf::type_id::BOOL8};
    // 'valid' is false where the string spells no value and null where the
    // input is null; 'values' holds the parsed values with the input's nulls.
    // A kernel that writes both allocates 'valid' from 'mr' along with them.
    std::unique_ptr<cudf::column> valid;
    std::unique_ptr<cudf::column> values;
    if (cudf::is_boolean(targetCudfType_)) {
      auto columns = transformStrings(
          kPrestoBooleanUdf, input, {boolType, boolType}, stream, mr);
      values = std::move(columns[0]);
      valid = std::move(columns[1]);
    } else if (cudf::is_floating_point(targetCudfType_)) {
      // TODO: Validate, trim and parse the strings in one kernel
      // (https://github.com/facebookincubator/velox/issues/19749).
      valid = std::move(
          transformStrings(
              kPrestoFloatingPointValidUdf, input, {boolType}, stream, tempMr)
              .front());
      // cudf::strings::to_floats() parses the trimmed string. It scales the
      // digits by a power of ten in double arithmetic, which is exact for a
      // significand of at most 15 digits scaled by at most 10^22 when exp10()
      // returns that power exactly, and can differ from the CPU's correctly
      // rounded parse in the last bit beyond that; a REAL is rounded through
      // a double.
      auto stripped = cudf::strings::strip(
          cudf::strings_column_view(input),
          cudf::strings::side_type::BOTH,
          cudf::string_scalar(
              std::string(kFloatingPointCastTrimCharacters),
              true,
              stream,
              tempMr),
          stream,
          tempMr);
      values = cudf::strings::to_floats(
          cudf::strings_column_view(stripped->view()),
          targetCudfType_,
          stream,
          mr);
    } else {
      auto columns = transformStrings(
          integralUdf_, input, {targetCudfType_, boolType}, stream, mr);
      values = std::move(columns[0]);
      valid = std::move(columns[1]);
    }

    if (isTryCast_) {
      // The rows that spell no value become NULL. bools_to_mask() clears a
      // false and a null alike, so a null input stays NULL as well. A count
      // of zero means every input row spelled a value, and 'values' stays as
      // it is.
      auto [nullMask, nullCount] =
          cudf::bools_to_mask(valid->view(), stream, mr);
      if (nullCount > 0) {
        values->set_null_mask(std::move(*nullMask), nullCount);
      }
      return values;
    }

    // The rows whose string spells no value; a null input is not one of them.
    auto invalid = cudf::binary_operation(
        valid->view(),
        cudf::numeric_scalar<bool>(false, true, stream, tempMr),
        cudf::binary_operator::NULL_EQUALS,
        boolType,
        stream,
        tempMr);
    auto anyInvalid = cudf::reduce(
        invalid->view(),
        *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
        boolType,
        stream,
        tempMr);
    const auto* anyInvalidScalar =
        static_cast<const cudf::numeric_scalar<bool>*>(anyInvalid.get());
    if (!anyInvalidScalar->is_valid(stream) ||
        !anyInvalidScalar->value(stream)) {
      return values;
    }
    // Name the first offending string, as the CPU cast does.
    auto invalidStrings = cudf::apply_retention_mask(
        cudf::table_view({input}), invalid->view(), stream, tempMr);
    auto first =
        cudf::get_element(invalidStrings->view().column(0), 0, stream, tempMr);
    VELOX_USER_FAIL(
        "Cannot cast VARCHAR '{}' to {}.",
        static_cast<const cudf::string_scalar*>(first.get())->to_string(stream),
        targetTypeName_);
  }

  cudf::data_type targetCudfType_;
  // Set when the source is VARCHAR.
  bool fromVarchar_{false};
  bool isTryCast_{false};
  std::string targetTypeName_;
  // The kernel casting strings to the integral target type.
  std::string integralUdf_;
};

class CardinalityFunction : public CudfFunction {
 public:
  CardinalityFunction(const core::TypedExprPtr& expr) {
    // Cardinality doesn't need any pre-computed scalars, just validates input
    // count
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "cardinality expects exactly 1 input");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::lists::count_elements(inputCol, stream, mr);
  }
};

class IsNotNullFunction : public CudfFunction {
 public:
  explicit IsNotNullFunction(const core::TypedExprPtr& expr) {
    VELOX_CHECK_EQ(expr->inputs().size(), 1, "isnotnull expects 1 input");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1, "isnotnull expects 1 input");
    return cudf::is_valid(asView(inputColumns[0]), stream, mr);
  }
};

class RoundFunction : public CudfFunction {
 public:
  RoundFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool) {
    const auto argSize = expr->inputs().size();
    VELOX_CHECK(argSize >= 1 && argSize <= 2, "round expects 1 or 2 inputs");
    VELOX_CHECK(
        !expr->inputs()[0]->isConstantKind(),
        "round expects first column is not literal");
    if (argSize == 2) {
      VELOX_CHECK(
          expr->inputs()[1]->isConstantKind(),
          "round scale must be a constant");
      scale_ = toConstantVector(expr->inputs()[1], pool)
                   ->as<SimpleVector<int32_t>>()
                   ->valueAt(0);
    }
    decimalsScalar_ = std::unique_ptr<cudf::numeric_scalar<int32_t>>(
        static_cast<cudf::numeric_scalar<int32_t>*>(
            makeScalarFromValue<int32_t>(INTEGER(), scale_, false).release()));
    factorScalar_ = std::unique_ptr<cudf::numeric_scalar<double>>(
        static_cast<cudf::numeric_scalar<double>*>(
            makeScalarFromValue<double>(
                DOUBLE(), std::pow(10.0, static_cast<double>(scale_)), false)
                .release()));
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    auto inputTypeId = inputCol.type().id();

    if (inputTypeId == cudf::type_id::FLOAT64) {
      // JIT-compile a CUDA UDF that mirrors the Velox CPU round implementation
      // (see velox/functions/prestosql/ArithmeticImpl.h). This makes
      // Velox-cuDF produce bit-identical results to the Velox CPU path for
      // round(double, scale) (e.g. round(2.675, 2) == 2.68) by relying on the
      // same `round(x * factor) / factor` trick.
      static constexpr char const* kUdf = R"***(
__device__ void velox_round_double(
    double* out, double number, int decimals, double factor) {
  if (!isfinite(number)) { *out = number; return; }
  if (decimals == 0) { *out = round(number); return; }
  if (decimals < 0) {
    *out = round(number * factor) / factor;
    return;
  }
  const double truncated = trunc(number);
  const double fraction = number - truncated;
  if (fraction == 0.0) { *out = number; return; }
  // Threshold matches Velox CPU: for small magnitudes the factor-multiply
  // path has less precision loss than the truncate + fraction path.
  if (fabs(number) < 17592186044415.0) {
    *out = round(number * factor) / factor;
    return;
  }
  const double roundedFractions = round(fraction * factor) / factor;
  *out = truncated + roundedFractions;
}
)***";

      // The scale-derived `decimals` and `factor` scalars are built once in
      // the constructor (see makeScalarFromValue). Each eval only constructs
      // two non-owning column_views over their device storage.
      const cudf::column_view decimalsView{
          cudf::data_type{cudf::type_id::INT32},
          /*size=*/1,
          decimalsScalar_->data(),
          /*null_mask=*/nullptr,
          /*null_count=*/0};
      const cudf::column_view factorView{
          cudf::data_type{cudf::type_id::FLOAT64},
          1,
          factorScalar_->data(),
          nullptr,
          0};

      const cudf::transform_input transformInputs[] = {
          inputCol,
          cudf::scalar_column_view(decimalsView),
          cudf::scalar_column_view(factorView),
      };
      const cudf::transform_output transformOutputs[] = {{
          cudf::data_type{cudf::type_id::FLOAT64},
          cudf::output_nullability::PRESERVE,
      }};
      auto transformed = cudf::transform(
          kUdf,
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

    return cudf::round_decimal(
        inputCol, scale_, cudf::rounding_method::HALF_UP, stream, mr);
  }

 private:
  int32_t scale_ = 0;
  std::unique_ptr<cudf::numeric_scalar<int32_t>> decimalsScalar_;
  std::unique_ptr<cudf::numeric_scalar<double>> factorScalar_;
};

// A short-decimal result can still have a long-decimal operand. The divide
// kernels require matching column widths and cannot write a DECIMAL64 result
// from DECIMAL128 input.
cudf::data_type decimalDivisionWorkingType(
    cudf::data_type lhsType,
    cudf::data_type rhsType,
    cudf::data_type resultType) {
  const auto typeId = lhsType.id() == cudf::type_id::DECIMAL128 ||
          rhsType.id() == cudf::type_id::DECIMAL128 ||
          resultType.id() == cudf::type_id::DECIMAL128
      ? cudf::type_id::DECIMAL128
      : cudf::type_id::DECIMAL64;
  return cudf::data_type{typeId, resultType.scale()};
}

std::unique_ptr<cudf::column> finalizeDecimalDivision(
    std::unique_ptr<cudf::column> result,
    cudf::data_type resultType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (result->type() == resultType) {
    return result;
  }
  return cudf::cast(result->view(), resultType, stream, mr);
}

class BinaryFunction : public CudfFunction {
 public:
  BinaryFunction(
      const core::TypedExprPtr& expr,
      cudf::binary_operator op,
      memory::MemoryPool* pool)
      : op_(op), type_(cudf_velox::veloxToCudfDataType(expr->type())) {
    if (cudf::is_fixed_point(type_) && isCheckedDecimalArithmeticOp(op_)) {
      decimalPrecision_ = getDecimalPrecision(expr->type());
    }
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "binary function expects exactly 2 inputs");
    if (expr->inputs()[0]->isConstantKind()) {
      left_ = makeScalarFromConstantExpr(expr->inputs()[0], pool);
    } else if (expr->inputs()[1]->isConstantKind()) {
      right_ = makeScalarFromConstantExpr(expr->inputs()[1], pool);
    }

    VELOX_CHECK(
        !(left_ != nullptr && right_ != nullptr),
        "Binary function on two literals is not supported");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto isComparisonOp = [](cudf::binary_operator op) {
      switch (op) {
        case cudf::binary_operator::EQUAL:
        case cudf::binary_operator::NOT_EQUAL:
        case cudf::binary_operator::GREATER:
        case cudf::binary_operator::GREATER_EQUAL:
        case cudf::binary_operator::LESS:
        case cudf::binary_operator::LESS_EQUAL:
          return true;
        default:
          return false;
      }
    };
    if (left_ == nullptr && right_ == nullptr) {
      if (op_ == cudf::binary_operator::DIV && cudf::is_fixed_point(type_)) {
        auto lhsView = asView(inputColumns[0]);
        auto rhsView = asView(inputColumns[1]);
        std::unique_ptr<cudf::column> lhsCast;
        std::unique_ptr<cudf::column> rhsCast;
        const auto workingType =
            decimalDivisionWorkingType(lhsView.type(), rhsView.type(), type_);
        if (workingType.id() == cudf::type_id::DECIMAL128) {
          if (lhsView.type().id() == cudf::type_id::DECIMAL64) {
            auto castType = cudf::data_type{
                cudf::type_id::DECIMAL128, lhsView.type().scale()};
            lhsCast = cudf::cast(lhsView, castType, stream, mr);
            lhsView = lhsCast->view();
          }
          if (rhsView.type().id() == cudf::type_id::DECIMAL64) {
            auto castType = cudf::data_type{
                cudf::type_id::DECIMAL128, rhsView.type().scale()};
            rhsCast = cudf::cast(rhsView, castType, stream, mr);
            rhsView = rhsCast->view();
          }
        }
        auto lhsScale = -lhsView.type().scale();
        auto rhsScale = -rhsView.type().scale();
        auto outScale = -type_.scale();
        auto aRescale = outScale - lhsScale + rhsScale;
        auto result =
            decimalDivide(lhsView, rhsView, workingType, aRescale, stream, mr);
        return finalizeDecimalDivision(std::move(result), type_, stream, mr);
      }
      auto lhsView = asView(inputColumns[0]);
      auto rhsView = asView(inputColumns[1]);
      if (isComparisonOp(op_) && cudf::is_fixed_point(lhsView.type()) &&
          cudf::is_fixed_point(rhsView.type())) {
        auto lhsScale = -lhsView.type().scale();
        auto rhsScale = -rhsView.type().scale();
        auto targetScale = lhsScale > rhsScale ? lhsScale : rhsScale;
        auto targetTypeId = (lhsView.type().id() == cudf::type_id::DECIMAL128 ||
                             rhsView.type().id() == cudf::type_id::DECIMAL128)
            ? cudf::type_id::DECIMAL128
            : cudf::type_id::DECIMAL64;
        auto targetType =
            cudf::data_type{targetTypeId, numeric::scale_type{-targetScale}};
        std::unique_ptr<cudf::column> lhsCast;
        std::unique_ptr<cudf::column> rhsCast;
        if (lhsView.type() != targetType) {
          lhsCast = cudf::cast(lhsView, targetType, stream, mr);
          lhsView = lhsCast->view();
        }
        if (rhsView.type() != targetType) {
          rhsCast = cudf::cast(rhsView, targetType, stream, mr);
          rhsView = rhsCast->view();
        }
        // @TODO Check for divide-by-zero as in the DECIMAL case above?
        return cudf::binary_operation(lhsView, rhsView, op_, type_, stream, mr);
      }
      if (cudf::is_fixed_point(type_)) {
        if (op_ == cudf::binary_operator::ADD ||
            op_ == cudf::binary_operator::SUB ||
            op_ == cudf::binary_operator::MUL ||
            op_ == cudf::binary_operator::MOD) {
          // Operands keep their own storage width and scale: the
          // overflow-checked kernel dispatches on the lhs, rhs and output
          // widths independently and rescales ADD/SUB/MOD operands to the
          // output scale itself, so both conversions stay inside the fail-fast
          // path instead of an unchecked cudf::cast here.
          // @TODO Check for divide-by-zero as in the DECIMAL case above?
          return decimalBinaryOperation(
              lhsView, rhsView, op_, type_, decimalPrecision_, stream, mr);
        }
      }
      // @TODO Check for divide-by-zero as in the DECIMAL case above?
      return cudf::binary_operation(lhsView, rhsView, op_, type_, stream, mr);
    } else if (left_ == nullptr) {
      if (op_ == cudf::binary_operator::DIV && cudf::is_fixed_point(type_)) {
        if (decimalScalarIsZero(*right_, stream)) {
          VELOX_USER_FAIL("Division by zero");
        }
        auto lhsView = asView(inputColumns[0]);
        const auto workingType =
            decimalDivisionWorkingType(lhsView.type(), right_->type(), type_);
        std::unique_ptr<cudf::column> lhsCast;
        if (lhsView.type().id() != workingType.id()) {
          lhsCast = cudf::cast(
              lhsView,
              cudf::data_type{workingType.id(), lhsView.type().scale()},
              stream,
              mr);
          lhsView = lhsCast->view();
        }
        auto lhsScale = -lhsView.type().scale();
        auto rhsScale = -right_->type().scale();
        auto outScale = -type_.scale();
        auto aRescale = outScale - lhsScale + rhsScale;
        auto result =
            decimalDivide(lhsView, *right_, workingType, aRescale, stream, mr);
        return finalizeDecimalDivision(std::move(result), type_, stream, mr);
      }
      auto lhsView = asView(inputColumns[0]);
      if (isComparisonOp(op_) && cudf::is_fixed_point(lhsView.type()) &&
          cudf::is_fixed_point(right_->type())) {
        auto lhsScale = -lhsView.type().scale();
        auto rhsScale = -right_->type().scale();
        auto targetScale = lhsScale > rhsScale ? lhsScale : rhsScale;
        auto targetTypeId = (lhsView.type().id() == cudf::type_id::DECIMAL128 ||
                             right_->type().id() == cudf::type_id::DECIMAL128)
            ? cudf::type_id::DECIMAL128
            : cudf::type_id::DECIMAL64;
        auto targetType =
            cudf::data_type{targetTypeId, numeric::scale_type{-targetScale}};
        std::unique_ptr<cudf::column> lhsCast;
        if (lhsView.type() != targetType) {
          lhsCast = cudf::cast(lhsView, targetType, stream, mr);
          lhsView = lhsCast->view();
        }
        if (right_->type() != targetType) {
          auto rhsScalar = castDecimalScalar(*right_, targetType, stream, mr);
          return cudf::binary_operation(
              lhsView, *rhsScalar, op_, type_, stream, mr);
        }
        return cudf::binary_operation(lhsView, *right_, op_, type_, stream, mr);
      }
      if (cudf::is_fixed_point(type_)) {
        if (op_ == cudf::binary_operator::ADD ||
            op_ == cudf::binary_operator::SUB ||
            op_ == cudf::binary_operator::MUL ||
            op_ == cudf::binary_operator::MOD) {
          // Operand widths and scales are handled by the checked kernel (see
          // the column/column path above).
          return decimalBinaryOperation(
              lhsView, *right_, op_, type_, decimalPrecision_, stream, mr);
        }
      }
      return cudf::binary_operation(
          asView(inputColumns[0]), *right_, op_, type_, stream, mr);
    }
    if (op_ == cudf::binary_operator::DIV && cudf::is_fixed_point(type_)) {
      auto rhsView = asView(inputColumns[0]);
      const auto workingType =
          decimalDivisionWorkingType(left_->type(), rhsView.type(), type_);
      std::unique_ptr<cudf::column> rhsCast;
      if (rhsView.type().id() != workingType.id()) {
        rhsCast = cudf::cast(
            rhsView,
            cudf::data_type{workingType.id(), rhsView.type().scale()},
            stream,
            mr);
        rhsView = rhsCast->view();
      }
      auto lhsScale = -left_->type().scale();
      auto rhsScale = -rhsView.type().scale();
      auto outScale = -type_.scale();
      auto aRescale = outScale - lhsScale + rhsScale;
      auto result =
          decimalDivide(*left_, rhsView, workingType, aRescale, stream, mr);
      return finalizeDecimalDivision(std::move(result), type_, stream, mr);
    }
    auto rhsView = asView(inputColumns[0]);
    if (isComparisonOp(op_) && cudf::is_fixed_point(left_->type()) &&
        cudf::is_fixed_point(rhsView.type())) {
      auto lhsScale = -left_->type().scale();
      auto rhsScale = -rhsView.type().scale();
      auto targetScale = lhsScale > rhsScale ? lhsScale : rhsScale;
      auto targetTypeId = (left_->type().id() == cudf::type_id::DECIMAL128 ||
                           rhsView.type().id() == cudf::type_id::DECIMAL128)
          ? cudf::type_id::DECIMAL128
          : cudf::type_id::DECIMAL64;
      auto targetType =
          cudf::data_type{targetTypeId, numeric::scale_type{-targetScale}};
      std::unique_ptr<cudf::column> rhsCast;
      if (rhsView.type() != targetType) {
        rhsCast = cudf::cast(rhsView, targetType, stream, mr);
        rhsView = rhsCast->view();
      }
      if (left_->type() != targetType) {
        auto lhsScalar = castDecimalScalar(*left_, targetType, stream, mr);
        return cudf::binary_operation(
            *lhsScalar, rhsView, op_, type_, stream, mr);
      }
      return cudf::binary_operation(*left_, rhsView, op_, type_, stream, mr);
    }
    if (cudf::is_fixed_point(type_)) {
      if (op_ == cudf::binary_operator::ADD ||
          op_ == cudf::binary_operator::SUB ||
          op_ == cudf::binary_operator::MUL ||
          op_ == cudf::binary_operator::MOD) {
        // Operand widths and scales are handled by the checked kernel (see the
        // column/column path above).
        return decimalBinaryOperation(
            *left_, rhsView, op_, type_, decimalPrecision_, stream, mr);
      }
    }
    return cudf::binary_operation(*left_, rhsView, op_, type_, stream, mr);
  }

 private:
  const cudf::binary_operator op_;
  const cudf::data_type type_;
  int32_t decimalPrecision_{0};
  std::unique_ptr<cudf::scalar> left_;
  std::unique_ptr<cudf::scalar> right_;
};

// @TODO 4/22/26
// Simplify or remove the logic in this class that handles constant-folding or
// short-circuiting of logical operations, once the cuDF expression optimizer
// enhancements land (Velox PR #17108, see also Velox Issue #17307).
class LogicalFunction : public CudfFunction {
 public:
  LogicalFunction(
      const core::TypedExprPtr& expr,
      cudf::binary_operator op,
      memory::MemoryPool* pool)
      : op_(op) {
    VELOX_CHECK_GE(
        expr->inputs().size(), 2, "Logical function expects at least 2 inputs");
    literals_.reserve(expr->inputs().size());
    for (const auto& input : expr->inputs()) {
      if (input->isConstantKind()) {
        VELOX_CHECK_EQ(
            input->type()->kind(),
            TypeKind::BOOLEAN,
            "Logical function only supports boolean literals");
        auto boolConst =
            toConstantVector(input, pool)->as<ConstantVector<bool>>();
        VELOX_CHECK_NOT_NULL(boolConst);
        if (!shortCircuitScalar_ && !boolConst->isNullAt(0)) {
          const bool v = boolConst->valueAt(0);
          if ((op_ == cudf::binary_operator::NULL_LOGICAL_AND && !v) ||
              (op_ == cudf::binary_operator::NULL_LOGICAL_OR && v)) {
            // If we encounter non-null false (for AND) or true (for OR), we
            // know what the final result must be, although it will still need
            // to be expanded to a column the same size as the input columns. No
            // need to continue capturing literals in that case.
            shortCircuitScalar_ = makeScalarFromConstantExpr(input, pool);
            break;
          }
        }
        literals_.push_back(makeScalarFromConstantExpr(input, pool));
      } else {
        literals_.push_back(nullptr);
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    // If there are no input columns, the result is a scalar.
    const size_t rowCount =
        inputColumns.empty() ? 1 : asView(inputColumns[0]).size();

    // If we determined a short-circuit result in the constructor, we
    // return it directly here, expanded to the size of the input
    // columns if not all the inputs are literals.
    if (shortCircuitScalar_) {
      return cudf::make_column_from_scalar(
          *shortCircuitScalar_, rowCount, stream, mr);
    }

    // Now build the vector of actual operands, each of which is either a
    // pre-computed literal or an input column.
    struct Operand {
      const cudf::scalar* scalar;
      cudf::column_view column;
    };
    std::vector<Operand> operands;
    operands.reserve(literals_.size());
    size_t columnIndex = 0;
    for (const auto& literal : literals_) {
      if (literal) {
        operands.push_back(Operand{literal.get(), {}});
      } else {
        VELOX_CHECK_LT(columnIndex, inputColumns.size());
        operands.push_back(
            Operand{nullptr, asView(inputColumns[columnIndex++])});
      }
    }

    // There must be at least one operand.
    VELOX_CHECK(!operands.empty());

    // If there is only one operand, we can return it directly,
    // again expanded to the size of the input columns if needed.
    if (operands.size() == 1) {
      const auto& only = operands[0];
      if (only.scalar) {
        return cudf::make_column_from_scalar(
            *only.scalar, rowCount, stream, mr);
      }
      return ColumnOrView(only.column);
    }

    // If we get this far, we have at least two operands. We can
    // now compute the result by iterating over the operands and
    // applying the binary operator to each pair of operands.
    const auto& left = operands[0];
    const auto& right = operands[1];
    std::unique_ptr<cudf::column> result;
    if (left.scalar && right.scalar) {
      // This case may still happen even in the case where a short-circuit
      // result was not determined in the constructor, for example, if the
      // inputs are 'true OR true' or 'false AND false'.
      auto tmp =
          cudf::make_column_from_scalar(*left.scalar, rowCount, stream, mr);
      result = cudf::binary_operation(
          tmp->view(), *right.scalar, op_, kBoolType, stream, mr);
    } else if (left.scalar) {
      result = cudf::binary_operation(
          *left.scalar, right.column, op_, kBoolType, stream, mr);
    } else if (right.scalar) {
      result = cudf::binary_operation(
          left.column, *right.scalar, op_, kBoolType, stream, mr);
    } else {
      result = cudf::binary_operation(
          left.column, right.column, op_, kBoolType, stream, mr);
    }
    for (size_t i = 2; i < operands.size(); ++i) {
      const auto& next = operands[i];
      if (next.scalar) {
        result = cudf::binary_operation(
            result->view(), *next.scalar, op_, kBoolType, stream, mr);
      } else {
        result = cudf::binary_operation(
            result->view(), next.column, op_, kBoolType, stream, mr);
      }
    }
    return result;
  }

 private:
  static constexpr cudf::data_type kBoolType{cudf::type_id::BOOL8};
  const cudf::binary_operator op_;
  std::unique_ptr<cudf::scalar> shortCircuitScalar_;
  std::vector<std::unique_ptr<cudf::scalar>> literals_;
};

class UnaryFunction : public CudfFunction {
 public:
  UnaryFunction(const core::TypedExprPtr& expr, cudf::unary_operator op)
      : op_(op) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "Unary function expects exactly 1 input");
    VELOX_CHECK(
        !expr->inputs()[0]->isConstantKind(),
        "Unary function on literal input is not supported");
    // @TODO (seves 1/28/26)
    // binary functions require at least ONE input to be non-literal
    // do we need to support unary functions with ONLY a literal input?
    // assuming not for now
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::unary_operation(asView(inputColumns[0]), op_, stream, mr);
  }

 private:
  const cudf::unary_operator op_;
};

class BetweenFunction : public CudfFunction {
 public:
  BetweenFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool) {
    // must have exactly three inputs: value, min, max
    VELOX_CHECK_EQ(
        expr->inputs().size(), 3, "Between function expects exactly 3 inputs");
    // value must not be a literal
    VELOX_CHECK(
        !expr->inputs()[0]->isConstantKind(),
        "Between function with literal input is not supported");
    if (expr->inputs()[1]->isConstantKind()) {
      // min is a literal
      minLiteral_ = makeScalarFromConstantExpr(expr->inputs()[1], pool);
    }
    if (expr->inputs()[2]->isConstantKind()) {
      // max is a literal
      maxLiteral_ = makeScalarFromConstantExpr(expr->inputs()[2], pool);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    // return (value >= min) && (value <= max)
    std::unique_ptr<cudf::column> geResultColumn, leResultColumn;
    if (minLiteral_) {
      geResultColumn = cudf::binary_operation(
          asView(inputColumns[0]),
          *minLiteral_,
          cudf::binary_operator::GREATER_EQUAL,
          kBoolType,
          stream,
          mr);
    } else {
      geResultColumn = cudf::binary_operation(
          asView(inputColumns[0]),
          asView(inputColumns[1]),
          cudf::binary_operator::GREATER_EQUAL,
          kBoolType,
          stream,
          mr);
    }
    if (maxLiteral_) {
      leResultColumn = cudf::binary_operation(
          asView(inputColumns[0]),
          *maxLiteral_,
          cudf::binary_operator::LESS_EQUAL,
          kBoolType,
          stream,
          mr);
    } else {
      leResultColumn = cudf::binary_operation(
          asView(inputColumns[0]),
          asView(inputColumns[minLiteral_ ? 1 : 2]),
          cudf::binary_operator::LESS_EQUAL,
          kBoolType,
          stream,
          mr);
    }
    return cudf::binary_operation(
        geResultColumn->view(),
        leResultColumn->view(),
        cudf::binary_operator::LOGICAL_AND,
        kBoolType,
        stream,
        mr);
  }

 private:
  static constexpr cudf::data_type kBoolType{cudf::type_id::BOOL8};
  std::unique_ptr<cudf::scalar> minLiteral_;
  std::unique_ptr<cudf::scalar> maxLiteral_;
};

template <TypeKind Kind>
static VectorPtr foldConstantPair(
    const VectorPtr& a,
    const VectorPtr& b,
    cudf::binary_operator op) {
  using T = typename TypeTraits<Kind>::NativeType;
  if (a->isNullAt(0))
    return b;
  if (b->isNullAt(0))
    return a;
  auto aVal = a->as<ConstantVector<T>>()->value();
  auto bVal = b->as<ConstantVector<T>>()->value();
  bool bWins =
      (op == cudf::binary_operator::NULL_MAX) ? (bVal > aVal) : (bVal < aVal);
  return bWins ? b : a;
}

class GreatestLeastFunction : public CudfFunction {
 public:
  GreatestLeastFunction(
      const core::TypedExprPtr& expr,
      cudf::binary_operator op,
      memory::MemoryPool* pool)
      : op_(op), type_(cudf_velox::veloxToCudfDataType(expr->type())) {
    VELOX_CHECK_GE(
        expr->inputs().size(),
        2,
        "Greatest/Least function expects at least 2 inputs");
    // Separate column inputs (into order_) from constant inputs (folded into
    // a single scalar). Column indices refer to positions in the packed
    // inputColumns vector that eval() receives (which excludes literals).
    std::vector<VectorPtr> constValues;
    size_t columnIndex = 0;
    for (const auto& input : expr->inputs()) {
      if (input->isConstantKind()) {
        constValues.push_back(toConstantVector(input, pool));
      } else {
        order_.push_back(columnIndex++);
      }
    }
    // Fold all constant values into a single scalar on the host.
    if (!constValues.empty()) {
      auto winner = constValues[0];
      for (size_t i = 1; i < constValues.size(); ++i) {
        winner = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            foldConstantPair, winner->typeKind(), winner, constValues[i], op);
      }
      foldedScalar_ = makeScalarFromConstantExpr(
          std::make_shared<core::ConstantTypedExpr>(winner), pool);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    // All inputs were constant -- return the pre-folded scalar as a column.
    if (order_.empty()) {
      return cudf::make_column_from_scalar(*foldedScalar_, 1, stream, mr);
    }

    // Accumulate across column inputs.
    std::unique_ptr<cudf::column> result;
    for (size_t i = 1; i < order_.size(); ++i) {
      cudf::column_view lhs =
          result ? result->view() : asView(inputColumns[order_[0]]);
      result = cudf::binary_operation(
          lhs, asView(inputColumns[order_[i]]), op_, type_, stream, mr);
    }

    // Apply the folded constant as a final (column, scalar) operation.
    if (foldedScalar_) {
      cudf::column_view lhs =
          result ? result->view() : asView(inputColumns[order_[0]]);
      result =
          cudf::binary_operation(lhs, *foldedScalar_, op_, type_, stream, mr);
    }
    return result;
  }

 private:
  const cudf::binary_operator op_;
  const cudf::data_type type_;
  std::unique_ptr<cudf::scalar> foldedScalar_;
  std::vector<size_t> order_;
};

namespace {

bool supportsImplicitNullElse(const TypePtr& type) {
  // Interval constant scalars differ from the cuDF storage type used for the
  // implicit null ELSE.
  if (type->isIntervalYearMonth() || type->isIntervalDayTime()) {
    return false;
  }
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::TIMESTAMP:
      return true;
    case TypeKind::HUGEINT:
      return type->isDecimal();
    default:
      return false;
  }
}

} // namespace

class SwitchFunction : public CudfFunction {
 public:
  SwitchFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool)
      : resultType_(veloxToCudfDataType(expr->type())),
        hasElseClause_(expr->inputs().size() == 3) {
    VELOX_CHECK(
        expr->inputs().size() == 2 || expr->inputs().size() == 3,
        "Single-branch CASE expects 2 or 3 inputs");
    VELOX_CHECK_EQ(
        expr->inputs()[0]->type()->kind(),
        TypeKind::BOOLEAN,
        "The switch condition result type should be boolean");
    VELOX_CHECK(
        !expr->inputs()[0]->isConstantKind(),
        "The condition should not be constant");
    if (expr->inputs()[1]->isConstantKind()) {
      left_ = makeScalarFromConstantExpr(expr->inputs()[1], pool);
    }
    if (hasElseClause_ && expr->inputs()[2]->isConstantKind()) {
      right_ = makeScalarFromConstantExpr(expr->inputs()[2], pool);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    std::unique_ptr<cudf::scalar> nullElse;
    const auto* right = right_.get();
    if (!hasElseClause_) {
      nullElse = cudf::make_default_constructed_scalar(resultType_, stream, mr);
      right = nullElse.get();
    }
    if (left_ == nullptr && right == nullptr) {
      return cudf::copy_if_else(
          asView(inputColumns[1]),
          asView(inputColumns[2]),
          asView(inputColumns[0]),
          stream,
          mr);
    } else if (left_ == nullptr) {
      return cudf::copy_if_else(
          asView(inputColumns[1]), *right, asView(inputColumns[0]), stream, mr);
    } else if (right == nullptr) {
      return cudf::copy_if_else(
          *left_, asView(inputColumns[1]), asView(inputColumns[0]), stream, mr);
    }
    // right != null and left != null
    return cudf::copy_if_else(
        *left_, *right, asView(inputColumns[0]), stream, mr);
  }

 private:
  const cudf::data_type resultType_;
  const bool hasElseClause_;
  std::unique_ptr<cudf::scalar> left_;
  std::unique_ptr<cudf::scalar> right_;
};

class CoalesceFunction : public CudfFunction {
 public:
  CoalesceFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool) {
    // Storing the first literal that appears in inputs because we don't need to
    // process after that. This is the last fallback.
    numColumnsBeforeLiteral_ = expr->inputs().size();
    for (size_t i = 0; i < expr->inputs().size(); ++i) {
      const auto& input = expr->inputs()[i];
      if (input->isConstantKind()) {
        const auto* c = input->asUnchecked<core::ConstantTypedExpr>();
        if (!c->isNull()) {
          literalScalar_ = makeScalarFromConstantExpr(input, pool);
          numColumnsBeforeLiteral_ = i;
          break;
        }
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    // Coalesce is practically a cudf::replace_nulls over multiple columns.
    // Starting from first column, we keep calling replace nulls with
    // subsequent cols until we get an all valid col or run out of columns

    // If a literal comes before any column input, fill the result with it.
    if (literalScalar_ && numColumnsBeforeLiteral_ == 0) {
      if (inputColumns.empty()) {
        // We need at least one column to tell us the required output size
        VELOX_NYI("coalesce with only literal inputs is not supported");
      }
      auto size = asView(inputColumns[0]).size();
      return cudf::make_column_from_scalar(*literalScalar_, size, stream, mr);
    }

    VELOX_CHECK(
        !inputColumns.empty(),
        "coalesce requires at least one non-literal input");
    ColumnOrView result = asView(inputColumns[0]);
    if (std::holds_alternative<std::unique_ptr<cudf::column>>(
            inputColumns[0])) {
      // Preserve an owned subexpression result instead of returning a view
      // that dangles when FunctionExpression destroys its temporary results.
      result =
          std::move(std::get<std::unique_ptr<cudf::column>>(inputColumns[0]));
    }
    size_t stop = std::min(numColumnsBeforeLiteral_, inputColumns.size());
    for (size_t i = 1; i < stop && asView(result).has_nulls(); ++i) {
      result = cudf::replace_nulls(
          asView(result), asView(inputColumns[i]), stream, mr);
    }

    if (literalScalar_ && asView(result).has_nulls()) {
      result = cudf::replace_nulls(asView(result), *literalScalar_, stream, mr);
    }

    return result;
  }

 private:
  size_t numColumnsBeforeLiteral_;
  std::unique_ptr<cudf::scalar> literalScalar_;
};

class ExtractComponentFunction : public CudfFunction {
 public:
  ExtractComponentFunction(
      const core::TypedExprPtr& expr,
      cudf::datetime::datetime_component component)
      : component_(component) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "extract expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::datetime::extract_datetime_component(
        inputCol, component_, stream, mr);
  }

 private:
  cudf::datetime::datetime_component component_;
};

// Builds an ExtractComponentFunction for a fixed datetime component, avoiding a
// near-identical registration lambda for every component (year, month, ...).
struct ExtractComponentFactory {
  cudf::datetime::datetime_component component;

  std::shared_ptr<CudfFunction> operator()(
      const std::string&,
      const core::TypedExprPtr& expr,
      memory::MemoryPool*) const {
    return std::make_shared<ExtractComponentFunction>(expr, component);
  }
};

class QuarterFunction : public CudfFunction {
 public:
  explicit QuarterFunction(const core::TypedExprPtr& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "quarter expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::datetime::extract_quarter(inputCol, stream, mr);
  }
};

class DayOfYearFunction : public CudfFunction {
 public:
  explicit DayOfYearFunction(const core::TypedExprPtr& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "day_of_year expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::datetime::day_of_year(inputCol, stream, mr);
  }
};

class WeekFunction : public CudfFunction {
 public:
  explicit WeekFunction(const core::TypedExprPtr& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "week expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    auto weekStrings = cudf::strings::from_timestamps(
        inputCol, "%V", cudf::strings_column_view{}, stream, mr);
    return cudf::strings::to_integers(
        cudf::strings_column_view(weekStrings->view()),
        cudf::data_type(cudf::type_id::INT32),
        stream,
        mr);
  }
};

class YearOfWeekFunction : public CudfFunction {
 public:
  explicit YearOfWeekFunction(const core::TypedExprPtr& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(),
        1,
        "year_of_week expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    auto yearStrings = cudf::strings::from_timestamps(
        inputCol, "%G", cudf::strings_column_view{}, stream, mr);
    return cudf::strings::to_integers(
        cudf::strings_column_view(yearStrings->view()),
        cudf::data_type(cudf::type_id::INT32),
        stream,
        mr);
  }
};

class LengthFunction : public CudfFunction {
 public:
  explicit LengthFunction(const core::TypedExprPtr& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "length expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::count_characters(inputCol, stream, mr);
  }
};

class LowerFunction : public CudfFunction {
 public:
  explicit LowerFunction(const core::TypedExprPtr& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "lower expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::to_lower(inputCol, stream, mr);
  }
};

class UpperFunction : public CudfFunction {
 public:
  explicit UpperFunction(const core::TypedExprPtr& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "upper expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::to_upper(inputCol, stream, mr);
  }
};

// replace(target_str, search_str[, replacement_str]): replaces every
// occurrence of search_str in target_str with replacement_str. With only
// search_str it removes search_str (i.e. an empty replacement). Backed by
// cudf::strings::replace with a maxrepl of -1 so all occurrences are replaced.
// Only a constant, non-empty search_str and constant replacement_str are
// allowed.
class ReplaceFunction : public CudfFunction {
 public:
  ReplaceFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool) {
    VELOX_CHECK(
        expr->inputs().size() == 2 || expr->inputs().size() == 3,
        "replace expects 2 or 3 inputs");

    // Search is a constant per the registered signatures; read it once here.
    // canEvaluate declines an empty search, so enforce that invariant here too
    // rather than let it surface as a raw cudf error deeper in eval.
    auto searchValue = toConstantVector(expr->inputs()[1], pool);
    searchIsNull_ = searchValue->isNullAt(0);
    if (!searchIsNull_) {
      search_ = searchValue->toString(0);
      VELOX_CHECK(!search_.empty(), "replace search must not be empty");
    }

    // The 3-argument form carries a constant replacement; the 2-argument form
    // removes the search, which is an empty, non-null replacement.
    if (expr->inputs().size() == 3) {
      auto replacementValue = toConstantVector(expr->inputs()[2], pool);
      replacementIsNull_ = replacementValue->isNullAt(0);
      if (!replacementIsNull_) {
        replacement_ = replacementValue->toString(0);
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(
        !inputColumns.empty(),
        "replace requires at least one non-literal input column");
    // inputColumns holds only non-literal children, so the string column is the
    // first (and only) entry and determines the output row count.
    auto inputCol = asView(inputColumns[0]);

    // replace(x, NULL, y) and replace(x, s, NULL) are NULL for every row.
    // Match Velox null propagation by producing an all-null varchar column.
    if (searchIsNull_ || replacementIsNull_) {
      cudf::string_scalar nullString("", false, stream, mr);
      return cudf::make_column_from_scalar(
          nullString, inputCol.size(), stream, mr);
    }

    const cudf::strings_column_view stringsView(inputCol);
    cudf::string_scalar searchScalar(search_, true, stream, mr);
    cudf::string_scalar replacementScalar(replacement_, true, stream, mr);
    // maxrepl == -1 replaces all occurrences. Nulls in the input column
    // propagate to a null result.
    return cudf::strings::replace(
        stringsView, searchScalar, replacementScalar, -1, stream, mr);
  }

 private:
  // Constant search value, valid only when searchIsNull_ is false.
  std::string search_;
  // Constant replacement value; empty for the 2-argument form, valid only when
  // replacementIsNull_ is false.
  std::string replacement_;
  // Set when the constant search argument is null.
  bool searchIsNull_{false};
  // Set when the constant replacement argument is null.
  bool replacementIsNull_{false};
};

class LikeFunction : public CudfFunction {
 public:
  LikeFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool) {
    VELOX_CHECK(
        expr->inputs().size() == 2 || expr->inputs().size() == 3,
        "like expects 2 or 3 inputs");

    if (expr->inputs()[0]->isConstantKind()) {
      inputIsConstant_ = true;
      auto inputValue = toConstantVector(expr->inputs()[0], pool);
      inputIsNull_ = inputValue->isNullAt(0);
      if (!inputIsNull_) {
        input_ = inputValue->toString(0);
      }
    }

    if (expr->inputs()[1]->isConstantKind()) {
      patternIsConstant_ = true;
      auto patternValue = toConstantVector(expr->inputs()[1], pool);
      patternIsNull_ = patternValue->isNullAt(0);
      if (!patternIsNull_) {
        pattern_ = patternValue->toString(0);
      }
    }

    VELOX_CHECK(
        !(inputIsConstant_ && patternIsConstant_),
        "like with constant input and pattern is not supported by the cuDF "
        "evaluator because there is no input column to derive the output row "
        "count from");

    hasEscape_ = expr->inputs().size() == 3;
    if (hasEscape_) {
      VELOX_CHECK(
          expr->inputs()[2]->isConstantKind(),
          "like escape must be a constant");
      auto escapeValue = toConstantVector(expr->inputs()[2], pool);
      escapeIsNull_ = escapeValue->isNullAt(0);
      if (!escapeIsNull_) {
        escape_ = escapeValue->toString(0);
        if (!patternIsConstant_ && escape_.size() == 1) {
          // Column-pattern LIKE ESCAPE reuses the same three legal escape
          // sequences for every batch, so cache those tiny helper columns once
          // here. Constant patterns are validated on the host and don't need
          // these columns.
          auto stream = getDefaultStreamForCurrentThread();
          auto mr = get_temp_mr();
          targetsColumn_ = makeEscapeTargetsColumn(escape_[0], stream, mr);
          replacementsColumn_ = makeEscapeReplacementsColumn(stream, mr);
        }
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    size_t nextInput = 0;
    VELOX_CHECK(
        !inputColumns.empty(),
        "like requires at least one non-literal input column");
    // inputColumns only contains non-literal children, so the first entry
    // determines the output row count even when the input itself is constant.
    auto outputRowCount = asView(inputColumns[0]).size();

    std::unique_ptr<cudf::column> inputColumnHolder;
    cudf::column_view inputCol;
    if (inputIsConstant_) {
      cudf::string_scalar inputScalar(input_, !inputIsNull_, stream, mr);
      inputColumnHolder = cudf::make_column_from_scalar(
          inputScalar, outputRowCount, stream, mr);
      inputCol = inputColumnHolder->view();
    } else {
      inputCol = asView(inputColumns[nextInput++]);
    }

    auto makeAllNullResult = [&]() {
      // Match Velox CPU null propagation for constant null pattern/escape
      // inputs by producing a fully null boolean column.
      auto nullScalar = cudf::numeric_scalar<bool>(false, false, stream, mr);
      return cudf::make_column_from_scalar(
          nullScalar, inputCol.size(), stream, mr);
    };

    if (patternIsConstant_ && patternIsNull_) {
      return makeAllNullResult();
    }
    if (hasEscape_ && escapeIsNull_) {
      return makeAllNullResult();
    }

    char escapeChar{0};
    if (hasEscape_) {
      VELOX_USER_CHECK_EQ(
          escape_.size(), 1, "Escape string must be a single character");
      escapeChar = escape_[0];
    }

    if (patternIsConstant_) {
      if (hasEscape_) {
        validateConstantPattern(pattern_, escapeChar);
      }
      return cudf::strings::like(inputCol, pattern_, escape_, stream, mr);
    }

    auto patternCol = asView(inputColumns[nextInput]);
    if (hasEscape_) {
      validatePatternColumn(patternCol, escapeChar, stream, mr);
    }

    std::unique_ptr<cudf::column> sanitizedPatternHolder;
    auto sanitizedPattern = patternCol;
    if (patternCol.has_nulls()) {
      // libcudf rejects null pattern rows for column/column LIKE. Replace them
      // only for the LIKE call after validation so the temporary column
      // overlaps less with the validation intermediates above.
      cudf::string_scalar emptyPattern("", true, stream, mr);
      sanitizedPatternHolder =
          cudf::replace_nulls(patternCol, emptyPattern, stream, mr);
      sanitizedPattern = sanitizedPatternHolder->view();
    }

    cudf::string_scalar escapeScalar(escape_, true, stream, mr);
    auto result = cudf::strings::like(
        inputCol, sanitizedPattern, escapeScalar, stream, mr);
    sanitizedPatternHolder.reset();
    // Velox returns null if either the input or pattern row is null. cuDF
    // already propagated input nulls into the result, so only merge the pattern
    // nulls back when needed.
    mergeNullSourceNullsIntoResult(*result, patternCol, stream, mr);
    return result;
  }

 private:
  static void validateConstantPattern(std::string_view pattern, char escape);

  static std::unique_ptr<cudf::column> makeEscapeTargetsColumn(
      char escape,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  static std::unique_ptr<cudf::column> makeEscapeReplacementsColumn(
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  void validatePatternColumn(
      cudf::column_view patternColumn,
      char escape,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const;

  bool inputIsConstant_{false};
  bool inputIsNull_{false};
  bool patternIsConstant_{false};
  bool patternIsNull_{false};
  bool hasEscape_{false};
  bool escapeIsNull_{false};
  std::string input_;
  std::string pattern_;
  std::string escape_;
  std::unique_ptr<cudf::column> targetsColumn_;
  std::unique_ptr<cudf::column> replacementsColumn_;
};

void LikeFunction::validateConstantPattern(
    std::string_view pattern,
    char escape) {
  // Match Velox CPU invalid escape validation before calling libcudf LIKE.
  for (size_t index = 0; index < pattern.size(); ++index) {
    if (pattern[index] != escape) {
      continue;
    }

    VELOX_USER_CHECK_LT(
        index + 1,
        pattern.size(),
        "Escape character must be followed by '%', '_' or the escape character itself");
    auto next = pattern[index + 1];
    if (next != escape && next != '%' && next != '_') {
      VELOX_USER_FAIL(
          "Escape character must be followed by '%', '_' or the escape character itself");
    }
    ++index;
  }
}

std::unique_ptr<cudf::column> LikeFunction::makeEscapeTargetsColumn(
    char escape,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  // Build the three legal escape sequences so column-pattern LIKE ESCAPE can
  // strip them before checking whether any invalid escape usage remains. This
  // keeps cuDF aligned with Velox CPU pattern validation semantics.
  cudf::string_scalar escapedEscapeScalar(
      std::string(2, escape), true, stream, mr);
  cudf::string_scalar escapedPercentScalar(
      std::string(1, escape) + '%', true, stream, mr);
  cudf::string_scalar escapedUnderscoreScalar(
      std::string(1, escape) + '_', true, stream, mr);
  auto escapedEscapeView = escapedEscapeScalar.value(stream);
  auto escapedPercentView = escapedPercentScalar.value(stream);
  auto escapedUnderscoreView = escapedUnderscoreScalar.value(stream);
  rmm::device_uvector<cudf::string_view> deviceTargetViews(3, stream, mr);
  deviceTargetViews.set_element_async(0, escapedEscapeView, stream);
  deviceTargetViews.set_element_async(1, escapedPercentView, stream);
  deviceTargetViews.set_element_async(2, escapedUnderscoreView, stream);
  auto targetsColumn = cudf::make_strings_column(
      cudf::device_span<cudf::string_view const>{deviceTargetViews},
      cudf::string_view{nullptr, 0},
      stream,
      mr);
  // The temporary scalars and string_view array above back async work used to
  // build the output column, so wait for the stream before returning.
  stream.sync();
  return targetsColumn;
}

std::unique_ptr<cudf::column> LikeFunction::makeEscapeReplacementsColumn(
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  cudf::string_scalar emptyString("", true, stream, mr);
  auto replacementsColumn =
      cudf::make_column_from_scalar(emptyString, 1, stream, mr);
  // make_column_from_scalar(string_scalar) reads the scalar's device string
  // data asynchronously, so keep the scalar alive until the stream completes.
  stream.sync();
  return replacementsColumn;
}

void LikeFunction::validatePatternColumn(
    cudf::column_view patternColumn,
    char escape,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK_NOT_NULL(targetsColumn_);
  VELOX_CHECK_NOT_NULL(replacementsColumn_);

  // Remove the three legal escape forms first. Any remaining escape character
  // must be dangling or followed by an unsupported character, which matches the
  // Velox CPU invalid-escape check.
  auto normalized = cudf::strings::replace_multiple(
      cudf::strings_column_view(patternColumn),
      cudf::strings_column_view(targetsColumn_->view()),
      cudf::strings_column_view(replacementsColumn_->view()),
      stream,
      mr);

  auto escapeScalar =
      cudf::string_scalar(std::string(1, escape), true, stream, mr);
  auto hasDanglingEscape = cudf::strings::contains(
      cudf::strings_column_view(normalized->view()), escapeScalar, stream, mr);
  normalized.reset();

  auto anyAggregation = cudf::make_any_aggregation<cudf::reduce_aggregation>();
  auto invalidScalar = cudf::reduce(
      hasDanglingEscape->view(),
      *anyAggregation,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  hasDanglingEscape.reset();

  auto const& invalidBool =
      static_cast<cudf::numeric_scalar<bool> const&>(*invalidScalar);
  auto hasInvalidEscapeUsageValue =
      invalidBool.is_valid(stream) && invalidBool.value(stream);
  invalidScalar.reset();

  VELOX_USER_CHECK(
      !hasInvalidEscapeUsageValue,
      "Escape character must be followed by '%', '_' or the escape character itself");
}

class StringPatternPredicateFunction : public CudfFunction {
 public:
  StringPatternPredicateFunction(
      const core::TypedExprPtr& expr,
      memory::MemoryPool* pool,
      std::string_view functionName) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "{} expects 2 inputs", functionName);

    if (expr->inputs()[0]->isConstantKind()) {
      inputIsConstant_ = true;
      auto inputValue = toConstantVector(expr->inputs()[0], pool);
      inputIsNull_ = inputValue->isNullAt(0);
      if (!inputIsNull_) {
        input_ = inputValue->toString(0);
      }
    }

    if (expr->inputs()[1]->isConstantKind()) {
      patternIsConstant_ = true;
      auto patternValue = toConstantVector(expr->inputs()[1], pool);
      patternIsNull_ = patternValue->isNullAt(0);
      if (!patternIsNull_) {
        pattern_ = patternValue->toString(0);
      }
    }

    // Fully constant string-match calls stay off the cuDF path because the
    // evaluator has no input column to derive the output row count from.
    VELOX_CHECK(
        !(inputIsConstant_ && patternIsConstant_),
        "{} with two constant inputs is not supported by the cuDF evaluator",
        functionName);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    size_t nextInput = 0;
    auto rowCount = inputColumns.empty() ? vector_size_t{1}
                                         : asView(inputColumns[0]).size();

    std::unique_ptr<cudf::column> inputColumnHolder;
    cudf::column_view inputCol;
    if (inputIsConstant_) {
      cudf::string_scalar inputScalar(input_, !inputIsNull_, stream, mr);
      inputColumnHolder =
          cudf::make_column_from_scalar(inputScalar, rowCount, stream, mr);
      inputCol = inputColumnHolder->view();
    } else {
      inputCol = asView(inputColumns[nextInput++]);
    }

    if (patternIsConstant_) {
      if (patternIsNull_) {
        auto nullScalar = cudf::numeric_scalar<bool>(false, false, stream, mr);
        return cudf::make_column_from_scalar(
            nullScalar, inputCol.size(), stream, mr);
      }
      cudf::string_scalar patternScalar(pattern_, true, stream, mr);
      return evaluateMatch(inputCol, patternScalar, stream, mr);
    }

    auto patternCol = asView(inputColumns[nextInput]);
    auto result = evaluateMatch(inputCol, patternCol, stream, mr);
    // Match Velox CPU null propagation for column/column evaluation: libcudf
    // can return a valid false when the pattern row is null, but Velox returns
    // null if either side is null. cuDF already propagated input nulls into the
    // result, so only merge the pattern nulls back when needed.
    mergeNullSourceNullsIntoResult(*result, patternCol, stream, mr);
    return result;
  }

 protected:
  virtual std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::string_scalar const& patternScalar,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const = 0;

  virtual std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::column_view patternCol,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const = 0;

  bool inputIsConstant_{false};
  bool inputIsNull_{false};
  bool patternIsNull_{false};
  bool patternIsConstant_{false};
  std::string input_;
  std::string pattern_;
};

class StartswithFunction : public StringPatternPredicateFunction {
 public:
  StartswithFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool)
      : StringPatternPredicateFunction(expr, pool, "startswith") {}

 protected:
  std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::string_scalar const& patternScalar,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::strings::starts_with(inputCol, patternScalar, stream, mr);
  }

  std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::column_view patternCol,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::strings::starts_with(inputCol, patternCol, stream, mr);
  }
};

class EndswithFunction : public StringPatternPredicateFunction {
 public:
  EndswithFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool)
      : StringPatternPredicateFunction(expr, pool, "endswith") {}

 protected:
  std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::string_scalar const& patternScalar,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::strings::ends_with(inputCol, patternScalar, stream, mr);
  }

  std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::column_view patternCol,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::strings::ends_with(inputCol, patternCol, stream, mr);
  }
};

class ContainsFunction : public StringPatternPredicateFunction {
 public:
  ContainsFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool)
      : StringPatternPredicateFunction(expr, pool, "contains") {}

 protected:
  std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::string_scalar const& patternScalar,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::strings::contains(inputCol, patternScalar, stream, mr);
  }

  std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::column_view patternCol,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::strings::contains(inputCol, patternCol, stream, mr);
  }
};

class ConcatFunction : public CudfFunction {
 public:
  ConcatFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool) {
    numInputs_ = expr->inputs().size();
    VELOX_CHECK_GE(numInputs_, 2, "concat expects at least 2 inputs");

    // Scan inputs for literals and store strings in map by input index.
    for (size_t i = 0; i < numInputs_; ++i) {
      if (expr->inputs()[i]->isConstantKind()) {
        inputIndexToLiteral_[i] = constantToString(expr->inputs()[i], pool);
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    // Validate sizes.
    VELOX_CHECK_EQ(
        inputColumns.size() + inputIndexToLiteral_.size(),
        numInputs_,
        "Unexpected number of input columns");

    // If there is at least one input column, fetch its size as the output size.
    // If there are no input columns, this means that all the inputs are
    // literals, and the output size will be 1.
    const size_t outputSize =
        inputColumns.empty() ? 1u : asView(inputColumns[0]).size();

    // Iterate the inputs, building a vector of column views, either a literal
    // from the map, or the next input column. We also keep a vector of the
    // columns created for literals, so that they persist while their views
    // are used in the concatenation.
    std::vector<cudf::column_view> columnViews;
    std::vector<std::unique_ptr<cudf::column>> literalColumns;
    size_t nextInputColumnIndex = 0u;
    for (size_t i = 0; i < numInputs_; ++i) {
      auto it = inputIndexToLiteral_.find(i);
      if (it == inputIndexToLiteral_.end()) {
        // No literal for this input. Use the next input column.
        auto& column = inputColumns[nextInputColumnIndex++];
        columnViews.push_back(asView(column));
      } else {
        // Create a column of the literal repeated for the entire output size.
        auto const& literal = it->second;
        cudf::string_scalar scalar(literal, true, stream, mr);
        auto col =
            cudf::make_column_from_scalar(scalar, outputSize, stream, mr);
        columnViews.push_back(col->view());
        literalColumns.emplace_back(std::move(col));
      }
    }

    // Concatenate the columns, nulls as empty strings, no separators.
    cudf::string_scalar emptyString("", true, stream, mr);
    return cudf::strings::concatenate(
        cudf::table_view(columnViews),
        emptyString,
        emptyString,
        cudf::strings::separator_on_nulls::YES,
        stream,
        mr);
  }

 private:
  std::map<int, std::string> inputIndexToLiteral_;
  size_t numInputs_{0};
};

class RowConstructorFunction : public CudfFunction {
 public:
  RowConstructorFunction(
      const core::TypedExprPtr& expr,
      memory::MemoryPool* pool) {
    VELOX_CHECK_GE(
        expr->inputs().size(), 1, "row_constructor expects at least 1 input");
    numInputs_ = expr->inputs().size();
    bool hasNonLiteralInput = false;
    literals_.reserve(numInputs_);
    for (const auto& input : expr->inputs()) {
      if (input->isConstantKind()) {
        literals_.push_back(makeScalarFromConstantExpr(input, pool));
      } else {
        hasNonLiteralInput = true;
        literals_.push_back(nullptr);
      }
    }
    if (!hasNonLiteralInput) {
      VELOX_NYI("row_constructor with only literal inputs is not supported");
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(
        !inputColumns.empty(),
        "row_constructor requires at least one non-literal input column");
    const cudf::size_type outputSize = asView(inputColumns[0]).size();

    std::vector<std::unique_ptr<cudf::column>> children;
    children.reserve(numInputs_);

    size_t nextInputColumnIndex = 0;
    for (const auto& literal : literals_) {
      if (literal) {
        children.push_back(
            cudf::make_column_from_scalar(*literal, outputSize, stream, mr));
      } else {
        VELOX_CHECK_LT(nextInputColumnIndex, inputColumns.size());
        children.push_back(
            makeOwnedColumn(inputColumns[nextInputColumnIndex++], stream, mr));
      }
    }

    VELOX_CHECK_EQ(nextInputColumnIndex, inputColumns.size());
    return cudf::make_structs_column(
        outputSize,
        std::move(children),
        0,
        cudf::create_null_mask(
            0,
            cudf::mask_state::UNALLOCATED,
            stream,
            cudf::memory_resources{mr, get_temp_mr()}),
        stream,
        mr);
  }

 private:
  static std::unique_ptr<cudf::column> makeOwnedColumn(
      ColumnOrView& holder,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  std::vector<std::unique_ptr<cudf::scalar>> literals_;
  size_t numInputs_{0};
};

std::unique_ptr<cudf::column> RowConstructorFunction::makeOwnedColumn(
    ColumnOrView& holder,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return std::visit(
      [&](auto& value) -> std::unique_ptr<cudf::column> {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, cudf::column_view>) {
          return std::make_unique<cudf::column>(value, stream, mr);
        } else {
          return std::move(value);
        }
      },
      holder);
}

bool registerCudfFunction(
    const std::string& name,
    CudfFunctionFactory factory,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite,
    CudfCanEvaluate canEvaluate) {
  auto& registry = getCudfFunctionRegistry();
  if (!overwrite && !registry[name].empty()) {
    return false;
  }
  registry[name].push_back(
      CudfFunctionSpec{std::move(factory), signatures, std::move(canEvaluate)});
  return true;
}

void registerCudfFunctions(
    const std::vector<std::string>& aliases,
    CudfFunctionFactory factory,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite,
    CudfCanEvaluate canEvaluate) {
  for (const auto& name : aliases) {
    registerCudfFunction(name, factory, signatures, overwrite, canEvaluate);
  }
}

std::shared_ptr<CudfFunction> createCudfFunction(
    const std::string& name,
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool) {
  auto& registry = getCudfFunctionRegistry();
  auto it = registry.find(name);
  if (it == registry.end()) {
    return nullptr;
  }
  for (const auto& spec : it->second) {
    // Empty signatures matching must be allowed to handle
    // the special case of cast.
    if (!spec.signatures.empty() &&
        !matchCallAgainstSignatures(expr, spec.signatures)) {
      continue;
    }
    if (spec.canEvaluate && !spec.canEvaluate(expr)) {
      continue;
    }
    return spec.factory(name, expr, pool);
  }
  return nullptr;
}

bool registerBuiltinFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  registerCudfFunction(
      prefix + "split",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<SplitFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("array(varchar)")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("array(varchar)")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           // cuDF expects cudf::size_type (int32) but we may get bigint from
           // presto. SplitFunction hacks around this by converting to string
           // and back
           .constantArgumentType("bigint")
           .build()});

  registerCudfFunction(
      prefix + "cardinality",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool*) {
        return std::make_shared<CardinalityFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("array(any)")
           .build()});

  // Coalesce is special form and doesn't have a prefix in its name.
  registerCudfFunction(
      "coalesce",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<CoalesceFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("T")
           .argumentType("T")
           .variableArity("T")
           .build()});

  // row_constructor is a special form and doesn't have a prefix in its name.
  registerCudfFunction(
      "row_constructor",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<RowConstructorFunction>(expr, pool);
      },
      {});

  registerCudfFunction(
      "and",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<LogicalFunction>(
            expr, cudf::binary_operator::NULL_LOGICAL_AND, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("boolean")
           .variableArity("boolean")
           .build()});

  registerCudfFunction(
      "or",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<LogicalFunction>(
            expr, cudf::binary_operator::NULL_LOGICAL_OR, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("boolean")
           .variableArity("boolean")
           .build()});

  registerCudfFunction(
      "not",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool*) {
        return std::make_shared<UnaryFunction>(expr, cudf::unary_operator::NOT);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("boolean")
           .build()});

  registerCudfFunction(
      "isnotnull",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool*) {
        return std::make_shared<IsNotNullFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("boolean")
           .argumentType("T")
           .build()});

  registerCudfFunction(
      prefix + "round",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<RoundFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .integerVariable("p")
           .integerVariable("s")
           .returnType("decimal(p,s)")
           .argumentType("decimal(p,s)")
           .build(),
       FunctionSignatureBuilder()
           .integerVariable("p")
           .integerVariable("s")
           .returnType("decimal(p,s)")
           .argumentType("decimal(p,s)")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("tinyint")
           .argumentType("tinyint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("tinyint")
           .argumentType("tinyint")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("smallint")
           .argumentType("smallint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("smallint")
           .argumentType("smallint")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("integer")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("bigint")
           .argumentType("bigint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("bigint")
           .argumentType("bigint")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .build(),
       FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .constantArgumentType("integer")
           .build()});

  const std::vector<exec::FunctionSignaturePtr> timestampDateIntegerSignatures{
      FunctionSignatureBuilder()
          .returnType("integer")
          .argumentType("timestamp")
          .build(),
      FunctionSignatureBuilder()
          .returnType("integer")
          .argumentType("date")
          .build()};

  registerCudfFunction(
      prefix + "year",
      ExtractComponentFactory{cudf::datetime::datetime_component::YEAR},
      timestampDateIntegerSignatures);

  registerCudfFunction(
      prefix + "month",
      ExtractComponentFactory{cudf::datetime::datetime_component::MONTH},
      timestampDateIntegerSignatures);

  registerCudfFunction(
      prefix + "day",
      ExtractComponentFactory{cudf::datetime::datetime_component::DAY},
      timestampDateIntegerSignatures);

  registerCudfFunctions(
      {prefix + "dow", prefix + "day_of_week"},
      ExtractComponentFactory{cudf::datetime::datetime_component::WEEKDAY},
      timestampDateIntegerSignatures);

  registerCudfFunctions(
      {prefix + "doy", prefix + "day_of_year"},
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool*) {
        return std::make_shared<DayOfYearFunction>(expr);
      },
      timestampDateIntegerSignatures);

  registerCudfFunctions(
      {prefix + "week", prefix + "week_of_year"},
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool*) { return std::make_shared<WeekFunction>(expr); },
      timestampDateIntegerSignatures);

  registerCudfFunction(
      prefix + "quarter",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool*) {
        return std::make_shared<QuarterFunction>(expr);
      },
      timestampDateIntegerSignatures);

  registerCudfFunctions(
      {prefix + "yow", prefix + "year_of_week"},
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool*) {
        return std::make_shared<YearOfWeekFunction>(expr);
      },
      timestampDateIntegerSignatures);

  registerCudfFunction(
      prefix + "hour",
      ExtractComponentFactory{cudf::datetime::datetime_component::HOUR},
      timestampDateIntegerSignatures);

  registerCudfFunction(
      prefix + "minute",
      ExtractComponentFactory{cudf::datetime::datetime_component::MINUTE},
      timestampDateIntegerSignatures);

  registerCudfFunction(
      prefix + "second",
      ExtractComponentFactory{cudf::datetime::datetime_component::SECOND},
      timestampDateIntegerSignatures);

  registerCudfFunction(
      prefix + "millisecond",
      ExtractComponentFactory{cudf::datetime::datetime_component::MILLISECOND},
      timestampDateIntegerSignatures);

  registerCudfFunction(
      prefix + "length",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool*) {
        return std::make_shared<LengthFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("bigint")
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "lower",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool*) { return std::make_shared<LowerFunction>(expr); },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "upper",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool*) { return std::make_shared<UpperFunction>(expr); },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "replace",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<ReplaceFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .constantArgumentType("varchar")
           .build(),
       FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .build()},
      /*overwrite=*/true,
      /*canEvaluate=*/
      [](const core::TypedExprPtr& expr) {
        // Decline an empty constant search_str: the expected semantics insert
        // the replacement_str around every character, i.e. replace('abc', '',
        // 'X') → 'XaXbXcX', which cudf::strings::replace does not implement.
        auto search = constantVarcharValue(expr->inputs()[1]);
        return !(search.has_value() && search->empty());
      });

  registerCudfFunction(
      prefix + "like",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<LikeFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("varchar")
           .argumentType("varchar")
           .build(),
       FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("varchar")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "startswith",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<StartswithFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("varchar")
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "endswith",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<EndswithFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("varchar")
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "contains",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<ContainsFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("varchar")
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "concat",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<ConcatFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .variableArity("varchar")
           .build()});

  // No prefix because switch and if are special forms. Only the two-argument
  // form needs a scalar for the implicit null ELSE.
  auto switchFactory = [](const std::string&,
                          const core::TypedExprPtr& expr,
                          memory::MemoryPool* pool) {
    return std::make_shared<SwitchFunction>(expr, pool);
  };
  registerCudfFunctions(
      {"switch", "if"},
      switchFactory,
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("T")
           .argumentType("boolean")
           .argumentType("T")
           .build()},
      /*overwrite=*/true,
      [](const core::TypedExprPtr& expr) {
        return !expr->inputs()[0]->isConstantKind() &&
            supportsImplicitNullElse(expr->type());
      });
  registerCudfFunctions(
      {"switch", "if"},
      switchFactory,
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("T")
           .argumentType("boolean")
           .argumentType("T")
           .argumentType("T")
           .build()},
      /*overwrite=*/true,
      [](const core::TypedExprPtr& expr) {
        return !expr->inputs()[0]->isConstantKind();
      });

  registerCudfFunctions(
      // No signatures required for cast and try_cast. They are special forms.
      {"try_cast", "cast"},
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool*) { return std::make_shared<CastFunction>(expr); },
      {
          // Cast needs special handling dynamically using cudf.
      });

  //
  // regular binary operators
  //

  auto registerBinaryOp = [&](const std::vector<std::string>& aliases,
                              cudf::binary_operator op) {
    auto decimalBinarySignature = [&]() {
      return FunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .integerVariable("b_precision")
          .integerVariable("b_scale")
          .integerVariable("r_precision")
          .integerVariable("r_scale")
          .returnType("decimal(r_precision, r_scale)")
          .argumentType("decimal(a_precision, a_scale)")
          .argumentType("decimal(b_precision, b_scale)")
          .build();
    };

    registerCudfFunctions(
        aliases,
        [op](
            const std::string&,
            const core::TypedExprPtr& expr,
            memory::MemoryPool* pool) {
          return std::make_shared<BinaryFunction>(expr, op, pool);
        },
        {FunctionSignatureBuilder()
             .returnType("double")
             .argumentType("double")
             .argumentType("double")
             .build(),
         decimalBinarySignature()});
  };

  registerBinaryOp(
      {prefix + "plus", prefix + "add"}, cudf::binary_operator::ADD);
  registerBinaryOp(
      {prefix + "minus", prefix + "subtract"}, cudf::binary_operator::SUB);
  registerBinaryOp({prefix + "multiply"}, cudf::binary_operator::MUL);
  registerBinaryOp({prefix + "divide"}, cudf::binary_operator::DIV);
  registerBinaryOp({prefix + "mod"}, cudf::binary_operator::MOD);

  //
  // regular comparison operators
  //

  const std::vector<exec::FunctionSignaturePtr> comparisonSignatures{
      FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("double")
          .argumentType("double")
          .build(),
      FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("timestamp")
          .argumentType("timestamp")
          .build(),
      FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("date")
          .argumentType("date")
          .build(),
      FunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .integerVariable("b_precision")
          .integerVariable("b_scale")
          .returnType("boolean")
          .argumentType("decimal(a_precision, a_scale)")
          .argumentType("decimal(b_precision, b_scale)")
          .build()};

  auto registerComparisonOp = [&](const std::vector<std::string>& aliases,
                                  cudf::binary_operator op) {
    registerCudfFunctions(
        aliases,
        [op](
            const std::string&,
            const core::TypedExprPtr& expr,
            memory::MemoryPool* pool) {
          return std::make_shared<BinaryFunction>(expr, op, pool);
        },
        comparisonSignatures);
  };

  registerComparisonOp(
      {prefix + "equalto", prefix + "eq"}, cudf::binary_operator::EQUAL);
  registerComparisonOp(
      {prefix + "notequalto", prefix + "neq"},
      cudf::binary_operator::NOT_EQUAL);
  registerComparisonOp(
      {prefix + "greaterthanorequal", prefix + "gte"},
      cudf::binary_operator::GREATER_EQUAL);
  registerComparisonOp(
      {prefix + "lessthanorequal", prefix + "lte"},
      cudf::binary_operator::LESS_EQUAL);
  registerComparisonOp(
      {prefix + "greaterthan", prefix + "gt"}, cudf::binary_operator::GREATER);
  registerComparisonOp(
      {prefix + "lessthan", prefix + "lt"}, cudf::binary_operator::LESS);

  //
  // regular unary operators
  //

  auto registerUnaryOp = [&](const std::vector<std::string>& aliases,
                             cudf::unary_operator op) {
    registerCudfFunctions(
        aliases,
        [op](
            const std::string&,
            const core::TypedExprPtr& expr,
            memory::MemoryPool*) {
          return std::make_shared<UnaryFunction>(expr, op);
        },
        {FunctionSignatureBuilder()
             .returnType("double")
             .argumentType("double")
             .build(),
         FunctionSignatureBuilder()
             .integerVariable("p")
             .integerVariable("s")
             .returnType("decimal(p,s)")
             .argumentType("decimal(p,s)")
             .build()});
  };

  registerUnaryOp({prefix + "abs"}, cudf::unary_operator::ABS);
  registerUnaryOp({prefix + "negate"}, cudf::unary_operator::NEGATE);
  registerUnaryOp({prefix + "floor"}, cudf::unary_operator::FLOOR);
  registerUnaryOp({prefix + "ceil"}, cudf::unary_operator::CEIL);

  // @TODO (seves 1/28/26)
  // truncate
  // no direct cudf mapping
  // perhaps a compound operation using round/round_decimal

  //
  // between
  //

  const std::vector<exec::FunctionSignaturePtr> betweenSignatures{
      FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("double")
          .argumentType("double")
          .argumentType("double")
          .build(),
      FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("timestamp")
          .argumentType("timestamp")
          .argumentType("timestamp")
          .build(),
      FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("date")
          .argumentType("date")
          .argumentType("date")
          .build(),
      FunctionSignatureBuilder()
          .integerVariable("p")
          .integerVariable("s")
          .returnType("boolean")
          .argumentType("decimal(p,s)")
          .argumentType("decimal(p,s)")
          .argumentType("decimal(p,s)")
          .build()};

  registerCudfFunction(
      prefix + "between",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<BetweenFunction>(expr, pool);
      },
      betweenSignatures);

  //
  // greatest & least
  //

  registerCudfFunction(
      prefix + "greatest",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<GreatestLeastFunction>(
            expr, cudf::binary_operator::NULL_MAX, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .variableArity("double")
           .build(),
       FunctionSignatureBuilder()
           .integerVariable("p")
           .integerVariable("s")
           .returnType("decimal(p,s)")
           .argumentType("decimal(p,s)")
           .variableArity("decimal(p,s)")
           .build()});

  registerCudfFunction(
      prefix + "least",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<GreatestLeastFunction>(
            expr, cudf::binary_operator::NULL_MIN, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .variableArity("double")
           .build(),
       FunctionSignatureBuilder()
           .integerVariable("p")
           .integerVariable("s")
           .returnType("decimal(p,s)")
           .argumentType("decimal(p,s)")
           .variableArity("decimal(p,s)")
           .build()});

  // Note: Spark and Presto functions are now registered separately via
  // registerSparkFunctions() and registerPrestoFunctions()
  return true;
}

/// Return the registry key for an expression node.
/// For calls: the function name. For casts: "cast"/"try_cast".
/// For concat: "row_constructor". Otherwise: toString().
std::string exprRegistryName(const core::TypedExprPtr& expr) {
  switch (expr->kind()) {
    case core::ExprKind::kCall:
      return expr->asUnchecked<core::CallTypedExpr>()->name();
    case core::ExprKind::kCast:
      return expr->asUnchecked<core::CastTypedExpr>()->isTryCast()
          ? expression::kTryCast
          : expression::kCast;
    case core::ExprKind::kConcat:
      return expression::kRowConstructor;
    default:
      return expr->toString();
  }
}

std::shared_ptr<FunctionExpression> FunctionExpression::create(
    const core::TypedExprPtr& expr,
    const RowTypePtr& inputRowSchema,
    memory::MemoryPool* pool,
    const core::QueryConfig& config) {
  auto node = std::make_shared<FunctionExpression>();
  node->expr_ = expr;
  node->inputRowSchema_ = inputRowSchema;

  auto name = exprRegistryName(expr);
  node->function_ = createCudfFunction(name, expr, pool);

  // For nested field accesses on computed ROW values (e.g. dereferencing the
  // result of row_constructor), pre-resolve the child index inside the parent
  // ROW so eval() can extract the right struct child without re-running the
  // CPU FieldReference resolution.
  const bool isFieldAccess = expr->isFieldAccessKind();
  const bool isDereference = expr->isDereferenceKind();
  if (isFieldAccess || isDereference) {
    if (!expr->inputs().empty() && !expr->inputs()[0]->isInputKind()) {
      const auto parentRowType = asRowType(expr->inputs()[0]->type());
      VELOX_CHECK_NOT_NULL(
          parentRowType,
          "Nested field reference parent must be a ROW: {}",
          expr->toString());
      if (isDereference) {
        // Dereference carries the child index directly; resolving by name
        // would be ambiguous for unnamed ROW fields (e.g. row_constructor).
        node->fieldIndex_ = static_cast<int32_t>(
            expr->asUnchecked<core::DereferenceTypedExpr>()->index());
      } else {
        node->fieldIndex_ = static_cast<int32_t>(parentRowType->getChildIdx(
            expr->asUnchecked<core::FieldAccessTypedExpr>()->name()));
      }
    }
  }

  if (node->function_ || isFieldAccess || isDereference) {
    for (const auto& input : expr->inputs()) {
      // Constant inputs are inlined by the cuDF function, and an InputTypedExpr
      // child marks an input-column field reference whose value eval() reads
      // directly from the input columns; neither needs its own subexpression.
      if (!input->isConstantKind() && !input->isInputKind()) {
        // Each child selects its own best evaluator via
        // createCudfExpression (e.g. AST for arithmetic, Function for
        // string ops).  Field references are handled as leaf
        // FunctionExpressions.
        node->subexpressions_.push_back(
            createCudfExpression(input, inputRowSchema, pool, config));
      }
    }
  }

  if (node->function_ && isConditionalFunction(expr)) {
    node->branchMasking_.reserve(node->subexpressions_.size());
    for (const auto& input : expr->inputs()) {
      if (input->isConstantKind() || input->isInputKind()) {
        continue;
      }
      auto& masking = node->branchMasking_.emplace_back();
      masking.mask = containsFailFastCall(input);
      if (!masking.mask) {
        continue;
      }
      masking.propagateActivity = !failFastCallsPreserveMaskedRows(input);
      masking.readsInput.resize(inputRowSchema->size(), false);
      std::unordered_set<std::string> fields;
      collectReferencedInputFields(input, fields);
      for (const auto& field : fields) {
        if (auto index = inputRowSchema->getChildIdxIfExists(field)) {
          masking.readsInput[*index] = true;
        }
      }
    }
  }

  return node;
}

std::unique_ptr<cudf::column> FunctionExpression::makeStructChildColumn(
    ColumnOrView& structColumn,
    cudf::size_type childIndex,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return std::visit(
      [&](auto& value) -> std::unique_ptr<cudf::column> {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, cudf::column_view>) {
          VELOX_CHECK(
              value.type().id() == cudf::type_id::STRUCT,
              "Nested field reference expects a ROW input");
          VELOX_CHECK_LT(static_cast<size_t>(childIndex), value.num_children());
          auto const childView =
              cudf::structs_column_view(value).get_sliced_child(
                  childIndex, stream);
          auto child = std::make_unique<cudf::column>(childView, stream, mr);
          mergeNullSourceNullsIntoResult(*child, value, stream, mr);
          return child;
        } else {
          auto const structView = value->view();
          VELOX_CHECK(
              structView.type().id() == cudf::type_id::STRUCT,
              "Nested field reference expects a ROW input");
          VELOX_CHECK_LT(
              static_cast<size_t>(childIndex), structView.num_children());

          // `structView` points into `value`. Keep `contents` alive until after
          // merging parent nulls so structView's null mask remains valid.
          auto contents = value->release();
          auto child = std::move(contents.children[childIndex]);
          mergeNullSourceNullsIntoResult(*child, structView, stream, mr);
          return child;
        }
      },
      structColumn);
}

namespace {

/// Maps each entry of subexpressions_ back to its expr_->inputs() index.
/// create() drops constant children, so `if(c, x, 1)` yields {0, 1}.
std::vector<size_t> conditionalOperandIndex(const core::TypedExprPtr& expr) {
  std::vector<size_t> indices;
  indices.reserve(expr->inputs().size());
  for (size_t input = 0; input < expr->inputs().size(); ++input) {
    const auto& child = expr->inputs()[input];
    if (!child->isConstantKind() && !child->isInputKind()) {
      indices.push_back(input);
    }
  }
  return indices;
}

/// Rows where copy_if_else takes the branch chosen by \p takeWhenTrue. It
/// selects `(valid(i) and condition[i]) ? then : else`, so a null condition row
/// belongs to the else branch alone. When \p activeRows is set, the result is
/// limited to those rows, since an enclosing branch has already excluded the
/// rest.
cuda::device_buffer<std::byte> makeBranchRowMask(
    const cudf::column_view& condition,
    bool takeWhenTrue,
    const cudf::bitmask_type* activeRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto limitToActiveRows = [&](cuda::device_buffer<std::byte> branchMask) {
    if (activeRows == nullptr || condition.size() == 0) {
      return branchMask;
    }
    const std::vector<const cudf::bitmask_type*> masks{
        activeRows,
        reinterpret_cast<const cudf::bitmask_type*>(branchMask.data())};
    const std::vector<cudf::size_type> beginBits{0, 0};
    return cudf::bitmask_and(masks, beginBits, condition.size(), stream, mr)
        .first;
  };

  // bools_to_mask clears false and null alike, already the then branch's set.
  if (takeWhenTrue) {
    auto [mask, unusedUnsetCount] = cudf::bools_to_mask(condition, stream, mr);
    return limitToActiveRows(std::move(*mask));
  }
  // The else branch also owns the null rows, and NOT maps null to null.
  auto negated =
      cudf::unary_operation(condition, cudf::unary_operator::NOT, stream, mr);
  std::unique_ptr<cudf::column> elseCondition;
  if (negated->view().has_nulls()) {
    const cudf::numeric_scalar<bool> nullsTakeElse(true, true, stream, mr);
    elseCondition =
        cudf::replace_nulls(negated->view(), nullsTakeElse, stream, mr);
  }
  auto [mask, unusedUnsetCount] = cudf::bools_to_mask(
      elseCondition ? elseCondition->view() : negated->view(), stream, mr);
  return limitToActiveRows(std::move(*mask));
}

/// Views over \p inputs' data with \p rowMask ANDed into the null mask of each
/// column \p readsInput marks, so a branch subtree sees the rows its
/// conditional discards as null. Columns the subtree never reads pass through
/// unchanged. The masks travel with the views because the views only borrow
/// them, and a subtree result can itself be a view (e.g. a bare field
/// reference).
struct MaskedInputs {
  std::vector<cudf::column_view> views;
  std::vector<cuda::device_buffer<std::byte>> masks;
};

MaskedInputs maskInputRows(
    const std::vector<cudf::column_view>& inputs,
    const std::vector<bool>& readsInput,
    const cudf::bitmask_type* rowMask,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  MaskedInputs masked;
  masked.views.reserve(inputs.size());
  masked.masks.reserve(inputs.size());
  for (size_t column = 0; column < inputs.size(); ++column) {
    const auto& input = inputs[column];
    const bool read = column >= readsInput.size() || readsInput[column];
    if (!read || input.size() == 0) {
      masked.views.push_back(input);
      continue;
    }
    // rowMask is indexed from row 0, so a view carrying its own offset would
    // read it out of step, so the branch would see the rows the conditional
    // discards: the abort this masking exists to prevent. No operator hands
    // one down today, because those that slice (CudfLimit, CudfTopN,
    // CudfLocalPartition) each materialize into a fresh cudf::table, so
    // CudfVector columns start at offset 0. Fail here rather than mask the
    // wrong rows if that ever changes; the fix is to teach this function to
    // apply the offset.
    VELOX_CHECK_EQ(
        input.offset(),
        0,
        "Non-zero column_view offset not yet supported in branch masking");
    std::vector<const cudf::bitmask_type*> masks{rowMask};
    std::vector<cudf::size_type> beginBits{0};
    if (input.nullable()) {
      masks.push_back(input.null_mask());
      beginBits.push_back(0);
    }
    auto [combined, nullCount] = cudf::bitmask_and(
        masks,
        beginBits,
        input.size(),
        stream,
        cudf::memory_resources{mr, get_temp_mr()});
    std::vector<cudf::column_view> children;
    children.reserve(input.num_children());
    for (cudf::size_type child = 0; child < input.num_children(); ++child) {
      children.push_back(input.child(child));
    }
    masked.masks.push_back(std::move(combined));
    masked.views.emplace_back(
        input.type(),
        input.size(),
        input.head<void>(),
        reinterpret_cast<const cudf::bitmask_type*>(masked.masks.back().data()),
        nullCount,
        0,
        children);
  }
  return masked;
}

/// ANDs \p activeRows into \p result's null mask. A borrowed view is copied
/// first because the combined mask must outlive this call.
ColumnOrView applyActivityMask(
    ColumnOrView result,
    const cudf::bitmask_type* activeRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (activeRows == nullptr || asView(result).size() == 0) {
    return result;
  }
  std::unique_ptr<cudf::column> owned;
  if (std::holds_alternative<std::unique_ptr<cudf::column>>(result)) {
    owned = std::move(std::get<std::unique_ptr<cudf::column>>(result));
  } else {
    owned = std::make_unique<cudf::column>(asView(result), stream, mr);
  }
  const auto view = owned->view();
  std::vector<const cudf::bitmask_type*> masks{activeRows};
  std::vector<cudf::size_type> beginBits{0};
  if (view.nullable()) {
    masks.push_back(view.null_mask());
    beginBits.push_back(0);
  }
  auto [combined, nullCount] =
      cudf::bitmask_and(masks, beginBits, view.size(), stream, mr);
  owned->set_null_mask(std::move(combined), nullCount);
  return owned;
}

} // namespace

ColumnOrView FunctionExpression::evalSubexpression(
    size_t index,
    std::vector<cudf::column_view> inputColumnViews,
    const cudf::bitmask_type* activeRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const auto& subexpr = subexpressions_[index];
  if (activeRows == nullptr) {
    return subexpr->eval(std::move(inputColumnViews), stream, mr);
  }
  if (auto* function = dynamic_cast<FunctionExpression*>(subexpr.get())) {
    return function->evalWithActivity(
        std::move(inputColumnViews), activeRows, stream, mr, false);
  }
  // AST and JIT evaluate their tree internally, so only their result can be
  // reapplied.
  return applyActivityMask(
      subexpr->eval(std::move(inputColumnViews), stream, mr),
      activeRows,
      stream,
      mr);
}

ColumnOrView FunctionExpression::eval(
    std::vector<cudf::column_view> inputColumnViews,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    bool finalize) {
  return evalWithActivity(
      std::move(inputColumnViews), nullptr, stream, mr, finalize);
}

ColumnOrView FunctionExpression::evalWithActivity(
    std::vector<cudf::column_view> inputColumnViews,
    const cudf::bitmask_type* activeRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    bool finalize) {
  // Top-level field access (or chain of field accesses on input columns) maps
  // directly to a column_view zero-copy.
  if (isInputFieldReference(expr_)) {
    auto path = extractFieldPath(expr_);
    VELOX_CHECK(path.has_value() && !path->empty());
    auto columnView =
        inputColumnViews[inputRowSchema_->getChildIdx(path->front())];
    // Walk nested struct fields using the extracted path. Use get_sliced_child
    // rather than child() so a sliced struct column's offset is applied to the
    // child; child() would return the unsliced child and misalign the rows.
    TypePtr current = inputRowSchema_->findChild(path->front());
    for (size_t i = 1; i < path->size(); ++i) {
      const auto& row = current->asRow();
      auto idx = row.getChildIdx(path->at(i));
      columnView =
          cudf::structs_column_view(columnView).get_sliced_child(idx, stream);
      current = row.childAt(idx);
    }
    return columnView;
  }

  // Nested field access on a computed ROW (e.g. dereference of
  // row_constructor): evaluate the parent expression and extract the cached
  // child index, propagating parent-struct nulls into the result.
  if (expr_->isFieldAccessKind() || expr_->isDereferenceKind()) {
    VELOX_CHECK_EQ(
        subexpressions_.size(),
        1,
        "Nested field reference expects exactly one subexpression");
    auto parent =
        evalSubexpression(0, inputColumnViews, activeRows, stream, mr);
    VELOX_DCHECK_GE(fieldIndex_, 0);
    auto child = FunctionExpression::makeStructChildColumn(
        parent, static_cast<cudf::size_type>(fieldIndex_), stream, mr);
    VELOX_DCHECK(
        child->view().type() == cudf_velox::veloxToCudfDataType(expr_->type()));
    return child;
  }

  if (function_) {
    std::vector<ColumnOrView> subexprResults;
    subexprResults.reserve(subexpressions_.size());

    // Borrowed by the masked views until after function_->eval.
    std::vector<MaskedInputs> branchInputs;
    std::vector<cuda::device_buffer<std::byte>> branchRowMasks;

    // Operand 0 is the condition, 1 and 2 the branches. Not positionally
    // aligned with subexpressions_, which omits constant branches.
    const auto operandIndex = isConditionalFunction(expr_)
        ? conditionalOperandIndex(expr_)
        : std::vector<size_t>{};

    const bool maskBranches = operandIndex.size() == subexpressions_.size() &&
        !operandIndex.empty() && operandIndex.front() == 0;

    if (maskBranches) {
      // Branches are materialized over every row before copy_if_else selects,
      // so a fail-fast call in one would abort the batch over rows the
      // conditional discards. Hand each such branch inputs whose null mask
      // excludes the rows it does not supply; the calls already skip null
      // rows. A branch without one cannot abort, and copy_if_else discards its
      // other rows. Velox CPU narrows a SelectivityVector per branch instead.
      auto condition =
          evalSubexpression(0, inputColumnViews, activeRows, stream, mr);
      const auto conditionView = asView(condition);
      subexprResults.push_back(std::move(condition));

      branchInputs.reserve(subexpressions_.size() - 1);
      branchRowMasks.reserve(subexpressions_.size() - 1);
      for (size_t branch = 1; branch < subexpressions_.size(); ++branch) {
        const auto& masking = branchMasking_[branch];
        if (!masking.mask) {
          subexprResults.push_back(
              subexpressions_[branch]->eval(inputColumnViews, stream, mr));
          continue;
        }
        branchRowMasks.push_back(makeBranchRowMask(
            conditionView,
            /*takeWhenTrue=*/operandIndex[branch] == 1,
            activeRows,
            stream,
            mr));
        const auto* branchRows = reinterpret_cast<const cudf::bitmask_type*>(
            branchRowMasks.back().data());
        branchInputs.push_back(maskInputRows(
            inputColumnViews, masking.readsInput, branchRows, stream, mr));
        // Masked inputs suffice while every path to a fail-fast call
        // propagates nulls. A null-eliminating node such as coalesce can turn
        // an excluded row valid again, so such a branch reapplies its mask
        // after every intermediate result instead.
        const bool propagate =
            activeRows != nullptr || masking.propagateActivity;
        subexprResults.push_back(evalSubexpression(
            branch,
            branchInputs.back().views,
            propagate ? branchRows : nullptr,
            stream,
            mr));
      }
    } else {
      for (size_t i = 0; i < subexpressions_.size(); ++i) {
        subexprResults.push_back(
            evalSubexpression(i, inputColumnViews, activeRows, stream, mr));
      }
    }

    auto result = function_->eval(subexprResults, stream, mr);
    if (finalize) {
      const auto requestedType = cudf_velox::veloxToCudfDataType(expr_->type());
      auto resultView = asView(result);
      if (resultView.type() != requestedType) {
        result = cudf::cast(resultView, requestedType, stream, mr);
      }
    }
    return applyActivityMask(std::move(result), activeRows, stream, mr);
  }

  VELOX_FAIL(
      "Unsupported expression for recursive evaluation: " +
      exprRegistryName(expr_));
}

void FunctionExpression::close() {
  function_.reset();
  subexpressions_.clear();
}

bool FunctionExpression::canEvaluate(const core::TypedExprPtr& expr) {
  if (isInputFieldReference(expr)) {
    return true;
  }

  // Field access / dereference on a computed ROW value (e.g. dereferencing the
  // result of row_constructor). The parent expression is validated separately
  // by the recursive deep check; here we only confirm the shape so eval() can
  // extract the struct child. Input-column references are handled above by
  // isInputFieldReference.
  if (expr->isFieldAccessKind() || expr->isDereferenceKind()) {
    return !expr->inputs().empty() && !expr->inputs()[0]->isInputKind() &&
        expr->inputs()[0]->type()->kind() == TypeKind::ROW;
  }

  const auto opName = exprRegistryName(expr);
  if (expr->isCastKind()) {
    const auto& srcType =
        expr->inputs().empty() ? nullptr : expr->inputs()[0]->type();
    const auto& dstType = expr->type();
    if (srcType == nullptr || dstType == nullptr) {
      return false;
    }
    auto src = cudf_velox::tryVeloxToCudfDataType(srcType);
    auto dst = cudf_velox::tryVeloxToCudfDataType(dstType);
    if (!src || !dst) {
      return false;
    }
    // After expression::optimize(), which canExprRunOnGpu() applies, a cast
    // of a constant is only left unfolded when it fails; the CPU raises its
    // error. CastFunction gets no column for a constant, so such a cast stays
    // on the CPU, as does one in a tree that was not optimized.
    if (expr->inputs()[0]->isConstantKind()) {
      return false;
    }
    if (srcType->isVarchar()) {
      return isCastFromVarcharEnabled(srcType, dstType);
    }
    return cudf::is_supported_cast(*src, *dst);
  }

  auto& registry = getCudfFunctionRegistry();
  auto it = registry.find(opName);
  if (it == registry.end()) {
    return false;
  }
  for (const auto& spec : it->second) {
    if (!spec.signatures.empty() &&
        !matchCallAgainstSignatures(expr, spec.signatures)) {
      continue;
    }
    if (spec.canEvaluate && !spec.canEvaluate(expr)) {
      continue;
    }
    return true;
  }
  return false;
}

std::optional<std::vector<std::string>> extractFieldPath(
    const core::TypedExprPtr& expr) {
  if (expr == nullptr) {
    return std::nullopt;
  }

  if (expr->isFieldAccessKind()) {
    const auto* field = expr->asUnchecked<core::FieldAccessTypedExpr>();
    if (field->inputs().empty() || field->inputs()[0]->isInputKind()) {
      return std::vector<std::string>{field->name()};
    }

    auto path = extractFieldPath(field->inputs()[0]);
    if (!path.has_value()) {
      return std::nullopt;
    }
    path->push_back(field->name());
    return path;
  }

  if (expr->isDereferenceKind()) {
    const auto* dereference = expr->asUnchecked<core::DereferenceTypedExpr>();
    auto path = extractFieldPath(dereference->inputs()[0]);
    if (!path.has_value()) {
      return std::nullopt;
    }
    path->push_back(dereference->name());
    return path;
  }

  return std::nullopt;
}

std::unordered_set<std::string> referencedInputFields(
    const core::TypedExprPtr& expr) {
  std::unordered_set<std::string> fields;
  collectReferencedInputFields(expr, fields);
  return fields;
}

namespace {

// True if the expression tree contains a timezone-sensitive date_trunc call.
// date_trunc on a timestamp needs the session timezone when
// adjust_timestamp_to_session_timezone is enabled, which cuDF cannot honor, so
// such trees must stay on CPU.
bool containsTimezoneSensitiveDateTrunc(const core::TypedExprPtr& expr) {
  const auto dateTruncName =
      CudfConfig::getInstance().functionNamePrefix + "date_trunc";
  if (expr->kind() == core::ExprKind::kCall &&
      exprRegistryName(expr) == dateTruncName &&
      DateTruncFunction::isTimezoneSensitive(expr)) {
    return true;
  }
  for (const auto& input : expr->inputs()) {
    if (containsTimezoneSensitiveDateTrunc(input)) {
      return true;
    }
  }
  return false;
}

// True if `expr` must fall back to CPU because it contains a timezone-sensitive
// date_trunc while the session enables adjust_timestamp_to_session_timezone,
// which cuDF cannot honor. False when `queryCtx` is null or the config is
// disabled.
bool requiresCpuForTimezone(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx) {
  if (queryCtx == nullptr ||
      !queryCtx->queryConfig().adjustTimestampToTimezone()) {
    return false;
  }
  if (containsTimezoneSensitiveDateTrunc(expr)) {
    LOG_FALLBACK(
        "date_trunc(timestamp) requires CPU evaluation when "
        "adjust_timestamp_to_session_timezone is enabled");
    return true;
  }
  return false;
}

// Returns true if 'expr' must stay on the CPU because it casts from VARCHAR
// while a session config selects rules CastFunction does not implement:
// legacy_cast, for every target, and
// cast_string_to_floating_point_lenient_inf_nan, which admits inf, infinity
// and nan in any letter case, for REAL and DOUBLE. False when 'queryCtx' is
// null or both configs are off.
bool requiresCpuForCastConfig(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx) {
  if (queryCtx == nullptr) {
    return false;
  }
  const auto& config = queryCtx->queryConfig();
  if (config.isLegacyCast() &&
      containsCastFromVarchar(expr, /*floatingPointOnly=*/false)) {
    LOG_FALLBACK("A cast from VARCHAR stays on the CPU under legacy_cast");
    return true;
  }
  if (config.isCastStringToFloatingPointLenientInfNan() &&
      containsCastFromVarchar(expr, /*floatingPointOnly=*/true)) {
    LOG_FALLBACK(
        "A cast from VARCHAR to REAL or DOUBLE stays on the CPU under "
        "cast_string_to_floating_point_lenient_inf_nan");
    return true;
  }
  return false;
}

// Optimizes 'expr' the way the operators do before compiling, so that a support
// check sees the same tree; see canExprRunOnGpu() in the header. Returns 'expr'
// itself without a query context or pool.
core::TypedExprPtr optimizeForCheck(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  return (queryCtx != nullptr && pool != nullptr)
      ? expression::optimize(expr, queryCtx, pool)
      : expr;
}

// Decides canExprRunOnGpu() for an expression optimizeForCheck() returned.
bool canOptimizedExprRunOnGpu(
    const core::TypedExprPtr& checked,
    core::QueryCtx* queryCtx) {
  return !requiresCpuForTimezone(checked, queryCtx) &&
      !requiresCpuForCastConfig(checked, queryCtx) &&
      canBeEvaluatedByCudf(checked);
}

} // namespace

bool containsCastFromVarchar(
    const core::TypedExprPtr& expr,
    bool floatingPointOnly) {
  if (expr->isCastKind() && !expr->inputs().empty() &&
      expr->inputs()[0]->type()->isVarchar() &&
      (!floatingPointOnly || expr->type()->isReal() ||
       expr->type()->isDouble())) {
    return true;
  }
  for (const auto& input : expr->inputs()) {
    if (containsCastFromVarchar(input, floatingPointOnly)) {
      return true;
    }
  }
  return false;
}

bool canExprRunOnGpu(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  return canOptimizedExprRunOnGpu(
      optimizeForCheck(expr, queryCtx, pool), queryCtx);
}

bool canJoinFilterRunOnGpu(
    const core::TypedExprPtr& filter,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  // The filter is optimized first, so that a cast of a literal folds away
  // before the check.
  const auto checked = optimizeForCheck(filter, queryCtx, pool);
  if (containsFailingCastFromVarchar(checked)) {
    LOG_FALLBACK(
        "CAST from VARCHAR in a join filter stays on the CPU, which casts "
        "only the pairs the join matches");
    return false;
  }
  return canOptimizedExprRunOnGpu(checked, queryCtx);
}

std::shared_ptr<CudfExpression> createCudfExpression(
    const core::TypedExprPtr& expr,
    const RowTypePtr& inputRowSchema,
    memory::MemoryPool* pool,
    const core::QueryConfig& config) {
  const auto* best = findBestEvaluator(expr);
  VELOX_CHECK_NOT_NULL(
      best, "No cuDF expression evaluator can handle: {}", expr->toString());
  return best->create(expr, inputRowSchema, pool, config);
}

void unregisterFunctions() {
  auto& registry = getCudfFunctionRegistry();
  registry.clear();
  castFromVarcharEnabled = false;
}

void setCastFromVarcharEnabled(bool enabled) {
  castFromVarcharEnabled = enabled;
}

} // namespace facebook::velox::cudf_velox
