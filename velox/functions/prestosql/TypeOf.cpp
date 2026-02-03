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
#include "velox/expression/VectorFunction.h"
#include "velox/functions/prestosql/types/BigintEnumType.h"
#include "velox/functions/prestosql/types/HyperLogLogType.h"
#include "velox/functions/prestosql/types/IPPrefixType.h"
#include "velox/functions/prestosql/types/KHyperLogLogType.h"
#include "velox/functions/prestosql/types/P4HyperLogLogType.h"
#include "velox/functions/prestosql/types/QDigestType.h"
#include "velox/functions/prestosql/types/SetDigestType.h"
#include "velox/functions/prestosql/types/TDigestType.h"
#include "velox/functions/prestosql/types/VarcharEnumType.h"

namespace facebook::velox::functions {
namespace {

// Converts a TypePtr to its string representation. Handles most types by
// default, but provides special handling for types with mixed casing, custom
// names, or complex formatting.
std::string typeName(const TypePtr& type) {
  // Handle decimal types with precision and scale
  if (type->isDecimal()) {
    auto precision = type->isShortDecimal() ? type->asShortDecimal().precision()
                                            : type->asLongDecimal().precision();
    auto scale = type->isShortDecimal() ? type->asShortDecimal().scale()
                                        : type->asLongDecimal().scale();
    return fmt::format("decimal({},{})", precision, scale);
  }

  // Handle complex types with recursive formatting
  if (type->kind() == TypeKind::ARRAY) {
    return fmt::format("array({})", typeName(type->childAt(0)));
  }

  if (type->kind() == TypeKind::MAP) {
    return fmt::format(
        "map({}, {})", typeName(type->childAt(0)), typeName(type->childAt(1)));
  }

  if (type->kind() == TypeKind::ROW) {
    if (isIPPrefixType(type)) {
      return "ipprefix";
    }
    const auto& rowType = type->asRow();
    std::ostringstream out;
    out << "row(";
    for (auto i = 0; i < type->size(); ++i) {
      if (i > 0) {
        out << ", ";
      }
      if (!rowType.nameOf(i).empty()) {
        out << "\"" << rowType.nameOf(i) << "\" ";
      }
      out << typeName(type->childAt(i));
    }
    out << ")";
    return out.str();
  }

  // Handle enum types that have custom names
  if (isBigintEnumType(*type)) {
    return asBigintEnum(type)->enumName();
  }
  if (isVarcharEnumType(*type)) {
    return asVarcharEnum(type)->enumName();
  }

  // Handle HyperLogLog types that use mixed case
  if (isHyperLogLogType(type)) {
    return "HyperLogLog";
  }
  if (isP4HyperLogLogType(type)) {
    return "P4HyperLogLog";
  }
  if (isKHyperLogLogType(type)) {
    return "KHyperLogLog";
  }

  // Handle SetDigest types
  if (isSetDigestType(type)) {
    return "SetDigest";
  }

  // Handle special digest types that need parameter formatting
  if (*type == *TDIGEST(DOUBLE())) {
    return "tdigest(double)";
  }
  if (*type == *QDIGEST(BIGINT())) {
    return "qdigest(bigint)";
  }
  if (*type == *QDIGEST(REAL())) {
    return "qdigest(real)";
  }
  if (*type == *QDIGEST(DOUBLE())) {
    return "qdigest(double)";
  }

  if (type->kind() == TypeKind::UNKNOWN) {
    return "unknown";
  }

  if (type->isVarcharN()) {
    auto length = getVarcharLength(*type);
    return fmt::format("varchar({})", length);
  }

  // Handle unsupported types
  if (type->kind() == TypeKind::OPAQUE || type->kind() == TypeKind::FUNCTION ||
      type->kind() == TypeKind::INVALID) {
    VELOX_UNSUPPORTED("Unsupported type: {}", type->toString());
  }

  // Default: use type->name() and lowercase it
  std::string name = type->name();
  folly::toLowerAscii(name);
  return name;
}

class TypeOfFunction : public exec::VectorFunction {
 public:
  TypeOfFunction(const TypePtr& type) : typeName_{typeName(type)} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    auto localResult = BaseVector::createConstant(
        VARCHAR(), typeName_, rows.size(), context.pool());
    context.moveOrCopyResult(localResult, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // T -> varchar
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("varchar")
                .argumentType("T")
                .build()};
  }

  static std::shared_ptr<exec::VectorFunction> create(
      const std::string& /*name*/,
      const std::vector<exec::VectorFunctionArg>& inputArgs,
      const core::QueryConfig& /*config*/) {
    try {
      return std::make_shared<TypeOfFunction>(inputArgs[0].type);
    } catch (...) {
      return std::make_shared<exec::AlwaysFailingVectorFunction>(
          std::current_exception());
    }
  }

 private:
  const std::string typeName_;
};
} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION_WITH_METADATA(
    udf_typeof,
    TypeOfFunction::signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    TypeOfFunction::create);

} // namespace facebook::velox::functions
