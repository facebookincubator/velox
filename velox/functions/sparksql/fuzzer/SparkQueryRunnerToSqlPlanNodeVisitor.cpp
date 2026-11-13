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

#include "velox/functions/sparksql/fuzzer/SparkQueryRunnerToSqlPlanNodeVisitor.h"

#include <cmath>
#include <iomanip>
#include <optional>

#include <folly/String.h>

#include "velox/exec/fuzzer/ReferenceQueryRunner.h"
#include "velox/type/Timestamp.h"

namespace facebook::velox::functions::sparksql::fuzzer {
namespace {

std::optional<std::string> toSparkTypeSql(const TypePtr& type) {
  if (type->isDecimal()) {
    const auto [precision, scale] = getDecimalPrecisionScale(*type);
    return fmt::format("DECIMAL({}, {})", precision, scale);
  }
  if (type->isDate()) {
    return "DATE";
  }
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return "BOOLEAN";
    case TypeKind::TINYINT:
      return "TINYINT";
    case TypeKind::SMALLINT:
      return "SMALLINT";
    case TypeKind::INTEGER:
      return "INT";
    case TypeKind::BIGINT:
      return "BIGINT";
    case TypeKind::REAL:
      return "FLOAT";
    case TypeKind::DOUBLE:
      return "DOUBLE";
    case TypeKind::VARCHAR:
      return "STRING";
    case TypeKind::VARBINARY:
      return "BINARY";
    case TypeKind::TIMESTAMP:
      return "TIMESTAMP";
    case TypeKind::ARRAY:
    case TypeKind::MAP:
    case TypeKind::ROW: {
      std::vector<std::string> children;
      for (auto i = 0; i < type->size(); ++i) {
        const auto child = toSparkTypeSql(type->childAt(i));
        if (!child) {
          return std::nullopt;
        }
        if (type->isRow()) {
          std::stringstream name;
          name << std::quoted(type->asRow().nameOf(i), '`', '`');
          children.push_back(fmt::format("{}: {}", name.str(), *child));
        } else {
          children.push_back(*child);
        }
      }
      const auto name = type->isArray() ? "ARRAY"
          : type->isMap()               ? "MAP"
                                        : "STRUCT";
      return fmt::format("{}<{}>", name, folly::join(", ", children));
    }
    default:
      return std::nullopt;
  }
}

std::optional<std::string> toSparkConstantSql(
    const core::ConstantTypedExpr& constant) {
  const auto& type = constant.type();
  const auto typeSql = toSparkTypeSql(type);
  if (!typeSql) {
    return std::nullopt;
  }
  const auto value = constant.toVariant();
  if (value.isNull()) {
    return fmt::format("cast(null as {})", *typeSql);
  }
  if (type->isTimestamp()) {
    const auto timestamp = value.value<TypeKind::TIMESTAMP>();
    // The Spark fuzzer generates timestamps representable in microseconds.
    VELOX_CHECK_EQ(timestamp.getNanos() % 1'000, 0);
    VELOX_CHECK(timestamp >= Timestamp::minMicros());
    VELOX_CHECK(timestamp <= Timestamp::maxMicros());
    return fmt::format("timestamp_micros({})", timestamp.toMicros());
  }
  if (type->isVarchar() || type->isVarbinary()) {
    const auto& bytes = type->isVarbinary() ? value.value<TypeKind::VARBINARY>()
                                            : value.value<TypeKind::VARCHAR>();
    // Avoid dependence on Spark's string-literal escape configuration.
    const auto binary = fmt::format("X'{}'", folly::hexlify(bytes));
    return type->isVarbinary() ? binary
                               : fmt::format("cast({} as STRING)", binary);
  }
  if (type->isArray() || type->isMap() || type->isRow()) {
    std::vector<std::string> elements;
    const auto append = [&](const TypePtr& childType, const Variant& child) {
      const auto sql =
          toSparkConstantSql(core::ConstantTypedExpr(childType, child));
      if (!sql) {
        return false;
      }
      elements.push_back(*sql);
      return true;
    };
    if (type->isArray()) {
      for (const auto& element : value.array()) {
        if (!append(type->childAt(0), element)) {
          return std::nullopt;
        }
      }
    } else if (type->isMap()) {
      for (const auto& [key, element] : value.map()) {
        if (!append(type->childAt(0), key) ||
            !append(type->childAt(1), element)) {
          return std::nullopt;
        }
      }
    } else {
      const auto& fields = value.row();
      for (auto i = 0; i < fields.size(); ++i) {
        if (!append(type->childAt(i), fields[i])) {
          return std::nullopt;
        }
      }
    }
    const auto name = type->isArray() ? "array"
        : type->isMap()               ? "map"
                                      : "struct";
    return fmt::format(
        "cast({}({}) as {})", name, folly::join(", ", elements), *typeSql);
  }
  const auto literal = value.toStringAsVector(type);
  if (type->isDate()) {
    return fmt::format("DATE '{}'", literal);
  }
  if (type->isDecimal()) {
    return fmt::format("cast({}BD as {})", literal, *typeSql);
  }
  if (type->isReal() || type->isDouble()) {
    const double number =
        type->isReal() ? value.value<float>() : value.value<double>();
    if (!std::isfinite(number)) {
      const auto special = std::isnan(number) ? "NaN"
          : number < 0                        ? "-Infinity"
                                              : "Infinity";
      return fmt::format("cast('{}' as {})", special, *typeSql);
    }
  }
  const auto suffix = type->isTinyint() ? "Y"
      : type->isSmallint()              ? "S"
      : type->isBigint()                ? "L"
      : type->isReal()                  ? "F"
      : type->isDouble()                ? "D"
                                        : "";
  return literal + suffix;
}

std::optional<std::string> toSparkAggregateCallSql(
    const core::AggregationNode::Aggregate& aggregate) {
  std::stringstream sql;
  sql << aggregate.call->name() << "(";
  if (aggregate.distinct) {
    sql << "distinct ";
  }
  const auto& inputs = aggregate.call->inputs();
  for (auto i = 0; i < inputs.size(); ++i) {
    exec::test::appendComma(i, sql);
    if (auto constant =
            std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
                inputs[i])) {
      const auto constantSql = toSparkConstantSql(*constant);
      if (!constantSql) {
        return std::nullopt;
      }
      sql << *constantSql;
    } else {
      exec::test::toCallInputsSql({inputs[i]}, sql);
    }
  }
  sql << ")";
  return sql.str();
}

} // namespace

void SparkQueryRunnerToSqlPlanNodeVisitor::visit(
    const core::AggregationNode& node,
    core::PlanNodeVisitorContext& ctx) const {
  // Assume plan is Aggregation over Values.
  VELOX_CHECK(node.step() == core::AggregationNode::Step::kSingle);

  exec::test::PrestoSqlPlanNodeVisitorContext& visitorContext =
      static_cast<exec::test::PrestoSqlPlanNodeVisitorContext&>(ctx);

  std::vector<std::string> groupingKeys;
  for (const auto& key : node.groupingKeys()) {
    groupingKeys.push_back(key->name());
  }

  std::stringstream sql;
  sql << "SELECT " << folly::join(", ", groupingKeys);

  const auto& aggregates = node.aggregates();
  if (!aggregates.empty()) {
    if (!groupingKeys.empty()) {
      sql << ", ";
    }

    for (auto i = 0; i < aggregates.size(); ++i) {
      exec::test::appendComma(i, sql);
      const auto& aggregate = aggregates[i];
      // Native bloom filters use a different binary format and hash scheme.
      // Compare equivalent native plans, not their serialized bytes to Spark.
      if (aggregate.call->name() == "bloom_filter_agg") {
        visitorContext.sql = std::nullopt;
        return;
      }
      VELOX_CHECK(
          aggregate.sortingKeys.empty(),
          "Sort key is not supported in Spark's aggregation. You may need to disable 'enable_sorted_aggregations' when running the fuzzer test.");
      const auto aggregateSql = toSparkAggregateCallSql(aggregate);
      if (!aggregateSql) {
        visitorContext.sql = std::nullopt;
        return;
      }
      sql << *aggregateSql;

      if (aggregate.mask != nullptr) {
        sql << " filter (where " << aggregate.mask->name() << ")";
      }
      sql << " as " << node.aggregateNames()[i];
    }
  }

  // AggregationNode should have a single source.
  const auto source = toSql(node.sources()[0]);
  if (!source) {
    visitorContext.sql = std::nullopt;
    return;
  }
  sql << " FROM (" << *source << ")";

  if (!groupingKeys.empty()) {
    sql << " GROUP BY " << folly::join(", ", groupingKeys);
  }

  visitorContext.sql = sql.str();
}

void SparkQueryRunnerToSqlPlanNodeVisitor::visit(
    const core::ProjectNode& node,
    core::PlanNodeVisitorContext& ctx) const {
  exec::test::PrestoSqlPlanNodeVisitorContext& visitorContext =
      static_cast<exec::test::PrestoSqlPlanNodeVisitorContext&>(ctx);

  const auto sourceSql = toSql(node.sources()[0]);
  if (!sourceSql.has_value()) {
    visitorContext.sql = std::nullopt;
    return;
  }

  std::stringstream sql;
  sql << "SELECT ";

  for (auto i = 0; i < node.names().size(); ++i) {
    exec::test::appendComma(i, sql);
    auto projection = node.projections()[i];
    if (auto field =
            std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
                projection)) {
      sql << field->name();
    } else if (
        auto call =
            std::dynamic_pointer_cast<const core::CallTypedExpr>(projection)) {
      sql << exec::test::toCallSql(call);
    } else {
      VELOX_NYI(
          "Unsupported projection {} in project node: {}.",
          projection->toString(),
          node.toString());
    }

    sql << " as " << node.names()[i];
  }

  sql << " FROM (" << sourceSql.value() << ")";
  visitorContext.sql = sql.str();
}

void SparkQueryRunnerToSqlPlanNodeVisitor::visit(
    const core::ValuesNode& node,
    core::PlanNodeVisitorContext& ctx) const {
  exec::test::PrestoSqlPlanNodeVisitorContext& visitorContext =
      static_cast<exec::test::PrestoSqlPlanNodeVisitorContext&>(ctx);
  visitorContext.sql = exec::test::ReferenceQueryRunner::getTableName(node);
}

} // namespace facebook::velox::functions::sparksql::fuzzer
