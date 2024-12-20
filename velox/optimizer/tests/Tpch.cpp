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

#include "velox/optimizer/tests/Tpch.h"
#include "velox/tpch/gen/TpchGen.h"

namespace facebook::velox::optimizer {

using namespace facebook::velox;
// Returns a map from column name to cardinality. Positive means fixed value
// count, negative means the count is scale * (-count).
const std::unordered_map<std::string, int64_t>& tpchColumns() {
  static std::unordered_map<std::string, int64_t> columns;
  if (columns.empty()) {
    columns["o_orderkey"] = -1500000;
    columns["o_custkey"] = -150000;

    columns["l_orderkey"] = -1500000;
    columns["l_linenumber"] = 4;
    columns["l_partkey"] = -200000;
    columns["l_suppkey"] = 10000;
    columns["l_discount"] = 10;
    columns["l_tax"] = 10;
    columns["l_shipmode"] = 7;
    columns["l_shipdate"] = 7 * 365;
    columns["l_commitdate"] = 7 * 365;
    columns["l_receiptdate"] = 7 * 365;

    columns["c_custkey"] = -150000;
    columns["c_mktsegment"] = 5;

    columns["p_partkey"] = -200000;
    columns["s_suppkey"] = -10000;
    columns["ps_partkey"] = -200000;
    columns["ps_suppkey"] = -10000;

    columns["n_nationkey"] = 25;
    columns["n_name"] = 25;
    columns["n_regionkey"] = 5;
    columns["r_regionkey"] = 5;
    columns["r_name"] = 5;
  }
  return columns;
}

Value columnValue(
    const std::string& name,
    const velox::TypePtr& type,
    int32_t scale,
    int64_t cardinality) {
  auto& columns = tpchColumns();
  auto it = columns.find(name);
  if (it == columns.end()) {
    return Value(type.get(), std::min<int64_t>(5, cardinality / 1000));
  }
  return Value(type.get(), it->second > 0 ? it->second : -it->second * scale);
}

SchemaTableCP makeTable(
    tpch::Table id,
    int32_t scale,
    bool partitioned,
    bool ordered,
    bool secondary) {
  VELOX_CHECK(!secondary, "Secondary indices not implemented");
  auto cardinality = tpch::getRowCount(id, scale);

  auto type = tpch::getTableSchema(id);
  auto tableName = tpch::toTableName(id);
  auto* table = make<SchemaTable>(
      toName(std::string(tableName.data(), tableName.size())), type);
  ColumnVector orderedColumns;
  for (auto i = 0; i < type->size(); ++i) {
    auto name = toName(type->nameOf(i));
    auto value = columnValue(name, type->childAt(i), scale, cardinality);
    orderedColumns.push_back(table->column(name, value));
  }
  int32_t numOrder =
      id == tpch::Table::TBL_LINEITEM || id == tpch::Table::TBL_PARTSUPP ? 2
                                                                         : 1;
  auto pkColumns = orderedColumns;
  if (id == tpch::Table::TBL_LINEITEM) {
    // Swap 2nd and 4th so that l_linenumber is second, since sorting or
    // uniqueness defining columns must be first in index.
    std::swap(pkColumns[1], pkColumns[3]);
  }
  ColumnVector partition;
  DistributionType dist;
  if (partitioned) {
    partition.push_back(pkColumns[0]);
    dist = DistributionType{ShuffleMode::kHive, 100};
  }
  table->addIndex(
      "pk",
      cardinality,
      numOrder,
      ordered ? numOrder : 0,
      pkColumns,
      dist,
      partition,
      orderedColumns);
  return table;
}

SchemaP
tpchSchema(int32_t scale, bool partitioned, bool ordered, bool secondary) {
  auto title =
      fmt::format("tpch{}{}", partitioned ? "p" : "", ordered ? "o" : "");
  std::vector<SchemaTableCP> tables{
      makeTable(
          tpch::Table::TBL_LINEITEM, scale, partitioned, ordered, secondary),
      makeTable(
          tpch::Table::TBL_ORDERS, scale, partitioned, ordered, secondary),
      makeTable(
          tpch::Table::TBL_CUSTOMER, scale, partitioned, ordered, secondary),
      makeTable(tpch::Table::TBL_PART, scale, partitioned, ordered, secondary),
      makeTable(
          tpch::Table::TBL_PARTSUPP, scale, partitioned, ordered, secondary),
      makeTable(
          tpch::Table::TBL_SUPPLIER, scale, partitioned, ordered, secondary),
      makeTable(
          tpch::Table::TBL_NATION, scale, partitioned, ordered, secondary),
      makeTable(
          tpch::Table::TBL_REGION, scale, partitioned, ordered, secondary),
  };

  return make<Schema>(toName(title), std::move(tables), nullptr);
}

} // namespace facebook::velox::optimizer
