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

#include "velox/tpch/gen/SsbGen.h"
#include "velox/external/duckdb/tpch/dbgen/include/dbgen/dbgen_gunk.hpp"
#include "velox/external/duckdb/tpch/dbgen/include/dbgen/dss.h"
#include "velox/external/duckdb/tpch/dbgen/include/dbgen/dsstypes.h"
#include "velox/tpch/gen/DBGenIterator.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::ssb {

namespace {

// The cardinality of the LINEITEM table is not a strict multiple of SF since
// the number of lineitems in an order is chosen at random with an average of
// four. This function contains the row count for all authorized scale factors
// (as described by the TPC-H spec), and approximates the remaining.
constexpr size_t getLineItemRowCount(double scaleFactor) {
  auto longScaleFactor = static_cast<long>(scaleFactor);
  switch (longScaleFactor) {
    case 1:
      return 6'001'215;
    case 10:
      return 59'986'052;
    case 30:
      return 179'998'372;
    case 100:
      return 600'037'902;
    case 300:
      return 1'799'989'091;
    case 1'000:
      return 5'999'989'709;
    case 3'000:
      return 18'000'048'306;
    case 10'000:
      return 59'999'994'267;
    case 30'000:
      return 179'999'978'268;
    case 100'000:
      return 599'999'969'200;
    default:
      break;
  }
  return 6'000'000 * scaleFactor;
}

size_t getVectorSize(size_t rowCount, size_t maxRows, size_t offset) {
  if (offset >= rowCount) {
    return 0;
  }
  return std::min(rowCount - offset, maxRows);
}

std::vector<VectorPtr> allocateVectors(
    const RowTypePtr& type,
    size_t vectorSize,
    memory::MemoryPool* pool) {
  std::vector<VectorPtr> vectors;
  vectors.reserve(type->size());

  for (const auto& childType : type->children()) {
    vectors.emplace_back(BaseVector::create(childType, vectorSize, pool));
  }
  return vectors;
}

double decimalToDouble(int64_t value) {
  return (double)value * 0.01;
}

Date toDate(std::string_view stringDate) {
  Date date;
  parseTo(stringDate, date);
  return date;
}

} // namespace

std::string_view toTableName(SSB_Table table) {
  switch (table) {
    case SSB_Table ::TBL_PART:
      return "part";
    case SSB_Table ::TBL_SUPPLIER:
      return "supplier";
    case SSB_Table ::TBL_CUSTOMER:
      return "customer";
    case SSB_Table ::TBL_LINEORDER:
      return "";
    case SSB_Table ::TBL_DATE:
      return "date";
  }
  return ""; // make gcc happy.
}

SSB_Table fromTableName(std::string_view tableName) {
  static std::unordered_map<std::string_view, SSB_Table > map{
      {"part", SSB_Table ::TBL_PART},
      {"supplier", SSB_Table ::TBL_SUPPLIER},
      {"customer", SSB_Table ::TBL_CUSTOMER},
      {"lineorder", SSB_Table ::TBL_LINEORDER},
      {"date", SSB_Table ::TBL_DATE},
  };

  auto it = map.find(tableName);
  if (it != map.end()) {
    return it->second;
  }
  throw std::invalid_argument(
      fmt::format("Invalid TPC-H table name: '{}'", tableName));
}
size_t getRowCount(SSB_Table table, double scaleFactor) {return 0;}
/* size_t getRowCount(SSB_Table table, double scaleFactor) {
  VELOX_CHECK_GE(scaleFactor, 0, "Ssb scale factor must be non-negative");
  switch (table) {
    case SSB_Table ::TBL_PART:
      return 200'000 * scaleFactor;
    case SSB_Table ::TBL_SUPPLIER:
      return 10'000 * scaleFactor;
    case SSB_Table ::TBL_PARTSUPP:
      return 800'000 * scaleFactor;
    case SSB_Table ::TBL_CUSTOMER:
      return 150'000 * scaleFactor;
    case SSB_Table ::TBL_ORDERS:
      return 1'500'000 * scaleFactor;
    case SSB_Table ::TBL_NATION:
      return 25;
    case SSB_Table ::TBL_REGION:
      return 5;
    case SSB_Table ::TBL_LINEITEM:
      return getLineItemRowCount(scaleFactor);
  }
  return 0; // make gcc happy.
} */

RowTypePtr getTableSchema(SSB_Table table) {
  switch (table) {
    case SSB_Table ::TBL_PART: {
      static RowTypePtr type = ROW(
          {
              "p_partkey",
              "p_name",
              "p_mfgr",
              "p_category",
              "p_brand1",
              "p_color",
              "p_type",
              "p_size",
              "p_container",
          },
          {
              BIGINT(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
              BIGINT(),
              VARCHAR(),
          });
      return type;
    }

    case SSB_Table ::TBL_SUPPLIER: {
      static RowTypePtr type = ROW(
          {
              "s_suppkey",
              "s_name",
              "s_address",
              "s_city",
              "s_nation",
              "s_region",
              "s_phone",
          },
          {
              BIGINT(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
          });
      return type;
    }
    case SSB_Table ::TBL_CUSTOMER: {
      static RowTypePtr type = ROW(
          {
              "c_custkey",
              "c_name",
              "c_address",
              "c_city",
              "c_nation",
              "c_region",
              "c_phone",
              "c_mktsegment",
          },
          {
              BIGINT(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
          });
      return type;
    }

    case SSB_Table ::TBL_LINEORDER: {
      static RowTypePtr type = ROW(
          {
              "lo_orderkey",
              "lo_linenumber",    
              "lo_custkey",
              "lo_partkey",     
              "lo_suppkey",     
              "lo_orderdate",     
              "lo_orderpriority",
              "lo_shippriority",  
              "lo_quantity",      
              "lo_extendedprice", 
              "lo_ordtotalprice", 
              "lo_discount",      
              "lo_revenue",       
              "lo_supplycost",    
              "lo_tax",           
              "lo_commitdate", 
              "lo_shipmod",        

          },
          {
              BIGINT(),
              INTEGER(),
              INTEGER(),
              INTEGER(),
              INTEGER(),
              INTEGER(),
              VARCHAR(),
              VARCHAR(),
              INTEGER(),
              INTEGER(),
              INTEGER(),
              INTEGER(),
              INTEGER(),
              INTEGER(),
              INTEGER(),
              INTEGER(),
              VARCHAR(),
          });
      return type;
    }

    case SSB_Table ::TBL_DATE: {
      static RowTypePtr type = ROW(
          {
              "d_datekey",
              "d_date",
              "d_dayofweek",
              "d_month",            
              "d_year",             
              "d_yearmonthnum",     
              "d_yearmonth",       
              "d_daynuminweek",     
              "d_daynuminmonth",    
              "d_daynuminyear",     
              "d_monthnuminyear",   
              "d_weeknuminyear",    
              "d_sellingseason",    
              "d_lastdayinweekfl",  
              "d_lastdayinmonthfl", 
              "d_holidayfl",       
              "d_weekdayfl",        
          },
          {
              INTEGER(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
              INTEGER(),
              INTEGER(),
              VARCHAR(),
              INTEGER(),
              INTEGER(),
              INTEGER(),
              INTEGER(),
              INTEGER(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
              VARCHAR(),
          });
      return type;
    }
  }
  return nullptr; // make gcc happy.
}

TypePtr resolveSsbColumn(SSB_Table table, const std::string& columnName) {
  return getTableSchema(table)->findChild(columnName);
}


} // namespace facebook::velox::ssb
