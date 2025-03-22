#
#  Copyright owned by the Transaction Processing Performance Council.
#
#  A copy of the license is included under external/dsdgen/LICENSE
#  in this repository.
#
#  You may not use this file except in compliance with the License.
#
#  THE TPC SOFTWARE IS AVAILABLE WITHOUT CHARGE FROM TPC.
#
cmake_minimum_required(VERSION 3.0.0)
cmake_policy(SET CMP0063 NEW)

project(dsdgen CXX C)

add_library(
  dsdgen OBJECT
  skip_days.cpp
  address.cpp
  build_support.cpp
  date.cpp
  dbgen_version.cpp
  decimal.cpp
  dist.cpp
  error_msg.cpp
  genrand.cpp
  join.cpp
  list.cpp
  load.cpp
  misc.cpp
  nulls.cpp
  parallel.cpp
  permute.cpp
  pricing.cpp
  r_params.cpp
  scaling.cpp
  scd.cpp
  sparse.cpp
  StringBuffer.cpp
  tdef_functions.cpp
  tdefs.cpp
  text.cpp
  w_call_center.cpp
  w_catalog_page.cpp
  w_catalog_returns.cpp
  w_catalog_sales.cpp
  w_customer.cpp
  w_customer_address.cpp
  w_customer_demographics.cpp
  w_datetbl.cpp
  w_household_demographics.cpp
  w_income_band.cpp
  w_inventory.cpp
  w_item.cpp
  w_promotion.cpp
  w_reason.cpp
  w_ship_mode.cpp
  w_store.cpp
  w_store_returns.cpp
  w_store_sales.cpp
  w_timetbl.cpp
  w_warehouse.cpp
  w_web_page.cpp
  w_web_returns.cpp
  w_web_sales.cpp
  w_web_site.cpp)

target_include_directories(dsdgen PRIVATE include)

# Include tpcds_idx.hpp
target_include_directories(dsdgen
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Suppress warnings when compiling dsdgen.
target_compile_options(dsdgen PRIVATE -w)
