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

// This file contains the definition of the FunctionBinder class extracted from
// the duckdb-internal.hpp file. See commit:
// https://github.com/duckdb/duckdb/commit/e9f04f9149081aeffa4a51672898be6839eba009
// It is not possible to include duckdb-internal.hpp directly because it
// contains an incompatible / outdated copy of fmt/format.h.

/*
Copyright 2018-2022 Stichting DuckDB Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "velox/external/duckdb/duckdb.hpp"

namespace duckdb {

//! The FunctionBinder class is responsible for binding functions
class FunctionBinder {
 public:
  DUCKDB_API explicit FunctionBinder(ClientContext& context);

  ClientContext& context;

 public:
  //! Bind a scalar function from the set of functions and input arguments.
  //! Returns the index of the chosen function, returns
  //! DConstants::INVALID_INDEX and sets error if none could be found
  DUCKDB_API idx_t BindFunction(
      const string& name,
      ScalarFunctionSet& functions,
      const vector<LogicalType>& arguments,
      string& error);
  DUCKDB_API idx_t BindFunction(
      const string& name,
      ScalarFunctionSet& functions,
      vector<unique_ptr<Expression>>& arguments,
      string& error);
  //! Bind an aggregate function from the set of functions and input arguments.
  //! Returns the index of the chosen function, returns
  //! DConstants::INVALID_INDEX and sets error if none could be found
  DUCKDB_API idx_t BindFunction(
      const string& name,
      AggregateFunctionSet& functions,
      const vector<LogicalType>& arguments,
      string& error);
  DUCKDB_API idx_t BindFunction(
      const string& name,
      AggregateFunctionSet& functions,
      vector<unique_ptr<Expression>>& arguments,
      string& error);
  //! Bind a table function from the set of functions and input arguments.
  //! Returns the index of the chosen function, returns
  //! DConstants::INVALID_INDEX and sets error if none could be found
  DUCKDB_API idx_t BindFunction(
      const string& name,
      TableFunctionSet& functions,
      const vector<LogicalType>& arguments,
      string& error);
  DUCKDB_API idx_t BindFunction(
      const string& name,
      TableFunctionSet& functions,
      vector<unique_ptr<Expression>>& arguments,
      string& error);
  //! Bind a pragma function from the set of functions and input arguments
  DUCKDB_API idx_t BindFunction(
      const string& name,
      PragmaFunctionSet& functions,
      PragmaInfo& info,
      string& error);

  DUCKDB_API unique_ptr<Expression> BindScalarFunction(
      const string& schema,
      const string& name,
      vector<unique_ptr<Expression>> children,
      string& error,
      bool is_operator = false,
      Binder* binder = nullptr);
  DUCKDB_API unique_ptr<Expression> BindScalarFunction(
      ScalarFunctionCatalogEntry& function,
      vector<unique_ptr<Expression>> children,
      string& error,
      bool is_operator = false,
      Binder* binder = nullptr);

  DUCKDB_API unique_ptr<BoundFunctionExpression> BindScalarFunction(
      ScalarFunction bound_function,
      vector<unique_ptr<Expression>> children,
      bool is_operator = false);

  DUCKDB_API unique_ptr<BoundAggregateExpression> BindAggregateFunction(
      AggregateFunction bound_function,
      vector<unique_ptr<Expression>> children,
      unique_ptr<Expression> filter = nullptr,
      AggregateType aggr_type = AggregateType::NON_DISTINCT,
      unique_ptr<BoundOrderModifier> order_bys = nullptr);

  DUCKDB_API unique_ptr<FunctionData> BindSortedAggregate(
      AggregateFunction& bound_function,
      vector<unique_ptr<Expression>>& children,
      unique_ptr<FunctionData> bind_info,
      unique_ptr<BoundOrderModifier> order_bys);

 private:
  //! Cast a set of expressions to the arguments of this function
  void CastToFunctionArguments(
      SimpleFunction& function,
      vector<unique_ptr<Expression>>& children);
  int64_t BindVarArgsFunctionCost(
      const SimpleFunction& func,
      const vector<LogicalType>& arguments);
  int64_t BindFunctionCost(
      const SimpleFunction& func,
      const vector<LogicalType>& arguments);

  template <class T>
  vector<idx_t> BindFunctionsFromArguments(
      const string& name,
      FunctionSet<T>& functions,
      const vector<LogicalType>& arguments,
      string& error);

  template <class T>
  idx_t MultipleCandidateException(
      const string& name,
      FunctionSet<T>& functions,
      vector<idx_t>& candidate_functions,
      const vector<LogicalType>& arguments,
      string& error);

  template <class T>
  idx_t BindFunctionFromArguments(
      const string& name,
      FunctionSet<T>& functions,
      const vector<LogicalType>& arguments,
      string& error);

  vector<LogicalType> GetLogicalTypesFromExpressions(
      vector<unique_ptr<Expression>>& arguments);
};

} // namespace duckdb
