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

#include "velox/expression/Expr.h"
#include "velox/expression/ExprCompiler.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"

#include <memory>
#include <string>

namespace facebook::velox::cudf_velox::test_utils {

// Parse SQL and infer types only, without compiling to exec::Expr
inline core::TypedExprPtr parseAndInferTypedExpr(
    const std::string& sql,
    const RowTypePtr& rowType,
    core::ExecCtx* execCtx,
    const parse::ParseOptions& options = {}) {
  auto untyped = parse::DuckSqlExpressionsParser(options).parseExpr(sql);
  return core::Expressions::inferTypes(untyped, rowType, execCtx->pool());
}

inline std::shared_ptr<exec::Expr> compileExecExpr(
    const std::string& sql,
    const RowTypePtr& rowType,
    core::ExecCtx* execCtx,
    const parse::ParseOptions& options = {}) {
  auto typed = parseAndInferTypedExpr(sql, rowType, execCtx, options);
  exec::ExprSet exprSet({typed}, execCtx, /*enableConstantFolding*/ false);
  return exprSet.expr(0);
}

} // namespace facebook::velox::cudf_velox::test_utils
