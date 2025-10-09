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

#include "velox/core/Expressions.h"

namespace facebook::velox::expression::utils {

/// Indicates the type of inputs of an expression.
/// kAllConstant: All inputs are of constant kind.
/// kAllField: All inputs are of field kind.
/// kConstantOrField: All inputs are of either constant or field kind.
/// kDefault: At least one input is not of constant or field kind.
enum ExprInputsKind { kAllConstant = 0, kAllField, kConstantOrField, kDefault };

/// Helper function to get ExprInputsKind for an expression.
ExprInputsKind getExprInputsKind(const core::TypedExprPtr& expr);

/// Returns true if expr is of type CallTypedExpr and the expr name matches
/// the passed name. Otherwise, returns false.
bool isCall(const core::TypedExprPtr& expr, const std::string& name);

/// Returns true when call expression contains an associative operator like and,
/// or, plus, multiply. The call expression name is checked to determine the
/// operator it represents, fully qualified names with a prefix of format
/// 'catalog.schema.' are handled.
bool isAssociativeOperator(const core::CallTypedExpr* call);

/// Utility method to check eligibility for flattening. Returns true if all
/// inputs to expr have the same type.
bool allInputTypesEquivalent(const core::TypedExprPtr& expr);

/// Recursively flattens nested ANDs, ORs or eligible callable expressions into
/// a vector of their inputs. Recursive flattening ceases exploring an input
/// branch if it encounters either an expression different from 'flattenCall' or
/// its inputs are not the same type. Examples:
/// flattenCall: AND
/// in: a AND (b AND (c AND d))
/// out: [a, b, c, d]
///
/// flattenCall: OR
/// in: (a OR b) OR (c OR d)
/// out: [a, b, c, d]
///
/// flattenCall: concat
/// in: (array1, concat(array2, concat(array3, array4))
/// out: [array1, array2, array3, array4]
void flattenInput(
    const core::TypedExprPtr& input,
    const std::string& flattenCall,
    std::vector<core::TypedExprPtr>& flat);

} // namespace facebook::velox::expression::utils
