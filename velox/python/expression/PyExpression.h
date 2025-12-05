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

#include <string>

#include "velox/parse/Expressions.h"
#include "velox/parse/IExpr.h"
#include "velox/type/Variant.h"

namespace facebook::velox::py {

/// A thin wrapper around Velox untyped expressions.
class PyExpr {
 public:
  explicit PyExpr(const core::ExprPtr expr = nullptr) : expr_(expr) {}

  std::string toString() const {
    if (!expr_) {
      return "[nullptr]";
    }
    return expr_->toString();
  }

  core::ExprPtr expr() const {
    return expr_;
  }

  static PyExpr createFieldAccess(const std::string& field) {
    return PyExpr{std::make_shared<core::FieldAccessExpr>(field, std::nullopt)};
  }

  static PyExpr createConstant(const TypePtr& type, const Variant& value) {
    return PyExpr{
        std::make_shared<core::ConstantExpr>(type, value, std::nullopt)};
  }

  static PyExpr createCall(
      const std::string& name,
      std::vector<core::ExprPtr> args) {
    return PyExpr{
        std::make_shared<core::CallExpr>(name, std::move(args), std::nullopt)};
  }

 private:
  core::ExprPtr expr_;
};

} // namespace facebook::velox::py
