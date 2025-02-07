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
#include "velox/core/PlanNode.h"
#include "velox/exec/Driver.h"
#include "velox/exec/FilterProject.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/expression/Expr.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

namespace facebook::velox::cudf_velox {

// Copied from cudf 24.12, TODO: remove this after cudf is updated
/**
 * @brief An AST expression tree. It owns and contains multiple dependent
 * expressions. All the expressions are destroyed when the tree is destroyed.
 */
class tree {
 public:
  /**
   * @brief construct an empty ast tree
   */
  tree() = default;

  /**
   * @brief Moves the ast tree
   */
  tree(tree&&) = default;

  /**
   * @brief move-assigns the AST tree
   * @returns a reference to the move-assigned tree
   */
  tree& operator=(tree&&) = default;

  ~tree() = default;

  // the tree is not copyable
  tree(tree const&) = delete;
  tree& operator=(tree const&) = delete;

  /**
   * @brief Add an expression to the AST tree
   * @param args Arguments to use to construct the ast expression
   * @returns a reference to the added expression
   */
  template <typename Expr, typename... Args>
  std::enable_if_t<std::is_base_of_v<cudf::ast::expression, Expr>, Expr const&>
  emplace(Args&&... args) {
    auto expr = std::make_unique<Expr>(std::forward<Args>(args)...);
    Expr const& expr_ref = *expr;
    expressions.emplace_back(std::move(expr));
    return expr_ref;
  }

  /**
   * @brief Add an expression to the AST tree
   * @param expr AST expression to be added
   * @returns a reference to the added expression
   */
  template <typename Expr>
  decltype(auto) push(Expr expr) {
    return emplace<Expr>(std::move(expr));
  }

  /**
   * @brief get the first expression in the tree
   * @returns the first inserted expression into the tree
   */
  [[nodiscard]] cudf::ast::expression const& front() const {
    return *expressions.front();
  }

  /**
   * @brief get the last expression in the tree
   * @returns the last inserted expression into the tree
   */
  [[nodiscard]] cudf::ast::expression const& back() const {
    return *expressions.back();
  }

  /**
   * @brief get the number of expressions added to the tree
   * @returns the number of expressions added to the tree
   */
  [[nodiscard]] size_t size() const {
    return expressions.size();
  }

  /**
   * @brief get the expression at an index in the tree. Index is checked.
   * @param index index of expression in the ast tree
   * @returns the expression at the specified index
   */
  cudf::ast::expression const& at(size_t index) {
    return *expressions.at(index);
  }

  /**
   * @brief get the expression at an index in the tree. Index is unchecked.
   * @param index index of expression in the ast tree
   * @returns the expression at the specified index
   */
  cudf::ast::expression const& operator[](size_t index) const {
    return *expressions[index];
  }

 private:
  // TODO: use better ownership semantics, the unique_ptr here is redundant.
  // Consider using a bump allocator with type-erased deleters.
  std::vector<std::unique_ptr<cudf::ast::expression>> expressions;
};

// TODO: Does not support Filter yet.
class CudfFilterProject : public exec::Operator {
 public:
  CudfFilterProject(
      int32_t operatorId,
      velox::exec::DriverCtx* driverCtx,
      const velox::exec::FilterProject::Export& info,
      std::vector<velox::exec::IdentityProjection> identityProjections,
      const std::shared_ptr<const core::FilterNode>& filter,
      const std::shared_ptr<const core::ProjectNode>& project);

  bool needsInput() const override {
    return !input_;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

  // TODO rewrite this.
  void close() override {
    Operator::close();
    projectAst_.clear();
    scalars_.clear();
  }

 private:
  bool allInputProcessed();
  // If true exprs_[0] is a filter and the other expressions are projections
  const bool hasFilter_{false};
  // Cached filter and project node for lazy initialization. After
  // initialization, they will be reset, and initialized_ will be set to true.
  std::shared_ptr<const core::ProjectNode> project_;
  std::shared_ptr<const core::FilterNode> filter_;
  std::vector<tree> projectAst_;
  std::vector<std::unique_ptr<cudf::scalar>> scalars_;

  std::vector<velox::exec::IdentityProjection> resultProjections_;
  std::vector<velox::exec::IdentityProjection> identityProjections_;
};

} // namespace facebook::velox::cudf_velox
