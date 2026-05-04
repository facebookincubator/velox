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
#include "velox/parse/ExprRewriter.h"

#include "velox/parse/Expressions.h"

namespace facebook::velox::core {

namespace {

std::vector<ExprPtr> rewriteAll(
    const std::vector<ExprPtr>& exprs,
    const ExprRewriter::Transform& transform) {
  std::vector<ExprPtr> result;
  result.reserve(exprs.size());
  for (const auto& expr : exprs) {
    result.push_back(ExprRewriter::rewrite(expr, transform));
  }
  return result;
}

std::vector<SortKey> rewriteSortKeys(
    const std::vector<SortKey>& keys,
    const ExprRewriter::Transform& transform) {
  std::vector<SortKey> result;
  result.reserve(keys.size());
  for (const auto& key : keys) {
    result.push_back(
        {ExprRewriter::rewrite(key.expr, transform),
         key.ascending,
         key.nullsFirst});
  }
  return result;
}

} // namespace

ExprPtr ExprRewriter::rewrite(const ExprPtr& expr, const Transform& transform) {
  switch (expr->kind()) {
    case IExpr::Kind::kAggregate: {
      auto* agg = expr->as<AggregateCallExpr>();
      auto rewritten = std::make_shared<AggregateCallExpr>(
          agg->name(),
          rewriteAll(agg->inputs(), transform),
          agg->isDistinct(),
          agg->filter() ? rewrite(agg->filter(), transform) : nullptr,
          rewriteSortKeys(agg->orderBy(), transform),
          agg->alias());
      return transform(std::move(rewritten));
    }
    case IExpr::Kind::kWindow: {
      auto* window = expr->as<WindowCallExpr>();
      std::optional<WindowCallExpr::Frame> frame = window->frame();
      if (frame.has_value()) {
        if (frame->startValue) {
          frame->startValue = rewrite(frame->startValue, transform);
        }
        if (frame->endValue) {
          frame->endValue = rewrite(frame->endValue, transform);
        }
      }
      auto rewritten = std::make_shared<WindowCallExpr>(
          window->name(),
          rewriteAll(window->inputs(), transform),
          rewriteAll(window->partitionKeys(), transform),
          rewriteSortKeys(window->orderByKeys(), transform),
          std::move(frame),
          window->isIgnoreNulls(),
          window->alias());
      return transform(std::move(rewritten));
    }
    default: {
      const auto& inputs = expr->inputs();
      if (inputs.empty()) {
        return transform(expr);
      }
      return transform(expr->replaceInputs(rewriteAll(inputs, transform)));
    }
  }
}

} // namespace facebook::velox::core
