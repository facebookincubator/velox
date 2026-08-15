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

#include "velox/core/ITypedExpr.h"

#include "velox/common/EnumDefine.h"

namespace facebook::velox::core {

namespace {
const auto& exprKindNames() {
  static const folly::F14FastMap<ExprKind, std::string_view> kNames = {
      {ExprKind::kInput, "INPUT"},
      {ExprKind::kFieldAccess, "FIELD"},
      {ExprKind::kDereference, "DEREFERENCE"},
      {ExprKind::kCall, "CALL"},
      {ExprKind::kCast, "CAST"},
      {ExprKind::kConstant, "CONSTANT"},
      {ExprKind::kConcat, "CONCAT"},
      {ExprKind::kLambda, "LAMBDA"},
  };
  return kNames;
}
} // namespace

VELOX_DEFINE_ENUM_NAME(ExprKind, exprKindNames);

namespace {
void countReferences(
    const ITypedExpr& expr,
    folly::F14FastMap<const ITypedExpr*, uint32_t>& refCounts) {
  auto [it, inserted] = refCounts.emplace(&expr, 1);
  if (!inserted) {
    // Already visited this node's subtree; just record the extra reference.
    ++it->second;
    return;
  }
  for (const auto& input : expr.inputs()) {
    countReferences(*input, refCounts);
  }
}
} // namespace

void ITypedExpr::appendToString(std::string& out, ExprToStringContext& /*ctx*/)
    const {
  out += toString();
}

void ITypedExpr::appendWithSharing(std::string& out, ExprToStringContext& ctx)
    const {
  if (auto it = ctx.labels.find(this); it != ctx.labels.end()) {
    out += '#';
    out += std::to_string(it->second);
    return;
  }

  appendToString(out, ctx);

  // Label a shared non-leaf node the first time it is printed, so later
  // occurrences reference it by "#N" instead of re-expanding it. Leaves are
  // cheap to reprint, so they are left unlabeled to avoid noise.
  if (!inputs().empty()) {
    if (auto it = ctx.refCounts.find(this);
        it != ctx.refCounts.end() && it->second > 1) {
      const uint32_t id = ctx.nextLabel++;
      ctx.labels.emplace(this, id);
      out += " as #";
      out += std::to_string(id);
    }
  }
}

std::string ITypedExpr::toStringWithSharing() const {
  ExprToStringContext ctx;
  countReferences(*this, ctx.refCounts);
  std::string out;
  appendWithSharing(out, ctx);
  return out;
}

size_t ITypedExprHasher::operator()(const ITypedExpr* expr) const {
  return expr->hash();
}

bool ITypedExprComparer::operator()(
    const ITypedExpr* lhs,
    const ITypedExpr* rhs) const {
  return *lhs == *rhs;
}
} // namespace facebook::velox::core
