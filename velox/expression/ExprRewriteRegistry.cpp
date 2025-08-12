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
#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/expression/FunctionSignature.h"

namespace facebook::velox::exec {

namespace {
ExpressionRewriteRegistry& expressionRewriteRegistryInternal() {
  static ExpressionRewriteRegistry instance;
  return instance;
}
} // namespace

void ExpressionRewriteRegistry::registerExpressionRewrite(
    const std::string& name,
    std::unique_ptr<expression::ExpressionRewrite> expressionRewrite) {
  const auto sanitizedName = sanitizeName(name);
  rewriteRegistry_.withWLock(
      [&](auto& map) { map[sanitizedName] = std::move(expressionRewrite); });
}

void ExpressionRewriteRegistry::unregisterExpressionRewrites() {
  rewriteRegistry_.withWLock([&](auto& map) { map.clear(); });
}

expression::ExpressionRewrite* FOLLY_NULLABLE
ExpressionRewriteRegistry::getExpressionRewrite(const std::string& name) const {
  const auto sanitizedName = sanitizeName(name);
  expression::ExpressionRewrite* rewrite = nullptr;
  rewriteRegistry_.withRLock([&](const auto& map) {
    auto it = map.find(sanitizedName);
    if (it != map.end()) {
      rewrite = it->second.get();
    }
  });
  return rewrite;
}

std::vector<std::string> ExpressionRewriteRegistry::getExpressionRewriteNames()
    const {
  std::vector<std::string> names;
  rewriteRegistry_.withRLock([&](const auto& map) {
    names.reserve(map.size());
    for (const auto& [name, _] : map) {
      names.push_back(name);
    }
  });
  return names;
}

const ExpressionRewriteRegistry& expressionRewriteRegistry() {
  return expressionRewriteRegistryInternal();
}

ExpressionRewriteRegistry& mutableExpressionRewriteRegistry() {
  return expressionRewriteRegistryInternal();
}

void registerExpressionRewrite(
    const std::string& name,
    std::unique_ptr<expression::ExpressionRewrite> expressionRewrite) {
  mutableExpressionRewriteRegistry().registerExpressionRewrite(
      name, std::move(expressionRewrite));
}

void unregisterExpressionRewrites() {
  mutableExpressionRewriteRegistry().unregisterExpressionRewrites();
}
} // namespace facebook::velox::exec
