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

#include "velox/functions/lib/SimpleComparisonMatcher.h"

namespace facebook::velox::functions::sparksql {
using namespace facebook::velox::functions;

namespace {
const std::string eq = "equalto";
const std::string lt = "lessthan";
const std::string gt = "greaterthan";
} // namespace

class SparkComparisonMatcher : public ComparisonMatcher {
 public:
  SparkComparisonMatcher(
      const std::string& prefix,
      std::vector<MatcherPtr> inputMatchers,
      std::string* op)
      : ComparisonMatcher(prefix, inputMatchers, op) {}

  bool exprNameMatch(const std::string& name) override {
    return name == prefix_ + eq || name == prefix_ + lt || name == prefix_ + gt;
  }
};

class SparkSimpleComparisonChecker : public SimpleComparisonChecker {
 public:
  ~SparkSimpleComparisonChecker() override = default;

 protected:
  MatcherPtr comparison(
      const std::string& prefix,
      const MatcherPtr& left,
      const MatcherPtr& right,
      std::string* op) override {
    return std::make_shared<SparkComparisonMatcher>(
        prefix, std::vector<MatcherPtr>{left, right}, op);
  }

  std::string eqName(const std::string& prefix) override {
    return prefix + eq;
  }

  std::string ltName(const std::string& prefix) override {
    return prefix + lt;
  }

  std::string gtName(const std::string& prefix) override {
    return prefix + gt;
  }
};

} // namespace facebook::velox::functions::sparksql
