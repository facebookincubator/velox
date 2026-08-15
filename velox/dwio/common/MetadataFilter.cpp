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

#include "velox/dwio/common/MetadataFilter.h"

#include <folly/container/F14Map.h>
#include "velox/dwio/common/ScanSpec.h"
#include "velox/expression/ExprConstants.h"
#include "velox/expression/ExprToSubfieldFilter.h"

namespace facebook::velox::common {

namespace {
using LeafResults =
    folly::F14FastMap<const MetadataFilter::LeafNode*, std::vector<uint64_t>*>;
}

struct MetadataFilter::Node {
  static std::unique_ptr<Node> fromExpression(
      const core::ITypedExpr&,
      core::ExpressionEvaluator*,
      bool negated);
  virtual ~Node() = default;
  virtual void addToScanSpec(ScanSpec&) const = 0;
  virtual uint64_t* eval(LeafResults&, int size) const = 0;
  virtual std::string toString() const = 0;
};

class MetadataFilter::LeafNode : public Node {
 public:
  LeafNode(Subfield&& field, std::unique_ptr<Filter> filter)
      : field_(std::move(field)), filter_(std::move(filter)) {}

  void addToScanSpec(ScanSpec& scanSpec) const override {
    scanSpec.getOrCreateChild(field_)->addMetadataFilter(this, filter_.get());
  }

  uint64_t* eval(LeafResults& leafResults, int) const override {
    if (auto it = leafResults.find(this); it != leafResults.end()) {
      return it->second->data();
    }
    return nullptr;
  }

  const Subfield& field() const {
    return field_;
  }

  std::string toString() const override {
    return field_.toString() + ":" + filter_->toString();
  }

 private:
  Subfield field_;
  std::unique_ptr<Filter> filter_;
};

struct MetadataFilter::ConditionNode : Node {
  static std::unique_ptr<Node> create(
      bool conjuction,
      std::vector<std::unique_ptr<Node>> args);

  static std::unique_ptr<Node> fromExpression(
      const std::vector<core::TypedExprPtr>& inputs,
      core::ExpressionEvaluator* evaluator,
      bool conjunction,
      bool negated) {
    conjunction = negated ? !conjunction : conjunction;
    std::vector<std::unique_ptr<Node>> args;
    args.reserve(inputs.size());
    for (const auto& input : inputs) {
      auto node = Node::fromExpression(*input, evaluator, negated);
      if (node) {
        args.push_back(std::move(node));
      } else if (!conjunction) {
        return nullptr;
      }
    }
    return create(conjunction, std::move(args));
  }

  explicit ConditionNode(std::vector<std::unique_ptr<Node>> args)
      : args_{std::move(args)} {}

  void addToScanSpec(ScanSpec& scanSpec) const final {
    for (const auto& arg : args_) {
      arg->addToScanSpec(scanSpec);
    }
  }

 protected:
  std::string ToStringImpl(std::string_view prefix) const {
    std::string result{prefix};
    for (size_t i = 0; i < args_.size(); ++i) {
      if (i != 0) {
        result += ",";
      }
      result += args_[i]->toString();
    }
    result += ")";
    return result;
  }

  std::vector<std::unique_ptr<Node>> args_;
};

struct MetadataFilter::AndNode final : ConditionNode {
  using ConditionNode::ConditionNode;

  uint64_t* eval(LeafResults& leafResults, int size) const final {
    uint64_t* result = nullptr;
    for (const auto& arg : args_) {
      auto* a = arg->eval(leafResults, size);
      if (!a) {
        continue;
      }
      if (!result) {
        result = a;
      } else {
        bits::orBits(result, a, 0, size);
      }
    }
    return result;
  }

  std::string toString() const final {
    return ToStringImpl("and(");
  }
};

struct MetadataFilter::OrNode final : ConditionNode {
  using ConditionNode::ConditionNode;

  uint64_t* eval(LeafResults& leafResults, int size) const final {
    uint64_t* result = nullptr;
    for (const auto& arg : args_) {
      auto* a = arg->eval(leafResults, size);
      if (!a) {
        return nullptr;
      }
      if (!result) {
        result = a;
      } else {
        bits::andBits(result, a, 0, size);
      }
    }
    return result;
  }

  std::string toString() const final {
    return ToStringImpl("or(");
  }
};

std::unique_ptr<MetadataFilter::Node> MetadataFilter::ConditionNode::create(
    bool conjunction,
    std::vector<std::unique_ptr<Node>> args) {
  if (args.empty()) {
    return nullptr;
  }
  if (args.size() == 1) {
    return std::move(args[0]);
  }
  if (conjunction) {
    return std::make_unique<AndNode>(std::move(args));
  }
  return std::make_unique<OrNode>(std::move(args));
}

namespace {

const core::CallTypedExpr* asCall(const core::ITypedExpr* expr) {
  return dynamic_cast<const core::CallTypedExpr*>(expr);
}

} // namespace

std::unique_ptr<MetadataFilter::Node> MetadataFilter::Node::fromExpression(
    const core::ITypedExpr& expr,
    core::ExpressionEvaluator* evaluator,
    bool negated) {
  auto* call = asCall(&expr);
  if (!call) {
    return nullptr;
  }
  if (call->name() == expression::kAnd) {
    return ConditionNode::fromExpression(
        call->inputs(), evaluator, true, negated);
  }
  if (call->name() == expression::kOr) {
    return ConditionNode::fromExpression(
        call->inputs(), evaluator, false, negated);
  }
  if (call->name() == "not") {
    return fromExpression(*call->inputs()[0], evaluator, !negated);
  }
  try {
    auto subfieldAndFilter =
        exec::ExprToSubfieldFilterParser::getInstance()
            ->leafCallToSubfieldFilter(*call, evaluator, negated);
    if (!subfieldAndFilter.has_value()) {
      return nullptr;
    }

    auto& [subfield, filter] = subfieldAndFilter.value();
    VELOX_CHECK(
        subfield.valid(),
        "Invalid subfield from expression: {}",
        expr.toString());
    return std::make_unique<LeafNode>(std::move(subfield), std::move(filter));
  } catch (const VeloxException&) {
    LOG(WARNING) << "Fail to convert expression to metadata filter: "
                 << expr.toString();
    return nullptr;
  }
}

MetadataFilter::MetadataFilter(
    ScanSpec& scanSpec,
    const core::ITypedExpr& expr,
    core::ExpressionEvaluator* evaluator)
    : root_(Node::fromExpression(expr, evaluator, false)) {
  if (root_) {
    root_->addToScanSpec(scanSpec);
  }
}

void MetadataFilter::eval(
    std::vector<std::pair<const LeafNode*, std::vector<uint64_t>>>&
        leafNodeResults,
    std::vector<uint64_t>& finalResult) {
  if (!root_) {
    return;
  }

  LeafResults leafResults;
  for (auto& [leaf, result] : leafNodeResults) {
    VELOX_CHECK_EQ(
        result.size(),
        finalResult.size(),
        "Result size mismatch: {}",
        leaf->field().toString());
    VELOX_CHECK(
        leafResults.emplace(leaf, &result).second,
        "Duplicate results: {}",
        leaf->field().toString());
  }
  const auto bitCount = finalResult.size() * 64;
  if (auto* combined = root_->eval(leafResults, bitCount)) {
    bits::orBits(finalResult.data(), combined, 0, bitCount);
  }
}

std::string MetadataFilter::toString() const {
  return !root_ ? "" : root_->toString();
}

} // namespace facebook::velox::common
