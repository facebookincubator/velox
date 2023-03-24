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

#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/common/Statistics.h"

namespace facebook::velox::common {

ScanSpec& ScanSpec::operator=(const ScanSpec& other) {
  if (this != &other) {
    numReads_ = other.numReads_;
    subscript_ = other.subscript_;
    fieldName_ = other.fieldName_;
    channel_ = other.channel_;
    constantValue_ = other.constantValue_;
    projectOut_ = other.projectOut_;
    extractValues_ = other.extractValues_;
    makeFlat_ = other.makeFlat_;
    filter_ = other.filter_;
    metadataFilters_ = other.metadataFilters_;
    selectivity_ = other.selectivity_;
    enableFilterReorder_ = other.enableFilterReorder_;
    children_ = other.children_;
    stableChildren_ = other.stableChildren_;
    valueHook_ = other.valueHook_;
    isArrayElementOrMapEntry_ = other.isArrayElementOrMapEntry_;
    maxArrayElementsCount_ = other.maxArrayElementsCount_;
  }
  return *this;
}

ScanSpec* ScanSpec::getOrCreateChild(const Subfield& subfield) {
  auto container = this;
  auto& path = subfield.path();
  for (size_t depth = 0; depth < path.size(); ++depth) {
    auto element = path[depth].get();
    bool found = false;
    for (auto& field : container->children_) {
      if (field->matches(*element)) {
        container = field.get();
        found = true;
        break;
      }
    }
    if (!found) {
      container->children_.push_back(std::make_unique<ScanSpec>(*element));
      container = container->children_.back().get();
    }
  }
  return container;
}

uint64_t ScanSpec::newRead() {
  if (!numReads_) {
    reorder();
  } else if (enableFilterReorder_) {
    for (auto i = 1; i < children_.size(); ++i) {
      if (!children_[i]->filter_) {
        break;
      }
      if (children_[i - 1]->selectivity_.timeToDropValue() >
          children_[i]->selectivity_.timeToDropValue()) {
        reorder();
        break;
      }
    }
  }
  return numReads_++;
}

void ScanSpec::reorder() {
  if (children_.empty()) {
    return;
  }
  // Make sure 'stableChildren_' is initialized.
  stableChildren();
  std::sort(
      children_.begin(),
      children_.end(),
      [this](
          const std::shared_ptr<ScanSpec>& left,
          const std::shared_ptr<ScanSpec>& right) {
        if (left->hasFilter() && right->hasFilter()) {
          if (enableFilterReorder_ &&
              (left->selectivity_.numIn() || right->selectivity_.numIn())) {
            return left->selectivity_.timeToDropValue() <
                right->selectivity_.timeToDropValue();
          }
          // Integer filters are before other filters if there is no
          // history data.
          if (left->filter_ && right->filter_) {
            return left->filter_->kind() < right->filter_->kind();
          }
          // If hasFilter() is true but 'filter_' is nullptr, we have a filter
          // on complex type members. The simple type filter goes first.
          if (left->filter_) {
            return true;
          }
          if (right->filter_) {
            return false;
          }
          return left->fieldName_ < right->fieldName_;
        }
        if (left->hasFilter()) {
          return true;
        }
        if (right->hasFilter()) {
          return false;
        }
        return left->fieldName_ < right->fieldName_;
      });
}

const std::vector<ScanSpec*>& ScanSpec::stableChildren() {
  std::lock_guard<std::mutex> l(mutex_);
  if (stableChildren_.empty()) {
    stableChildren_.reserve(children_.size());
    for (auto& child : children_) {
      stableChildren_.push_back(child.get());
    }
  }
  return stableChildren_;
}

bool ScanSpec::hasFilter() const {
  if (hasFilter_.has_value()) {
    return hasFilter_.value();
  }
  if (!isConstant() && filter_) {
    hasFilter_ = true;
    return true;
  }
  for (auto& child : children_) {
    if (!child->isArrayElementOrMapEntry_ && child->hasFilter()) {
      hasFilter_ = true;
      return true;
    }
  }
  hasFilter_ = false;
  return false;
}

void ScanSpec::moveAdaptationFrom(ScanSpec& other) {
  // moves the filters and filter order from 'other'.
  std::vector<std::shared_ptr<ScanSpec>> newChildren;
  for (auto& otherChild : other.children_) {
    bool found = false;
    for (auto& child : children_) {
      if (child && child->fieldName_ == otherChild->fieldName_) {
        if (!child->isConstant() && !otherChild->isConstant()) {
          // If other child is constant, a possible filter on a
          // constant will have been evaluated at split start time. If
          // 'child' is constant there is no adaptation that can be
          // received.
          child->filter_ = std::move(otherChild->filter_);
          child->selectivity_ = otherChild->selectivity_;
        }
        newChildren.push_back(std::move(child));
        found = true;
        break;
      }
    }
    VELOX_CHECK(found);
  }
  children_ = std::move(newChildren);
  stableChildren_.clear();
  for (auto& otherChild : other.stableChildren_) {
    auto child = childByName(otherChild->fieldName_);
    VELOX_CHECK(child);
    stableChildren_.push_back(child);
  }
}

namespace {
bool testIntFilter(
    common::Filter* filter,
    dwio::common::IntegerColumnStatistics* intStats,
    bool mayHaveNull) {
  if (!intStats) {
    return true;
  }

  if (intStats->getMinimum().has_value() &&
      intStats->getMaximum().has_value()) {
    return filter->testInt64Range(
        intStats->getMinimum().value(),
        intStats->getMaximum().value(),
        mayHaveNull);
  }

  // only min value
  if (intStats->getMinimum().has_value()) {
    return filter->testInt64Range(
        intStats->getMinimum().value(),
        std::numeric_limits<int64_t>::max(),
        mayHaveNull);
  }

  // only max value
  if (intStats->getMaximum().has_value()) {
    return filter->testInt64Range(
        std::numeric_limits<int64_t>::min(),
        intStats->getMaximum().value(),
        mayHaveNull);
  }

  return true;
}

bool testDoubleFilter(
    common::Filter* filter,
    dwio::common::DoubleColumnStatistics* doubleStats,
    bool mayHaveNull) {
  if (!doubleStats) {
    return true;
  }

  if (doubleStats->getMinimum().has_value() &&
      doubleStats->getMaximum().has_value()) {
    return filter->testDoubleRange(
        doubleStats->getMinimum().value(),
        doubleStats->getMaximum().value(),
        mayHaveNull);
  }

  // only min value
  if (doubleStats->getMinimum().has_value()) {
    return filter->testDoubleRange(
        doubleStats->getMinimum().value(),
        std::numeric_limits<double>::max(),
        mayHaveNull);
  }

  // only max value
  if (doubleStats->getMaximum().has_value()) {
    return filter->testDoubleRange(
        std::numeric_limits<double>::lowest(),
        doubleStats->getMaximum().value(),
        mayHaveNull);
  }

  return true;
}

bool testStringFilter(
    common::Filter* filter,
    dwio::common::StringColumnStatistics* stringStats,
    bool mayHaveNull) {
  if (!stringStats) {
    return true;
  }

  if (stringStats->getMinimum().has_value() &&
      stringStats->getMaximum().has_value()) {
    const auto& min = stringStats->getMinimum().value();
    const auto& max = stringStats->getMaximum().value();
    return filter->testBytesRange(min, max, mayHaveNull);
  }

  // only min value
  if (stringStats->getMinimum().has_value()) {
    const auto& min = stringStats->getMinimum().value();
    return filter->testBytesRange(min, std::nullopt, mayHaveNull);
  }

  // only max value
  if (stringStats->getMaximum().has_value()) {
    const auto& max = stringStats->getMaximum().value();
    return filter->testBytesRange(std::nullopt, max, mayHaveNull);
  }

  return true;
}

bool testBoolFilter(
    common::Filter* filter,
    dwio::common::BooleanColumnStatistics* boolStats) {
  auto trueCount = boolStats->getTrueCount();
  auto falseCount = boolStats->getFalseCount();
  if (trueCount.has_value() && falseCount.has_value()) {
    if (trueCount.value() == 0) {
      if (!filter->testBool(false)) {
        return false;
      }
    } else if (falseCount.value() == 0) {
      if (!filter->testBool(true)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

bool testFilter(
    common::Filter* filter,
    dwio::common::ColumnStatistics* stats,
    uint64_t totalRows,
    const TypePtr& type) {
  bool mayHaveNull =
      stats->hasNull().has_value() ? stats->hasNull().value() : true;

  // Has-null statistics is often not set. Hence, we supplement it with
  // number-of-values statistic to detect no-null columns more often.
  // Number-of-values is the number of non-null values. When it is equal to
  // total number of values, we know there are no nulls.
  if (stats->getNumberOfValues().has_value()) {
    if (stats->getNumberOfValues().value() == 0) {
      // Column is all null.
      return filter->testNull();
    }

    if (stats->getNumberOfValues().value() == totalRows) {
      // Column has no nulls.
      mayHaveNull = false;
    }
  }

  if (!mayHaveNull && filter->kind() == common::FilterKind::kIsNull) {
    // IS NULL filter cannot pass.
    return false;
  }
  if (mayHaveNull && filter->testNull()) {
    return true;
  }
  switch (type->kind()) {
    case TypeKind::BIGINT:
    case TypeKind::INTEGER:
    case TypeKind::SMALLINT:
    case TypeKind::TINYINT: {
      auto intStats =
          dynamic_cast<dwio::common::IntegerColumnStatistics*>(stats);
      return testIntFilter(filter, intStats, mayHaveNull);
    }
    case TypeKind::REAL:
    case TypeKind::DOUBLE: {
      auto doubleStats =
          dynamic_cast<dwio::common::DoubleColumnStatistics*>(stats);
      return testDoubleFilter(filter, doubleStats, mayHaveNull);
    }
    case TypeKind::BOOLEAN: {
      auto boolStats =
          dynamic_cast<dwio::common::BooleanColumnStatistics*>(stats);
      return testBoolFilter(filter, boolStats);
    }
    case TypeKind::VARCHAR: {
      auto stringStats =
          dynamic_cast<dwio::common::StringColumnStatistics*>(stats);
      return testStringFilter(filter, stringStats, mayHaveNull);
    }
    default:
      break;
  }

  return true;
}

ScanSpec& ScanSpec::getChildByChannel(column_index_t channel) {
  for (auto& child : children_) {
    if (child->channel_ == channel) {
      return *child;
    }
  }
  VELOX_FAIL("No ScanSpec produces channel {}", channel);
}

std::string ScanSpec::toString() const {
  std::stringstream out;
  if (!fieldName_.empty()) {
    out << fieldName_;
    if (filter_) {
      out << " filter " << filter_->toString();
    }
  }
  if (!children_.empty()) {
    out << " (";
    for (auto& child : children_) {
      out << child->toString() << ", ";
    }
    out << ")";
  }
  return out.str();
}

std::shared_ptr<ScanSpec> ScanSpec::removeChild(const ScanSpec* child) {
  for (auto it = children_.begin(); it != children_.end(); ++it) {
    if (it->get() == child) {
      auto removed = std::move(*it);
      children_.erase(it);
      return removed;
    }
  }
  return nullptr;
}

void ScanSpec::addFilter(const Filter& filter) {
  filter_ = filter_ ? filter_->mergeWith(&filter) : filter.clone();
}

ScanSpec* ScanSpec::addField(const std::string& name, column_index_t channel) {
  auto child = getOrCreateChild(Subfield(name));
  child->setProjectOut(true);
  child->setChannel(channel);
  return child;
}

ScanSpec* ScanSpec::addFieldRecursively(
    const std::string& name,
    const Type& type,
    column_index_t channel) {
  auto* child = addField(name, channel);
  child->addAllChildFields(type);
  return child;
}

ScanSpec* ScanSpec::addMapKeyField() {
  auto* child = addField(kMapKeysFieldName, kNoChannel);
  child->isArrayElementOrMapEntry_ = true;
  return child;
}

ScanSpec* ScanSpec::addMapKeyFieldRecursively(const Type& type) {
  auto* child = addFieldRecursively(kMapKeysFieldName, type, kNoChannel);
  child->isArrayElementOrMapEntry_ = true;
  return child;
}

ScanSpec* ScanSpec::addMapValueField() {
  auto* child = addField(kMapValuesFieldName, kNoChannel);
  child->isArrayElementOrMapEntry_ = true;
  return child;
}

ScanSpec* ScanSpec::addMapValueFieldRecursively(const Type& type) {
  auto* child = addFieldRecursively(kMapValuesFieldName, type, kNoChannel);
  child->isArrayElementOrMapEntry_ = true;
  return child;
}

ScanSpec* ScanSpec::addArrayElementField() {
  auto* child = addField(kArrayElementsFieldName, kNoChannel);
  child->isArrayElementOrMapEntry_ = true;
  return child;
}

ScanSpec* ScanSpec::addArrayElementFieldRecursively(const Type& type) {
  auto* child = addFieldRecursively(kArrayElementsFieldName, type, kNoChannel);
  child->isArrayElementOrMapEntry_ = true;
  return child;
}

void ScanSpec::addAllChildFields(const Type& type) {
  switch (type.kind()) {
    case TypeKind::ROW: {
      auto& rowType = type.asRow();
      for (auto i = 0; i < type.size(); ++i) {
        addFieldRecursively(rowType.nameOf(i), *type.childAt(i), i);
      }
      break;
    }
    case TypeKind::MAP:
      addMapKeyFieldRecursively(*type.childAt(0));
      addMapValueFieldRecursively(*type.childAt(1));
      break;
    case TypeKind::ARRAY:
      addArrayElementFieldRecursively(*type.childAt(0));
      break;
    default:
      break;
  }
}

} // namespace facebook::velox::common
