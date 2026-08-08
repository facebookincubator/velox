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
//
// Originally authored by Oleksii PELYKH for pcre4j; ported from
// org.pcre4j.regex.translate.ClassNode (Java) under Apache-2.0 by the
// same author for inclusion in Velox.
//
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace facebook::velox::functions::java_pcre2_translator {

struct ClassNode;
using ClassNodePtr = std::shared_ptr<const ClassNode>;

struct Literal {
  std::int32_t cp;
  explicit Literal(std::int32_t cpIn) : cp(cpIn) {}
  bool operator==(const Literal& other) const {
    return cp == other.cp;
  }
};

struct Range {
  std::int32_t lo;
  std::int32_t hi;
  Range(std::int32_t loIn, std::int32_t hiIn) : lo(loIn), hi(hiIn) {}
  bool operator==(const Range& other) const {
    return lo == other.lo && hi == other.hi;
  }
};

struct PropertyLeaf {
  std::string pcre2Token;
  bool negated;
  PropertyLeaf(std::string tokenIn, bool negatedIn)
      : pcre2Token(std::move(tokenIn)), negated(negatedIn) {}
  bool operator==(const PropertyLeaf& other) const {
    return pcre2Token == other.pcre2Token && negated == other.negated;
  }
};

struct Negated {
  ClassNodePtr child;
  explicit Negated(ClassNodePtr childIn) : child(std::move(childIn)) {}
  explicit Negated(const ClassNode& childIn);
  bool operator==(const Negated& other) const;
};

struct Union {
  std::vector<ClassNodePtr> children;
  explicit Union(std::vector<ClassNodePtr> childrenIn)
      : children(std::move(childrenIn)) {}
  explicit Union(const std::vector<ClassNode>& childrenIn);
  bool operator==(const Union& other) const;
};

struct Intersection {
  std::vector<ClassNodePtr> operands;
  explicit Intersection(std::vector<ClassNodePtr> operandsIn)
      : operands(std::move(operandsIn)) {}
  explicit Intersection(const std::vector<ClassNode>& operandsIn);
  bool operator==(const Intersection& other) const;
};

struct ClassNode {
  using Variant =
      std::variant<Literal, Range, PropertyLeaf, Negated, Union, Intersection>;

  Variant value;

  ClassNode(Literal v) : value(std::move(v)) {}
  ClassNode(Range v) : value(std::move(v)) {}
  ClassNode(PropertyLeaf v) : value(std::move(v)) {}
  ClassNode(Negated v) : value(std::move(v)) {}
  ClassNode(Union v) : value(std::move(v)) {}
  ClassNode(Intersection v) : value(std::move(v)) {}

  template <typename T>
  const T* getIf() const {
    return std::get_if<T>(&value);
  }

  template <typename T>
  bool is() const {
    return std::holds_alternative<T>(value);
  }

  bool operator==(const ClassNode& other) const {
    return value == other.value;
  }
  bool operator!=(const ClassNode& other) const {
    return !(*this == other);
  }
};

inline ClassNodePtr nodePtr(const ClassNode& node) {
  return std::make_shared<const ClassNode>(node);
}

inline Negated::Negated(const ClassNode& childIn) : child(nodePtr(childIn)) {}

inline Union::Union(const std::vector<ClassNode>& childrenIn) {
  children.reserve(childrenIn.size());
  for (const auto& child : childrenIn) {
    children.push_back(nodePtr(child));
  }
}

inline Intersection::Intersection(const std::vector<ClassNode>& operandsIn) {
  operands.reserve(operandsIn.size());
  for (const auto& operand : operandsIn) {
    operands.push_back(nodePtr(operand));
  }
}

inline bool Negated::operator==(const Negated& other) const {
  if (child == nullptr || other.child == nullptr) {
    return child == other.child;
  }
  return *child == *other.child;
}

inline bool Union::operator==(const Union& other) const {
  if (children.size() != other.children.size()) {
    return false;
  }
  for (std::size_t i = 0; i < children.size(); ++i) {
    if ((children[i] == nullptr) != (other.children[i] == nullptr)) {
      return false;
    }
    if (children[i] != nullptr && *children[i] != *other.children[i]) {
      return false;
    }
  }
  return true;
}

inline bool Intersection::operator==(const Intersection& other) const {
  if (operands.size() != other.operands.size()) {
    return false;
  }
  for (std::size_t i = 0; i < operands.size(); ++i) {
    if ((operands[i] == nullptr) != (other.operands[i] == nullptr)) {
      return false;
    }
    if (operands[i] != nullptr && *operands[i] != *other.operands[i]) {
      return false;
    }
  }
  return true;
}

} // namespace facebook::velox::functions::java_pcre2_translator
