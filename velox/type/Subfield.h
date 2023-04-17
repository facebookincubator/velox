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

#include <boost/algorithm/string/replace.hpp>
#include <folly/json.h>
#include "velox/common/base/Exceptions.h"
#include "velox/common/serialization/Serializable.h"

namespace facebook::velox::common {

enum SubfieldKind {
  kAllSubscripts,
  kNestedField,
  kStringSubscript,
  kLongSubscript
};

namespace {
std::unordered_map<SubfieldKind, std::string> subfieldKinds() {
  return {
      {kAllSubscripts, "kAllSubscripts"},
      {kNestedField, "kNestedField"},
      {kStringSubscript, "kStringSubscript"},
      {kLongSubscript, "kLongSubscript"},
  };
}

template <typename K, typename V>
std::unordered_map<V, K> invertMap(const std::unordered_map<K, V>& mapping) {
  std::unordered_map<V, K> inverted;
  for (const auto& [key, value] : mapping) {
    inverted.emplace(value, key);
  }
  return inverted;
}

const char* subfieldKind(SubfieldKind kind) {
  static const auto kinds = subfieldKinds();
  return kinds.at(kind).c_str();
}

SubfieldKind subfieldKindFromName(const std::string& name) {
  static const auto kinds = invertMap(subfieldKinds());
  return kinds.at(name);
}

} // namespace

class Subfield : public velox::ISerializable {
 public:
  class PathElement : public velox::ISerializable {
   public:
    virtual ~PathElement() = default;
    virtual SubfieldKind kind() const = 0;
    virtual bool isSubscript() const = 0;
    virtual std::string toString() const = 0;
    virtual size_t hash() const = 0;
    virtual bool operator==(const PathElement& other) const = 0;
    virtual std::unique_ptr<PathElement> clone() = 0;

    static std::unique_ptr<PathElement> create(const folly::dynamic& obj) {
      auto kindName = obj["kind"].asString();
      auto kind = subfieldKindFromName(kindName);
      switch (kind) {
        case SubfieldKind::kAllSubscripts: {
          return std::make_unique<AllSubscripts>();
        }
        case SubfieldKind::kNestedField: {
          auto name = obj["name"].asString();
          return std::make_unique<NestedField>(name);
        }
        case SubfieldKind::kStringSubscript: {
          auto index = obj["index"].asString();
          return std::make_unique<StringSubscript>(index);
        }
        case SubfieldKind::kLongSubscript: {
          auto index = obj["index"].asInt();
          return std::make_unique<LongSubscript>(index);
        }
      }
      VELOX_UNREACHABLE();
    }
  };

  class AllSubscripts final : public PathElement {
   public:
    SubfieldKind kind() const override {
      return kAllSubscripts;
    }

    bool isSubscript() const override {
      return true;
    }

    std::string toString() const override {
      return "[*]";
    }

    size_t hash() const override {
      return 0;
    }

    bool operator==(const PathElement& other) const override {
      return other.kind() == kAllSubscripts;
    }

    folly::dynamic serialize() const override {
      folly::dynamic obj = folly::dynamic::object;
      obj["kind"] = subfieldKind(kAllSubscripts);
      return obj;
    }

    std::unique_ptr<PathElement> clone() override {
      return std::make_unique<AllSubscripts>();
    }
  };

  class NestedField final : public PathElement {
   public:
    explicit NestedField(const std::string& name) : name_(name) {}

    SubfieldKind kind() const override {
      return kNestedField;
    }

    const std::string& name() const {
      return name_;
    }

    bool operator==(const PathElement& other) const override {
      if (this == &other) {
        return true;
      }
      return other.kind() == kNestedField &&
          reinterpret_cast<const NestedField*>(&other)->name_ == name_;
    }

    size_t hash() const override {
      std::hash<std::string> hash;
      return hash(name_);
    }

    std::string toString() const override {
      return "." + name_;
    }

    bool isSubscript() const override {
      return false;
    }

    folly::dynamic serialize() const override {
      folly::dynamic obj = folly::dynamic::object;
      obj["kind"] = subfieldKind(kNestedField);
      obj["name"] = name_;
      return obj;
    }

    std::unique_ptr<PathElement> clone() override {
      return std::make_unique<NestedField>(name_);
    }

   private:
    const std::string name_;
  };

  class LongSubscript final : public PathElement {
   public:
    explicit LongSubscript(long index) : index_(index) {}

    SubfieldKind kind() const override {
      return kLongSubscript;
    }

    long index() const {
      return index_;
    }

    bool operator==(const PathElement& other) const override {
      if (this == &other) {
        return true;
      }
      return other.kind() == kLongSubscript &&
          reinterpret_cast<const LongSubscript*>(&other)->index_ == index_;
    }

    size_t hash() const override {
      std::hash<long> hash;
      return hash(index_);
    }

    std::string toString() const override {
      return "[" + std::to_string(index_) + "]";
    }

    bool isSubscript() const override {
      return true;
    }

    folly::dynamic serialize() const override {
      folly::dynamic obj = folly::dynamic::object;
      obj["kind"] = subfieldKind(kLongSubscript);
      obj["index"] = index_;
      return obj;
    }

    std::unique_ptr<PathElement> clone() override {
      return std::make_unique<LongSubscript>(index_);
    }

   private:
    const long index_;
  };

  class StringSubscript final : public PathElement {
   public:
    explicit StringSubscript(const std::string& index) : index_(index) {}

    SubfieldKind kind() const override {
      return kStringSubscript;
    }

    const std::string index() const {
      return index_;
    }

    bool operator==(const PathElement& other) const override {
      if (this == &other) {
        return true;
      }
      return other.kind() == kStringSubscript &&
          reinterpret_cast<const StringSubscript*>(&other)->index_ == index_;
    }

    size_t hash() const override {
      std::hash<std::string> hash;
      return hash(index_);
    }

    std::string toString() const override {
      return "[\"" + boost::replace_all_copy(index_, "\"", "\\\"") + "\"]";
    }

    bool isSubscript() const override {
      return true;
    }

    folly::dynamic serialize() const override {
      folly::dynamic obj = folly::dynamic::object;
      obj["kind"] = subfieldKind(kStringSubscript);
      obj["index"] = index_;
      return obj;
    }

    std::unique_ptr<PathElement> clone() override {
      return std::make_unique<StringSubscript>(index_);
    }

   private:
    const std::string index_;
  };

 public:
  explicit Subfield(const std::string& path);

  explicit Subfield(std::vector<std::unique_ptr<PathElement>>&& path);

  Subfield() = default;

  static std::unique_ptr<Subfield> create(const std::string& field) {
    std::vector<std::unique_ptr<PathElement>> path;
    path.push_back(std::make_unique<NestedField>(field));
    return std::make_unique<Subfield>(std::move(path));
  }

  folly::dynamic serialize() const override;
  static void registerSerDe();

  static std::unique_ptr<const Subfield> create(const folly::dynamic& obj) {
    std::vector<std::unique_ptr<PathElement>> path;
    auto paths = obj["path"];
    for (const auto& p : paths) {
      path.emplace_back(PathElement::create(p));
    }
    return std::make_unique<const Subfield>(std::move(path));
  }

  const std::vector<std::unique_ptr<PathElement>>& path() const {
    return path_;
  }

  std::vector<std::unique_ptr<PathElement>>& path() {
    return path_;
  }

  bool isPrefix(const Subfield& other) const {
    if (path_.size() < other.path_.size()) {
      for (int i = 0; i < path_.size(); ++i) {
        if (!(*path_[i].get() == *other.path_[i].get())) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  std::string toString() const {
    std::ostringstream out;
    out << static_cast<const NestedField*>(path_[0].get())->name();
    for (int i = 1; i < path_.size(); i++) {
      out << path_[i]->toString();
    }
    return out.str();
  }

  bool operator==(const Subfield& other) const {
    if (this == &other) {
      return true;
    }

    if (path_.size() != other.path_.size()) {
      return false;
    }
    for (int i = 0; i < path_.size(); ++i) {
      if (!(*path_[i].get() == *other.path_[i].get())) {
        return false;
      }
    }
    return true;
  }

  size_t hash() const {
    size_t result = 1;
    for (int i = 0; i < path_.size(); ++i) {
      result = result * 31 + path_[i]->hash();
    }
    return result;
  }

 private:
  std::vector<std::unique_ptr<PathElement>> path_;
};
} // namespace facebook::velox::common

namespace std {
template <>
struct hash<::facebook::velox::common::Subfield> {
  size_t operator()(const ::facebook::velox::common::Subfield& subfield) const {
    return subfield.hash();
  }
};
} // namespace std
