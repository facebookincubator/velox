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

#include <boost/asio/ip/address.hpp>
#include <boost/asio/ip/network_v4.hpp>
#include <boost/asio/ip/network_v6.hpp>
#include <boost/lexical_cast.hpp>
#include "velox/expression/CastExpr.h"
#include "velox/functions/prestosql/types/IPAddressType.h"
#include "velox/type/OpaqueCustomTypes.h"
#include "velox/type/SimpleFunctionApi.h"
#include "velox/type/Type.h"

#include <iostream>

static constexpr int kIPPrefixIndex = 16;
static constexpr int kIPPrefixBytes = 17;

namespace facebook::velox {

class IPPrefixType : public VarbinaryType {
  IPPrefixType() = default;

 public:
  static const std::shared_ptr<const IPPrefixType>& get() {
    static const std::shared_ptr<const IPPrefixType> instance{
        new IPPrefixType()};

    return instance;
  }

  bool equivalent(const Type& other) const override {
    // Pointer comparison works since this type is a singleton.
    return this == &other;
  }

  const char* name() const override {
    return "IPPREFIX";
  }

  std::string toString() const override {
    return name();
  }

  folly::dynamic serialize() const override {
    folly::dynamic obj = folly::dynamic::object;
    obj["name"] = "Type";
    obj["type"] = name();
    return obj;
  }
};

FOLLY_ALWAYS_INLINE bool isIPPrefixType(const TypePtr& type) {
  // Pointer comparison works since this type is a singleton.
  return IPPrefixType::get() == type;
}

FOLLY_ALWAYS_INLINE std::shared_ptr<const IPPrefixType> IPPREFIX() {
  return IPPrefixType::get();
}

struct IPPrefixT {
  using type = Varbinary;
  static constexpr const char* typeName = "ipprefix";
};

using IPPrefix = CustomType<IPPrefixT>;

void registerIPPrefixType();

} // namespace facebook::velox
