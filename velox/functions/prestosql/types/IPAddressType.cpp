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

#include "velox/functions/prestosql/types/IPAddressType.h"
#include <iostream>

namespace facebook::velox {

namespace {

class IPAddressCastOperator : public exec::CastOperator {
 public:
  bool isSupportedFromType(const TypePtr& other) const override {
    return VARCHAR()->equivalent(*other);
  }

  bool isSupportedToType(const TypePtr& other) const override {
    return VARCHAR()->equivalent(*other);
  }

  void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);

    if (input.typeKind() == TypeKind::VARCHAR) {
      castFromString(input, context, rows, *result);
    } else {
      VELOX_UNSUPPORTED(
          "Cast from {} to IPAddress not yet supported",
          resultType->toString());
    }
  }

  void castFrom(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);

    if (resultType->kind() == TypeKind::VARCHAR) {
      castToString(input, context, rows, *result);
    } else {
      VELOX_UNSUPPORTED(
          "Cast from IPAddress to {} not yet supported",
          resultType->toString());
    }
  }

 private:
  static void castToString(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* ipaddresses = input.as<SimpleVector<int128_t>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto intAddr = ipaddresses->valueAt(row);
      boost::asio::ip::address_v6::bytes_type addrBytes;
      std::string s;
      memcpy(&addrBytes, &intAddr, 16);

      bigEndianByteArray(addrBytes);
      auto v6Addr = boost::asio::ip::make_address_v6(addrBytes);

      if (v6Addr.is_v4_mapped()) {
        auto v4Addr = boost::asio::ip::make_address_v4(
            boost::asio::ip::v4_mapped, v6Addr);
        s = boost::lexical_cast<std::string>(v4Addr);
      } else {
        s = boost::lexical_cast<std::string>(v6Addr);
      }

      exec::StringWriter<false> result(flatResult, row);
      result.append(s);
      result.finalize();
    });
  }

  static void castFromString(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<int128_t>>();
    const auto* ipAddressStrings = input.as<SimpleVector<StringView>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto ipAddressString = ipAddressStrings->valueAt(row);
      boost::asio::ip::address_v6::bytes_type addrBytes;
      auto addr = boost::asio::ip::make_address(ipAddressString);
      int128_t intAddr;
      if (addr.is_v4()) {
        addrBytes = boost::asio::ip::make_address_v6(
                        boost::asio::ip::v4_mapped, addr.to_v4())
                        .to_bytes();
      } else {
        addrBytes = addr.to_v6().to_bytes();
      }

      bigEndianByteArray(addrBytes);
      memcpy(&intAddr, &addrBytes, 16);

      flatResult->set(row, intAddr);
    });
  }
};

class IPAddressTypeFactories : public CustomTypeFactories {
 public:
  IPAddressTypeFactories() = default;

  TypePtr getType() const override {
    return IPADDRESS();
  }

  exec::CastOperatorPtr getCastOperator() const override {
    return std::make_shared<IPAddressCastOperator>();
  }
};

} // namespace

void registerIPAddressType() {
  registerCustomType(
      "ipaddress", std::make_unique<const IPAddressTypeFactories>());
}

} // namespace facebook::velox
