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

#include "velox/functions/prestosql/types/IPPrefixType.h"
#include "velox/functions/prestosql/types/IPAddressType.h"

namespace facebook::velox {

namespace {

class IPPrefixCastOperator : public exec::CastOperator {
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
          "Cast from {} to IPPrefix not yet supported", resultType->toString());
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
          "Cast from IPPrefix to {} not yet supported", resultType->toString());
    }
  }

 private:
  static void castToString(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* ipaddresses = input.as<SimpleVector<std::shared_ptr<void>>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto intAddr =
          std::static_pointer_cast<IPPrefix>(ipaddresses->valueAt(row));
      boost::asio::ip::address_v6::bytes_type addrBytes;
      std::string s;

      memcpy(&addrBytes, &intAddr->ip, 16);
      bigEndianByteArray(addrBytes);
      auto v6Addr = boost::asio::ip::make_address_v6(addrBytes);

      if (v6Addr.is_v4_mapped()) {
        auto v4Addr = boost::asio::ip::make_address_v4(
            boost::asio::ip::v4_mapped, v6Addr);
        auto v4Net =
            boost::asio::ip::network_v4(v4Addr, (uint8_t)intAddr->prefix);
        s = boost::lexical_cast<std::string>(v4Net);
      } else {
        auto v6Net =
            boost::asio::ip::network_v6(v6Addr, (uint8_t)intAddr->prefix);
        s = boost::lexical_cast<std::string>(v6Net);
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
    auto* flatResult = result.as<FlatVector<std::shared_ptr<void>>>();
    const auto* ipAddressStrings = input.as<SimpleVector<StringView>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto ipAddressString = ipAddressStrings->valueAt(row);
      int slashPos = ipAddressString.str().find_last_of('/');
      std::string ipOnly = ipAddressString.str().substr(0, slashPos);
      boost::asio::ip::address_v6 v6Addr;
      boost::asio::ip::address_v6::bytes_type addrBytes;

      auto addr = boost::asio::ip::make_address(ipOnly);
      IPPrefix res(0, 0);
      if (addr.is_v4()) {
        auto v4Net = boost::asio::ip::make_network_v4(ipAddressString);
        res.prefix = (uint8_t)v4Net.prefix_length();
        addrBytes = boost::asio::ip::make_address_v6(
                        boost::asio::ip::v4_mapped, v4Net.canonical().address())
                        .to_bytes();
      } else {
        auto v6Net = boost::asio::ip::make_network_v6(ipAddressString);
        if (addr.to_v6().is_v4_mapped()) {
          auto v4Addr = boost::asio::ip::make_address_v4(
              boost::asio::ip::v4_mapped, addr.to_v6());
          auto v4Net = boost::asio::ip::make_network_v4(
              v4Addr, (uint8_t)v6Net.prefix_length());
          addrBytes =
              boost::asio::ip::make_address_v6(
                  boost::asio::ip::v4_mapped, v4Net.canonical().address())
                  .to_bytes();
        } else {
          addrBytes = v6Net.canonical().address().to_bytes();
        }
        res.prefix = (uint8_t)v6Net.prefix_length();
      }

      bigEndianByteArray(addrBytes);
      memcpy(&res.ip, &addrBytes, 16);

      flatResult->set(
          row, std::make_shared<IPPrefix>(res.ip, (uint8_t)res.prefix));
    });
  }
};

class IPPrefixTypeFactories : public CustomTypeFactories {
 public:
  TypePtr getType() const override {
    return IPPrefixType::get();
  }

  exec::CastOperatorPtr getCastOperator() const override {
    return std::make_shared<IPPrefixCastOperator>();
  }
};

} // namespace

void registerIPPrefixType() {
  registerCustomType(
      "ipprefix", std::make_unique<const IPPrefixTypeFactories>());
}

} // namespace facebook::velox
