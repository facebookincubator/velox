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

namespace facebook::velox {

namespace {

class IPPrefixCastOperator : public exec::CastOperator {
 public:
  bool isSupportedFromType(const TypePtr& other) const override {
    switch (other->kind()) {
      case TypeKind::VARCHAR:
        return true;
      case TypeKind::HUGEINT:
        if (isIPAddressType(other)) {
          return true;
        }
        return false;
      default:
        return false;
    }
  }

  bool isSupportedToType(const TypePtr& other) const override {
    switch (other->kind()) {
      case TypeKind::VARCHAR:
        return true;
      case TypeKind::HUGEINT:
        if (isIPAddressType(other)) {
          return true;
        }
        return false;
      default:
        return false;
    }
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
    } else if (isIPAddressType(input.type())) {
      castFromIPAddress(input, context, rows, *result);
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
    } else if (isIPAddressType(resultType)) {
      castToIPAddress(input, context, rows, *result);
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
    const auto* ipaddresses = input.as<SimpleVector<StringView>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto intAddr = ipaddresses->valueAt(row);
      folly::ByteArray16 addrBytes;
      std::string s;

      memcpy(&addrBytes, intAddr.data(), kIPAddressBytes);
      folly::IPAddressV6 v6Addr(addrBytes);

      if (v6Addr.isIPv4Mapped()) {
        s = v6Addr.createIPv4().str();
      } else {
        s = v6Addr.str();
      }
      s += "/" + std::to_string((uint8_t)intAddr.data()[kIPAddressBytes]);
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
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* ipAddressStrings = input.as<SimpleVector<StringView>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      auto ipAddressString = ipAddressStrings->valueAt(row);
      if (ipAddressString.str().find('/') == std::string::npos) {
        context.setStatus(row, Status::UserError("String missing '/'"));
      }
      folly::CIDRNetwork net =
          folly::IPAddress::createNetwork(ipAddressString, -1, false);
      uint8_t prefix = 0;
      folly::ByteArray16 addrBytes;

      if (net.first.isIPv4Mapped() || net.first.isV4()) {
        addrBytes = folly::IPAddress::createIPv4(net.first)
                        .mask(net.second)
                        .createIPv6()
                        .toByteArray();
      } else {
        addrBytes = folly::IPAddress::createIPv6(net.first)
                        .mask(net.second)
                        .toByteArray();
      }

      exec::StringWriter<false> result(flatResult, row);
      result.resize(kIPPrefixBytes);
      memcpy(result.data(), &addrBytes, kIPAddressBytes);
      result.data()[kIPAddressBytes] = net.second;
      result.finalize();
    });
  }

  static void castToIPAddress(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<int128_t>>();
    const auto* ipaddresses = input.as<SimpleVector<StringView>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto intAddr = ipaddresses->valueAt(row);
      int128_t addrResult = 0;
      folly::ByteArray16 addrBytes;

      memcpy(&addrBytes, intAddr.data(), kIPAddressBytes);
      bigEndianByteArray(addrBytes);

      memcpy(&addrResult, &addrBytes, kIPAddressBytes);
      flatResult->set(row, addrResult);
    });
  }

  static void castFromIPAddress(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* ipAddresses = input.as<SimpleVector<int128_t>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      auto ipAddress = ipAddresses->valueAt(row);
      folly::ByteArray16 addrBytes;

      exec::StringWriter<false> result(flatResult, row);
      result.resize(kIPPrefixBytes);

      memcpy(&addrBytes, &ipAddress, kIPAddressBytes);
      bigEndianByteArray(addrBytes);
      memcpy(result.data(), &addrBytes, kIPAddressBytes);

      folly::IPAddressV6 v6Addr(addrBytes);
      if (v6Addr.isIPv4Mapped()) {
        result.data()[kIPAddressBytes] = kIPV4Bits;
      } else {
        result.data()[kIPAddressBytes] = (unsigned char)kIPV6Bits;
      }

      result.finalize();
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
