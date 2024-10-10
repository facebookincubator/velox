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
#include <folly/IPAddress.h>
#include <folly/small_vector.h>
#include "velox/expression/CastExpr.h"
#include "velox/functions/prestosql/types/IPAddressType.h"

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
          "Cast from {} to IPPrefix not yet supported",
          input.type()->toString());
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

      memcpy(&addrBytes, intAddr.data(), kIPAddressBytes);
      folly::IPAddressV6 v6Addr(addrBytes);

      exec::StringWriter<false> resultWriter(flatResult, row);
      if (v6Addr.isIPv4Mapped()) {
        resultWriter.append(fmt::format(
            "{}/{}",
            v6Addr.createIPv4().str(),
            (uint8_t)intAddr.data()[kIPAddressBytes]));
      } else {
        resultWriter.append(fmt::format(
            "{}/{}", v6Addr.str(), (uint8_t)intAddr.data()[kIPAddressBytes]));
      }
      resultWriter.finalize();
    });
  }

  static folly::small_vector<folly::StringPiece, 2> splitIpSlashCidr(
      const folly::StringPiece& ipSlashCidr) {
    folly::small_vector<folly::StringPiece, 2> vec;
    folly::split('/', ipSlashCidr, vec);
    return vec;
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

      // Folly allows for creation of networks without a "/" so check to make
      // sure that we have one.
      if (ipAddressString.str().find('/') == std::string::npos) {
        context.setStatus(
            row,
            threadSkipErrorDetails()
                ? Status::UserError()
                : Status::UserError(
                      "Invalid CIDR IP address specified. Expected IP/PREFIX format, got '{}'",
                      ipAddressString.str()));
        return;
      }

      folly::ByteArray16 addrBytes;
      auto const maybeNet =
          folly::IPAddress::tryCreateNetwork(ipAddressString, -1, false);

      if (maybeNet.hasError()) {
        if (threadSkipErrorDetails()) {
          context.setStatus(row, Status::UserError());
        } else {
          switch (maybeNet.error()) {
            case folly::CIDRNetworkError::INVALID_DEFAULT_CIDR:
              context.setStatus(
                  row, Status::UserError("defaultCidr must be <= UINT8_MAX"));
              break;
            case folly::CIDRNetworkError::INVALID_IP_SLASH_CIDR:
              context.setStatus(
                  row,
                  Status::UserError(
                      "Invalid CIDR IP address specified. Expected IP/PREFIX format, got '{}'",
                      ipAddressString.str()));
              break;
            case folly::CIDRNetworkError::INVALID_IP: {
              auto const vec = splitIpSlashCidr(ipAddressString);
              context.setStatus(
                  row,
                  Status::UserError(
                      "Invalid IP address '{}'",
                      vec.size() > 0 ? vec.at(0) : ""));
              break;
            }
            case folly::CIDRNetworkError::INVALID_CIDR: {
              auto const vec = splitIpSlashCidr(ipAddressString);
              context.setStatus(
                  row,
                  Status::UserError(
                      "Mask value '{}' not a valid mask",
                      vec.size() > 1 ? vec.at(1) : ""));
              break;
            }
            case folly::CIDRNetworkError::CIDR_MISMATCH: {
              auto const vec = splitIpSlashCidr(ipAddressString);
              auto const subnet =
                  folly::IPAddress::tryFromString(vec.at(0)).value();
              context.setStatus(
                  row,
                  Status::UserError(
                      "CIDR value '{}' is > network bit count '{}'",
                      vec.size() == 2
                          ? vec.at(1)
                          : folly::to<std::string>(
                                subnet.isV4() ? kIPV4Bits : kIPV6Bits),
                      subnet.bitCount()));
              break;
            }
            default:
              context.setStatus(row, Status::UserError());
              break;
          }
        }
        return;
      }

      auto net = maybeNet.value();
      if (net.first.isIPv4Mapped() || net.first.isV4()) {
        if (net.second > kIPV4Bits) {
          context.setStatus(
              row,
              threadSkipErrorDetails()
                  ? Status::UserError()
                  : Status::UserError(
                        "CIDR value '{}' is > network bit count '{}'",
                        net.second,
                        kIPV4Bits));
          return;
        }
        addrBytes = folly::IPAddress::createIPv4(net.first)
                        .mask(net.second)
                        .createIPv6()
                        .toByteArray();
      } else {
        if (net.second > kIPV6Bits) {
          context.setStatus(
              row,
              threadSkipErrorDetails()
                  ? Status::UserError()
                  : Status::UserError(
                        "CIDR value '{}' is > network bit count '{}'",
                        net.second,
                        kIPV6Bits));
          return;
        }
        addrBytes = folly::IPAddress::createIPv6(net.first)
                        .mask(net.second)
                        .toByteArray();
      }

      exec::StringWriter<false> resultWriter(flatResult, row);
      resultWriter.resize(kIPPrefixBytes);
      memcpy(resultWriter.data(), &addrBytes, kIPAddressBytes);
      resultWriter.data()[kIPAddressBytes] = net.second;
      resultWriter.finalize();
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
      std::reverse(addrBytes.begin(), addrBytes.end());

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

      exec::StringWriter<false> resultWriter(flatResult, row);
      resultWriter.resize(kIPPrefixBytes);

      memcpy(&addrBytes, &ipAddress, kIPAddressBytes);
      std::reverse(addrBytes.begin(), addrBytes.end());
      memcpy(resultWriter.data(), &addrBytes, kIPAddressBytes);

      folly::IPAddressV6 v6Addr(addrBytes);
      if (v6Addr.isIPv4Mapped()) {
        resultWriter.data()[kIPAddressBytes] = kIPV4Bits;
      } else {
        resultWriter.data()[kIPAddressBytes] = kIPV6Bits;
      }

      resultWriter.finalize();
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
