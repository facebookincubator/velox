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

#include <folly/IPAddress.h>
#include <folly/small_vector.h>

#include "velox/expression/CastExpr.h"
#include "velox/functions/prestosql/types/IPAddressType.h"
#include "velox/functions/prestosql/types/IPPrefixType.h"

static constexpr uint8_t kIPV4Bits = 32;
static constexpr uint8_t kIPV6Bits = 128;

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
    } else {
      VELOX_NYI(
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
    } else {
      VELOX_NYI(
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
    const auto* ipprefixes = input.as<RowVector>();
    const auto* ip = ipprefixes->childAt(0)->as<SimpleVector<int128_t>>();
    const auto* prefix = ipprefixes->childAt(1)->as<SimpleVector<int8_t>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto intAddr = ip->valueAt(row);
      folly::ByteArray16 addrBytes;

      memcpy(&addrBytes, &intAddr, kIPAddressBytes);
      std::reverse(addrBytes.begin(), addrBytes.end());
      folly::IPAddressV6 v6Addr(addrBytes);

      exec::StringWriter<false> resultWriter(flatResult, row);
      if (v6Addr.isIPv4Mapped()) {
        resultWriter.append(fmt::format(
            "{}/{}", v6Addr.createIPv4().str(), prefix->valueAt(row)));
      } else {
        resultWriter.append(
            fmt::format("{}/{}", v6Addr.str(), (uint8_t)prefix->valueAt(row)));
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
    int128_t intAddr;
    folly::ByteArray16 addrBytes;
    auto* rowResult = result.as<RowVector>();
    const auto* ipAddressStrings = input.as<SimpleVector<StringView>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      auto ipAddressString = ipAddressStrings->valueAt(row);

      // Folly allows for creation of networks without a "/" so check to make
      // sure that we have one.
      if (ipAddressString.str().find('/') == std::string::npos) {
        context.setStatus(
            row,
            threadSkipErrorDetails() ? Status::UserError()
                                     : Status::UserError(
                                           "Cannot cast value to IPPREFIX: {}",
                                           ipAddressString.str()));
        return;
      }

      auto const maybeNet =
          folly::IPAddress::tryCreateNetwork(ipAddressString, -1, false);

      if (maybeNet.hasError()) {
        context.setStatus(
            row,
            threadSkipErrorDetails() ? Status::UserError()
                                     : Status::UserError(
                                           "Cannot cast value to IPPREFIX: {}",
                                           ipAddressString.str()));
        return;
      }

      auto [ip, prefix] = maybeNet.value();
      if (prefix > ((ip.isIPv4Mapped() || ip.isV4()) ? kIPV4Bits : kIPV6Bits)) {
        context.setStatus(
            row,
            threadSkipErrorDetails() ? Status::UserError()
                                     : Status::UserError(
                                           "Cannot cast value to IPPREFIX: {}",
                                           ipAddressString.str()));
        return;
      }

      addrBytes = (ip.isIPv4Mapped() || ip.isV4())
          ? folly::IPAddress::createIPv4(ip)
                .mask(prefix)
                .createIPv6()
                .toByteArray()
          : folly::IPAddress::createIPv6(ip).mask(prefix).toByteArray();

      std::reverse(addrBytes.begin(), addrBytes.end());
      memcpy(&intAddr, &addrBytes, kIPAddressBytes);
      rowResult->childAt(0)->as<FlatVector<int128_t>>()->set(row, intAddr);
      rowResult->childAt(1)->as<FlatVector<int8_t>>()->set(row, prefix);
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
