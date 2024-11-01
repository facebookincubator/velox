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

#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/types/IPAddressType.h"
#include "velox/functions/prestosql/types/IPPrefixType.h"

namespace facebook::velox::functions {

template <typename T>
struct IPPrefixFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<IPPrefix>& result,
      const arg_type<IPAddress>& ip,
      const arg_type<int64_t> prefixBits) {
    folly::ByteArray16 addrBytes;
    folly::ByteArray16 canonicalBytes;
    int128_t intAddr;

    memcpy(&addrBytes, &ip, kIPAddressBytes);
    std::reverse(addrBytes.begin(), addrBytes.end());
    folly::IPAddressV6 v6Addr(addrBytes);

    if (v6Addr.isIPv4Mapped()) {
      canonicalBytes =
          v6Addr.createIPv4().mask(prefixBits).createIPv6().toByteArray();
    } else {
      canonicalBytes = v6Addr.mask(prefixBits).toByteArray();
    }
    std::reverse(canonicalBytes.begin(), canonicalBytes.end());
    memcpy(&intAddr, &canonicalBytes, kIPAddressBytes);
    result = std::make_tuple(intAddr, (int8_t)prefixBits);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<IPPrefix>& result,
      const arg_type<Varchar>& ip,
      const arg_type<int64_t> prefixBits) {
    int128_t intAddr;
    folly::IPAddress addr(ip);
    auto addrBytes = folly::IPAddress::createIPv6(addr).toByteArray();

    std::reverse(addrBytes.begin(), addrBytes.end());
    memcpy(&intAddr, &addrBytes, kIPAddressBytes);

    call(result, intAddr, prefixBits);
  }
};

template <typename T>
struct IPSubnetMinFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<IPAddress>& result,
      const arg_type<IPPrefix>& ipPrefix) {
    // IPPrefix type stores the smallest(canonical) IP already
    result = *ipPrefix.template at<0>();
  }
};

static int128_t getIPSubnetMax(int128_t ip, uint8_t prefix) {
  uint128_t mask = 1;
  int128_t result;
  memcpy(&result, &ip, kIPAddressBytes);

  if (isIPv4(ip)) {
    result |= (mask << (kIPV4Bits - prefix)) - 1;
  } else {
    // Special case: Overflow to all 0 subtracting 1 does not work.
    if (prefix == 0) {
      result = -1;
    } else {
      result |= (mask << (kIPV6Bits - prefix)) - 1;
    }
  }
  return result;
}

template <typename T>
struct IPSubnetMaxFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<IPAddress>& result,
      const arg_type<IPPrefix>& ipPrefix) {
    result =
        getIPSubnetMax(*ipPrefix.template at<0>(), *ipPrefix.template at<1>());
  }
};

template <typename T>
struct IPSubnetRangeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Array<IPAddress>>& result,
      const arg_type<IPPrefix>& ipPrefix) {
    result.push_back(*ipPrefix.template at<0>());
    result.push_back(
        getIPSubnetMax(*ipPrefix.template at<0>(), *ipPrefix.template at<1>()));
  }
};

template <typename T>
struct IPSubnetOfFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  FOLLY_ALWAYS_INLINE void call(
      out_type<bool>& result,
      const arg_type<IPPrefix>& ipPrefix,
      const arg_type<IPAddress>& ip) {
    uint128_t mask = 1;
    const uint8_t prefix = *ipPrefix.template at<1>();
    int128_t checkIP = ip;

    if (isIPv4(*ipPrefix.template at<0>())) {
      checkIP &= ((mask << (kIPV4Bits - prefix)) - 1) ^ -1;
    } else {
      // Special case: Overflow to all 0 subtracting 1 does not work.
      if (prefix == 0) {
        checkIP = 0;
      } else {
        checkIP &= ((mask << (kIPV6Bits - prefix)) - 1) ^ -1;
      }
    }
    result = (*ipPrefix.template at<0>() == checkIP);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<bool>& result,
      const arg_type<IPPrefix>& ipPrefix,
      const arg_type<IPPrefix>& ipPrefix2) {
    call(result, ipPrefix, *ipPrefix2.template at<0>());
    result =
        result && (*ipPrefix2.template at<1>() >= *ipPrefix.template at<1>());
  }
};

void registerIPAddressFunctions(const std::string& prefix) {
  registerIPAddressType();
  registerIPPrefixType();
  registerFunction<IPPrefixFunction, IPPrefix, IPAddress, int64_t>(
      {prefix + "ip_prefix"});
  registerFunction<IPPrefixFunction, IPPrefix, Varchar, int64_t>(
      {prefix + "ip_prefix"});
  registerFunction<IPSubnetMinFunction, IPAddress, IPPrefix>(
      {prefix + "ip_subnet_min"});
  registerFunction<IPSubnetMaxFunction, IPAddress, IPPrefix>(
      {prefix + "ip_subnet_max"});
  registerFunction<IPSubnetRangeFunction, Array<IPAddress>, IPPrefix>(
      {prefix + "ip_subnet_range"});
  registerFunction<IPSubnetOfFunction, bool, IPPrefix, IPAddress>(
      {prefix + "is_subnet_of"});
  registerFunction<IPSubnetOfFunction, bool, IPPrefix, IPPrefix>(
      {prefix + "is_subnet_of"});
}

} // namespace facebook::velox::functions
