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
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/functions/prestosql/types/IPPrefixType.h"

namespace facebook::velox::functions {

inline bool isIPV4(int128_t ip) {
  int128_t ipV4 = 0x0000FFFF00000000;
  uint128_t mask = 0xFFFFFFFFFFFFFFFF;
  mask = (mask << kIPV6HalfBits) | 0xFFFFFFFF00000000;
  return (ip & mask) == ipV4;
}

inline int128_t getIPFromIPPrefix(const facebook::velox::StringView ipPrefix) {
  int128_t result = 0;
  folly::ByteArray16 addrBytes;
  memcpy(&addrBytes, ipPrefix.data(), kIPAddressBytes);
  bigEndianByteArray(addrBytes);
  memcpy(&result, &addrBytes, kIPAddressBytes);
  return result;
}

template <typename T>
struct IPPrefixFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<IPPrefix>& result,
      const arg_type<IPAddress>& ip,
      const arg_type<int64_t> prefixBits) {
    // Presto stores prefixBits in one signed byte. Cast to unsigned
    folly::ByteArray16 addrBytes;
    memcpy(&addrBytes, &ip, kIPAddressBytes);
    bigEndianByteArray(addrBytes);

    // All IPs are stored as V6
    folly::IPAddressV6 v6Addr(addrBytes);

    // For return
    folly::ByteArray16 canonicalBytes;

    if (v6Addr.isIPv4Mapped()) {
      canonicalBytes =
          v6Addr.createIPv4().mask(prefixBits).createIPv6().toByteArray();
    } else {
      canonicalBytes = v6Addr.mask(prefixBits).toByteArray();
    }
    result.resize(kIPPrefixBytes);
    memcpy(result.data(), &canonicalBytes, kIPAddressBytes);
    result.data()[kIPPrefixIndex] = prefixBits;
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<IPPrefix>& result,
      const arg_type<Varchar>& ip,
      const arg_type<int64_t> prefixBits) {
    int128_t intAddr;
    folly::IPAddress addr(ip);
    auto addrBytes = folly::IPAddress::createIPv6(addr).toByteArray();

    bigEndianByteArray(addrBytes);
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
    result = getIPFromIPPrefix(ipPrefix);
  }
};

inline int128_t getIPSubnetMax(int128_t ip, uint8_t prefix) {
  uint128_t mask = 1;
  int128_t result;
  memcpy(&result, &ip, kIPAddressBytes);

  if (isIPV4(ip)) {
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
    result = getIPSubnetMax(
        getIPFromIPPrefix(ipPrefix), ipPrefix.data()[kIPPrefixIndex]);
  }
};

template <typename T>
struct IPSubnetRangeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Array<IPAddress>>& result,
      const arg_type<IPPrefix>& ipPrefix) {
    result.push_back(getIPFromIPPrefix(ipPrefix));
    result.push_back(getIPSubnetMax(
        getIPFromIPPrefix(ipPrefix), ipPrefix.data()[kIPPrefixIndex]));
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
    uint8_t prefix = ipPrefix.data()[kIPPrefixIndex];
    int128_t checkIP = ip;

    if (isIPV4(getIPFromIPPrefix(ipPrefix))) {
      checkIP &= ((mask << (kIPV4Bits - prefix)) - 1) ^ -1;
    } else {
      // Special case: Overflow to all 0 subtracting 1 does not work.
      if (prefix == 0) {
        checkIP = 0;
      } else {
        checkIP &= ((mask << (kIPV6Bits - prefix)) - 1) ^ -1;
      }
    }
    result = (getIPFromIPPrefix(ipPrefix) == checkIP);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<bool>& result,
      const arg_type<IPPrefix>& ipPrefix,
      const arg_type<IPPrefix>& ipPrefix2) {
    call(result, ipPrefix, getIPFromIPPrefix(ipPrefix2));
    result = result &&
        (ipPrefix2.data()[kIPPrefixIndex] >= ipPrefix.data()[kIPPrefixIndex]);
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
