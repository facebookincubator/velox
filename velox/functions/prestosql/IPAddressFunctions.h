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
#include "velox/functions/prestosql/types/IPAddressType.h"
#include "velox/functions/prestosql/types/IPPrefixType.h"

#include <iostream>

namespace facebook::velox::functions {

inline bool isIPV4(int128_t ip) {
  int128_t ipV4 = 0x0000FFFF00000000;
  int128_t mask = 0xFFFFFFFFFFFFFFFF;
  mask = (mask << 64) | 0xFFFFFFFF00000000;
  return (ip & mask) == ipV4;
}

template <typename T>
struct IPPrefixFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<TheIPPrefix>& result,
      const arg_type<IPAddress>& ip,
      const arg_type<int8_t> prefixBits) {
    // Presto stores prefixBits in one signed byte. Cast to unsigned
    uint8_t prefix = (uint8_t)prefixBits;
    boost::asio::ip::address_v6::bytes_type addrBytes;
    memcpy(&addrBytes, &ip, 16);
    bigEndianByteArray(addrBytes);

    // All IPs are stored as V6
    auto v6Addr = boost::asio::ip::make_address_v6(addrBytes);
    boost::asio::ip::address_v6 v6CanonicalAddr;

    // For return
    int128_t canonicalAddrInt;

    // Convert to V4/V6 respectively and create network to get canonical
    // address as well as check validity of the prefix.
    if (v6Addr.is_v4_mapped()) {
      auto v4Addr =
          boost::asio::ip::make_address_v4(boost::asio::ip::v4_mapped, v6Addr);
      auto v4Network = boost::asio::ip::make_network_v4(v4Addr, prefix);
      v6CanonicalAddr = boost::asio::ip::make_address_v6(
          boost::asio::ip::v4_mapped, v4Network.canonical().address());
    } else {
      auto v6Network = boost::asio::ip::make_network_v6(v6Addr, prefix);
      v6CanonicalAddr = v6Network.canonical().address();
    }

    auto canonicalBytes = v6CanonicalAddr.to_bytes();
    bigEndianByteArray(canonicalBytes);
    memcpy(&canonicalAddrInt, &canonicalBytes, 16);

    result = std::make_shared<IPPrefix>(canonicalAddrInt, prefix);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<TheIPPrefix>& result,
      const arg_type<Varchar>& ip,
      const arg_type<int8_t> prefixBits) {
    boost::asio::ip::address_v6::bytes_type addrBytes;
    auto addr = boost::asio::ip::make_address(ip);
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

    call(result, intAddr, prefixBits);
  }
};

template <typename T>
struct IPSubnetMinFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<IPAddress>& result,
      const arg_type<TheIPPrefix>& ipPrefix) {
    // IPPrefix type should store the smallest(canonical) IP already
    memcpy(&result, &ipPrefix->ip, 16);
  }
};

inline int128_t getIPSubnetMax(int128_t ip, uint8_t prefix) {
  uint128_t mask = 1;
  int128_t result;
  boost::asio::ip::address_v6::bytes_type addrBytes;
  memcpy(&result, &ip, 16);

  if (isIPV4(ip)) {
    result |= (mask << (32 - prefix)) - 1;
  } else {
    // Special case: Overflow to all 0 subtracting 1 does not work.
    if (prefix == 0) {
      result = -1;
    } else {
      result |= (mask << (128 - prefix)) - 1;
    }
  }
  return result;
}

template <typename T>
struct IPSubnetMaxFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<IPAddress>& result,
      const arg_type<TheIPPrefix>& ipPrefix) {
    result = getIPSubnetMax(ipPrefix->ip, (uint8_t)ipPrefix->prefix);
  }
};

template <typename T>
struct IPSubnetRangeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Array<IPAddress>>& result,
      const arg_type<TheIPPrefix>& ipPrefix) {
    result.push_back(ipPrefix->ip);
    result.push_back(getIPSubnetMax(ipPrefix->ip, (uint8_t)ipPrefix->prefix));
  }
};

template <typename T>
struct IPSubnetOfFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  FOLLY_ALWAYS_INLINE void call(
      out_type<bool>& result,
      const arg_type<TheIPPrefix>& ipPrefix,
      const arg_type<IPAddress>& ip) {
    uint128_t mask = 1;
    uint8_t prefix = (uint8_t)ipPrefix->prefix;
    int128_t checkIP = ip;

    if (isIPV4(ipPrefix->ip)) {
      checkIP &= ((mask << (32 - prefix)) - 1) ^ -1;
    } else {
      // Special case: Overflow to all 0 subtracting 1 does not work.
      if (prefix == 0) {
        checkIP = 0;
      } else {
        checkIP &= ((mask << (128 - prefix)) - 1) ^ -1;
      }
    }
    result = (ipPrefix->ip == checkIP);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<bool>& result,
      const arg_type<TheIPPrefix>& ipPrefix,
      const arg_type<TheIPPrefix>& ipPrefix2) {
    call(result, ipPrefix, ipPrefix2->ip);
    result = result && (ipPrefix2->prefix >= ipPrefix->prefix);
  }
};

void registerIPAddressFunctions(const std::string& prefix) {
  registerIPAddressType();
  registerIPPrefixType();
  registerFunction<IPPrefixFunction, TheIPPrefix, IPAddress, int8_t>(
      {prefix + "ip_prefix"});
  registerFunction<IPPrefixFunction, TheIPPrefix, Varchar, int8_t>(
      {prefix + "ip_prefix"});
  registerFunction<IPSubnetMinFunction, IPAddress, TheIPPrefix>(
      {prefix + "ip_subnet_min"});
  registerFunction<IPSubnetMaxFunction, IPAddress, TheIPPrefix>(
      {prefix + "ip_subnet_max"});
  registerFunction<IPSubnetRangeFunction, Array<IPAddress>, TheIPPrefix>(
      {prefix + "ip_subnet_range"});
  registerFunction<IPSubnetOfFunction, bool, TheIPPrefix, IPAddress>(
      {prefix + "is_subnet_of"});
  registerFunction<IPSubnetOfFunction, bool, TheIPPrefix, TheIPPrefix>(
      {prefix + "is_subnet_of"});
}

} // namespace facebook::velox::functions
