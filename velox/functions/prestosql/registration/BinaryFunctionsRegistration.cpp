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
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/BinaryFunctions.h"

namespace facebook::velox::functions {

namespace {
void registerSimpleFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  // Register hash functions.
  registerFunction<CRC32Function, int64_t, Varbinary>(
      {prefix + "crc32"}, {}, true, defaultOwner);
  registerFunction<XxHash64Function, Varbinary, Varbinary>(
      {prefix + "xxhash64"}, {}, true, defaultOwner);
  registerFunction<XxHash64Function, Varbinary, Varbinary, int64_t>(
      {prefix + "xxhash64"}, {}, true, defaultOwner);
  registerFunction<XxHash128Function, Varbinary, Varbinary>(
      {prefix + "xxhash128"}, {}, true, defaultOwner);
  registerFunction<XxHash128Function, Varbinary, Varbinary, int64_t>(
      {prefix + "xxhash128"}, {}, true, defaultOwner);
  registerFunction<Md5Function, Varbinary, Varbinary>(
      {prefix + "md5"}, {}, true, defaultOwner);
  registerFunction<Murmur3X64_128Function, Varbinary, Varbinary>(
      {prefix + "murmur3_x64_128"}, {}, true, defaultOwner);
  registerFunction<Sha1Function, Varbinary, Varbinary>(
      {prefix + "sha1"}, {}, true, defaultOwner);
  registerFunction<Sha256Function, Varbinary, Varbinary>(
      {prefix + "sha256"}, {}, true, defaultOwner);
  registerFunction<Sha512Function, Varbinary, Varbinary>(
      {prefix + "sha512"}, {}, true, defaultOwner);
  registerFunction<HmacSha1Function, Varbinary, Varbinary, Varbinary>(
      {prefix + "hmac_sha1"}, {}, true, defaultOwner);
  registerFunction<HmacSha256Function, Varbinary, Varbinary, Varbinary>(
      {prefix + "hmac_sha256"}, {}, true, defaultOwner);
  registerFunction<HmacSha512Function, Varbinary, Varbinary, Varbinary>(
      {prefix + "hmac_sha512"}, {}, true, defaultOwner);
  registerFunction<HmacMd5Function, Varbinary, Varbinary, Varbinary>(
      {prefix + "hmac_md5"}, {}, true, defaultOwner);
  registerFunction<SpookyHashV232Function, Varbinary, Varbinary>(
      {prefix + "spooky_hash_v2_32"}, {}, true, defaultOwner);
  registerFunction<SpookyHashV264Function, Varbinary, Varbinary>(
      {prefix + "spooky_hash_v2_64"}, {}, true, defaultOwner);
  registerFunction<Fnv1_32Function, int32_t, Varbinary>(
      {prefix + "fnv1_32"}, {}, true, defaultOwner);
  registerFunction<Fnv1_64Function, int64_t, Varbinary>(
      {prefix + "fnv1_64"}, {}, true, defaultOwner);
  registerFunction<Fnv1a_32Function, int32_t, Varbinary>(
      {prefix + "fnv1a_32"}, {}, true, defaultOwner);
  registerFunction<Fnv1a_64Function, int64_t, Varbinary>(
      {prefix + "fnv1a_64"}, {}, true, defaultOwner);

  registerFunction<ToHexFunction, Varchar, Varbinary>(
      {prefix + "to_hex"}, {}, true, defaultOwner);
  registerFunction<FromHexFunction, Varbinary, Varchar>(
      {prefix + "from_hex"}, {}, true, defaultOwner);
  registerFunction<FromHexFunction, Varbinary, Varbinary>(
      {prefix + "from_hex"}, {}, true, defaultOwner);
  registerFunction<ToBase64Function, Varchar, Varbinary>(
      {prefix + "to_base64"}, {}, true, defaultOwner);

  registerFunction<FromBase64Function, Varbinary, Varchar>(
      {prefix + "from_base64"}, {}, true, defaultOwner);
  registerFunction<FromBase64Function, Varbinary, Varbinary>(
      {prefix + "from_base64"}, {}, true, defaultOwner);

  registerFunction<FromBase32Function, Varbinary, Varchar>(
      {prefix + "from_base32"}, {}, true, defaultOwner);
  registerFunction<FromBase32Function, Varbinary, Varbinary>(
      {prefix + "from_base32"}, {}, true, defaultOwner);

  registerFunction<ToBase64UrlFunction, Varchar, Varbinary>(
      {prefix + "to_base64url"}, {}, true, defaultOwner);
  registerFunction<FromBase64UrlFunction, Varbinary, Varchar>(
      {prefix + "from_base64url"}, {}, true, defaultOwner);
  registerFunction<FromBase64UrlFunction, Varbinary, Varbinary>(
      {prefix + "from_base64url"}, {}, true, defaultOwner);

  registerFunction<FromBigEndian32, int32_t, Varbinary>(
      {prefix + "from_big_endian_32"}, {}, true, defaultOwner);
  registerFunction<ToBigEndian32, Varbinary, int32_t>(
      {prefix + "to_big_endian_32"}, {}, true, defaultOwner);
  registerFunction<FromBigEndian64, int64_t, Varbinary>(
      {prefix + "from_big_endian_64"}, {}, true, defaultOwner);
  registerFunction<ToBigEndian64, Varbinary, int64_t>(
      {prefix + "to_big_endian_64"}, {}, true, defaultOwner);
  registerFunction<ToIEEE754Bits64, Varbinary, double>(
      {prefix + "to_ieee754_64"}, {}, true, defaultOwner);
  registerFunction<FromIEEE754Bits64, double, Varbinary>(
      {prefix + "from_ieee754_64"}, {}, true, defaultOwner);
  registerFunction<ToIEEE754Bits32, Varbinary, float>(
      {prefix + "to_ieee754_32"}, {}, true, defaultOwner);
  registerFunction<FromIEEE754Bits32, float, Varbinary>(
      {prefix + "from_ieee754_32"}, {}, true, defaultOwner);
  registerFunction<
      LPadVarbinaryFunction,
      Varbinary,
      Varbinary,
      int64_t,
      Varbinary>({prefix + "lpad"}, {}, true, defaultOwner);
  registerFunction<
      RPadVarbinaryFunction,
      Varbinary,
      Varbinary,
      int64_t,
      Varbinary>({prefix + "rpad"}, {}, true, defaultOwner);
}
} // namespace

void registerBinaryFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerSimpleFunctions(prefix, defaultOwner);
}
} // namespace facebook::velox::functions
