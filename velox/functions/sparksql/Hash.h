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

#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions::sparksql {

/// Computes the hash of a single primitive value using the hash function in
/// HashClass (e.g. XxHash64 or Murmur3Hash). HashClass provides the
/// Spark-compatible hashInt32, hashInt64, hashFloat, hashDouble,
/// hashLongDecimal, hashTimestamp and hashBytes functions. Shared by the hash
/// vector functions and by aggregates that need Spark's hash of a value.
template <typename HashClass>
typename HashClass::ReturnType hashOne(
    int32_t input,
    typename HashClass::SeedType seed) {
  return HashClass::hashInt32(input, seed);
}

template <typename HashClass>
typename HashClass::ReturnType hashOne(
    int64_t input,
    typename HashClass::SeedType seed) {
  return HashClass::hashInt64(input, seed);
}

template <typename HashClass>
typename HashClass::ReturnType hashOne(
    float input,
    typename HashClass::SeedType seed) {
  return HashClass::hashFloat(input, seed);
}

template <typename HashClass>
typename HashClass::ReturnType hashOne(
    double input,
    typename HashClass::SeedType seed) {
  return HashClass::hashDouble(input, seed);
}

template <typename HashClass>
typename HashClass::ReturnType hashOne(
    int128_t input,
    typename HashClass::SeedType seed) {
  return HashClass::hashLongDecimal(input, seed);
}

template <typename HashClass>
typename HashClass::ReturnType hashOne(
    Timestamp input,
    typename HashClass::SeedType seed) {
  return HashClass::hashTimestamp(input, seed);
}

template <typename HashClass>
typename HashClass::ReturnType hashOne(
    StringView input,
    typename HashClass::SeedType seed) {
  return HashClass::hashBytes(input, seed);
}

template <typename HashClass>
typename HashClass::ReturnType hashOne(
    UnknownValue /*input*/,
    typename HashClass::SeedType seed) {
  return seed;
}

// Supported types:
//   - Boolean
//   - Integer types (tinyint, smallint, integer, bigint)
//   - Varchar, varbinary
//   - Real, double
//   - Decimal
//   - Date
//   - Timestamp
//
// TODO:
//   - Row, Array: hash the elements in order
//   - Map: iterate over map, hashing key then value. Since map ordering is
//        unspecified, hashing logically equivalent maps may result in
//        different hash values.

std::vector<std::shared_ptr<exec::FunctionSignature>> hashSignatures();

std::vector<std::shared_ptr<exec::FunctionSignature>> hashWithSeedSignatures();

std::shared_ptr<exec::VectorFunction> makeHash(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

std::shared_ptr<exec::VectorFunction> makeHashWithSeed(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

// Supported types:
//   - Bools
//   - Integer types (byte, short, int, long)
//   - String, Binary
//   - Float, Double
//   - Decimal
//   - Date
//   - Timestamp
//   - UnknownType
//   - Struct
//   - Array
//   - Map

std::vector<std::shared_ptr<exec::FunctionSignature>> xxhash64Signatures();

std::vector<std::shared_ptr<exec::FunctionSignature>>
xxhash64WithSeedSignatures();

std::shared_ptr<exec::VectorFunction> makeXxHash64(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

std::shared_ptr<exec::VectorFunction> makeXxHash64WithSeed(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

exec::VectorFunctionMetadata hashMetadata();

} // namespace facebook::velox::functions::sparksql
