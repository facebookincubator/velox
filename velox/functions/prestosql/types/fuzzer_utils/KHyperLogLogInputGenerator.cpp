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

#include "velox/functions/prestosql/types/fuzzer_utils/KHyperLogLogInputGenerator.h"

#include "velox/common/fuzzer/Utils.h"
#include "velox/functions/lib/KHyperLogLog.h"
#include "velox/functions/prestosql/types/KHyperLogLogType.h"
#include "velox/type/Variant.h"

namespace facebook::velox::fuzzer {

KHyperLogLogInputGenerator::KHyperLogLogInputGenerator(
    const size_t seed,
    const double nullRatio,
    memory::MemoryPool* pool,
    int32_t minNumValues)
    : AbstractInputGenerator{seed, KHYPERLOGLOG(), nullptr, nullRatio},
      minNumValues_{minNumValues},
      pool_{pool} {
  static const std::vector<TypePtr> kBaseTypes{
      BIGINT(), VARCHAR(), DOUBLE(), UNKNOWN()};
  baseType_ = kBaseTypes[rand<int32_t>(rng_, 0, kBaseTypes.size() - 1)];

  maxSize_ = rand<int32_t>(rng_, 100, 5000);
  static const std::vector<int32_t> kValidBuckets{64, 128, 256, 512, 1024};
  hllBuckets_ = kValidBuckets[rand<int32_t>(rng_, 0, kValidBuckets.size() - 1)];
}

// General template for numeric types (double)
template <typename T>
variant KHyperLogLogInputGenerator::generateTyped() {
  HashStringAllocator allocator{pool_};
  common::hll::KHyperLogLog<int64_t, HashStringAllocator> khll{
      maxSize_, hllBuckets_, &allocator};

  auto numPairs = rand<int32_t>(rng_, minNumValues_, 10000);
  for (auto i = 0; i < numPairs; ++i) {
    auto value = rand<T>(rng_);
    int64_t joinKey = std::hash<T>{}(value);
    int64_t uii = rand<int64_t>(rng_);
    khll.add(joinKey, uii);
  }

  auto size = khll.estimatedSerializedSize();
  std::string buff(size, '\0');
  khll.serialize(buff.data());
  return variant::binary(std::move(buff));
}

// Specialization for int64_t
template <>
variant KHyperLogLogInputGenerator::generateTyped<int64_t>() {
  HashStringAllocator allocator{pool_};
  common::hll::KHyperLogLog<int64_t, HashStringAllocator> khll{
      maxSize_, hllBuckets_, &allocator};

  auto numPairs = rand<int32_t>(rng_, minNumValues_, 10000);
  for (auto i = 0; i < numPairs; ++i) {
    int64_t joinKey = rand<int64_t>(rng_);
    int64_t uii = rand<int64_t>(rng_);
    khll.add(joinKey, uii);
  }

  auto size = khll.estimatedSerializedSize();
  std::string buff(size, '\0');
  khll.serialize(buff.data());
  return variant::binary(std::move(buff));
}

// Specialization for std::string (VARCHAR)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template <>
variant KHyperLogLogInputGenerator::generateTyped<std::string>() {
  HashStringAllocator allocator{pool_};
  common::hll::KHyperLogLog<int64_t, HashStringAllocator> khll{
      maxSize_, hllBuckets_, &allocator};

  static const std::vector<UTF8CharList> encodings{
      UTF8CharList::ASCII,
      UTF8CharList::UNICODE_CASE_SENSITIVE,
      UTF8CharList::EXTENDED_UNICODE,
      UTF8CharList::MATHEMATICAL_SYMBOLS};
  std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t> converter;

  auto numPairs = rand<int32_t>(rng_, minNumValues_, 10000);
  for (auto i = 0; i < numPairs; ++i) {
    auto size = rand<int32_t>(rng_, 0, 100);
    std::string result;
    auto str = randString(rng_, size, encodings, result, converter);
    int64_t joinKey = std::hash<std::string>{}(str);
    int64_t uii = rand<int64_t>(rng_);
    khll.add(joinKey, uii);
  }

  auto size = khll.estimatedSerializedSize();
  std::string buff(size, '\0');
  khll.serialize(buff.data());
  return variant::binary(std::move(buff));
}
#pragma GCC diagnostic pop

// Specialization for UnknownValue (NULL type)
template <>
variant KHyperLogLogInputGenerator::generateTyped<UnknownValue>() {
  HashStringAllocator allocator{pool_};
  common::hll::KHyperLogLog<int64_t, HashStringAllocator> khll{
      maxSize_, hllBuckets_, &allocator};

  // UnknownValue represents NULL, so create an empty KHyperLogLog
  // since KHyperLogLog ignores NULL inputs

  auto size = khll.estimatedSerializedSize();
  std::string buff(size, '\0');
  khll.serialize(buff.data());
  return variant::binary(std::move(buff));
}

variant KHyperLogLogInputGenerator::generate() {
  if (coinToss(rng_, nullRatio_)) {
    return variant::null(type_->kind());
  }

  if (baseType_->isBigint()) {
    return generateTyped<int64_t>();
  } else if (baseType_->isVarchar()) {
    return generateTyped<std::string>();
  } else if (baseType_->isDouble()) {
    return generateTyped<double>();
  } else {
    return generateTyped<UnknownValue>();
  }
}

} // namespace facebook::velox::fuzzer
