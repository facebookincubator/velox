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

#include <boost/random/uniform_int_distribution.hpp>

#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/tz/TimeZoneMap.h"
#include "velox/vector/fuzzer/CustomVectorFuzzer.h"

namespace facebook::velox {
/// A CustomVectorFuzzer for TimestampWithTimeZoneType. The millisUtc is random,
/// and the time zone is selected randomly from the list of known time zones.
class TimestampWithTimeZoneVectorFuzzer : public CustomVectorFuzzer {
 public:
  TimestampWithTimeZoneVectorFuzzer()
      : CustomVectorFuzzer(), timeZoneIds_(tz::getTimeZoneIDs()) {}

  const VectorPtr fuzzFlat(
      memory::MemoryPool* pool,
      const TypePtr& type,
      vector_size_t size,
      FuzzerGenerator& rng) override {
    VELOX_CHECK(isTimestampWithTimeZoneType(type));

    auto result = BaseVector::create(type, size, pool);
    auto flatResult = result->asFlatVector<int64_t>();
    for (auto i = 0; i < size; ++i) {
      int16_t timeZoneId = timeZoneIds_
          [boost::random::uniform_int_distribution<size_t>()(rng) %
           timeZoneIds_.size()];
      flatResult->set(
          i,
          pack(
              boost::random::uniform_int_distribution<int64_t>()(rng),
              timeZoneId));
    }
    return result;
  }

  const VectorPtr fuzzConstant(
      memory::MemoryPool* pool,
      const TypePtr& type,
      vector_size_t size,
      FuzzerGenerator& rng) override {
    VELOX_CHECK(isTimestampWithTimeZoneType(type));

    int16_t timeZoneId = timeZoneIds_
        [boost::random::uniform_int_distribution<size_t>()(rng) %
         timeZoneIds_.size()];

    return std::make_shared<ConstantVector<int64_t>>(
        pool,
        size,
        false,
        type,
        pack(
            boost::random::uniform_int_distribution<int64_t>()(rng),
            timeZoneId));
  }

 private:
  const std::vector<int16_t> timeZoneIds_;
};
} // namespace facebook::velox
