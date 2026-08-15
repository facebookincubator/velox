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

namespace facebook::velox::functions {

/// Sequence function that generates an array of values from start to stop
/// (inclusive) with an optional step. Supports integer types (tinyint,
/// smallint, integer, bigint), date, and timestamp.
template <typename T, typename K>
class SequenceFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override;

 private:
  static vector_size_t checkArguments(
      DecodedVector* startVector,
      DecodedVector* stopVector,
      DecodedVector* stepVector,
      vector_size_t row,
      bool isDate,
      bool isYearMonth,
      int32_t maxElementsSize);

  static void writeToElements(
      T* elements,
      bool isDate,
      bool isYearMonth,
      vector_size_t sequenceCount,
      DecodedVector* startVector,
      DecodedVector* stopVector,
      DecodedVector* stepVector,
      vector_size_t row);

  static K getStep(
      int64_t start,
      int64_t stop,
      DecodedVector* stepVector,
      vector_size_t row,
      bool isDate,
      bool isYearMonth);
};

} // namespace facebook::velox::functions
