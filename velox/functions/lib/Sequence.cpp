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

#include "velox/functions/lib/Sequence.h"
#include "velox/expression/DecodedArgs.h"
#include "velox/functions/prestosql/DateTimeImpl.h"

namespace facebook::velox::functions {
namespace {

template <typename T>
int64_t toInt64(T value);

template <>
int64_t toInt64(int8_t value) {
  return value;
}

template <>
int64_t toInt64(int16_t value) {
  return value;
}

template <>
int64_t toInt64(int32_t value) {
  return value;
}

template <>
int64_t toInt64(int64_t value) {
  return value;
}

template <>
int64_t toInt64(Timestamp value) {
  return value.toMillis();
}

using Days = int64_t;
using Months = int32_t;
template <typename T, typename K>
T add(T value, K step, int32_t sequence);

template <>
int8_t add(int8_t value, int8_t step, int32_t sequence) {
  const auto delta =
      static_cast<int128_t>(step) * static_cast<int128_t>(sequence);
  return value + delta;
}

template <>
int16_t add(int16_t value, int16_t step, int32_t sequence) {
  const auto delta =
      static_cast<int128_t>(step) * static_cast<int128_t>(sequence);
  return value + delta;
}

template <>
int64_t add(int64_t value, int64_t step, int32_t sequence) {
  const auto delta =
      static_cast<int128_t>(step) * static_cast<int128_t>(sequence);
  // Since step is calculated from start and stop,
  // the sum of 'value' and 'add' is within int64_t.
  return value + delta;
}

template <>
int32_t add(int32_t value, int64_t step, int32_t sequence) {
  const auto delta =
      static_cast<int128_t>(step) * static_cast<int128_t>(sequence);
  return value + delta;
}

template <>
Timestamp add(Timestamp value, int64_t step, int32_t sequence) {
  const auto delta =
      static_cast<int128_t>(step) * static_cast<int128_t>(sequence);
  return Timestamp::fromMillis(value.toMillis() + delta);
}

template <>
int32_t add(int32_t value, Months step, int32_t sequence) {
  return addToDate(value, DateTimeUnit::kMonth, step * sequence);
}

template <>
Timestamp add(Timestamp value, Months step, int32_t sequence) {
  return addToTimestamp(value, DateTimeUnit::kMonth, step * sequence);
}

template <typename T>
int128_t getStepCount(T start, T end, int32_t step) {
  VELOX_FAIL("Unexpected start/end type for argument INTERVAL_YEAR_MONTH");
}

template <>
int128_t getStepCount(int32_t start, int32_t end, int32_t step) {
  return diffDate(DateTimeUnit::kMonth, start, end) / step + 1;
}

template <>
int128_t getStepCount(Timestamp start, Timestamp end, int32_t step) {
  return diffTimestamp(DateTimeUnit::kMonth, start, end) / step + 1;
}

} // namespace

template <typename T, typename K>
void SequenceFunction<T, K>::apply(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) const {
  exec::DecodedArgs decodedArgs(rows, args, context);
  auto startVector = decodedArgs.at(0);
  auto stopVector = decodedArgs.at(1);
  DecodedVector* stepVector = nullptr;
  bool isIntervalYearMonth = false;
  if (args.size() == 3) {
    stepVector = decodedArgs.at(2);
    isIntervalYearMonth = args[2]->type()->isIntervalYearMonth();
  }

  const auto numRows = rows.end();
  auto pool = context.pool();
  size_t numElements = 0;

  BufferPtr sizes = allocateSizes(numRows, pool);
  BufferPtr offsets = allocateOffsets(numRows, pool);
  auto rawSizes = sizes->asMutable<vector_size_t>();
  auto rawOffsets = offsets->asMutable<vector_size_t>();

  const bool isDate = args[0]->type()->isDate();
  context.applyToSelectedNoThrow(rows, [&](auto row) {
    rawSizes[row] = checkArguments(
        startVector,
        stopVector,
        stepVector,
        row,
        isDate,
        isIntervalYearMonth,
        context.execCtx()
            ->queryCtx()
            ->queryConfig()
            .maxElementsSizeInRepeatAndSequence());
    numElements += rawSizes[row];
  });

  // We could overflow int32 if total number of elements is too large,
  // potentially causing a small buffer allocation followed by a large write,
  // resulting in SIGSEGV.
  VELOX_USER_CHECK_LE(
      numElements,
      std::numeric_limits<vector_size_t>::max(),
      "SEQUENCE result too large: {} elements exceeds maximum {}",
      numElements,
      std::numeric_limits<vector_size_t>::max());

  VectorPtr elements =
      BaseVector::create(outputType->childAt(0), numElements, pool);
  auto rawElements = elements->asFlatVector<T>()->mutableRawValues();

  vector_size_t elementsOffset = 0;
  context.applyToSelectedNoThrow(rows, [&](auto row) {
    const auto sequenceCount = rawSizes[row];
    if (sequenceCount) {
      rawOffsets[row] = elementsOffset;
      writeToElements(
          rawElements + elementsOffset,
          isDate,
          isIntervalYearMonth,
          sequenceCount,
          startVector,
          stopVector,
          stepVector,
          row);
      elementsOffset += rawSizes[row];
    }
  });
  context.moveOrCopyResult(
      std::make_shared<ArrayVector>(
          pool, outputType, nullptr, numRows, offsets, sizes, elements),
      rows,
      result);
}

template <typename T, typename K>
vector_size_t SequenceFunction<T, K>::checkArguments(
    DecodedVector* startVector,
    DecodedVector* stopVector,
    DecodedVector* stepVector,
    vector_size_t row,
    bool isDate,
    bool isYearMonth,
    int32_t maxElementsSize) {
  T start = startVector->valueAt<T>(row);
  T stop = stopVector->valueAt<T>(row);
  auto step = getStep(
      toInt64(start), toInt64(stop), stepVector, row, isDate, isYearMonth);
  VELOX_USER_CHECK_NE(step, 0, "step must not be zero");
  VELOX_USER_CHECK(
      step > 0 ? stop >= start : stop <= start,
      "sequence stop value should be greater than or equal to start value if "
      "step is greater than zero otherwise stop should be less than or equal to start");
  int128_t sequenceCount;
  if (isYearMonth) {
    sequenceCount = getStepCount(start, stop, step);
  } else {
    sequenceCount = (static_cast<int128_t>(toInt64(stop)) -
                     static_cast<int128_t>(toInt64(start))) /
            step +
        1; // prevent overflow
  }

  VELOX_USER_CHECK_LE(
      sequenceCount,
      maxElementsSize,
      "result of sequence function must not have more than {} entries",
      maxElementsSize);
  return sequenceCount;
}

template <typename T, typename K>
void SequenceFunction<T, K>::writeToElements(
    T* elements,
    bool isDate,
    bool isYearMonth,
    vector_size_t sequenceCount,
    DecodedVector* startVector,
    DecodedVector* stopVector,
    DecodedVector* stepVector,
    vector_size_t row) {
  auto start = startVector->valueAt<T>(row);
  auto stop = stopVector->valueAt<T>(row);
  auto step = getStep(
      toInt64(start), toInt64(stop), stepVector, row, isDate, isYearMonth);
  for (auto sequence = 0; sequence < sequenceCount; ++sequence) {
    elements[sequence] = add(start, step, sequence);
  }
}

template <typename T, typename K>
K SequenceFunction<T, K>::getStep(
    int64_t start,
    int64_t stop,
    DecodedVector* stepVector,
    vector_size_t row,
    bool isDate,
    bool isYearMonth) {
  if (!stepVector) {
    return (stop >= start ? 1 : -1);
  }
  // When element type T differs from step type K (e.g., integer sequence
  // using T=int32_t, K=int64_t to avoid conflicting with date
  // specializations), read step as T and cast to K.
  if constexpr (std::is_integral_v<T> && !std::is_same_v<T, K>) {
    if (!isDate) {
      return static_cast<K>(stepVector->valueAt<T>(row));
    }
  }
  auto step = stepVector->valueAt<K>(row);
  if (!isDate || isYearMonth) {
    return step;
  }
  // Handle Date
  VELOX_USER_CHECK(
      step % kMillisInDay == 0,
      "sequence step must be a day interval if start and end values are dates");
  return step / kMillisInDay;
}

// Explicit instantiations for all supported types.
// Presto types:
template class SequenceFunction<int64_t, int64_t>;
template class SequenceFunction<int32_t, int64_t>;
template class SequenceFunction<int32_t, int32_t>;
template class SequenceFunction<Timestamp, int64_t>;
template class SequenceFunction<Timestamp, int32_t>;
// Spark-specific types:
template class SequenceFunction<int8_t, int8_t>;
template class SequenceFunction<int16_t, int16_t>;

} // namespace facebook::velox::functions
