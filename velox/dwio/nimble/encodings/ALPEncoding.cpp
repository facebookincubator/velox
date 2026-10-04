/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/nimble/encodings/ALPEncoding.h"

namespace facebook::nimble {

template <typename T>
void ALPEncoding<T>::decodeBulkValues(
    const uint64_t* encodedValues,
    vector_size_t numValues,
    int exponent,
    int factor,
    cppDataType* output) {
  using UnsignedBatch = xsimd::batch<uint64_t>;
  using SignedBatch = xsimd::batch<int64_t>;
  using DoubleBatch = xsimd::batch<double>;
  constexpr auto kDecodeBatchSize =
      static_cast<vector_size_t>(DoubleBatch::size);
  const DoubleBatch exponentMultiplier(kPow10Double[exponent]);
  const DoubleBatch factorMultiplier(kPow10Double[factor]);
  vector_size_t row = 0;
  for (; row <= numValues - kDecodeBatchSize; row += kDecodeBatchSize) {
    // ALP stores transformed values as ZigZag-encoded uint64_t lanes for both
    // float and double inputs. Shifting removes the sign bit; XOR with the
    // zero or all-ones sign mask restores each signed integer.
    const auto zigZag = UnsignedBatch::load_unaligned(encodedValues + row);
    const auto signedBits =
        (zigZag >> 1) ^ (UnsignedBatch(0) - (zigZag & UnsignedBatch(1)));
    const auto integers = xsimd::bitwise_cast<SignedBatch>(signedBits);
    const auto restored = xsimd::batch_cast<double>(integers) *
        factorMultiplier / exponentMultiplier;
    restored.store_unaligned(output + row);
  }
  for (; row < numValues; ++row) {
    output[row] = decodeValue(
        velox::ZigZag::decode(encodedValues[row]), exponent, factor);
  }
}

template void ALPEncoding<float>::decodeBulkValues(
    const uint64_t*,
    vector_size_t,
    int,
    int,
    float*);
template void ALPEncoding<double>::decodeBulkValues(
    const uint64_t*,
    vector_size_t,
    int,
    int,
    double*);

} // namespace facebook::nimble
