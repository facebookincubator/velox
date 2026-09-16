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
