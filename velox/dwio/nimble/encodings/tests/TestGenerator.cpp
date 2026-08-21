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
#include <glog/logging.h>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#include "common/init/light.h"
#include "folly/io/FsUtil.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;

DEFINE_string(output_dir, "", "Output directory to write test artifacts to.");
DEFINE_string(
    encoding_file,
    "",
    "If porivided, prints the content of the encoding file.");
DEFINE_string(
    encoding_filter,
    "",
    "If provided, only generate files for this encoding type "
    "(e.g. BlockBitPacking, Trivial, Dictionary). "
    "Skips directory cleanup so existing files are preserved.");

namespace {
template <typename T, typename RNG, bool SmallNumbers = false>
T randomValue(RNG&& rng, nimble::Buffer& buffer) {
  if constexpr (std::is_same_v<bool, T>) {
    return static_cast<bool>(folly::Random::rand32(2, std::forward<RNG>(rng)));
  } else if constexpr (std::is_same_v<std::string_view, T>) {
    auto size = folly::Random::rand32(32, std::forward<RNG>(rng));
    auto data = buffer.reserve(size);
    for (auto i = 0; i < size; ++i) {
      data[i] = static_cast<char>(folly::Random::rand32(
          std::numeric_limits<uint8_t>::max(),
          // NOLINTNEXTLINE(bugprone-use-after-move)
          std::forward<RNG>(rng)));
    }
    return std::string_view{data, size};
  } else {
    uint64_t rand = SmallNumbers
        ? folly::Random::rand64(15, std::forward<RNG>(rng))
        : folly::Random::rand64(std::forward<RNG>(rng));
    return *reinterpret_cast<const T*>(&rand);
  }
}

template <typename T, typename RNG, bool SmallNumbers = false>
inline nimble::Vector<T>
generateRandomData(RNG&& rng, nimble::Buffer& buffer, int count) {
  nimble::Vector<T> data{&buffer.getMemoryPool()};
  data.reserve(count);

  for (int i = 0; i < count; ++i) {
    data.push_back(
        // NOLINTNEXTLINE(bugprone-use-after-move)
        randomValue<T, RNG, SmallNumbers>(std::forward<RNG>(rng), buffer));
  }

  return data;
}

template <typename T, typename RNG>
inline nimble::Vector<T>
generateConstData(RNG&& rng, nimble::Buffer& buffer, int count) {
  nimble::Vector<T> data{&buffer.getMemoryPool()};
  data.reserve(count);

  T value = randomValue<T>(std::forward<RNG>(rng), buffer);
  for (int i = 0; i < count; ++i) {
    data.push_back(value);
  }

  return data;
}

template <typename T, typename RNG>
inline nimble::Vector<T>
generateFixedBitWidthData(RNG&& rng, nimble::Buffer& buffer, int count) {
  nimble::Vector<T> data{&buffer.getMemoryPool()};
  data.reserve(count);

  using physicalType = typename nimble::TypeTraits<T>::physicalType;

  auto bits = 4 * sizeof(T) - 1;
  physicalType mask = (1 << bits) - 1u;
  LOG(INFO) << velox::bits::printBits(mask);
  for (int i = 0; i < count; ++i) {
    physicalType value =
        // NOLINTNEXTLINE(bugprone-use-after-move)
        randomValue<physicalType>(std::forward<RNG>(rng), buffer) & mask;
    value = value ? value : 1;
    data.push_back(value << bits);
  }

  return data;
}

template <typename T, typename RNG>
inline void randomZero(RNG&& rng, std::vector<T>& data) {
  for (int i = 0; i < data.size(); ++i) {
    if (folly::Random::rand64(std::forward<RNG>(rng)) % 2) {
      data[i] = 0;
    }
  }
}

} // namespace

template <typename E, typename RNG>
void writeFile(
    RNG&& rng,
    const std::string& path,
    uint32_t rowCount,
    std::function<nimble::Vector<
        typename E::cppDataType>(RNG&&, nimble::Buffer&, int)> dataGenerator =
        [](RNG&& rng, nimble::Buffer& buffer, int count) {
          return generateRandomData<typename E::cppDataType>(
              std::forward<RNG>(rng), buffer, count);
        },
    std::string suffix = "") {
  auto identifier = fmt::format(
      "{}_{}_{}",
      nimble::test::Encoder<E>::encodingType(),
      toString(nimble::TypeTraits<typename E::cppDataType>::dataType),
      rowCount);

  if (!suffix.empty()) {
    identifier += "_" + suffix;
  }

  LOG(INFO) << "Writing " << identifier;

  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  auto data = dataGenerator(rng, buffer, rowCount);

  {
    std::ofstream file{
        fmt::format("{}/{}.data", path, identifier),
        std::ios::out | std::ios::binary | std::ios::trunc};
    for (const auto& value : data) {
      if constexpr (std::is_same_v<std::string_view, typename E::cppDataType>) {
        auto size = value.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
        file.write(value.data(), value.size());
      } else {
        file.write(
            reinterpret_cast<const char*>(&value),
            sizeof(typename E::cppDataType));
      }
    }
  }

  std::optional<nimble::Vector<bool>> nulls;

  if (nimble::test::Encoder<E>::encodingType() ==
      nimble::EncodingType::Nullable) {
    nulls = generateRandomData<bool>(std::forward<RNG>(rng), buffer, rowCount);
  }

  if (nulls) {
    std::ofstream file{
        fmt::format("{}/{}.nulls", path, identifier),
        std::ios::out | std::ios::binary | std::ios::trunc};
    for (const auto& value : nulls.value()) {
      file.write(reinterpret_cast<const char*>(&value), sizeof(uint8_t));
    }
  }

  for (auto compressionType :
       {nimble::CompressionType::Uncompressed,
        nimble::CompressionType::Zstd,
        nimble::CompressionType::MetaInternal}) {
    std::string_view encoded;
    if constexpr (
        nimble::test::Encoder<E>::encodingType() ==
        nimble::EncodingType::Nullable) {
      encoded = nimble::test::Encoder<E>::encodeNullable(
          buffer, data, nulls.value(), compressionType);
    } else {
      encoded = nimble::test::Encoder<E>::encode(buffer, data, compressionType);
    }
    {
      std::ofstream file{
          fmt::format(
              "{}/{}_{}.encoding",
              path,
              identifier,
              compressionType == nimble::CompressionType::MetaInternal
                  ? "Zstrong"
                  : toString(compressionType)),
          std::ios::out | std::ios::binary | std::ios::trunc};
      file << encoded;
    }
  }
}

template <typename E, typename RNG>
void writeFileSmallNumbers(
    RNG&& rng,
    const std::string& path,
    uint32_t rowCount) {
  writeFile<E, RNG>(
      std::forward<RNG>(rng),
      path,
      rowCount,
      [](RNG&& rng, nimble::Buffer& buffer, int count) {
        return generateRandomData<
            typename E::cppDataType,
            RNG,
            /* SmallNumbers */ true>(std::forward<RNG>(rng), buffer, count);
      },
      "small-numbers");
}

template <typename E, typename RNG>
void writeFileConstant(RNG&& rng, const std::string& path, uint32_t rowCount) {
  writeFile<E, RNG>(
      std::forward<RNG>(rng),
      path,
      rowCount,
      [](RNG&& rng, nimble::Buffer& buffer, int count) {
        return generateConstData<typename E::cppDataType>(
            std::forward<RNG>(rng), buffer, count);
      },
      "constant");
}

template <typename E, typename RNG>
void writeFileFixedBitWidth(
    RNG&& rng,
    const std::string& path,
    uint32_t rowCount) {
  writeFile<E, RNG>(
      std::forward<RNG>(rng),
      path,
      rowCount,
      [](RNG&& rng, nimble::Buffer& buffer, int count) {
        return generateFixedBitWidthData<typename E::cppDataType>(
            std::forward<RNG>(rng), buffer, count);
      },
      "bits");
}

template <typename T>
void printScalarData(
    std::ostream& ostream,
    velox::memory::MemoryPool& pool,
    nimble::Encoding& stream,
    uint32_t rowCount) {
  nimble::Vector<T> buffer(&pool);
  nimble::Vector<char> nulls(&pool);
  buffer.resize(rowCount);
  nulls.resize((nimble::FixedBitArray::bufferSize(rowCount, 1)));
  nulls.zero_out();
  if (stream.isNullable()) {
    stream.materializeNullable(
        rowCount, buffer.data(), [&]() { return nulls.data(); });
  } else {
    stream.materialize(rowCount, buffer.data());
    nulls.fill(0xff);
  }
  for (uint32_t i = 0; i < rowCount; ++i) {
    if (stream.isNullable() &&
        velox::bits::isBitSet(
            reinterpret_cast<const uint8_t*>(nulls.data()), i) == 0) {
      ostream << "NULL" << std::endl;
    } else {
      ostream << folly::to<std::string>(buffer[i])
              << std::endl; // Have to use folly::to as Int8 was getting
                            // converted to char.
    }
  }
}

void printScalarType(
    std::ostream& ostream,
    velox::memory::MemoryPool& pool,
    nimble::Encoding& stream,
    uint32_t rowCount) {
  switch (stream.dataType()) {
#define CASE(KIND, cppType)                                    \
  case nimble::DataType::KIND: {                               \
    printScalarData<cppType>(ostream, pool, stream, rowCount); \
    break;                                                     \
  }
    CASE(Int8, int8_t);
    CASE(Uint8, uint8_t);
    CASE(Int16, int16_t);
    CASE(Uint16, uint16_t);
    CASE(Int32, int32_t);
    CASE(Uint32, uint32_t);
    CASE(Int64, int64_t);
    CASE(Uint64, uint64_t);
    CASE(Float, float);
    CASE(Double, double);
    CASE(Bool, bool);
    CASE(String, std::string_view);
#undef CASE
    case nimble::DataType::Undefined: {
      NIMBLE_UNREACHABLE(
          fmt::format("Undefined type for stream: {}", stream.dataType()));
    }
  }
}

#define _WRITE_NUMERIC_FILES_(encoding, rowCount, type)                        \
  writeFile##type<nimble::encoding<int8_t>>(rng, FLAGS_output_dir, rowCount);  \
  writeFile##type<nimble::encoding<uint8_t>>(rng, FLAGS_output_dir, rowCount); \
  writeFile##type<nimble::encoding<int16_t>>(rng, FLAGS_output_dir, rowCount); \
  writeFile##type<nimble::encoding<uint16_t>>(                                 \
      rng, FLAGS_output_dir, rowCount);                                        \
  writeFile##type<nimble::encoding<int32_t>>(rng, FLAGS_output_dir, rowCount); \
  writeFile##type<nimble::encoding<uint32_t>>(                                 \
      rng, FLAGS_output_dir, rowCount);                                        \
  writeFile##type<nimble::encoding<int64_t>>(rng, FLAGS_output_dir, rowCount); \
  writeFile##type<nimble::encoding<uint64_t>>(                                 \
      rng, FLAGS_output_dir, rowCount);                                        \
  writeFile##type<nimble::encoding<float>>(rng, FLAGS_output_dir, rowCount);   \
  writeFile##type<nimble::encoding<double>>(rng, FLAGS_output_dir, rowCount);

#define _WRITE_FILES_(encoding, rowCount, type)                             \
  _WRITE_NUMERIC_FILES_(encoding, rowCount, type)                           \
  writeFile##type<nimble::encoding<bool>>(rng, FLAGS_output_dir, rowCount); \
  writeFile##type<nimble::encoding<std::string_view>>(                      \
      rng, FLAGS_output_dir, rowCount);

#define WRITE_FILES(encoding, rowCount) _WRITE_FILES_(encoding, rowCount, )

#define WRITE_NUMERIC_FILES(encoding, rowCount) \
  _WRITE_NUMERIC_FILES_(encoding, rowCount, )

#define WRITE_CONSTANT_FILES(rowCount)                 \
  _WRITE_FILES_(ConstantEncoding, rowCount, Constant); \
  _WRITE_FILES_(RLEEncoding, rowCount, Constant)

#define WRITE_FBW_FILES(rowCount)                                        \
  _WRITE_NUMERIC_FILES_(FixedBitWidthEncoding, rowCount, FixedBitWidth); \
  _WRITE_NUMERIC_FILES_(FixedBitWidthEncoding, rowCount, SmallNumbers)

#define WRITE_NON_BOOL_FILES(encoding, rowCount) \
  _WRITE_NUMERIC_FILES_(encoding, rowCount, )    \
  writeFile<nimble::encoding<std::string_view>>( \
      rng, FLAGS_output_dir, rowCount);

int main(int argc, char* argv[]) {
  init::initFacebookLight(&argc, &argv);

  if (!FLAGS_encoding_file.empty()) {
    std::ifstream stream{FLAGS_encoding_file, std::ios::binary};
    stream.seekg(0, std::ios::end);
    auto size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    stream.read(buffer.data(), buffer.size());

    auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
    std::vector<velox::BufferPtr> newStringBuffers;
    const auto stringBufferFactory = [&](uint32_t totalLength) {
      auto& buffer = newStringBuffers.emplace_back(
          velox::AlignedBuffer::allocate<char>(totalLength, pool.get()));
      return buffer->asMutable<void>();
    };
    auto encoding = nimble::EncodingFactory().create(
        *pool, {buffer.data(), buffer.size()}, stringBufferFactory);
    auto rowCount = encoding->rowCount();
    printScalarType(std::cout, *pool, *encoding, rowCount);
    return 0;
  }

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  if (FLAGS_output_dir.empty()) {
    LOG(ERROR) << "Output directory is empty";
    return 1;
  }

  const auto& filter = FLAGS_encoding_filter;
  const auto match = [&](const std::string& name) {
    return filter.empty() || name == filter;
  };

  if (filter.empty()) {
    if (folly::fs::exists(FLAGS_output_dir)) {
      folly::fs::remove_all(FLAGS_output_dir);
    }
    folly::fs::create_directory(FLAGS_output_dir);
  } else if (!folly::fs::exists(FLAGS_output_dir)) {
    folly::fs::create_directory(FLAGS_output_dir);
  }

  if (match("Constant") || match("RLE")) {
    WRITE_CONSTANT_FILES(256);
  }
  if (match("FixedBitWidth")) {
    WRITE_FBW_FILES(256);
  }
  if (match("MainlyConstant")) {
    WRITE_NON_BOOL_FILES(MainlyConstantEncoding, 256);
  }

  for (auto rowCount : {0, 256}) {
    if (match("Trivial")) {
      WRITE_FILES(TrivialEncoding, rowCount);
    }
    if (match("Dictionary")) {
      WRITE_FILES(DictionaryEncoding, rowCount);
    }
    if (match("RLE")) {
      WRITE_FILES(RLEEncoding, rowCount);
    }
    if (match("Nullable")) {
      WRITE_FILES(NullableEncoding, rowCount);
    }
    if (match("FixedBitWidth")) {
      WRITE_NUMERIC_FILES(FixedBitWidthEncoding, rowCount);
    }
    if (match("BlockBitPacking")) {
      WRITE_NUMERIC_FILES(BlockBitPackingEncoding, rowCount);
    }
    if (match("SparseBool")) {
      writeFile<nimble::SparseBoolEncoding>(rng, FLAGS_output_dir, rowCount);
    }
    if (match("Varint")) {
      writeFile<nimble::VarintEncoding<int32_t>>(
          rng, FLAGS_output_dir, rowCount);
      writeFile<nimble::VarintEncoding<uint32_t>>(
          rng, FLAGS_output_dir, rowCount);
      writeFile<nimble::VarintEncoding<int64_t>>(
          rng, FLAGS_output_dir, rowCount);
      writeFile<nimble::VarintEncoding<uint64_t>>(
          rng, FLAGS_output_dir, rowCount);
      writeFile<nimble::VarintEncoding<float>>(rng, FLAGS_output_dir, rowCount);
      writeFile<nimble::VarintEncoding<double>>(
          rng, FLAGS_output_dir, rowCount);
    }
  }

  return 0;
}
