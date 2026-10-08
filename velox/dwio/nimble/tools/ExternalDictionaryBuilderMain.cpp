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

#include <fstream>
#include <istream>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

#include <glog/logging.h>
#include "folly/Conv.h"

#include "common/init/light.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/DataTypeDispatch.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/tools/ExternalDictionaryBuilder.h"

DEFINE_string(
    input,
    "",
    "Input file containing whitespace-delimited dictionary values.");
DEFINE_string(output, "", "Output external dictionary artifact file.");
DEFINE_string(data_type, "", "Integer data type, e.g. Int32 or Uint64.");
DEFINE_bool(
    sort_values,
    true,
    "Sort the unique alphabet values before encoding.");
DEFINE_string(
    alphabet_encoding,
    "",
    "Forced alphabet encoding. Empty uses encoding selection.");
DEFINE_string(read_factors, "", "Read factors to use for encoding selection.");

namespace facebook::nimble {
namespace {

template <typename T>
T parseValue(std::string_view text) {
  static_assert(isIntegralType<T>() && !std::is_same_v<T, bool>);
  if constexpr (std::is_signed_v<T>) {
    const auto value = folly::to<int64_t>(text);
    NIMBLE_USER_CHECK_GE(
        value,
        static_cast<int64_t>(std::numeric_limits<T>::min()),
        "Value {} is below {} range.",
        text,
        toString(TypeTraits<T>::dataType));
    NIMBLE_USER_CHECK_LE(
        value,
        static_cast<int64_t>(std::numeric_limits<T>::max()),
        "Value {} is above {} range.",
        text,
        toString(TypeTraits<T>::dataType));
    return static_cast<T>(value);
  } else {
    const auto value = folly::to<uint64_t>(text);
    NIMBLE_USER_CHECK_LE(
        value,
        static_cast<uint64_t>(std::numeric_limits<T>::max()),
        "Value {} is above {} range.",
        text,
        toString(TypeTraits<T>::dataType));
    return static_cast<T>(value);
  }
}

template <typename T>
Vector<T> readValues(std::istream& input, velox::memory::MemoryPool* pool) {
  Vector<T> values{pool};
  std::string value;
  // Treat spaces and newlines the same, so callers can use one value per line
  // or a compact space-delimited file.
  while (input >> value) {
    values.push_back(parseValue<T>(value));
  }
  return values;
}

template <typename T>
Vector<T> readInput(velox::memory::MemoryPool* pool) {
  NIMBLE_USER_CHECK(!FLAGS_input.empty(), "input is required.");
  std::ifstream file{FLAGS_input};
  NIMBLE_USER_CHECK(
      file.is_open(),
      "Unable to open shared dictionary input '{}'.",
      FLAGS_input);
  return readValues<T>(file, pool);
}

DataType parseDataType(std::string_view dataType) {
  NIMBLE_USER_CHECK(!dataType.empty(), "data_type is required.");
#define DATA_TYPE(type)                                   \
  if (dataType == toString(TypeTraits<type>::dataType)) { \
    return TypeTraits<type>::dataType;                    \
  }
  DATA_TYPE(int8_t)
  DATA_TYPE(uint8_t)
  DATA_TYPE(int16_t)
  DATA_TYPE(uint16_t)
  DATA_TYPE(int32_t)
  DATA_TYPE(uint32_t)
  DATA_TYPE(int64_t)
  DATA_TYPE(uint64_t)
#undef DATA_TYPE
  NIMBLE_USER_FAIL("Unsupported external dictionary data type '{}'.", dataType);
}

template <typename T>
std::string buildExternalDictionaryArtifact(
    const ExternalDictionaryBuilder::Options& options) {
  auto pool =
      velox::memory::memoryManager()->addLeafPool("external_dictionary_build");
  const ExternalDictionaryBuilder builder{pool.get()};
  const auto source = readInput<T>(pool.get());
  const auto alphabet = builder.build(source, options);
  return builder.serialize(alphabet);
}

std::string buildExternalDictionaryArtifact(
    DataType dataType,
    const ExternalDictionaryBuilder::Options& options) {
  NIMBLE_RETURN_BY_INTEGER_DATA_TYPE_OR(
      dataType,
      Type,
      buildExternalDictionaryArtifact<Type>(options),
      NIMBLE_USER_FAIL(
          "Unsupported external dictionary data type '{}'.",
          toString(dataType)));
}

void writeOutput(std::string_view serializedDictionary) {
  NIMBLE_USER_CHECK(!FLAGS_output.empty(), "output is required.");
  std::ofstream output{FLAGS_output, std::ios::binary};
  NIMBLE_USER_CHECK(
      output.is_open(),
      "Unable to open external dictionary output '{}'.",
      FLAGS_output);
  output.write(serializedDictionary.data(), serializedDictionary.size());
  NIMBLE_USER_CHECK(
      output.good(),
      "Failed to write external dictionary output '{}'.",
      FLAGS_output);
}

} // namespace
} // namespace facebook::nimble

int main(int argc, char* argv[]) {
  auto init = facebook::init::InitFacebookLight{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});
  facebook::velox::filesystems::registerLocalFileSystem();
  using namespace facebook::nimble;

  ExternalDictionaryBuilder::Options options{
      .sortValues = FLAGS_sort_values,
  };
  if (!FLAGS_alphabet_encoding.empty()) {
    options.alphabetEncoding = toEncodingType(FLAGS_alphabet_encoding);
  }
  if (!FLAGS_read_factors.empty()) {
    options.readFactors =
        ManualEncodingSelectionPolicyFactory::parseEncodingReadFactors(
            FLAGS_read_factors);
  }

  const auto dictionaryArtifact =
      buildExternalDictionaryArtifact(parseDataType(FLAGS_data_type), options);
  writeOutput(dictionaryArtifact);
  LOG(INFO) << "Wrote external dictionary artifact " << FLAGS_output << " ("
            << dictionaryArtifact.size() << " bytes).";
  return 0;
}
