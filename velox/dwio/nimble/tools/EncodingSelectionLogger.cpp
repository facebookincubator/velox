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
#define NIMBLE_ENCODING_SELECTION_DEBUG

#include <fstream>
#include <iostream>
#include "common/init/light.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/tools/EncodingUtilities.h"

DEFINE_string(
    data_type,
    "",
    "If provided, used as the data type for the encoded values. "
    "If not provided, first input line is parsed to try and find the data type.");

DEFINE_string(
    file,
    "",
    "If provided, used as the input file to read from. Otherwise, cin is used.");

DEFINE_string(read_factors, "", "Read factors to use for encoding selection.");
DEFINE_double(compression_acceptance_ratio, 0.9, "");

using namespace facebook;

template <typename T>
void logEncodingSelection(const std::vector<std::string>& source) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Vector<T> values{pool.get()};
  values.reserve(source.size());
  for (const auto& value : source) {
    values.push_back(folly::to<T>(value));
  }

  auto policy = std::unique_ptr<nimble::ManualEncodingSelectionPolicy<T>>(
      static_cast<nimble::ManualEncodingSelectionPolicy<T>*>(
          facebook::nimble::ManualEncodingSelectionPolicyFactory{
              FLAGS_read_factors.empty()
                  ? nimble::ManualEncodingSelectionPolicyFactory::
                        defaultEncodingReadFactors()
                  : nimble::ManualEncodingSelectionPolicyFactory::
                        parseEncodingReadFactors(FLAGS_read_factors),
              nimble::CompressionOptions{
                  .compressionAcceptRatio =
                      folly::to<float>(FLAGS_compression_acceptance_ratio)}}
              .createPolicy(nimble::TypeTraits<T>::dataType)
              .release()));
  nimble::Buffer buffer{*pool};

  auto serialized =
      nimble::EncodingFactory::encode<T>(std::move(policy), values, buffer);

  LOG(INFO) << "Encoding: " << nimble::tools::getEncodingLabel(serialized);
}

int main(int argc, char* argv[]) {
  auto init = init::InitFacebookLight{&argc, &argv};

  std::istream* stream{&std::cin};
  std::ifstream file;
  if (!FLAGS_file.empty()) {
    file = std::ifstream{FLAGS_file};
    stream = &file;
  }

  std::string dataTypeString = FLAGS_data_type;
  if (dataTypeString.empty()) {
    std::string line;
    std::getline(*stream, line);

    const std::string kDataTypePrefix = "DataType ";
    auto index = line.find(kDataTypePrefix);
    if (index == std::string::npos) {
      LOG(FATAL)
          << "Unable to find data type prefix '" << kDataTypePrefix
          << "' in first input row. Consider providing the data type using "
             "the 'data_type' command line argument.";
    }

    line = line.substr(index + kDataTypePrefix.size());
    auto end = line.find(',');
    if (end == std::string::npos) {
      dataTypeString = line;
    } else {
      dataTypeString = line.substr(0, end);
    }
  }

#define DATA_TYPE(type) \
  {toString(nimble::TypeTraits<type>::dataType), logEncodingSelection<type>}
  std::unordered_map<
      std::string,
      std::function<void(const std::vector<std::string>&)>>
      dataTypes{
          DATA_TYPE(int8_t),
          DATA_TYPE(uint8_t),
          DATA_TYPE(int16_t),
          DATA_TYPE(uint16_t),
          DATA_TYPE(int32_t),
          DATA_TYPE(uint32_t),
          DATA_TYPE(int64_t),
          DATA_TYPE(uint64_t),
          DATA_TYPE(float),
          DATA_TYPE(double),
          DATA_TYPE(bool),
          DATA_TYPE(std::string_view),
      };
#undef DATA_TYPE

  auto it = dataTypes.find(dataTypeString);
  if (it == dataTypes.end()) {
    LOG(FATAL) << "Unknown data type: " << dataTypeString;
  }

  std::vector<std::string> values;
  std::string value;
  while (*stream >> value) {
    values.push_back(value);
  }

  it->second(values);
}
