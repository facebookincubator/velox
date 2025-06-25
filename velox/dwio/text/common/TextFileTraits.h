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

#include <string>

#include "velox/dwio/common/Options.h"

namespace facebook::velox::text {

using dwio::common::SerDeOptions;

class TextFileTraits {
 public:
  //// The following constants define the delimiters used by TextFile format.
  /// Each row is separated by 'kNewLine'.
  /// Each column is separated by 'kSOH' within each row.

  TextFileTraits(
      SerDeOptions serDeOptions = SerDeOptions(),
      const std::string& nullString = "\\N",
      char newLine = '\n')
      : serDeOptions_(serDeOptions), kNewLine_(newLine) {
    serDeOptions_.nullString = nullString;
  }

  std::string& getKNullData() {
    return serDeOptions_.nullString;
  }

  char getKNewLine() {
    return kNewLine_;
  }

  char getFieldDelim() {
    return serDeOptions_.separators[0];
  }

  char getKSOH() {
    return getFieldDelim();
  }

  char getCollectionDelim() {
    return serDeOptions_.separators[1];
  }

  char getMapKeyDelim() {
    return serDeOptions_.separators[2];
  }

  SerDeOptions serDeOptions_;

 private:
  char kNewLine_;
};

} // namespace facebook::velox::text
