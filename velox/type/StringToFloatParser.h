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

#include "velox/common/base/Status.h"

namespace facebook::velox {

class StringToFloatParser {
 public:
  /// Converts a string to a double/float precision floating point value.
  /// Returns UserError/Invalid status if the data is invalid.
  ///
  /// Leading and trailing whitespace characters in str are ignored.
  /// Whitespace is removed as if by the java's String#trim method.
  /// that is, both ASCII space and control characters are removed.
  ///
  /// The supported string formats is:
  /// ([\\x00-\\x20]*                 - Optional leading whitespace.
  /// [+-]?(                          - Optional sign character.
  /// NaN|"                           - "NaN" string, ignoring case.
  /// Infinity|                       - "Infinity" string, ignoring case.
  /// (((Digits(\\.)?(Digits?)(Exp)?)|
  /// (\\.(Digits)(Exp)?)))
  /// [\\x00-\\x20]*)                 - Optional trailing whitespace.
  ///
  /// @tparam T Either float or double.
  /// @param str A string to convert.
  /// @param out The double/float precision value to be output.
  template <typename T>
  static Status parse(const std::string_view& str, T& out);
};

} // namespace facebook::velox