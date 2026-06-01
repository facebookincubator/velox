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
//
// Originally authored by Oleksii PELYKH for pcre4j; ported from
// org.pcre4j.regex.translate.JdkPropertyExpander (Java) under Apache-2.0 by
// the same author for inclusion in Velox.
//
#pragma once

#include "velox/functions/lib/java_pcre2_translator/RangeSet.h"

#include <optional>
#include <string_view>

namespace facebook::velox::functions::java_pcre2_translator {

class JdkPropertyExpander {
 public:
  static std::optional<RangeSet> expand(std::string_view pcre2Token);

 private:
  JdkPropertyExpander() = delete;
};

} // namespace facebook::velox::functions::java_pcre2_translator
