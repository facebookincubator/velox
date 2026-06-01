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
// org.pcre4j.regex.translate.ClassRenderer (Java) under Apache-2.0 by the
// same author for inclusion in Velox.
//
#pragma once

#include "velox/functions/lib/java_pcre2_translator/ClassNode.h"

#include <cstdint>
#include <string>

namespace facebook::velox::functions::java_pcre2_translator {

class ClassRenderer {
 public:
  struct RenderResult {
    std::string text;
    bool intersectionUnresolved{false};
  };

  static std::string render(const ClassNode& node);
  static RenderResult renderWithSignal(const ClassNode& node);
  static void emitLiteralInClass(std::int32_t cp, std::string& sb);
  static bool containsIntersection(const ClassNode& node);

 private:
  ClassRenderer() = delete;
};

} // namespace facebook::velox::functions::java_pcre2_translator
