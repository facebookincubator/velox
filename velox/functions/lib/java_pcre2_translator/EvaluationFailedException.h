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
// org.pcre4j.regex.translate.EvaluationFailedException (Java) under
// Apache-2.0 by the same author for inclusion in Velox.
//
#pragma once

#include <stdexcept>
#include <string>

namespace facebook::velox::functions::java_pcre2_translator {

/// Thrown by the translator pipeline when a Java regex feature cannot be
/// represented in the target engine's syntax (e.g. when the target is
/// asked to express something it has no equivalent for, like an
/// unsupported character-class intersection).
class EvaluationFailedException : public std::runtime_error {
 public:
  explicit EvaluationFailedException(const std::string& msg)
      : std::runtime_error(msg) {}
};

} // namespace facebook::velox::functions::java_pcre2_translator
