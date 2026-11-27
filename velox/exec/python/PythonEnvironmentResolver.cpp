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
#include "velox/exec/python/PythonEnvironmentResolver.h"

#include <unistd.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec::python {
namespace {

constexpr std::string_view kSchemeSeparator{"://"};

} // namespace

PythonInterpreter FileEnvironmentResolver::resolveUri(
    std::string_view uri) const {
  std::string path{
      uri.substr(uri.find(kSchemeSeparator) + kSchemeSeparator.size())};
  VELOX_USER_CHECK_EQ(
      ::access(path.c_str(), X_OK),
      0,
      "Python interpreter must be an executable file: {}",
      path);
  return PythonInterpreter{.path = std::move(path)};
}

PythonInterpreter PythonEnvironmentResolvers::resolve(
    std::string_view uri) const {
  const auto separator = uri.find(kSchemeSeparator);
  VELOX_USER_CHECK_NE(
      separator,
      std::string_view::npos,
      "Python environment must be a URI with a scheme: {}",
      uri);
  const std::string scheme{uri.substr(0, separator)};
  const auto resolver = find(scheme);
  VELOX_USER_CHECK_NOT_NULL(
      resolver, "No resolver for Python environment: {}", scheme);
  return resolver->resolveUri(uri);
}

} // namespace facebook::velox::exec::python
