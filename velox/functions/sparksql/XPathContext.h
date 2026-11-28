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

#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <libxml/xmlmemory.h>
#include <libxml/xpath.h>

namespace facebook::velox::functions::sparksql::xpath {
using CompiledXPathPtr =
    std::unique_ptr<xmlXPathCompExpr, decltype(&xmlXPathFreeCompExpr)>;
using XPathObjectPtr =
    std::unique_ptr<xmlXPathObject, decltype(&xmlXPathFreeObject)>;
using XmlStringPtr = std::unique_ptr<xmlChar, decltype(xmlFree)>;

/// Carries callback failures and cached expression data through libxml2
/// contexts.
struct XPathEvaluationState {
  /// Defers C++ exceptions until control returns from libxml2.
  std::exception_ptr error;
  /// Preserves the first nested selector's diagnostic.
  std::optional<int> selectorErrorCode;
  /// Borrows selectors for the duration of a compiled-expression evaluation.
  const std::vector<CompiledXPathPtr>* namespaceSelectors{nullptr};
  /// Borrows QName prefixes shared by the outer and nested contexts.
  const std::vector<std::string>* prefixes{nullptr};
  /// Borrows parsed numeric literals owned by the compiled expression.
  const std::vector<double>* literals{nullptr};
};

} // namespace facebook::velox::functions::sparksql::xpath
