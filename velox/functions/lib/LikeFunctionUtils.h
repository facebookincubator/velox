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
#include <memory>
#include <vector>

#include "velox/expression/VectorFunction.h"
#include "velox/functions/Udf.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::functions {
namespace likeutils {

bool checkWildcardInPattern(const std::string& pattern);
bool matchExactPattern(
    const std::string& inputString,
    const std::string& pattern);

bool checkPrefixPattern(const std::string& pattern);
bool matchPrefixPattern(
    const std::string& inputString,
    const std::string& pattern);

bool checkSuffixPattern(const std::string& pattern);
bool matchSuffixPattern(
    const std::string& inputString,
    const std::string& pattern);

void kmpSearchPreprocessPattern(
    const std::string& pattern,
    std::vector<int>& lps);

} // namespace likeutils
} // namespace facebook::velox::functions
