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

namespace facebook::velox::functions {

/// Registers the array_top_n function with a comparator lambda.
/// Signature: array_top_n(array(T), integer, function(T, T, bigint)) ->
/// array(T) Returns the top n elements of the array sorted in descending order
/// according to the comparator function.
void registerArrayTopNComparatorFunction(const std::string& prefix);

} // namespace facebook::velox::functions
