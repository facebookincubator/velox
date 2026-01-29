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

/// Registers the array_top_n function with a transform lambda.
/// Signature: array_top_n(array(T), integer, function(T, U)) -> array(T)
/// Returns the top n elements of the array sorted in descending order
/// according to values computed by the transform function.
/// The transform function computes a sorting key for each element.
/// Elements are sorted by keys in descending order, with nulls at the end.
void registerArrayTopNComparatorFunction(const std::string& prefix);

} // namespace facebook::velox::functions
