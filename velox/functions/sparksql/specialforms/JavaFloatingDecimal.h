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

namespace facebook::velox::functions::sparksql {

/// Appends 'value' to 'output' the way Java's Double.toString renders it,
/// which is what Spark's JSON writer emits when a JSON number reaches a
/// string-typed field.
///
/// This is a port of jdk.internal.math.FloatingDecimal as shipped in JDK 11
/// and 17. That algorithm is not shortest-round-trip, so std::to_chars, fmt
/// and Ryu all disagree with it on four families of values: large integral
/// doubles, exact powers of two, decimals that sit exactly halfway between two
/// doubles, and the binade starting at 2^84, where the JDK's own 64-bit
/// overflow changes the digits it prints. JDK 19 replaced the algorithm with
/// Schubfach, so a JVM upgrade past 17 changes the expected output here.
///
/// 'value' must be finite; the caller renders NaN and infinity.
void appendJavaDouble(double value, std::string& output);

} // namespace facebook::velox::functions::sparksql
