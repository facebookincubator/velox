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

struct SparkRegistrationConfig {
  /// When true, establishing the result type of an arithmetic operation
  /// according to Hive behavior and SQL ANSI 2011 specification, i.e.
  /// rounding the decimal part of the result if an exact representation is not
  /// possible. Otherwise, NULL is returned when the actual result cannot be
  /// represented with the calculated decimal type. Now we support add,
  /// subtract, multiply and divide operations.
  bool allowPrecisionLoss = true;
};
