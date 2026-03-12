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

#include "velox/serializers/CompactRowSerializer.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/serializers/UnsafeRowSerializer.h"

namespace facebook::velox {

/// Registers all built-in named vector serdes (Presto, CompactRow, UnsafeRow).
/// Safe to call multiple times; skips already-registered serdes.
inline void registerAllNamedVectorSerdes() {
  if (!isRegisteredNamedVectorSerde(
          serializer::presto::PrestoVectorSerde::name())) {
    serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde(
          serializer::CompactRowVectorSerde::name())) {
    serializer::CompactRowVectorSerde::registerNamedVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde(
          serializer::spark::UnsafeRowVectorSerde::name())) {
    serializer::spark::UnsafeRowVectorSerde::registerNamedVectorSerde();
  }
}

} // namespace facebook::velox
