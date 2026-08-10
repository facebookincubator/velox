/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <memory>

#include "velox/dwio/nimble/common/DataTypeDispatch.h"

namespace facebook::nimble {

#define RETURN_ENCODING_BY_DATA_TYPE(Encoding, dataType)                       \
  NIMBLE_RETURN_BY_DATA_TYPE_OR(                                               \
      dataType,                                                                \
      T,                                                                       \
      std::make_unique<Encoding<T>>(pool, data, stringBufferFactory, options), \
      NIMBLE_UNREACHABLE("Unknown encoding type {}.", dataType))

#define RETURN_ENCODING_BY_VARINT_TYPE(Encoding, dataType)                     \
  NIMBLE_RETURN_BY_VARINT_DATA_TYPE_OR(                                        \
      dataType,                                                                \
      T,                                                                       \
      std::make_unique<Encoding<T>>(pool, data, stringBufferFactory, options), \
      NIMBLE_UNREACHABLE(                                                      \
          "Trying to deserialize a varint stream for "                         \
          "an incompatible data type {}.",                                     \
          toString(dataType)))

#define RETURN_ENCODING_BY_NON_BOOL_TYPE(Encoding, dataType)                   \
  NIMBLE_RETURN_BY_NON_BOOL_DATA_TYPE_OR(                                      \
      dataType,                                                                \
      T,                                                                       \
      std::make_unique<Encoding<T>>(pool, data, stringBufferFactory, options), \
      NIMBLE_UNREACHABLE(                                                      \
          "Trying to deserialize a non-bool stream for "                       \
          "the bool data type {}.",                                            \
          toString(dataType)))

#define RETURN_ENCODING_BY_NUMERIC_TYPE(Encoding, dataType)                    \
  NIMBLE_RETURN_BY_NUMERIC_DATA_TYPE_OR(                                       \
      dataType,                                                                \
      T,                                                                       \
      std::make_unique<Encoding<T>>(pool, data, stringBufferFactory, options), \
      NIMBLE_UNREACHABLE(                                                      \
          "Trying to deserialize a non-numeric stream for "                    \
          "a numeric data type {}.",                                           \
          toString(dataType)))

#define RETURN_ENCODING_BY_INTEGER_TYPE(Encoding, dataType)                    \
  NIMBLE_RETURN_BY_INTEGER_DATA_TYPE_OR(                                       \
      dataType,                                                                \
      T,                                                                       \
      std::make_unique<Encoding<T>>(pool, data, stringBufferFactory, options), \
      NIMBLE_UNREACHABLE(                                                      \
          "Trying to deserialize an integer stream for "                       \
          "an incompatible data type {}.",                                     \
          toString(dataType)))

} // namespace facebook::nimble
