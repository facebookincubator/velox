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

#include <cstdint>
#include <string_view>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"

namespace facebook::nimble {

#define NIMBLE_RETURN_BY_DATA_TYPE_OR(dataType, Type, expression, ...) \
  switch (dataType) {                                                  \
    case ::facebook::nimble::DataType::Int8: {                         \
      using Type = int8_t;                                             \
      return (expression);                                             \
    }                                                                  \
    case ::facebook::nimble::DataType::Uint8: {                        \
      using Type = uint8_t;                                            \
      return (expression);                                             \
    }                                                                  \
    case ::facebook::nimble::DataType::Int16: {                        \
      using Type = int16_t;                                            \
      return (expression);                                             \
    }                                                                  \
    case ::facebook::nimble::DataType::Uint16: {                       \
      using Type = uint16_t;                                           \
      return (expression);                                             \
    }                                                                  \
    case ::facebook::nimble::DataType::Int32: {                        \
      using Type = int32_t;                                            \
      return (expression);                                             \
    }                                                                  \
    case ::facebook::nimble::DataType::Uint32: {                       \
      using Type = uint32_t;                                           \
      return (expression);                                             \
    }                                                                  \
    case ::facebook::nimble::DataType::Int64: {                        \
      using Type = int64_t;                                            \
      return (expression);                                             \
    }                                                                  \
    case ::facebook::nimble::DataType::Uint64: {                       \
      using Type = uint64_t;                                           \
      return (expression);                                             \
    }                                                                  \
    case ::facebook::nimble::DataType::Float: {                        \
      using Type = float;                                              \
      return (expression);                                             \
    }                                                                  \
    case ::facebook::nimble::DataType::Double: {                       \
      using Type = double;                                             \
      return (expression);                                             \
    }                                                                  \
    case ::facebook::nimble::DataType::Bool: {                         \
      using Type = bool;                                               \
      return (expression);                                             \
    }                                                                  \
    case ::facebook::nimble::DataType::String: {                       \
      using Type = std::string_view;                                   \
      return (expression);                                             \
    }                                                                  \
    default: {                                                         \
      __VA_ARGS__;                                                     \
    }                                                                  \
  }

#define NIMBLE_RETURN_BY_DATA_TYPE(dataType, Type, expression) \
  NIMBLE_RETURN_BY_DATA_TYPE_OR(                               \
      dataType,                                                \
      Type,                                                    \
      expression,                                              \
      NIMBLE_UNREACHABLE("Unsupported data type {}.", dataType))

#define NIMBLE_TRY_RETURN_BY_DATA_TYPE(dataType, Type, expression) \
  NIMBLE_RETURN_BY_DATA_TYPE_OR(dataType, Type, expression, return {})

#define NIMBLE_RETURN_BY_VARINT_DATA_TYPE_OR(dataType, Type, expression, ...) \
  switch (dataType) {                                                         \
    case ::facebook::nimble::DataType::Int32: {                               \
      using Type = int32_t;                                                   \
      return (expression);                                                    \
    }                                                                         \
    case ::facebook::nimble::DataType::Uint32: {                              \
      using Type = uint32_t;                                                  \
      return (expression);                                                    \
    }                                                                         \
    case ::facebook::nimble::DataType::Int64: {                               \
      using Type = int64_t;                                                   \
      return (expression);                                                    \
    }                                                                         \
    case ::facebook::nimble::DataType::Uint64: {                              \
      using Type = uint64_t;                                                  \
      return (expression);                                                    \
    }                                                                         \
    case ::facebook::nimble::DataType::Float: {                               \
      using Type = float;                                                     \
      return (expression);                                                    \
    }                                                                         \
    case ::facebook::nimble::DataType::Double: {                              \
      using Type = double;                                                    \
      return (expression);                                                    \
    }                                                                         \
    default: {                                                                \
      __VA_ARGS__;                                                            \
    }                                                                         \
  }

#define NIMBLE_RETURN_BY_VARINT_DATA_TYPE(dataType, Type, expression) \
  NIMBLE_RETURN_BY_VARINT_DATA_TYPE_OR(                               \
      dataType,                                                       \
      Type,                                                           \
      expression,                                                     \
      NIMBLE_UNREACHABLE("Unsupported varint data type {}.", dataType))

#define NIMBLE_TRY_RETURN_BY_VARINT_DATA_TYPE(dataType, Type, expression) \
  NIMBLE_RETURN_BY_VARINT_DATA_TYPE_OR(dataType, Type, expression, return {})

#define NIMBLE_RETURN_BY_NON_BOOL_DATA_TYPE_OR(  \
    dataType, Type, expression, ...)             \
  switch (dataType) {                            \
    case ::facebook::nimble::DataType::Int8: {   \
      using Type = int8_t;                       \
      return (expression);                       \
    }                                            \
    case ::facebook::nimble::DataType::Uint8: {  \
      using Type = uint8_t;                      \
      return (expression);                       \
    }                                            \
    case ::facebook::nimble::DataType::Int16: {  \
      using Type = int16_t;                      \
      return (expression);                       \
    }                                            \
    case ::facebook::nimble::DataType::Uint16: { \
      using Type = uint16_t;                     \
      return (expression);                       \
    }                                            \
    case ::facebook::nimble::DataType::Int32: {  \
      using Type = int32_t;                      \
      return (expression);                       \
    }                                            \
    case ::facebook::nimble::DataType::Uint32: { \
      using Type = uint32_t;                     \
      return (expression);                       \
    }                                            \
    case ::facebook::nimble::DataType::Int64: {  \
      using Type = int64_t;                      \
      return (expression);                       \
    }                                            \
    case ::facebook::nimble::DataType::Uint64: { \
      using Type = uint64_t;                     \
      return (expression);                       \
    }                                            \
    case ::facebook::nimble::DataType::Float: {  \
      using Type = float;                        \
      return (expression);                       \
    }                                            \
    case ::facebook::nimble::DataType::Double: { \
      using Type = double;                       \
      return (expression);                       \
    }                                            \
    case ::facebook::nimble::DataType::String: { \
      using Type = std::string_view;             \
      return (expression);                       \
    }                                            \
    default: {                                   \
      __VA_ARGS__;                               \
    }                                            \
  }

#define NIMBLE_RETURN_BY_NON_BOOL_DATA_TYPE(dataType, Type, expression) \
  NIMBLE_RETURN_BY_NON_BOOL_DATA_TYPE_OR(                               \
      dataType,                                                         \
      Type,                                                             \
      expression,                                                       \
      NIMBLE_UNREACHABLE("Unsupported non-bool data type {}.", dataType))

#define NIMBLE_TRY_RETURN_BY_NON_BOOL_DATA_TYPE(dataType, Type, expression) \
  NIMBLE_RETURN_BY_NON_BOOL_DATA_TYPE_OR(dataType, Type, expression, return {})

#define NIMBLE_RETURN_BY_NUMERIC_DATA_TYPE_OR(dataType, Type, expression, ...) \
  switch (dataType) {                                                          \
    case ::facebook::nimble::DataType::Int8: {                                 \
      using Type = int8_t;                                                     \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Uint8: {                                \
      using Type = uint8_t;                                                    \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Int16: {                                \
      using Type = int16_t;                                                    \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Uint16: {                               \
      using Type = uint16_t;                                                   \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Int32: {                                \
      using Type = int32_t;                                                    \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Uint32: {                               \
      using Type = uint32_t;                                                   \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Int64: {                                \
      using Type = int64_t;                                                    \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Uint64: {                               \
      using Type = uint64_t;                                                   \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Float: {                                \
      using Type = float;                                                      \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Double: {                               \
      using Type = double;                                                     \
      return (expression);                                                     \
    }                                                                          \
    default: {                                                                 \
      __VA_ARGS__;                                                             \
    }                                                                          \
  }

#define NIMBLE_RETURN_BY_NUMERIC_DATA_TYPE(dataType, Type, expression) \
  NIMBLE_RETURN_BY_NUMERIC_DATA_TYPE_OR(                               \
      dataType,                                                        \
      Type,                                                            \
      expression,                                                      \
      NIMBLE_UNREACHABLE("Unsupported numeric data type {}.", dataType))

#define NIMBLE_TRY_RETURN_BY_NUMERIC_DATA_TYPE(dataType, Type, expression) \
  NIMBLE_RETURN_BY_NUMERIC_DATA_TYPE_OR(dataType, Type, expression, return {})

#define NIMBLE_RETURN_BY_FLOATING_POINT_DATA_TYPE_OR( \
    dataType, Type, expression, ...)                  \
  switch (dataType) {                                 \
    case ::facebook::nimble::DataType::Float: {       \
      using Type = float;                             \
      return (expression);                            \
    }                                                 \
    case ::facebook::nimble::DataType::Double: {      \
      using Type = double;                            \
      return (expression);                            \
    }                                                 \
    default: {                                        \
      __VA_ARGS__;                                    \
    }                                                 \
  }

#define NIMBLE_RETURN_BY_FLOATING_POINT_DATA_TYPE(dataType, Type, expression) \
  NIMBLE_RETURN_BY_FLOATING_POINT_DATA_TYPE_OR(                               \
      dataType,                                                               \
      Type,                                                                   \
      expression,                                                             \
      NIMBLE_UNREACHABLE(                                                     \
          "Unsupported floating point data type {}.", dataType))

#define NIMBLE_TRY_RETURN_BY_FLOATING_POINT_DATA_TYPE( \
    dataType, Type, expression)                        \
  NIMBLE_RETURN_BY_FLOATING_POINT_DATA_TYPE_OR(        \
      dataType, Type, expression, return {})

#define NIMBLE_RETURN_BY_INTEGER_DATA_TYPE_OR(dataType, Type, expression, ...) \
  switch (dataType) {                                                          \
    case ::facebook::nimble::DataType::Int8: {                                 \
      using Type = int8_t;                                                     \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Uint8: {                                \
      using Type = uint8_t;                                                    \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Int16: {                                \
      using Type = int16_t;                                                    \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Uint16: {                               \
      using Type = uint16_t;                                                   \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Int32: {                                \
      using Type = int32_t;                                                    \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Uint32: {                               \
      using Type = uint32_t;                                                   \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Int64: {                                \
      using Type = int64_t;                                                    \
      return (expression);                                                     \
    }                                                                          \
    case ::facebook::nimble::DataType::Uint64: {                               \
      using Type = uint64_t;                                                   \
      return (expression);                                                     \
    }                                                                          \
    default: {                                                                 \
      __VA_ARGS__;                                                             \
    }                                                                          \
  }

#define NIMBLE_RETURN_BY_INTEGER_DATA_TYPE(dataType, Type, expression) \
  NIMBLE_RETURN_BY_INTEGER_DATA_TYPE_OR(                               \
      dataType,                                                        \
      Type,                                                            \
      expression,                                                      \
      NIMBLE_UNREACHABLE("Unsupported integer data type {}.", dataType))

#define NIMBLE_TRY_RETURN_BY_INTEGER_DATA_TYPE(dataType, Type, expression) \
  NIMBLE_RETURN_BY_INTEGER_DATA_TYPE_OR(dataType, Type, expression, return {})

} // namespace facebook::nimble
