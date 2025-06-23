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

// Adapted from Apache DataSketches

#pragma once

#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "MemoryOperations.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common::theta {

/// Interface for serializing and deserializing items
template <typename T, typename Enable = void>
struct serde {
  /**
   * Stream serialization
   * @param os output stream
   * @param items pointer to array of items
   * @param num number of items
   */
  void serialize(std::ostream& os, const T* items, unsigned num) const;

  /**
   * Stream deserialization
   * @param is input stream
   * @param items pointer to array of items (items in the array are allocated
   * but not initialized)
   * @param num number of items
   */
  void deserialize(std::istream& is, T* items, unsigned num) const;

  /**
   * Raw bytes serialization
   * @param ptr pointer to output buffer
   * @param capacity size of the buffer in bytes
   * @param items pointer to array of items
   * @param num number of items
   */
  size_t serialize(void* ptr, size_t capacity, const T* items, unsigned num)
      const;

  /**
   * Raw bytes deserialization
   * @param ptr pointer to input buffer
   * @param capacity size of the buffer in bytes
   * @param items pointer to array of items (items in the array are allocated
   * but not initialized)
   * @param num number of items
   */
  size_t deserialize(const void* ptr, size_t capacity, T* items, unsigned num)
      const;

  /**
   * Size of the given item
   * @param item to be sized
   * @return size of the given item in bytes
   */
  size_t size_of_item(const T& item) const;
};

/// serde for all fixed-size arithmetic types (int and float of different
/// sizes). in particular, kll_sketch<int64_t> should produce sketches
/// binary-compatible with LongsSketch and ItemsSketch<Long> with
/// ArrayOfLongsSerDe in Java
template <typename T>
struct serde<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  /// @copydoc serde::serialize
  void serialize(std::ostream& os, const T* items, unsigned num) const {
    bool failure = false;
    try {
      os.write(reinterpret_cast<const char*>(items), sizeof(T) * num);
    } catch (std::ostream::failure&) {
      failure = true;
    }
    if (failure || !os.good()) {
      throw VeloxRuntimeError(
          __FILE__,
          __LINE__,
          __FUNCTION__,
          "",
          "error writing to std::ostream with " + std::to_string(num) +
              " items",
          error_source::kErrorSourceUser,
          error_code::kGenericUserError,
          false /*retriable*/);
    }
  }

  void deserialize(std::istream& is, T* items, unsigned num) const {
    bool failure = false;
    try {
      is.read((char*)items, sizeof(T) * num);
    } catch (std::istream::failure&) {
      failure = true;
    }
    if (failure || !is.good()) {
      throw VeloxRuntimeError(
          __FILE__,
          __LINE__,
          __FUNCTION__,
          "",
          "error reading from std::istream with " + std::to_string(num) +
              " items",
          error_source::kErrorSourceUser,
          error_code::kGenericUserError,
          false /*retriable*/);
    }
  }

  /// @copydoc serde::serialize(void*,size_t,const T*,unsigned) const
  size_t serialize(void* ptr, size_t capacity, const T* items, unsigned num)
      const {
    const size_t bytes_written = sizeof(T) * num;
    checkMemorySize(bytes_written, capacity);
    memcpy(ptr, items, bytes_written);
    return bytes_written;
  }

  /// @copydoc serde::deserialize(const void*,size_t,T*,unsigned) const
  size_t deserialize(const void* ptr, size_t capacity, T* items, unsigned num)
      const {
    const size_t bytes_read = sizeof(T) * num;
    checkMemorySize(bytes_read, capacity);
    memcpy(items, ptr, bytes_read);
    return bytes_read;
  }

  /// @copydoc serde::size_of_item
  size_t size_of_item(const T& item) const {
    unused(item);
    return sizeof(T);
  }
};

/// serde for std::string items.
/// This should produce sketches binary-compatible with
/// ItemsSketch<String> with ArrayOfStringsSerDe in Java.
/// The length of each string is stored as a 32-bit integer (historically),
/// which may be too wasteful. Treat this as an example.
template <>
struct serde<std::string> {
  /// @copydoc serde::serialize
  void serialize(std::ostream& os, const std::string* items, unsigned num)
      const {
    unsigned i = 0;
    bool failure = false;
    try {
      for (; i < num && os.good(); i++) {
        uint32_t length = static_cast<uint32_t>(items[i].size());
        os.write((char*)&length, sizeof(length));
        os.write(items[i].c_str(), length);
      }
    } catch (std::ostream::failure&) {
      failure = true;
    }
    if (failure || !os.good()) {
      throw VeloxRuntimeError(
          __FILE__,
          __LINE__,
          __FUNCTION__,
          "",
          "error writing to std::ostream at item " + std::to_string(i),
          error_source::kErrorSourceUser,
          error_code::kGenericUserError,
          false /*retriable*/);
    }
  }

  /// @copydoc serde::deserialize
  void deserialize(std::istream& is, std::string* items, unsigned num) const {
    unsigned i = 0;
    bool failure = false;
    try {
      for (; i < num; i++) {
        uint32_t length;
        is.read((char*)&length, sizeof(length));
        if (!is.good()) {
          break;
        }
        std::string str;
        str.reserve(length);
        for (uint32_t j = 0; j < length; j++) {
          str.push_back(static_cast<char>(is.get()));
        }
        if (!is.good()) {
          break;
        }
        new (&items[i]) std::string(std::move(str));
      }
    } catch (std::istream::failure&) {
      failure = true;
    }
    if (failure || !is.good()) {
      // clean up what we've already allocated
      for (unsigned j = 0; j < i; ++j) {
        items[j].~basic_string();
      }
      throw VeloxRuntimeError(
          __FILE__,
          __LINE__,
          __FUNCTION__,
          "",
          "error reading from std::istream at item " + std::to_string(i),
          error_source::kErrorSourceUser,
          error_code::kGenericUserError,
          false /*retriable*/);
    }
  }

  /// @copydoc serde::serialize(void*,size_t,const T*,unsigned) const
  size_t serialize(
      void* ptr,
      size_t capacity,
      const std::string* items,
      unsigned num) const {
    size_t bytes_written = 0;
    for (unsigned i = 0; i < num; ++i) {
      const uint32_t length = static_cast<uint32_t>(items[i].size());
      const size_t new_bytes = length + sizeof(length);
      checkMemorySize(bytes_written + new_bytes, capacity);
      memcpy(ptr, &length, sizeof(length));
      ptr = static_cast<char*>(ptr) + sizeof(uint32_t);
      memcpy(ptr, items[i].c_str(), length);
      ptr = static_cast<char*>(ptr) + length;
      bytes_written += new_bytes;
    }
    return bytes_written;
  }

  /// @copydoc serde::deserialize(const void*,size_t,T*,unsigned) const
  size_t deserialize(
      const void* ptr,
      size_t capacity,
      std::string* items,
      unsigned num) const {
    size_t bytes_read = 0;
    unsigned i = 0;
    bool failure = false;
    for (; i < num && !failure; ++i) {
      uint32_t length;
      if (bytes_read + sizeof(length) > capacity) {
        bytes_read += sizeof(length); // we'll use this to report the error
        failure = true;
        break;
      }
      memcpy(&length, ptr, sizeof(length));
      ptr = static_cast<const char*>(ptr) + sizeof(uint32_t);
      bytes_read += sizeof(length);

      if (bytes_read + length > capacity) {
        bytes_read += length; // we'll use this to report the error
        failure = true;
        break;
      }
      new (&items[i]) std::string(static_cast<const char*>(ptr), length);
      ptr = static_cast<const char*>(ptr) + length;
      bytes_read += length;
    }

    if (failure) {
      // clean up what we've already allocated
      for (unsigned j = 0; j < i; ++j)
        items[j].~basic_string();
      // using this for a consistent error message
      checkMemorySize(bytes_read, capacity);
    }

    return bytes_read;
  }

  /// @copydoc serde::size_of_item
  size_t size_of_item(const std::string& item) const {
    return sizeof(uint32_t) + item.size();
  }
};

} // namespace facebook::velox::common::theta
