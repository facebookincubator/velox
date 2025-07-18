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

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>

namespace facebook::velox::common::theta {

static const uint64_t DEFAULT_SEED = 9001;

enum resizeFactor { X1 = 0, X2, X4, X8 };

template <typename A>
using string = std::basic_string<
    char,
    std::char_traits<char>,
    typename std::allocator_traits<A>::template rebind_alloc<char>>;

// common random declarations
namespace randomUtils {
static std::random_device rd; // possibly unsafe in MinGW with GCC < 9.2
static thread_local std::mt19937_64 rand(rd());
static thread_local std::uniform_real_distribution<> next_double(0.0, 1.0);
static thread_local std::uniform_int_distribution<uint64_t> next_uint64(
    0,
    UINT64_MAX);

// thread-safe random bit
static thread_local std::independent_bits_engine<std::mt19937, 1, uint32_t>
    random_bit(static_cast<uint32_t>(
        std::chrono::system_clock::now().time_since_epoch().count() +
        std::hash<std::thread::id>{}(std::this_thread::get_id())));

inline void overrideSeed(uint64_t s) {
  rand.seed(s);
}
} // namespace randomUtils

// utility function to hide unused compiler warning
// usually has no additional cost
template <typename T>
void unused(T&&...) {}

// common helping functions
// TODO: find a better place for them

constexpr uint8_t log2(uint32_t n) {
  return (n > 1) ? 1 + log2(n >> 1) : 0;
}

constexpr uint8_t lgSizeFromCount(uint32_t n, double load_factor) {
  return log2(n) +
      ((n > static_cast<uint32_t>((1 << (log2(n) + 1)) * load_factor)) ? 2 : 1);
}

// stream helpers to hide casts
template <typename T>
static inline T read(std::istream& is) {
  T value;
  is.read(reinterpret_cast<char*>(&value), sizeof(T));
  return value;
}

template <typename T>
static inline void read(std::istream& is, T* ptr, size_t size_bytes) {
  is.read(reinterpret_cast<char*>(ptr), size_bytes);
}

template <typename T>
static inline void write(std::ostream& os, T value) {
  os.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
static inline void write(std::ostream& os, const T* ptr, size_t size_bytes) {
  os.write(reinterpret_cast<const char*>(ptr), size_bytes);
}

template <typename T>
T byteswap(T value) {
  char* ptr = static_cast<char*>(static_cast<void*>(&value));
  const int len = sizeof(T);
  for (size_t i = 0; i < len / 2; ++i) {
    std::swap(ptr[i], ptr[len - i - 1]);
  }
  return value;
}

template <typename T>
static inline T readBigEndian(std::istream& is) {
  T value;
  is.read(reinterpret_cast<char*>(&value), sizeof(T));
  return byteswap(value);
}

// wrapper for iterators to implement operator-> returning temporary value
template <typename T>
class returnValueHolder {
 public:
  returnValueHolder(T value) : value_(value) {}
  const T* operator->() const {
    return std::addressof(value_);
  }

 private:
  T value_;
};

} // namespace facebook::velox::common::theta
