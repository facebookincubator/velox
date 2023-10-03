// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#pragma once

#include "duckdb.hpp"
#include <exception>

namespace duckdb {

class ByteBuffer { // on to the 10 thousandth impl
 public:
  ByteBuffer() {};
  ByteBuffer(char *ptr, uint64_t len) : ptr(ptr), len(len) {};

  char *ptr = nullptr;
  uint64_t len = 0;

 public:
  void inc(uint64_t increment) {
    available(increment);
    len -= increment;
    ptr += increment;
  }

  template <class T>
  T read() {
    T val = get<T>();
    inc(sizeof(T));
    return val;
  }

  template <class T>
  T get() {
    available(sizeof(T));
    T val = Load<T>((data_ptr_t)ptr);
    return val;
  }

  void copy_to(char *dest, uint64_t len) {
    available(len);
    std::memcpy(dest, ptr, len);
  }

  void zero() {
    std::memset(ptr, 0, len);
  }

  void available(uint64_t req_len) {
    if (req_len > len) {
      throw std::runtime_error("Out of buffer");
    }
  }
};

} // namespace duckdb
