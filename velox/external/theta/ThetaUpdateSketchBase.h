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

#include <cmath>
#include <iterator>

#include "MurmurHash3.h"
#include "ThetaComparators.h"
#include "ThetaConstants.h"

namespace facebook::velox::common::theta {

template <typename Entry, typename ExtractKey, typename Allocator>
struct ThetaUpdateSketchBase {
  using resizeFactor = ThetaConstants::resizeFactor;
  using comparator = compareByKey<ExtractKey>;

  ThetaUpdateSketchBase(
      uint8_t lg_cur_size,
      uint8_t lg_nom_size,
      resizeFactor rf,
      float p,
      uint64_t theta,
      uint64_t seed,
      const Allocator& allocator,
      bool is_empty = true);
  ThetaUpdateSketchBase(const ThetaUpdateSketchBase& other);
  ThetaUpdateSketchBase(ThetaUpdateSketchBase&& other) noexcept;
  ~ThetaUpdateSketchBase();
  ThetaUpdateSketchBase& operator=(const ThetaUpdateSketchBase& other);
  ThetaUpdateSketchBase& operator=(ThetaUpdateSketchBase&& other);

  using iterator = Entry*;

  inline uint64_t hashAndScreen(const void* data, size_t length);

  inline std::pair<iterator, bool> find(uint64_t key) const;
  static inline std::pair<iterator, bool>
  find(Entry* entries, uint8_t lg_size, uint64_t key);

  template <typename FwdEntry>
  inline void insert(iterator it, FwdEntry&& entry);

  iterator begin() const;
  iterator end() const;

  // resize threshold = 0.5 tuned for speed
  static constexpr double RESIZE_THRESHOLD = 0.5;
  // hash table rebuild threshold = 15/16
  static constexpr double REBUILD_THRESHOLD = 15.0 / 16.0;

  static constexpr uint8_t STRIDE_HASH_BITS = 7;
  static constexpr uint32_t STRIDE_MASK = (1 << STRIDE_HASH_BITS) - 1;

  Allocator allocator_;
  bool isEmpty_;
  uint8_t lgCurSize_;
  uint8_t lgNomSize_;
  resizeFactor rf_;
  float p_;
  uint32_t numEntries_;
  uint64_t theta_;
  uint64_t seed_;
  Entry* entries_;

  void resize();
  void rebuild();
  void trim();
  void reset();

  static inline uint32_t getCapacity(uint8_t lg_cur_size, uint8_t lg_nom_size);
  static inline uint32_t getStride(uint64_t key, uint8_t lg_size);
  static void consolidateNonEmpty(Entry* entries, size_t size, size_t num);
};

/// Theta base builder
template <typename Derived, typename Allocator>
class ThetaBaseBuilder {
 public:
  /**
   * Creates and instance of the builder with default parameters.
   * @param allocator instance of an Allocator to pass to created sketches
   */
  ThetaBaseBuilder(const Allocator& allocator);

  /**
   * Set log2(k), where k is a nominal number of entries in the sketch
   * @param lg_k base 2 logarithm of nominal number of entries
   * @return this builder
   */
  Derived& set_lg_k(uint8_t lg_k);

  /**
   * Set resize factor for the internal hash table (defaults to 8)
   * @param rf resize factor
   * @return this builder
   */
  Derived& setResizeFactor(resizeFactor rf);

  /**
   * Set sampling probability (initial theta). The default is 1, so the sketch
   * retains all entries until it reaches the limit, at which point it goes into
   * the estimation mode and reduces the effective sampling probability (theta)
   * as necessary.
   * @param p sampling probability
   * @return this builder
   */
  Derived& setP(float p);

  /**
   * Set the seed for the hash function. Should be used carefully if needed.
   * Sketches produced with different seed are not compatible
   * and cannot be mixed in set operations.
   * @param seed hash seed
   * @return this builder
   */
  Derived& setSeed(uint64_t seed);

 protected:
  Allocator allocator_;
  uint8_t lg_k_;
  resizeFactor rf_;
  float p_;
  uint64_t seed_;

  uint64_t startingTheta() const;
  uint8_t startingLgSize() const;
};

// key extractor

struct trivialExtractKey {
  template <typename T>
  auto operator()(T&& entry) const -> decltype(std::forward<T>(entry)) {
    return std::forward<T>(entry);
  }
};

// key not zero

template <typename Entry, typename ExtractKey>
class keyNotZero {
 public:
  bool operator()(const Entry& entry) const {
    return ExtractKey()(entry) != 0;
  }
};

template <typename Key, typename Entry, typename ExtractKey>
class keyNotZeroLessThan {
 public:
  explicit keyNotZeroLessThan(const Key& key) : key(key) {}
  bool operator()(const Entry& entry) const {
    return ExtractKey()(entry) != 0 && ExtractKey()(entry) < this->key;
  }

 private:
  Key key;
};

// MurMur3 hash functions

static inline uint64_t
computeHash(const void* data, size_t length, uint64_t seed) {
  HashState hashes;
  MurmurHash3_x64_128(data, length, seed, hashes);
  return (
      hashes.h1 >>
      1); // Java implementation does unsigned shift >>> to make values positive
}

// iterators

template <typename Entry, typename ExtractKey>
class ThetaIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = Entry;
  using difference_type = std::ptrdiff_t;
  using pointer = Entry*;
  using reference = Entry&;

  ThetaIterator(Entry* entries, uint32_t size, uint32_t index);
  ThetaIterator& operator++();
  ThetaIterator operator++(int);
  bool operator==(const ThetaIterator& other) const;
  bool operator!=(const ThetaIterator& other) const;
  reference operator*() const;
  pointer operator->() const;

 private:
  Entry* entries_;
  uint32_t size_;
  uint32_t index_;
};

template <typename Entry, typename ExtractKey>
class ThetaConstIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = const Entry;
  using difference_type = std::ptrdiff_t;
  using pointer = const Entry*;
  using reference = const Entry&;

  ThetaConstIterator(const Entry* entries, uint32_t size, uint32_t index);
  ThetaConstIterator& operator++();
  ThetaConstIterator operator++(int);
  bool operator==(const ThetaConstIterator& other) const;
  bool operator!=(const ThetaConstIterator& other) const;
  reference operator*() const;
  pointer operator->() const;

 private:
  const Entry* entries_;
  uint32_t size_;
  uint32_t index_;
};

// double value canonicalization for compatibility with Java
static inline int64_t canonical_double(double value) {
  union {
    int64_t long_value;
    double double_value;
  } long_double_union;

  if (value == 0.0) {
    long_double_union.double_value = 0.0; // canonicalize -0.0 to 0.0
  } else if (std::isnan(value)) {
    long_double_union.long_value =
        0x7ff8000000000000L; // canonicalize NaN using value from Java's
                             // Double.doubleToLongBits()
  } else {
    long_double_union.double_value = value;
  }
  return long_double_union.long_value;
}

} // namespace facebook::velox::common::theta

#include "ThetaUpdateSketchBase.cpp"
