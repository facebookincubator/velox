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

#ifndef THETA_UPDATE_SKETCH_BASE_CPP
#define THETA_UPDATE_SKETCH_BASE_CPP

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "ThetaHelpers.h"
#include "ThetaUpdateSketchBase.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common::theta {

template <typename EN, typename EK, typename A>
ThetaUpdateSketchBase<EN, EK, A>::ThetaUpdateSketchBase(
    uint8_t lgCurSize,
    uint8_t lgNomSize,
    resizeFactor rf,
    float p,
    uint64_t theta,
    uint64_t seed,
    const A& allocator,
    bool isEmpty)
    : allocator_(allocator),
      isEmpty_(isEmpty),
      lgCurSize_(lgCurSize),
      lgNomSize_(lgNomSize),
      rf_(rf),
      p_(p),
      numEntries_(0),
      theta_(theta),
      seed_(seed),
      entries_(nullptr) {
  if (lgCurSize > 0) {
    const size_t size = 1ULL << lgCurSize;
    entries_ = allocator_.allocate(size);
    for (size_t i = 0; i < size; ++i)
      EK()(entries_[i]) = 0;
  }
}

template <typename EN, typename EK, typename A>
ThetaUpdateSketchBase<EN, EK, A>::ThetaUpdateSketchBase(
    const ThetaUpdateSketchBase& other)
    : allocator_(other.allocator_),
      isEmpty_(other.isEmpty_),
      lgCurSize_(other.lgCurSize_),
      lgNomSize_(other.lgNomSize_),
      rf_(other.rf_),
      p_(other.p_),
      numEntries_(other.numEntries_),
      theta_(other.theta_),
      seed_(other.seed_),
      entries_(nullptr) {
  if (other.entries_ != nullptr) {
    const size_t size = 1ULL << lgCurSize_;
    entries_ = allocator_.allocate(size);
    for (size_t i = 0; i < size; ++i) {
      if (EK()(other.entries_[i]) != 0) {
        new (&entries_[i]) EN(other.entries_[i]);
      } else {
        EK()(entries_[i]) = 0;
      }
    }
  }
}

template <typename EN, typename EK, typename A>
ThetaUpdateSketchBase<EN, EK, A>::ThetaUpdateSketchBase(
    ThetaUpdateSketchBase&& other) noexcept
    : allocator_(std::move(other.allocator_)),
      isEmpty_(other.isEmpty_),
      lgCurSize_(other.lgCurSize_),
      lgNomSize_(other.lgNomSize_),
      rf_(other.rf_),
      p_(other.p_),
      numEntries_(other.numEntries_),
      theta_(other.theta_),
      seed_(other.seed_),
      entries_(other.entries_) {
  other.entries_ = nullptr;
}

template <typename EN, typename EK, typename A>
ThetaUpdateSketchBase<EN, EK, A>::~ThetaUpdateSketchBase() {
  if (entries_ != nullptr) {
    const size_t size = 1ULL << lgCurSize_;
    for (size_t i = 0; i < size; ++i) {
      if (EK()(entries_[i]) != 0)
        entries_[i].~EN();
    }
    allocator_.deallocate(entries_, size);
  }
}

template <typename EN, typename EK, typename A>
ThetaUpdateSketchBase<EN, EK, A>& ThetaUpdateSketchBase<EN, EK, A>::operator=(
    const ThetaUpdateSketchBase& other) {
  ThetaUpdateSketchBase<EN, EK, A> copy(other);
  std::swap(allocator_, copy.allocator_);
  std::swap(isEmpty_, copy.isEmpty_);
  std::swap(lgCurSize_, copy.lgCurSize_);
  std::swap(lgNomSize_, copy.lgNomSize_);
  std::swap(rf_, copy.rf_);
  std::swap(p_, copy.p_);
  std::swap(numEntries_, copy.numEntries_);
  std::swap(theta_, copy.theta_);
  std::swap(seed_, copy.seed_);
  std::swap(entries_, copy.entries_);
  return *this;
}

template <typename EN, typename EK, typename A>
ThetaUpdateSketchBase<EN, EK, A>& ThetaUpdateSketchBase<EN, EK, A>::operator=(
    ThetaUpdateSketchBase&& other) {
  std::swap(allocator_, other.allocator_);
  std::swap(isEmpty_, other.isEmpty_);
  std::swap(lgCurSize_, other.lgCurSize_);
  std::swap(lgNomSize_, other.lgNomSize_);
  std::swap(rf_, other.rf_);
  std::swap(p_, other.p_);
  std::swap(numEntries_, other.numEntries_);
  std::swap(theta_, other.theta_);
  std::swap(seed_, other.seed_);
  std::swap(entries_, other.entries_);
  return *this;
}

template <typename EN, typename EK, typename A>
uint64_t ThetaUpdateSketchBase<EN, EK, A>::hashAndScreen(
    const void* data,
    size_t length) {
  isEmpty_ = false;
  const uint64_t hash = computeHash(data, length, seed_);
  if (hash >= theta_)
    return 0; // hash == 0 is reserved to mark empty slots in the table
  return hash;
}

template <typename EN, typename EK, typename A>
auto ThetaUpdateSketchBase<EN, EK, A>::find(uint64_t key) const
    -> std::pair<iterator, bool> {
  return find(entries_, lgCurSize_, key);
}

template <typename EN, typename EK, typename A>
auto ThetaUpdateSketchBase<EN, EK, A>::find(
    EN* entries,
    uint8_t lg_size,
    uint64_t key) -> std::pair<iterator, bool> {
  const uint32_t size = 1 << lg_size;
  const uint32_t mask = size - 1;
  const uint32_t stride = getStride(key, lg_size);
  uint32_t index = static_cast<uint32_t>(key) & mask;
  // search for duplicate or zero
  const uint32_t loop_index = index;
  do {
    const uint64_t probe = EK()(entries[index]);
    if (probe == 0) {
      return std::pair<iterator, bool>(&entries[index], false);
    } else if (probe == key) {
      return std::pair<iterator, bool>(&entries[index], true);
    }
    index = (index + stride) & mask;
  } while (index != loop_index);
  throw VeloxRuntimeError(
      __FILE__,
      __LINE__,
      __FUNCTION__,
      "",
      "key not found and no empty slots!",
      error_source::kErrorSourceRuntime,
      error_code::kInvalidArgument,
      false /*retriable*/);
}

template <typename EN, typename EK, typename A>
template <typename Fwd>
void ThetaUpdateSketchBase<EN, EK, A>::insert(iterator it, Fwd&& entry) {
  new (it) EN(std::forward<Fwd>(entry));
  ++numEntries_;
  if (numEntries_ > getCapacity(lgCurSize_, lgNomSize_)) {
    if (lgCurSize_ <= lgNomSize_) {
      resize();
    } else {
      rebuild();
    }
  }
}

template <typename EN, typename EK, typename A>
auto ThetaUpdateSketchBase<EN, EK, A>::begin() const -> iterator {
  return entries_;
}

template <typename EN, typename EK, typename A>
auto ThetaUpdateSketchBase<EN, EK, A>::end() const -> iterator {
  return entries_ + (1ULL << lgCurSize_);
}

template <typename EN, typename EK, typename A>
uint32_t ThetaUpdateSketchBase<EN, EK, A>::getCapacity(
    uint8_t lg_cur_size,
    uint8_t lg_nom_size) {
  const double fraction =
      (lg_cur_size <= lg_nom_size) ? RESIZE_THRESHOLD : REBUILD_THRESHOLD;
  return static_cast<uint32_t>(std::floor(fraction * (1 << lg_cur_size)));
}

template <typename EN, typename EK, typename A>
uint32_t ThetaUpdateSketchBase<EN, EK, A>::getStride(
    uint64_t key,
    uint8_t lg_size) {
  // odd and independent of index assuming lg_size lowest bits of the key were
  // used for the index
  return (2 * static_cast<uint32_t>((key >> lg_size) & STRIDE_MASK)) + 1;
}

template <typename EN, typename EK, typename A>
void ThetaUpdateSketchBase<EN, EK, A>::resize() {
  const size_t old_size = 1ULL << lgCurSize_;
  const uint8_t lg_new_size =
      std::min<uint8_t>(lgCurSize_ + static_cast<uint8_t>(rf_), lgNomSize_ + 1);
  const size_t new_size = 1ULL << lg_new_size;
  EN* new_entries = allocator_.allocate(new_size);
  for (size_t i = 0; i < new_size; ++i)
    EK()(new_entries[i]) = 0;
  for (size_t i = 0; i < old_size; ++i) {
    const uint64_t key = EK()(entries_[i]);
    if (key != 0) {
      // always finds an empty slot in a larger table
      new (find(new_entries, lg_new_size, key).first)
          EN(std::move(entries_[i]));
      entries_[i].~EN();
      EK()(entries_[i]) = 0;
    }
  }
  std::swap(entries_, new_entries);
  lgCurSize_ = lg_new_size;
  allocator_.deallocate(new_entries, old_size);
}

// assumes number of entries > nominal size
template <typename EN, typename EK, typename A>
void ThetaUpdateSketchBase<EN, EK, A>::rebuild() {
  const size_t size = 1ULL << lgCurSize_;
  const uint32_t nominal_size = 1 << lgNomSize_;

  // empty entries have uninitialized payloads
  // TODO: avoid this for empty or trivial payloads (arithmetic types)
  consolidateNonEmpty(entries_, size, numEntries_);

  std::nth_element(
      entries_, entries_ + nominal_size, entries_ + numEntries_, comparator());
  this->theta_ = EK()(entries_[nominal_size]);
  EN* old_entries = entries_;
  const size_t num_old_entries = numEntries_;
  entries_ = allocator_.allocate(size);
  for (size_t i = 0; i < size; ++i)
    EK()(entries_[i]) = 0;
  numEntries_ = nominal_size;
  // relies on consolidating non-empty entries to the front
  for (size_t i = 0; i < nominal_size; ++i) {
    new (find(EK()(old_entries[i])).first) EN(std::move(old_entries[i]));
    old_entries[i].~EN();
  }
  for (size_t i = nominal_size; i < num_old_entries; ++i)
    old_entries[i].~EN();
  allocator_.deallocate(old_entries, size);
}

template <typename EN, typename EK, typename A>
void ThetaUpdateSketchBase<EN, EK, A>::trim() {
  if (numEntries_ > static_cast<uint32_t>(1 << lgNomSize_))
    rebuild();
}

template <typename EN, typename EK, typename A>
void ThetaUpdateSketchBase<EN, EK, A>::reset() {
  const size_t cur_size = 1ULL << lgCurSize_;
  for (size_t i = 0; i < cur_size; ++i) {
    if (EK()(entries_[i]) != 0) {
      entries_[i].~EN();
      EK()(entries_[i]) = 0;
    }
  }
  const uint8_t starting_lg_size = ThetaBuildHelper<true>::startingSubMultiple(
      lgNomSize_ + 1, ThetaConstants::MIN_LG_K, static_cast<uint8_t>(rf_));
  if (starting_lg_size != lgCurSize_) {
    allocator_.deallocate(entries_, cur_size);
    lgCurSize_ = starting_lg_size;
    const size_t new_size = 1ULL << starting_lg_size;
    entries_ = allocator_.allocate(new_size);
    for (size_t i = 0; i < new_size; ++i)
      EK()(entries_[i]) = 0;
  }
  numEntries_ = 0;
  theta_ = ThetaBuildHelper<true>::startingThetaFromP(p_);
  isEmpty_ = true;
}

template <typename EN, typename EK, typename A>
void ThetaUpdateSketchBase<EN, EK, A>::consolidateNonEmpty(
    EN* entries,
    size_t size,
    size_t num) {
  // find the first empty slot
  size_t i = 0;
  while (i < size) {
    if (EK()(entries[i]) == 0)
      break;
    ++i;
  }
  // scan the rest and move non-empty entries to the front
  for (size_t j = i + 1; j < size; ++j) {
    if (EK()(entries[j]) != 0) {
      new (&entries[i]) EN(std::move(entries[j]));
      entries[j].~EN();
      EK()(entries[j]) = 0;
      ++i;
      if (i == num)
        break;
    }
  }
}

// builder

template <typename Derived, typename Allocator>
ThetaBaseBuilder<Derived, Allocator>::ThetaBaseBuilder(
    const Allocator& allocator)
    : allocator_(allocator),
      lg_k_(ThetaConstants::DEFAULT_LG_K),
      rf_(ThetaConstants::DEFAULT_RESIZE_FACTOR),
      p_(1),
      seed_(DEFAULT_SEED) {}

template <typename Derived, typename Allocator>
Derived& ThetaBaseBuilder<Derived, Allocator>::set_lg_k(uint8_t lg_k) {
  if (lg_k < ThetaConstants::MIN_LG_K) {
    throw VeloxRuntimeError(
        __FILE__,
        __LINE__,
        __FUNCTION__,
        "",
        "lg_k must not be less than " +
            std::to_string(ThetaConstants::MIN_LG_K) + ": " + std::to_string(lg_k),
        error_source::kErrorSourceRuntime,
        error_code::kInvalidArgument,
        false /*retriable*/);
  }
  if (lg_k > ThetaConstants::MAX_LG_K) {
    throw VeloxRuntimeError(
        __FILE__,
        __LINE__,
        __FUNCTION__,
        "",
        "lg_k must not be greater than " +
            std::to_string(ThetaConstants::MAX_LG_K) + ": " + std::to_string(lg_k),
        error_source::kErrorSourceRuntime,
        error_code::kInvalidArgument,
        false /*retriable*/);
  }
  lg_k_ = lg_k;
  return static_cast<Derived&>(*this);
}

template <typename Derived, typename Allocator>
Derived& ThetaBaseBuilder<Derived, Allocator>::setResizeFactor(
    resizeFactor rf) {
  rf_ = rf;
  return static_cast<Derived&>(*this);
}

template <typename Derived, typename Allocator>
Derived& ThetaBaseBuilder<Derived, Allocator>::setP(float p) {
  if (p <= 0 || p > 1) {
    throw VeloxRuntimeError(
        __FILE__,
        __LINE__,
        __FUNCTION__,
        "",
        "sampling probability must be between 0 and 1",
        error_source::kErrorSourceRuntime,
        error_code::kInvalidArgument,
        false /*retriable*/);
  }
  p_ = p;
  return static_cast<Derived&>(*this);
}

template <typename Derived, typename Allocator>
Derived& ThetaBaseBuilder<Derived, Allocator>::setSeed(uint64_t seed) {
  seed_ = seed;
  return static_cast<Derived&>(*this);
}

template <typename Derived, typename Allocator>
uint64_t ThetaBaseBuilder<Derived, Allocator>::startingTheta() const {
  return ThetaBuildHelper<true>::startingThetaFromP(p_);
}

template <typename Derived, typename Allocator>
uint8_t ThetaBaseBuilder<Derived, Allocator>::startingLgSize() const {
  return ThetaBuildHelper<true>::startingSubMultiple(
      lg_k_ + 1, ThetaConstants::MIN_LG_K, static_cast<uint8_t>(rf_));
}

// iterator

template <typename Entry, typename ExtractKey>
ThetaIterator<Entry, ExtractKey>::ThetaIterator(
    Entry* entries,
    uint32_t size,
    uint32_t index)
    : entries_(entries), size_(size), index_(index) {
  while (index_ < size_ && ExtractKey()(entries_[index_]) == 0)
    ++index_;
}

template <typename Entry, typename ExtractKey>
auto ThetaIterator<Entry, ExtractKey>::operator++() -> ThetaIterator& {
  ++index_;
  while (index_ < size_ && ExtractKey()(entries_[index_]) == 0)
    ++index_;
  return *this;
}

template <typename Entry, typename ExtractKey>
auto ThetaIterator<Entry, ExtractKey>::operator++(int) -> ThetaIterator {
  ThetaIterator tmp(*this);
  operator++();
  return tmp;
}

template <typename Entry, typename ExtractKey>
bool ThetaIterator<Entry, ExtractKey>::operator!=(
    const ThetaIterator& other) const {
  return index_ != other.index_;
}

template <typename Entry, typename ExtractKey>
bool ThetaIterator<Entry, ExtractKey>::operator==(
    const ThetaIterator& other) const {
  return index_ == other.index_;
}

template <typename Entry, typename ExtractKey>
auto ThetaIterator<Entry, ExtractKey>::operator*() const -> reference {
  return entries_[index_];
}

template <typename Entry, typename ExtractKey>
auto ThetaIterator<Entry, ExtractKey>::operator->() const -> pointer {
  return entries_ + index_;
}

// const iterator

template <typename Entry, typename ExtractKey>
ThetaConstIterator<Entry, ExtractKey>::ThetaConstIterator(
    const Entry* entries,
    uint32_t size,
    uint32_t index)
    : entries_(entries), size_(size), index_(index) {
  while (index_ < size_ && ExtractKey()(entries_[index_]) == 0)
    ++index_;
}

template <typename Entry, typename ExtractKey>
auto ThetaConstIterator<Entry, ExtractKey>::operator++()
    -> ThetaConstIterator& {
  ++index_;
  while (index_ < size_ && ExtractKey()(entries_[index_]) == 0)
    ++index_;
  return *this;
}

template <typename Entry, typename ExtractKey>
auto ThetaConstIterator<Entry, ExtractKey>::operator++(int)
    -> ThetaConstIterator {
  ThetaConstIterator tmp(*this);
  operator++();
  return tmp;
}

template <typename Entry, typename ExtractKey>
bool ThetaConstIterator<Entry, ExtractKey>::operator!=(
    const ThetaConstIterator& other) const {
  return index_ != other.index_;
}

template <typename Entry, typename ExtractKey>
bool ThetaConstIterator<Entry, ExtractKey>::operator==(
    const ThetaConstIterator& other) const {
  return index_ == other.index_;
}

template <typename Entry, typename ExtractKey>
auto ThetaConstIterator<Entry, ExtractKey>::operator*() const -> reference {
  return entries_[index_];
}

template <typename Entry, typename ExtractKey>
auto ThetaConstIterator<Entry, ExtractKey>::operator->() const -> pointer {
  return entries_ + index_;
}

} // namespace facebook::velox::common::theta

#endif
