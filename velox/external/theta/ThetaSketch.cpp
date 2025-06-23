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

#ifndef THETA_SKETCH_CPP
#define THETA_SKETCH_CPP

#include <sstream>
#include <vector>

#include "BinomialBounds.h"
#include "BitPacking.h"
#include "CompactThetaSketchParser.h"
#include "CountZeros.h"
#include "MemoryOperations.h"
#include "ThetaSketch.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common::theta {

template <typename A>
bool BaseThetaSketchAlloc<A>::isEstimationMode() const {
  return getTheta64() < ThetaConstants::MAX_THETA && !isEmpty();
}

template <typename A>
double BaseThetaSketchAlloc<A>::getTheta() const {
  return static_cast<double>(getTheta64()) /
      static_cast<double>(ThetaConstants::MAX_THETA);
}

template <typename A>
double BaseThetaSketchAlloc<A>::getEstimate() const {
  return getNumRetained() / getTheta();
}

template <typename A>
double BaseThetaSketchAlloc<A>::getLowerBound(uint8_t num_std_devs) const {
  if (!isEstimationMode())
    return getNumRetained();
  return BinomialBounds::getLowerBound(
      getNumRetained(), getTheta(), num_std_devs);
}

template <typename A>
double BaseThetaSketchAlloc<A>::getUpperBound(uint8_t num_std_devs) const {
  if (!isEstimationMode())
    return getNumRetained();
  return BinomialBounds::getUpperBound(
      getNumRetained(), getTheta(), num_std_devs);
}

template <typename A>
string<A> BaseThetaSketchAlloc<A>::toString(bool print_details) const {
  // Using a temporary stream for implementation here does not comply with
  // AllocatorAwareContainer requirements. The stream does not support passing
  // an allocator instance, and alternatives are complicated.
  std::ostringstream os;
  os << "### Theta sketch summary:" << std::endl;
  os << "   num retained entries : " << this->getNumRetained() << std::endl;
  os << "   seed hash            : " << this->getSeedHash() << std::endl;
  os << "   empty?               : " << (this->isEmpty() ? "true" : "false")
     << std::endl;
  os << "   ordered?             : " << (this->isOrdered() ? "true" : "false")
     << std::endl;
  os << "   estimation mode?     : "
     << (this->isEstimationMode() ? "true" : "false") << std::endl;
  os << "   theta (fraction)     : " << this->getTheta() << std::endl;
  os << "   theta (raw 64-bit)   : " << this->getTheta64() << std::endl;
  os << "   estimate             : " << this->getEstimate() << std::endl;
  os << "   lower bound 95% conf : " << this->getLowerBound(2) << std::endl;
  os << "   upper bound 95% conf : " << this->getUpperBound(2) << std::endl;
  printSpecifics(os);
  os << "### End sketch summary" << std::endl;
  if (print_details) {
    printItems(os);
  }
  return string<A>(os.str().c_str(), this->getAllocator());
}

template <typename A>
void ThetaSketchAlloc<A>::printItems(std::ostringstream& os) const {
  os << "### Retained entries" << std::endl;
  for (const auto& hash : *this) {
    os << hash << std::endl;
  }
  os << "### End retained entries" << std::endl;
}

// update sketch

template <typename A>
UpdateThetaSketchAlloc<A>::UpdateThetaSketchAlloc(
    uint8_t lgCurSize,
    uint8_t lgNomSize,
    resizeFactor rf,
    float p,
    uint64_t theta,
    uint64_t seed,
    const A& allocator)
    : table_(lgCurSize, lgNomSize, rf, p, theta, seed, allocator) {}

template <typename A>
A UpdateThetaSketchAlloc<A>::getAllocator() const {
  return table_.allocator_;
}

template <typename A>
bool UpdateThetaSketchAlloc<A>::isEmpty() const {
  return table_.isEmpty_;
}

template <typename A>
bool UpdateThetaSketchAlloc<A>::isOrdered() const {
  return table_.numEntries_ > 1 ? false : true;
}

template <typename A>
uint64_t UpdateThetaSketchAlloc<A>::getTheta64() const {
  return isEmpty() ? ThetaConstants::MAX_THETA : table_.theta_;
}

template <typename A>
uint32_t UpdateThetaSketchAlloc<A>::getNumRetained() const {
  return table_.numEntries_;
}

template <typename A>
uint16_t UpdateThetaSketchAlloc<A>::getSeedHash() const {
  return compute_seed_hash(table_.seed_);
}

template <typename A>
uint8_t UpdateThetaSketchAlloc<A>::getLgK() const {
  return table_.lgNomSize_;
}

template <typename A>
auto UpdateThetaSketchAlloc<A>::getRf() const -> resizeFactor {
  return table_.rf_;
}

template <typename A>
void UpdateThetaSketchAlloc<A>::update(uint64_t value) {
  update(&value, sizeof(value));
}

template <typename A>
void UpdateThetaSketchAlloc<A>::update(int64_t value) {
  update(&value, sizeof(value));
}

template <typename A>
void UpdateThetaSketchAlloc<A>::update(uint32_t value) {
  update(static_cast<int32_t>(value));
}

template <typename A>
void UpdateThetaSketchAlloc<A>::update(int32_t value) {
  update(static_cast<int64_t>(value));
}

template <typename A>
void UpdateThetaSketchAlloc<A>::update(uint16_t value) {
  update(static_cast<int16_t>(value));
}

template <typename A>
void UpdateThetaSketchAlloc<A>::update(int16_t value) {
  update(static_cast<int64_t>(value));
}

template <typename A>
void UpdateThetaSketchAlloc<A>::update(uint8_t value) {
  update(static_cast<int8_t>(value));
}

template <typename A>
void UpdateThetaSketchAlloc<A>::update(int8_t value) {
  update(static_cast<int64_t>(value));
}

template <typename A>
void UpdateThetaSketchAlloc<A>::update(double value) {
  update(canonical_double(value));
}

template <typename A>
void UpdateThetaSketchAlloc<A>::update(float value) {
  update(static_cast<double>(value));
}

template <typename A>
void UpdateThetaSketchAlloc<A>::update(const std::string& value) {
  if (value.empty())
    return;
  update(value.c_str(), value.length());
}

template <typename A>
void UpdateThetaSketchAlloc<A>::update(const void* data, size_t length) {
  const uint64_t hash = table_.hashAndScreen(data, length);
  if (hash == 0)
    return;
  auto result = table_.find(hash);
  if (!result.second) {
    table_.insert(result.first, hash);
  }
}

template <typename A>
void UpdateThetaSketchAlloc<A>::trim() {
  table_.trim();
}

template <typename A>
void UpdateThetaSketchAlloc<A>::reset() {
  table_.reset();
}

template <typename A>
auto UpdateThetaSketchAlloc<A>::begin() -> iterator {
  return iterator(table_.entries_, 1 << table_.lgCurSize_, 0);
}

template <typename A>
auto UpdateThetaSketchAlloc<A>::end() -> iterator {
  return iterator(nullptr, 0, 1 << table_.lgCurSize_);
}

template <typename A>
auto UpdateThetaSketchAlloc<A>::begin() const -> const_iterator {
  return const_iterator(table_.entries_, 1 << table_.lgCurSize_, 0);
}

template <typename A>
auto UpdateThetaSketchAlloc<A>::end() const -> const_iterator {
  return const_iterator(nullptr, 0, 1 << table_.lgCurSize_);
}

template <typename A>
CompactThetaSketchAlloc<A> UpdateThetaSketchAlloc<A>::compact(
    bool ordered) const {
  return CompactThetaSketchAlloc<A>(*this, ordered);
}

template <typename A>
void UpdateThetaSketchAlloc<A>::printSpecifics(std::ostringstream& os) const {
  os << "   lg nominal size      : " << static_cast<int>(table_.lgNomSize_)
     << std::endl;
  os << "   lg current size      : " << static_cast<int>(table_.lgCurSize_)
     << std::endl;
  os << "   resize factor        : " << (1 << table_.rf_) << std::endl;
}

// builder

template <typename A>
UpdateThetaSketchAlloc<A>::builder::builder(const A& allocator)
    : ThetaBaseBuilder<builder, A>(allocator) {}

template <typename A>
UpdateThetaSketchAlloc<A> UpdateThetaSketchAlloc<A>::builder::build() const {
  return UpdateThetaSketchAlloc(
      this->startingLgSize(),
      this->lg_k_,
      this->rf_,
      this->p_,
      this->startingTheta(),
      this->seed_,
      this->allocator_);
}

// compact sketch

template <typename A>
template <typename Other>
CompactThetaSketchAlloc<A>::CompactThetaSketchAlloc(
    const Other& other,
    bool ordered)
    : isEmpty_(other.isEmpty()),
      isOrdered_(other.isOrdered() || ordered),
      seedHash_(other.getSeedHash()),
      theta_(other.getTheta64()),
      entries_(other.getAllocator()) {
  if (!other.isEmpty()) {
    entries_.reserve(other.getNumRetained());
    std::copy(other.begin(), other.end(), std::back_inserter(entries_));
    if (ordered && !other.isOrdered())
      std::sort(entries_.begin(), entries_.end());
  }
}

template <typename A>
CompactThetaSketchAlloc<A>::CompactThetaSketchAlloc(
    bool isEmpty,
    bool isOrdered,
    uint16_t seedHash,
    uint64_t theta,
    std::vector<uint64_t, A>&& entries)
    : isEmpty_(isEmpty),
      isOrdered_(isOrdered || (entries.size() <= 1ULL)),
      seedHash_(seedHash),
      theta_(theta),
      entries_(std::move(entries)) {}

template <typename A>
A CompactThetaSketchAlloc<A>::getAllocator() const {
  return entries_.get_allocator();
}

template <typename A>
bool CompactThetaSketchAlloc<A>::isEmpty() const {
  return isEmpty_;
}

template <typename A>
bool CompactThetaSketchAlloc<A>::isOrdered() const {
  return isOrdered_;
}

template <typename A>
uint64_t CompactThetaSketchAlloc<A>::getTheta64() const {
  return theta_;
}

template <typename A>
uint32_t CompactThetaSketchAlloc<A>::getNumRetained() const {
  return static_cast<uint32_t>(entries_.size());
}

template <typename A>
uint16_t CompactThetaSketchAlloc<A>::getSeedHash() const {
  return seedHash_;
}

template <typename A>
auto CompactThetaSketchAlloc<A>::begin() -> iterator {
  return iterator(entries_.data(), static_cast<uint32_t>(entries_.size()), 0);
}

template <typename A>
auto CompactThetaSketchAlloc<A>::end() -> iterator {
  return iterator(nullptr, 0, static_cast<uint32_t>(entries_.size()));
}

template <typename A>
auto CompactThetaSketchAlloc<A>::begin() const -> const_iterator {
  return const_iterator(
      entries_.data(), static_cast<uint32_t>(entries_.size()), 0);
}

template <typename A>
auto CompactThetaSketchAlloc<A>::end() const -> const_iterator {
  return const_iterator(nullptr, 0, static_cast<uint32_t>(entries_.size()));
}

template <typename A>
void CompactThetaSketchAlloc<A>::printSpecifics(std::ostringstream&) const {}

template <typename A>
uint8_t CompactThetaSketchAlloc<A>::getPreambleLongs(bool compressed) const {
  if (compressed) {
    return this->isEstimationMode() ? 2 : 1;
  }
  return this->isEstimationMode()               ? 3
      : this->isEmpty() || entries_.size() == 1 ? 1
                                                : 2;
}

template <typename A>
size_t CompactThetaSketchAlloc<A>::getMaxSerializedSizeBytes(uint8_t lg_k) {
  return sizeof(uint64_t) *
      (3 + UpdateThetaSketchAlloc<A>::ThetaTable::getCapacity(lg_k + 1, lg_k));
}

template <typename A>
size_t CompactThetaSketchAlloc<A>::getSerializedSizeBytes(
    bool compressed) const {
  if (compressed && isSuitableForCompression()) {
    return getCompressedSerializedSizeBytes(
        computeEntryBits(), getNumEntriesBytes());
  }
  return sizeof(uint64_t) * getPreambleLongs(false) +
      sizeof(uint64_t) * entries_.size();
}

// store num_entries as whole bytes since whole-byte blocks will follow (most
// probably)
template <typename A>
uint8_t CompactThetaSketchAlloc<A>::getNumEntriesBytes() const {
  return wholeBytesToHoldBits<uint8_t>(
      32 - countLeadingZerosInU32(static_cast<uint32_t>(entries_.size())));
}

template <typename A>
size_t CompactThetaSketchAlloc<A>::getCompressedSerializedSizeBytes(
    uint8_t entry_bits,
    uint8_t num_entries_bytes) const {
  const size_t compressed_bits = entry_bits * entries_.size();
  return sizeof(uint64_t) * getPreambleLongs(true) + num_entries_bytes +
      wholeBytesToHoldBits(compressed_bits);
}

template <typename A>
void CompactThetaSketchAlloc<A>::serialize(std::ostream& os) const {
  const uint8_t preamble_longs = this->isEstimationMode() ? 3
      : this->isEmpty() || entries_.size() == 1           ? 1
                                                          : 2;
  write(os, preamble_longs);
  write(os, UNCOMPRESSED_SERIAL_VERSION);
  write(os, SKETCH_TYPE);
  write<uint16_t>(os, 0); // unused
  const uint8_t flags_byte(
      (1 << flags::IS_COMPACT) | (1 << flags::IS_READ_ONLY) |
      (this->isEmpty() ? 1 << flags::IS_EMPTY : 0) |
      (this->isOrdered() ? 1 << flags::IS_ORDERED : 0));
  write(os, flags_byte);
  write(os, getSeedHash());
  if (preamble_longs > 1) {
    write(os, static_cast<uint32_t>(entries_.size()));
    write<uint32_t>(os, 0); // unused
  }
  if (this->isEstimationMode())
    write(os, this->theta_);
  if (entries_.size() > 0)
    write(os, entries_.data(), entries_.size() * sizeof(uint64_t));
}

template <typename A>
auto CompactThetaSketchAlloc<A>::serialize(unsigned header_size_bytes) const
    -> vector_bytes {
  const size_t size = getSerializedSizeBytes() + header_size_bytes;
  vector_bytes bytes(size, 0, entries_.get_allocator());
  uint8_t* ptr = bytes.data() + header_size_bytes;
  const uint8_t preamble_longs = getPreambleLongs(false);
  *ptr++ = preamble_longs;
  *ptr++ = UNCOMPRESSED_SERIAL_VERSION;
  *ptr++ = SKETCH_TYPE;
  ptr += sizeof(uint16_t); // unused
  const uint8_t flags_byte(
      (1 << flags::IS_COMPACT) | (1 << flags::IS_READ_ONLY) |
      (this->isEmpty() ? 1 << flags::IS_EMPTY : 0) |
      (this->isOrdered() ? 1 << flags::IS_ORDERED : 0));
  *ptr++ = flags_byte;
  ptr += copyToMem(getSeedHash(), ptr);
  if (preamble_longs > 1) {
    ptr += copyToMem(static_cast<uint32_t>(entries_.size()), ptr);
    ptr += sizeof(uint32_t); // unused
  }
  if (this->isEstimationMode())
    ptr += copyToMem(theta_, ptr);
  if (entries_.size() > 0)
    ptr += copyToMem(entries_.data(), ptr, entries_.size() * sizeof(uint64_t));
  return bytes;
}

template <typename A>
bool CompactThetaSketchAlloc<A>::isSuitableForCompression() const {
  if (!this->isOrdered() || entries_.size() == 0 ||
      (entries_.size() == 1 && !this->isEstimationMode()))
    return false;
  return true;
}

template <typename A>
void CompactThetaSketchAlloc<A>::serializeCompressed(std::ostream& os) const {
  if (isSuitableForCompression())
    return serializeVersion4(os);
  return serialize(os);
}

template <typename A>
auto CompactThetaSketchAlloc<A>::serializeCompressed(
    unsigned header_size_bytes) const -> vector_bytes {
  if (isSuitableForCompression())
    return serializeVersion4(header_size_bytes);
  return serialize(header_size_bytes);
}

template <typename A>
uint8_t CompactThetaSketchAlloc<A>::computeEntryBits() const {
  // compression is based on leading zeros in deltas between ordered hash values
  // assumes ordered sketch
  uint64_t previous = 0;
  uint64_t ored = 0;
  for (const uint64_t entry : entries_) {
    const uint64_t delta = entry - previous;
    ored |= delta;
    previous = entry;
  }
  return 64 - countLeadingZerosInU64(ored);
}

template <typename A>
void CompactThetaSketchAlloc<A>::serializeVersion4(std::ostream& os) const {
  const uint8_t preamble_longs = this->isEstimationMode() ? 2 : 1;
  const uint8_t entry_bits = computeEntryBits();
  const uint8_t num_entries_bytes = getNumEntriesBytes();

  write(os, preamble_longs);
  write(os, COMPRESSED_SERIAL_VERSION);
  write(os, SKETCH_TYPE);
  write(os, entry_bits);
  write(os, num_entries_bytes);
  const uint8_t flags_byte(
      (1 << flags::IS_COMPACT) | (1 << flags::IS_READ_ONLY) |
      (1 << flags::IS_ORDERED));
  write(os, flags_byte);
  write(os, getSeedHash());
  if (this->isEstimationMode())
    write(os, this->theta_);
  uint32_t num_entries = static_cast<uint32_t>(entries_.size());
  for (unsigned i = 0; i < num_entries_bytes; ++i) {
    write<uint8_t>(os, num_entries & 0xff);
    num_entries >>= 8;
  }

  uint64_t previous = 0;
  uint64_t deltas[8];
  vector_bytes buffer(
      entry_bits,
      0,
      entries_.get_allocator()); // block of 8 entries takes entry_bits bytes

  // pack blocks of 8 deltas
  unsigned i;
  for (i = 0; i + 7 < entries_.size(); i += 8) {
    for (unsigned j = 0; j < 8; ++j) {
      deltas[j] = entries_[i + j] - previous;
      previous = entries_[i + j];
    }
    packBitsBlock8(deltas, buffer.data(), entry_bits);
    write(os, buffer.data(), buffer.size());
  }

  // pack extra deltas if fewer than 8 of them left
  if (i < entries_.size()) {
    uint8_t offset = 0;
    uint8_t* ptr = buffer.data();
    for (; i < entries_.size(); ++i) {
      const uint64_t delta = entries_[i] - previous;
      previous = entries_[i];
      offset = packBits(delta, entry_bits, ptr, offset);
    }
    if (offset > 0)
      ++ptr;
    write(os, buffer.data(), ptr - buffer.data());
  }
}

template <typename A>
auto CompactThetaSketchAlloc<A>::serializeVersion4(
    unsigned header_size_bytes) const -> vector_bytes {
  const uint8_t entry_bits = computeEntryBits();
  const uint8_t num_entries_bytes = getNumEntriesBytes();
  const size_t size =
      getCompressedSerializedSizeBytes(entry_bits, num_entries_bytes) +
      header_size_bytes;
  vector_bytes bytes(size, 0, entries_.get_allocator());
  uint8_t* ptr = bytes.data() + header_size_bytes;

  *ptr++ = getPreambleLongs(true);
  *ptr++ = COMPRESSED_SERIAL_VERSION;
  *ptr++ = SKETCH_TYPE;
  *ptr++ = entry_bits;
  *ptr++ = num_entries_bytes;
  const uint8_t flags_byte(
      (1 << flags::IS_COMPACT) | (1 << flags::IS_READ_ONLY) |
      (1 << flags::IS_ORDERED));
  *ptr++ = flags_byte;
  ptr += copyToMem(getSeedHash(), ptr);
  if (this->isEstimationMode()) {
    ptr += copyToMem(theta_, ptr);
  }
  uint32_t num_entries = static_cast<uint32_t>(entries_.size());
  for (unsigned i = 0; i < num_entries_bytes; ++i) {
    *ptr++ = num_entries & 0xff;
    num_entries >>= 8;
  }

  uint64_t previous = 0;
  uint64_t deltas[8];

  // pack blocks of 8 deltas
  unsigned i;
  for (i = 0; i + 7 < entries_.size(); i += 8) {
    for (unsigned j = 0; j < 8; ++j) {
      deltas[j] = entries_[i + j] - previous;
      previous = entries_[i + j];
    }
    packBitsBlock8(deltas, ptr, entry_bits);
    ptr += entry_bits;
  }

  // pack extra deltas if fewer than 8 of them left
  uint8_t offset = 0;
  for (; i < entries_.size(); ++i) {
    const uint64_t delta = entries_[i] - previous;
    previous = entries_[i];
    offset = packBits(delta, entry_bits, ptr, offset);
  }
  return bytes;
}

template <typename A>
CompactThetaSketchAlloc<A> CompactThetaSketchAlloc<A>::deserialize(
    std::istream& is,
    uint64_t seed,
    const A& allocator) {
  const auto preamble_longs = read<uint8_t>(is);
  const auto serial_version = read<uint8_t>(is);
  const auto type = read<uint8_t>(is);
  checker<true>::checkSketchType(type, SKETCH_TYPE);
  switch (serial_version) {
    case 4:
      return deserializeV4(preamble_longs, is, seed, allocator);
    case 3:
      return deserializeV3(preamble_longs, is, seed, allocator);
    case 1:
      return deserializeV1(preamble_longs, is, seed, allocator);
    case 2:
      return deserializeV2(preamble_longs, is, seed, allocator);
    default:
      throw VeloxUserError(
          __FILE__,
          __LINE__,
          __FUNCTION__,
          "",
          "unexpected sketch serialization version " +
              std::to_string(serial_version),
          error_source::kErrorSourceRuntime,
          error_code::kInvalidArgument,
          false /*retriable*/);
  }
}

template <typename A>
CompactThetaSketchAlloc<A> CompactThetaSketchAlloc<A>::deserializeV1(
    uint8_t,
    std::istream& is,
    uint64_t seed,
    const A& allocator) {
  const auto seed_hash = compute_seed_hash(seed);
  read<uint8_t>(is); // unused
  read<uint32_t>(is); // unused
  const auto num_entries = read<uint32_t>(is);
  read<uint32_t>(is); // unused
  const auto theta = read<uint64_t>(is);
  std::vector<uint64_t, A> entries(num_entries, 0, allocator);
  bool isEmpty = (num_entries == 0) && (theta == ThetaConstants::MAX_THETA);
  if (!isEmpty)
    read(is, entries.data(), sizeof(uint64_t) * entries.size());
  if (!is.good()) {
    throw VeloxRuntimeError(
        __FILE__,
        __LINE__,
        __FUNCTION__,
        "",
        "error reading from std::istream",
        error_source::kErrorSourceRuntime,
        error_code::kInvalidArgument,
        false /*retriable*/);
  }
  return CompactThetaSketchAlloc(
      isEmpty, true, seed_hash, theta, std::move(entries));
}

template <typename A>
CompactThetaSketchAlloc<A> CompactThetaSketchAlloc<A>::deserializeV2(
    uint8_t preamble_longs,
    std::istream& is,
    uint64_t seed,
    const A& allocator) {
  read<uint8_t>(is); // unused
  read<uint16_t>(is); // unused
  const uint16_t seed_hash = read<uint16_t>(is);
  checker<true>::checkSeedHash(seed_hash, compute_seed_hash(seed));
  if (preamble_longs == 1) {
    if (!is.good()) {
      throw VeloxRuntimeError(
          __FILE__,
          __LINE__,
          __FUNCTION__,
          "",
          "error reading from std::istream",
          error_source::kErrorSourceRuntime,
          error_code::kInvalidArgument,
          false /*retriable*/);
    }
    std::vector<uint64_t, A> entries(0, 0, allocator);
    return CompactThetaSketchAlloc(
        true, true, seed_hash, ThetaConstants::MAX_THETA, std::move(entries));
  } else if (preamble_longs == 2) {
    const uint32_t num_entries = read<uint32_t>(is);
    read<uint32_t>(is); // unused
    std::vector<uint64_t, A> entries(num_entries, 0, allocator);
    if (num_entries == 0) {
      return CompactThetaSketchAlloc(
          true, true, seed_hash, ThetaConstants::MAX_THETA, std::move(entries));
    }
    read(is, entries.data(), entries.size() * sizeof(uint64_t));
    if (!is.good()) {
      throw VeloxRuntimeError(
          __FILE__,
          __LINE__,
          __FUNCTION__,
          "",
          "error reading from std::istream",
          error_source::kErrorSourceRuntime,
          error_code::kInvalidArgument,
          false /*retriable*/);
    }
    return CompactThetaSketchAlloc(
        false, true, seed_hash, ThetaConstants::MAX_THETA, std::move(entries));
  } else if (preamble_longs == 3) {
    const uint32_t num_entries = read<uint32_t>(is);
    read<uint32_t>(is); // unused
    const auto theta = read<uint64_t>(is);
    bool isEmpty = (num_entries == 0) && (theta == ThetaConstants::MAX_THETA);
    std::vector<uint64_t, A> entries(num_entries, 0, allocator);
    if (isEmpty) {
      if (!is.good()) {
        throw VeloxRuntimeError(
            __FILE__,
            __LINE__,
            __FUNCTION__,
            "",
            "error reading from std::istream",
            error_source::kErrorSourceRuntime,
            error_code::kInvalidArgument,
            false /*retriable*/);
      }
      return CompactThetaSketchAlloc(
          true, true, seed_hash, theta, std::move(entries));
    } else {
      read(is, entries.data(), sizeof(uint64_t) * entries.size());
      if (!is.good()) {
        throw VeloxRuntimeError(
            __FILE__,
            __LINE__,
            __FUNCTION__,
            "",
            "error reading from std::istream",
            error_source::kErrorSourceRuntime,
            error_code::kInvalidArgument,
            false /*retriable*/);
      }
      return CompactThetaSketchAlloc(
          false, true, seed_hash, theta, std::move(entries));
    }
  } else {
    throw VeloxRuntimeError(
        __FILE__,
        __LINE__,
        __FUNCTION__,
        "",
        std::to_string(preamble_longs) +
            " longs of premable, but expected 1, 2, or 3",
        error_source::kErrorSourceRuntime,
        error_code::kInvalidArgument,
        false /*retriable*/);
  }
}

template <typename A>
CompactThetaSketchAlloc<A> CompactThetaSketchAlloc<A>::deserializeV3(
    uint8_t preamble_longs,
    std::istream& is,
    uint64_t seed,
    const A& allocator) {
  read<uint16_t>(is); // unused
  const auto flags_byte = read<uint8_t>(is);
  const auto seed_hash = read<uint16_t>(is);
  const bool isEmpty = flags_byte & (1 << flags::IS_EMPTY);
  if (!isEmpty)
    checker<true>::checkSeedHash(seed_hash, compute_seed_hash(seed));
  uint64_t theta = ThetaConstants::MAX_THETA;
  uint32_t num_entries = 0;
  if (!isEmpty) {
    if (preamble_longs == 1) {
      num_entries = 1;
    } else {
      num_entries = read<uint32_t>(is);
      read<uint32_t>(is); // unused
      if (preamble_longs > 2)
        theta = read<uint64_t>(is);
    }
  }
  std::vector<uint64_t, A> entries(num_entries, 0, allocator);
  if (!isEmpty)
    read(is, entries.data(), sizeof(uint64_t) * entries.size());
  const bool isOrdered = flags_byte & (1 << flags::IS_ORDERED);
  if (!is.good()) {
    throw VeloxRuntimeError(
        __FILE__,
        __LINE__,
        __FUNCTION__,
        "",
        "error reading from std::istream",
        error_source::kErrorSourceRuntime,
        error_code::kInvalidArgument,
        false /*retriable*/);
  }
  return CompactThetaSketchAlloc(
      isEmpty, isOrdered, seed_hash, theta, std::move(entries));
}

template <typename A>
CompactThetaSketchAlloc<A> CompactThetaSketchAlloc<A>::deserializeV4(
    uint8_t preamble_longs,
    std::istream& is,
    uint64_t seed,
    const A& allocator) {
  const auto entry_bits = read<uint8_t>(is);
  const auto num_entries_bytes = read<uint8_t>(is);
  const auto flags_byte = read<uint8_t>(is);
  const auto seed_hash = read<uint16_t>(is);
  const bool isEmpty = flags_byte & (1 << flags::IS_EMPTY);
  if (!isEmpty)
    checker<true>::checkSeedHash(seed_hash, compute_seed_hash(seed));
  uint64_t theta = ThetaConstants::MAX_THETA;
  if (preamble_longs > 1)
    theta = read<uint64_t>(is);
  uint32_t num_entries = 0;
  for (unsigned i = 0; i < num_entries_bytes; ++i) {
    num_entries |= read<uint8_t>(is) << (i << 3);
  }
  vector_bytes buffer(
      entry_bits, 0, allocator); // block of 8 entries takes entry_bits bytes
  std::vector<uint64_t, A> entries(num_entries, 0, allocator);

  // unpack blocks of 8 deltas
  unsigned i;
  for (i = 0; i + 7 < num_entries; i += 8) {
    read(is, buffer.data(), buffer.size());
    unpackBitsBlock8(&entries[i], buffer.data(), entry_bits);
  }
  // unpack extra deltas if fewer than 8 of them left
  if (i < num_entries)
    read(
        is,
        buffer.data(),
        wholeBytesToHoldBits((num_entries - i) * entry_bits));
  if (!is.good()) {
    throw VeloxRuntimeError(
        __FILE__,
        __LINE__,
        __FUNCTION__,
        "",
        "error reading from std::istream",
        error_source::kErrorSourceRuntime,
        error_code::kInvalidArgument,
        false /*retriable*/);
  }
  const uint8_t* ptr = buffer.data();
  uint8_t offset = 0;
  for (; i < num_entries; ++i) {
    offset = unpackBits(entries[i], entry_bits, ptr, offset);
  }
  // undo deltas
  uint64_t previous = 0;
  for (i = 0; i < num_entries; ++i) {
    entries[i] += previous;
    previous = entries[i];
  }
  const bool isOrdered = flags_byte & (1 << flags::IS_ORDERED);
  return CompactThetaSketchAlloc(
      isEmpty, isOrdered, seed_hash, theta, std::move(entries));
}

template <typename A>
CompactThetaSketchAlloc<A> CompactThetaSketchAlloc<A>::deserialize(
    const void* bytes,
    size_t size,
    uint64_t seed,
    const A& allocator) {
  auto data = CompactThetaSketchParser<true>::parse(bytes, size, seed, false);
  if (data.entryBits == 64) { // versions 1 to 3
    const uint64_t* entries =
        reinterpret_cast<const uint64_t*>(data.entriesStartPtr);
    return CompactThetaSketchAlloc(
        data.isEmpty,
        data.isOrdered,
        data.seedHash,
        data.theta,
        std::vector<uint64_t, A>(
            entries, entries + data.numEntries, allocator));
  } else { // version 4
    std::vector<uint64_t, A> entries(data.numEntries, 0, allocator);
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data.entriesStartPtr);
    // unpack blocks of 8 deltas
    unsigned i;
    for (i = 0; i + 7 < data.numEntries; i += 8) {
      unpackBitsBlock8(&entries[i], ptr, data.entryBits);
      ptr += data.entryBits;
    }
    // unpack extra deltas if fewer than 8 of them left
    uint8_t offset = 0;
    for (; i < data.numEntries; ++i) {
      offset = unpackBits(entries[i], data.entryBits, ptr, offset);
    }
    // undo deltas
    uint64_t previous = 0;
    for (i = 0; i < data.numEntries; ++i) {
      entries[i] += previous;
      previous = entries[i];
    }
    return CompactThetaSketchAlloc(
        data.isEmpty,
        data.isOrdered,
        data.seedHash,
        data.theta,
        std::move(entries));
  }
}

// wrapped compact sketch

template <typename A>
WrappedCompactThetaSketchAlloc<A>::WrappedCompactThetaSketchAlloc(
    const data_type& data)
    : data_(data) {}

template <typename A>
const WrappedCompactThetaSketchAlloc<A> WrappedCompactThetaSketchAlloc<A>::wrap(
    const void* bytes,
    size_t size,
    uint64_t seed,
    bool dump_on_error) {
  return WrappedCompactThetaSketchAlloc(
      CompactThetaSketchParser<true>::parse(bytes, size, seed, dump_on_error));
}

template <typename A>
A WrappedCompactThetaSketchAlloc<A>::getAllocator() const {
  return A();
}

template <typename A>
bool WrappedCompactThetaSketchAlloc<A>::isEmpty() const {
  return data_.isEmpty;
}

template <typename A>
bool WrappedCompactThetaSketchAlloc<A>::isOrdered() const {
  return data_.isOrdered;
}

template <typename A>
uint64_t WrappedCompactThetaSketchAlloc<A>::getTheta64() const {
  return data_.theta;
}

template <typename A>
uint32_t WrappedCompactThetaSketchAlloc<A>::getNumRetained() const {
  return data_.numEntries;
}

template <typename A>
uint16_t WrappedCompactThetaSketchAlloc<A>::getSeedHash() const {
  return data_.seedHash;
}

template <typename A>
auto WrappedCompactThetaSketchAlloc<A>::begin() const -> const_iterator {
  return const_iterator(
      data_.entriesStartPtr, data_.entryBits, data_.numEntries, 0);
}

template <typename A>
auto WrappedCompactThetaSketchAlloc<A>::end() const -> const_iterator {
  return const_iterator(
      data_.entriesStartPtr,
      data_.entryBits,
      data_.numEntries,
      data_.numEntries);
}

template <typename A>
void WrappedCompactThetaSketchAlloc<A>::printSpecifics(
    std::ostringstream&) const {}

template <typename A>
void WrappedCompactThetaSketchAlloc<A>::printItems(
    std::ostringstream& os) const {
  os << "### Retained entries" << std::endl;
  for (const auto hash : *this) {
    os << hash << std::endl;
  }
  os << "### End retained entries" << std::endl;
}

// assumes index == 0 or index == num_entries
template <typename Allocator>
WrappedCompactThetaSketchAlloc<Allocator>::const_iterator::const_iterator(
    const void* ptr,
    uint8_t entry_bits,
    uint32_t num_entries,
    uint32_t index)
    : ptr_(ptr),
      entry_bits_(entry_bits),
      num_entries_(num_entries),
      index_(index),
      previous_(0),
      is_block_mode_(num_entries_ >= 8),
      offset_(0) {
  if (entry_bits == 64) { // no compression
    ptr_ = reinterpret_cast<const uint64_t*>(ptr) + index;
  } else if (index < num_entries) {
    if (is_block_mode_) {
      unpack8();
    } else {
      unpack1();
    }
  }
}

template <typename Allocator>
auto WrappedCompactThetaSketchAlloc<Allocator>::const_iterator::operator++()
    -> const_iterator& {
  if (entry_bits_ == 64) { // no compression
    ptr_ = reinterpret_cast<const uint64_t*>(ptr_) + 1;
    return *this;
  }
  if (++index_ < num_entries_) {
    if (is_block_mode_) {
      if ((index_ & 7) == 0) {
        if (num_entries_ - index_ >= 8) {
          unpack8();
        } else {
          is_block_mode_ = false;
          unpack1();
        }
      }
    } else {
      unpack1();
    }
  }
  return *this;
}

template <typename Allocator>
void WrappedCompactThetaSketchAlloc<Allocator>::const_iterator::unpack1() {
  const uint32_t i = index_ & 7;
  offset_ = unpackBits(
      buffer_[i],
      entry_bits_,
      reinterpret_cast<const uint8_t*&>(ptr_),
      offset_);
  buffer_[i] += previous_;
  previous_ = buffer_[i];
}

template <typename Allocator>
void WrappedCompactThetaSketchAlloc<Allocator>::const_iterator::unpack8() {
  unpackBitsBlock8(
      buffer_, reinterpret_cast<const uint8_t*>(ptr_), entry_bits_);
  ptr_ = reinterpret_cast<const uint8_t*>(ptr_) + entry_bits_;
  for (int i = 0; i < 8; ++i) {
    buffer_[i] += previous_;
    previous_ = buffer_[i];
  }
}

template <typename Allocator>
auto WrappedCompactThetaSketchAlloc<Allocator>::const_iterator::operator++(int)
    -> const_iterator {
  const_iterator tmp(*this);
  operator++();
  return tmp;
}

template <typename Allocator>
bool WrappedCompactThetaSketchAlloc<Allocator>::const_iterator::operator!=(
    const const_iterator& other) const {
  if (entry_bits_ == 64)
    return ptr_ != other.ptr_;
  return index_ != other.index_;
}

template <typename Allocator>
bool WrappedCompactThetaSketchAlloc<Allocator>::const_iterator::operator==(
    const const_iterator& other) const {
  if (entry_bits_ == 64)
    return ptr_ == other.ptr_;
  return index_ == other.index_;
}

template <typename Allocator>
auto WrappedCompactThetaSketchAlloc<Allocator>::const_iterator::operator*()
    const -> reference {
  if (entry_bits_ == 64)
    return *reinterpret_cast<const uint64_t*>(ptr_);
  return buffer_[index_ & 7];
}

template <typename Allocator>
auto WrappedCompactThetaSketchAlloc<Allocator>::const_iterator::operator->()
    const -> pointer {
  if (entry_bits_ == 64)
    return reinterpret_cast<const uint64_t*>(ptr_);
  return buffer_ + (index_ & 7);
}

} // namespace facebook::velox::common::theta

#endif
