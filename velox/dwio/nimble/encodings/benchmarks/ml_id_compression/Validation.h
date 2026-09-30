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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <cstring>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>
#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/velox/RowRange.h"

// Correctness checks for the ML ID benchmark drivers. A timed cell is only
// reported when the reads it times return the input: before timing, the exact
// reads the timed loop issues are replayed and compared element-wise against
// the column, and after timing, the output buffer of the last timed iteration
// is compared again. Every check runs outside the timed region.

DECLARE_bool(mlidc_validate);

namespace facebook::nimble::mlidc {

/// Returns a description of the first element where actual differs from
/// expected, or nullopt when every element matches. firstRow is the row the
/// first element stands for, so the message names a row of the column.
///
/// Compared bit for bit rather than with operator==: the encodings preserve
/// the bit pattern, and a NaN in a real double column would otherwise never
/// compare equal to itself.
template <typename T>
std::optional<std::string> firstMismatch(
    std::span<const T> actual,
    std::span<const T> expected,
    size_t firstRow = 0) {
  if (actual.size() != expected.size()) {
    return fmt::format(
        "{} rows returned, {} expected", actual.size(), expected.size());
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (std::memcmp(&actual[i], &expected[i], sizeof(T)) != 0) {
      std::ostringstream message;
      message << "row " << firstRow + i << ": got " << actual[i]
              << ", expected " << expected[i];
      return message.str();
    }
  }
  return std::nullopt;
}

/// The values a target's reads are checked against.
///
/// For every arm that preserves row order this is the input column. An arm
/// that does not, FPE/fpe_noindex, encodes the multiset rather than the
/// sequence: its reference is its own full decode, accepted only once that
/// decode is shown to be a permutation of the input, and its partial reads are
/// then checked against that permutation.
template <typename T>
class ValidationReference {
 public:
  ValidationReference() = default;
  // Not copyable: values() may point into this object's own buffer.
  ValidationReference(const ValidationReference&) = delete;
  ValidationReference& operator=(const ValidationReference&) = delete;

  /// Builds the reference for target, or returns a description of why the
  /// target's full decode does not reproduce data. data must outlive out.
  static std::optional<std::string> build(
      NimbleBenchTargetBase<T>& target,
      const EncoderEntry<T>& encoder,
      const Vector<T>& data,
      ValidationReference& out) {
    if (encoder.preservesRowOrder) {
      out.values_.clear();
      out.data_ = std::span<const T>(data.data(), data.size());
      return std::nullopt;
    }
    out.values_.assign(data.size(), T{});
    target.materializeAll(
        out.values_.data(), static_cast<uint32_t>(data.size()));
    out.data_ = out.values_;
    std::vector<T> decoded = out.values_;
    std::vector<T> input(data.begin(), data.end());
    const auto bitLess = [](const T& a, const T& b) {
      return std::memcmp(&a, &b, sizeof(T)) < 0;
    };
    std::sort(decoded.begin(), decoded.end(), bitLess);
    std::sort(input.begin(), input.end(), bitLess);
    if (auto mismatch = firstMismatch<T>(decoded, input)) {
      return "decode is not a permutation of the input: " + *mismatch;
    }
    return std::nullopt;
  }

  std::span<const T> values() const {
    return data_;
  }

 private:
  std::vector<T> values_;
  std::span<const T> data_;
};

/// Checks the rows a gather over ranges wrote to output.
template <typename T>
std::optional<std::string> checkGatherOutput(
    std::span<const T> output,
    std::span<const T> reference,
    std::span<const nimble::RowRange> ranges) {
  size_t offset = 0;
  for (const auto& range : ranges) {
    const size_t count = range.numRows();
    if (offset + count > output.size()) {
      return fmt::format(
          "output holds {} rows, the ranges select more", output.size());
    }
    if (auto mismatch = firstMismatch<T>(
            output.subspan(offset, count),
            reference.subspan(range.startRow, count),
            range.startRow)) {
      return mismatch;
    }
    offset += count;
  }
  return std::nullopt;
}

/// Decodes the whole column and compares it with the reference.
template <typename T>
std::optional<std::string> validateMaterializeAll(
    NimbleBenchTargetBase<T>& target,
    std::span<const T> reference) {
  std::vector<T> decoded(reference.size());
  target.materializeAll(decoded.data(), static_cast<uint32_t>(decoded.size()));
  return firstMismatch<T>(decoded, reference);
}

/// Decodes the whole column and checks it against data: element-wise for an
/// arm that preserves row order, as a permutation for one that does not.
template <typename T>
std::optional<std::string> validateRoundTrip(
    NimbleBenchTargetBase<T>& target,
    const EncoderEntry<T>& encoder,
    const Vector<T>& data) {
  ValidationReference<T> reference;
  if (auto mismatch =
          ValidationReference<T>::build(target, encoder, data, reference)) {
    return mismatch;
  }
  return validateMaterializeAll<T>(target, reference.values());
}

/// Issues one-row reads at each of rows, in order, and compares each.
template <typename T>
std::optional<std::string> validatePoints(
    NimbleBenchTargetBase<T>& target,
    std::span<const T> reference,
    std::span<const size_t> rows) {
  for (const size_t row : rows) {
    T value{};
    target.materializeRange(static_cast<uint32_t>(row), 1, &value);
    if (auto mismatch = firstMismatch<T>(
            std::span<const T>(&value, 1), reference.subspan(row, 1), row)) {
      return mismatch;
    }
  }
  return std::nullopt;
}

/// Issues the contiguous read [begin, begin + count) and compares it.
template <typename T>
std::optional<std::string> validateRange(
    NimbleBenchTargetBase<T>& target,
    std::span<const T> reference,
    size_t begin,
    size_t count) {
  std::vector<T> decoded(count);
  target.materializeRange(
      static_cast<uint32_t>(begin),
      static_cast<uint32_t>(count),
      decoded.data());
  return firstMismatch<T>(decoded, reference.subspan(begin, count), begin);
}

/// Issues the gather over ranges and compares every row it returns.
template <typename T>
std::optional<std::string> validateGather(
    NimbleBenchTargetBase<T>& target,
    std::span<const T> reference,
    std::span<const nimble::RowRange> ranges) {
  size_t rows = 0;
  for (const auto& range : ranges) {
    rows += range.numRows();
  }
  std::vector<T> decoded(rows);
  target.skipThenMaterialize(ranges, decoded.data());
  return checkGatherOutput<T>(decoded, reference, ranges);
}

/// Overwrites an output buffer before a timed loop, so that a read which
/// writes nothing cannot pass the post-timing check on rows an earlier arm
/// left behind.
template <typename T>
void poisonOutput(std::span<T> output) {
  std::memset(output.data(), 0xA5, output.size_bytes());
}

/// Counts the cells whose reads did not match the input, so a driver can exit
/// non-zero at the end and a sweep cannot publish numbers from a wrong arm.
class ValidationLedger {
 public:
  /// Whether cells are checked at all. --mlidc_validate=false turns checking
  /// off for profiling runs; every row then reports validated=0.
  bool enabled() const {
    return FLAGS_mlidc_validate;
  }

  /// Runs check, which returns a mismatch description or nullopt, and
  /// returns whether it passed. A failure is counted and reported. A check
  /// that throws has failed too: a read that cannot complete is as wrong as
  /// one that returns the wrong rows.
  template <typename Check>
  bool check(
      std::string_view encoder,
      std::string_view dataset,
      std::string_view cell,
      Check&& check) {
    std::optional<std::string> mismatch;
    try {
      mismatch = check();
    } catch (const std::exception& e) {
      mismatch = fmt::format("read threw: {}", e.what());
    }
    if (!mismatch.has_value()) {
      return true;
    }
    ++failures_;
    std::cerr << "  [VALIDATE FAIL] " << encoder << " / " << dataset;
    if (!cell.empty()) {
      std::cerr << " / " << cell;
    }
    std::cerr << ": " << *mismatch << "\n";
    return false;
  }

  int failures() const {
    return failures_;
  }

  /// The process exit code: 0, or 2 when any cell failed validation.
  int exitCode() const {
    if (failures_ == 0) {
      return 0;
    }
    std::cerr << failures_ << " validation failure(s)\n";
    return 2;
  }

 private:
  int failures_{0};
};

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
