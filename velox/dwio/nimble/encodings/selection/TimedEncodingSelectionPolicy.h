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
#include <optional>
#include <span>

#include "folly/Synchronized.h"
#include "velox/common/time/CpuWallTimer.h"
#include "velox/dwio/nimble/common/DataTypeDispatch.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble {

/// Wraps an encoding selection policy and adds the CPU and wall time of each
/// of its select() calls to a shared timing. Nested policies it creates are
/// wrapped the same way. A nested encode starts only after its parent's
/// select() returns, so no time is counted twice. Compression trials run while
/// encoding, after selection, so they are not counted.
template <typename T>
class TimedEncodingSelectionPolicy : public EncodingSelectionPolicy<T> {
 public:
  using physicalType = typename TypeTraits<T>::physicalType;

  /// Adds the selection time of 'policy' to 'selectionTiming', which must
  /// outlive this policy and every nested policy it creates.
  TimedEncodingSelectionPolicy(
      std::unique_ptr<EncodingSelectionPolicy<T>> policy,
      folly::Synchronized<velox::CpuWallTiming>& selectionTiming)
      : policy_{std::move(policy)}, selectionTiming_{selectionTiming} {}

  EncodingSelectionResult select(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) override {
    return timed([&] { return policy_->select(values, statistics, options); });
  }

  std::optional<EncodingSelectionResult> select(
      std::span<const physicalType> sampleValues,
      const Statistics<physicalType>& statistics,
      size_t numValues,
      const Encoding::Options& options) override {
    return timed([&] {
      return policy_->select(sampleValues, statistics, numValues, options);
    });
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> values,
      std::span<const bool> nulls,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) override {
    return timed([&] {
      return policy_->selectNullable(values, nulls, statistics, options);
    });
  }

  bool hasAlpLikeCandidates() const override {
    return policy_->hasAlpLikeCandidates();
  }

 protected:
  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType parentEncodingType,
      NestedEncodingIdentifier nestedEncodingIdentifier,
      DataType nestedDataType) override {
    NIMBLE_RETURN_BY_DATA_TYPE(
        nestedDataType,
        NestedT,
        createTimedNested<NestedT>(
            parentEncodingType, nestedEncodingIdentifier));
  }

 private:
  template <typename NestedT>
  std::unique_ptr<EncodingSelectionPolicyBase> createTimedNested(
      EncodingType parentEncodingType,
      NestedEncodingIdentifier nestedEncodingIdentifier) {
    return std::make_unique<TimedEncodingSelectionPolicy<NestedT>>(
        unique_ptr_cast<EncodingSelectionPolicy<NestedT>>(
            policy_->template create<NestedT>(
                parentEncodingType, nestedEncodingIdentifier)),
        selectionTiming_);
  }

  template <typename Select>
  auto timed(Select&& select) {
    velox::DeltaCpuWallTimer timer{[this](const velox::CpuWallTiming& timing) {
      selectionTiming_.wlock()->add(timing);
    }};
    return select();
  }

  // Policy whose selections are timed.
  const std::unique_ptr<EncodingSelectionPolicy<T>> policy_;
  // Shared by every policy of one writer, across encoding threads.
  folly::Synchronized<velox::CpuWallTiming>& selectionTiming_;
};

} // namespace facebook::nimble
