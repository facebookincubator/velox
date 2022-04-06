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

#pragma once

namespace facebook::velox::dwio::common {

struct StripeProgress {
  explicit StripeProgress() = default;
  StripeProgress(
      int64_t stripeIndex,
      int64_t totalMemoryUsage,
      int64_t stripeSizeEstimate)
      : stripeIndex{stripeIndex},
        totalMemoryUsage{totalMemoryUsage},
        stripeSizeEstimate{stripeSizeEstimate} {}

  int64_t stripeIndex{0};
  int64_t totalMemoryUsage{0};
  // hide first stripe special case in customer side.
  int64_t stripeSizeEstimate{0};

  bool compare(const StripeProgress& other) const {
    return stripeSizeEstimate < other.stripeSizeEstimate;
  }
};

// Specific formats can extend this interface to do additional
// checks and customize how the decisions are combined.
class IFlushPolicy {
 public:
  virtual ~IFlushPolicy() = default;
  virtual bool shouldFlush(StripeProgress&& stripeProgress) = 0;
  virtual void close() = 0;
};

} // namespace facebook::velox::dwio::common
