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

#include "velox/dwio/dwrf/writer/FlushPolicy.h"

namespace {
static constexpr size_t kNumDictionaryChecksPerStripe = 3UL;
} // namespace

namespace facebook::velox::dwrf {

DefaultFlushPolicy::DefaultFlushPolicy(
    uint64_t stripeSizeThreshold,
    uint64_t dictionarySizeThreshold)
    : stripeSizeThreshold_{stripeSizeThreshold},
      dictionarySizeThreshold_{dictionarySizeThreshold},
      dictionaryCheckIncrement_{
          stripeSizeThreshold_ / kNumDictionaryChecksPerStripe} {
  VELOX_CHECK_GT(dictionaryCheckIncrement_, 0);
  setNextDictionaryCheckThreshold();
}

void DefaultFlushPolicy::setNextDictionaryCheckThreshold(
    uint64_t stripeSizeEstimate) {
  // Add 1 so we can round up to the next increment above the current estimate.
  dictionaryCheckThreshold_ =
      bits::roundUp(stripeSizeEstimate + 1, dictionaryCheckIncrement_);
}

FlushDecision DefaultFlushPolicy::shouldFlushDictionary(
    bool flushStripe,
    bool overMemoryBudget,
    const dwio::common::StripeProgress& stripeProgress,
    int64_t dictionaryMemoryUsage) {
  if (flushStripe) {
    return FlushDecision::SKIP;
  }

  if (dictionaryMemoryUsage > dictionarySizeThreshold_) {
    return FlushDecision::FLUSH_DICTIONARY;
  }
  if (stripeIndex_ < stripeProgress.stripeIndex) {
    // Reset dictionary check threshold for the new stripe.
    setNextDictionaryCheckThreshold();
    stripeIndex_ = stripeProgress.stripeIndex;
  }
  if (stripeProgress.stripeSizeEstimate >= dictionaryCheckThreshold_) {
    setNextDictionaryCheckThreshold(stripeProgress.stripeSizeEstimate);
    return FlushDecision::CHECK_DICTIONARY;
  }
  return FlushDecision::SKIP;
}

FlushDecision DefaultFlushPolicy::shouldFlushDictionary(
    bool flushStripe,
    bool overMemoryBudget,
    const dwio::common::StripeProgress& stripeProgress,
    const WriterContext& context) {
  return shouldFlushDictionary(
      flushStripe,
      overMemoryBudget,
      stripeProgress,
      context.getMemoryUsage(MemoryUsageCategory::DICTIONARY));
}

RowsPerStripeFlushPolicy::RowsPerStripeFlushPolicy(
    std::vector<uint64_t> rowsPerStripe)
    : rowsPerStripe_{std::move(rowsPerStripe)} {
  // Note: Vector will be empty for empty files.
  for (auto i = 0; i < rowsPerStripe_.size(); i++) {
    DWIO_ENSURE_GT(
        rowsPerStripe_.at(i),
        0,
        "More than 0 rows expected in the stripe at ",
        i,
        folly::join(",", rowsPerStripe_));
  }
}

// We can throw if writer reported the incoming write to be over memory budget.
bool RowsPerStripeFlushPolicy::shouldFlush(
    const dwio::common::StripeProgress& stripeProgress) {
  const auto stripeIndex = stripeProgress.stripeIndex;
  const auto stripeRowCount = stripeProgress.stripeRowCount;
  DWIO_ENSURE_LT(
      stripeIndex,
      rowsPerStripe_.size(),
      "Stripe index is bigger than expected");

  DWIO_ENSURE_LE(
      stripeRowCount,
      rowsPerStripe_.at(stripeIndex),
      "More rows in Stripe than expected ",
      stripeIndex);

  if ((stripeIndex + 1) == rowsPerStripe_.size()) {
    // Last Stripe is always flushed at the time of close.
    return false;
  }

  return stripeRowCount == rowsPerStripe_.at(stripeIndex);
}
} // namespace facebook::velox::dwrf
