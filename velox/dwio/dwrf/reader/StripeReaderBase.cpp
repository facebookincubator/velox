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

#include "velox/dwio/dwrf/reader/StripeReaderBase.h"

namespace facebook::velox::dwrf {

using dwio::common::LogType;

const proto::StripeInformation& StripeReaderBase::loadStripe(
    uint32_t index,
    bool& preload) {
  auto& footer = reader_->getFooter();
  DWIO_ENSURE_LT(index, footer.stripes_size(), "invalid stripe index");
  auto& stripe = footer.stripes(index);
  auto& cache = reader_->getMetadataCache();

  stripeInput_.reset();
  uint64_t offset = stripe.offset();
  uint64_t length =
      stripe.indexlength() + stripe.datalength() + stripe.footerlength();
  if (reader_->getBufferedInput().isBuffered(offset, length)) {
    // if file is preloaded, return stripe is preloaded
    preload = true;
  } else {
    stripeInput_ = reader_->bufferedInputFactory().create(
        reader_->getStream(),
        reader_->getMemoryPool(),
        reader_->getDataCacheConfig());

    if (preload) {
      // If metadata cache exists, adjust read position to avoid re-reading
      // metadata sections
      if (cache) {
        if (cache->has(proto::StripeCacheMode::INDEX, index)) {
          offset += stripe.indexlength();
          length -= stripe.indexlength();
        }
        if (cache->has(proto::StripeCacheMode::FOOTER, index)) {
          length -= stripe.footerlength();
        }
      }

      stripeInput_->enqueue({offset, length});
      stripeInput_->load(LogType::STRIPE);
    }
  }

  // load stripe footer
  std::unique_ptr<SeekableInputStream> stream;
  if (cache) {
    stream = cache->get(proto::StripeCacheMode::FOOTER, index);
  }

  if (!stream) {
    if (reader_->getDataCacheConfig()) {
      stream = getStripeInput().enqueue(
          {stripe.offset() + stripe.indexlength() + stripe.datalength(),
           stripe.footerlength()});
      // It will not load anything if we hit the cache while enqueuing
      getStripeInput().load(LogType::STRIPE_FOOTER);
    } else {
      stream = getStripeInput().read(
          stripe.offset() + stripe.indexlength() + stripe.datalength(),
          stripe.footerlength(),
          LogType::STRIPE_FOOTER);
    }
  }

  // Reuse footer_'s memory to avoid expensive destruction
  if (!footer_) {
    footer_ = google::protobuf::Arena::CreateMessage<proto::StripeFooter>(
        reader_->arena());
  }

  auto streamDebugInfo = fmt::format("Stripe {} Footer ", index);
  ProtoUtils::readProtoInto<proto::StripeFooter>(
      reader_->createDecompressedStream(std::move(stream), streamDebugInfo),
      footer_);

  // refresh stripe encryption key if necessary
  loadEncryptionKeys(index);
  lastStripeIndex_ = index;

  return stripe;
}

void StripeReaderBase::loadEncryptionKeys(uint32_t index) {
  if (!handler_->isEncrypted()) {
    return;
  }

  DWIO_ENSURE_EQ(
      footer_->encryptiongroups_size(), handler_->getEncryptionGroupCount());
  auto& footer = reader_->getFooter();
  DWIO_ENSURE_LT(index, footer.stripes_size(), "invalid stripe index");

  auto& stripe = footer.stripes(index);
  // If current stripe has keys, load these keys.
  if (stripe.keymetadata_size() > 0) {
    handler_->setKeys(stripe.keymetadata());
  } else {
    // If current stripe doesn't have keys, then:
    //  1. If it's sequential read (ie. we've just finished reading one stripe
    //  and are now trying to read the stripe right after it), we can reuse the
    //  loaded keys.
    //  2. If it's not sequential read (which means we performed a skip/seek
    //  into a random stripe in the file), we need to sequentially lookup
    //  previous stripes, until we find a stripe with keys.
    DWIO_ENSURE_GT(index, 0, "first stripe must have key");
    bool isSequentialRead =
        (lastStripeIndex_ && lastStripeIndex_.value() == index - 1);
    if (!isSequentialRead) {
      uint32_t prevIndex = index - 1;
      while (true) {
        auto prev = footer.stripes(prevIndex);
        if (prev.keymetadata_size() > 0) {
          handler_->setKeys(prev.keymetadata());
          break;
        }
        DWIO_ENSURE_GE(prevIndex, 0, "key not found");
        --prevIndex;
      }
    }
  }
}

} // namespace facebook::velox::dwrf
