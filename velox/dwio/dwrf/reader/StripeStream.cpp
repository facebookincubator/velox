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

#include <folly/ScopeGuard.h>
#include <folly/container/F14Set.h>

#include "velox/dwio/common/exception/Exception.h"
#include "velox/dwio/dwrf/common/DecoderUtil.h"
#include "velox/dwio/dwrf/common/wrap/coded-stream-wrapper.h"
#include "velox/dwio/dwrf/reader/StripeStream.h"

namespace facebook::velox::dwrf {

using dwio::common::LogType;
using dwio::common::TypeWithId;

namespace {

template <typename IsProjected>
void findProjectedNodes(
    BitSet& projectedNodes,
    const dwio::common::TypeWithId& expected,
    const dwio::common::TypeWithId& actual,
    IsProjected&& isProjected) {
  // we don't need to perform schema compatibility check since reader should
  // have already done that before reaching here.
  // if a leaf node is projected, all the intermediate node from root to the
  // node should also be projected. So we can return as soon as seeing node that
  // is not projected
  if (!isProjected(expected.id())) {
    return;
  }
  projectedNodes.insert(actual.id());
  switch (actual.type()->kind()) {
    case TypeKind::ROW: {
      uint64_t childCount = std::min(expected.size(), actual.size());
      for (uint64_t i = 0; i < childCount; ++i) {
        findProjectedNodes(
            projectedNodes,
            *expected.childAt(i),
            *actual.childAt(i),
            std::forward<IsProjected>(isProjected));
      }
      break;
    }
    case TypeKind::ARRAY:
      findProjectedNodes(
          projectedNodes,
          *expected.childAt(0),
          *actual.childAt(0),
          std::forward<IsProjected>(isProjected));
      break;
    case TypeKind::MAP: {
      findProjectedNodes(
          projectedNodes,
          *expected.childAt(0),
          *actual.childAt(0),
          std::forward<IsProjected>(isProjected));
      findProjectedNodes(
          projectedNodes,
          *expected.childAt(1),
          *actual.childAt(1),
          std::forward<IsProjected>(isProjected));
      break;
    }
    default:
      break;
  }
}

template <typename T>
static inline void ensureCapacity(
    BufferPtr& data,
    size_t capacity,
    velox::memory::MemoryPool* pool) {
  if (!data || data->capacity() < BaseVector::byteSize<T>(capacity)) {
    data = AlignedBuffer::allocate<T>(capacity, pool);
  }
}

template <typename T>
BufferPtr readDict(
    dwio::common::IntDecoder<true>* dictReader,
    int64_t dictionarySize,
    velox::memory::MemoryPool* pool) {
  BufferPtr dictionaryBuffer = AlignedBuffer::allocate<T>(dictionarySize, pool);
  dictReader->bulkRead(dictionarySize, dictionaryBuffer->asMutable<T>());
  return dictionaryBuffer;
}
} // namespace

std::function<BufferPtr()>
StripeStreamsBase::getIntDictionaryInitializerForNode(
    const EncodingKey& encodingKey,
    uint64_t elementWidth,
    const StreamLabels& streamLabels,
    uint64_t dictionaryWidth) {
  // Create local copy for manipulation
  EncodingKey dictEncodingKey{encodingKey};
  auto dictDataSi = StripeStreamsUtil::getStreamForKind(
      *this,
      dictEncodingKey,
      proto::Stream_Kind_DICTIONARY_DATA,
      proto::orc::Stream_Kind_DICTIONARY_DATA);

  auto dictDataStream = getStream(dictDataSi, streamLabels.label(), false);
  const auto dictionarySize = format() == DwrfFormat::kDwrf
      ? getEncoding(dictEncodingKey).dictionarysize()
      : getEncodingOrc(dictEncodingKey).dictionarysize();
  // Try fetching shared dictionary streams instead.
  if (!dictDataStream) {
    // Get the label of the top level column, since this dictionary is shared by
    // the entire column.
    auto label = streamLabels.label();
    // Shouldn't be empty, but just in case
    if (!label.empty()) {
      // Ex: "/5/1759392083" -> "/5"
      label = label.substr(0, label.find('/', 1));
    }
    dictEncodingKey = EncodingKey(encodingKey.node(), 0);
    dictDataSi = StripeStreamsUtil::getStreamForKind(
        *this,
        dictEncodingKey,
        proto::Stream_Kind_DICTIONARY_DATA,
        proto::orc::Stream_Kind_DICTIONARY_DATA);
    dictDataStream = getStream(dictDataSi, label, false);
  }

  const bool dictVInts = getUseVInts(dictDataSi);
  VELOX_CHECK_NOT_NULL(dictDataStream);
  stripeDictionaryCache_->registerIntDictionary(
      dictEncodingKey,
      [dictReader = createDirectDecoder</* isSigned = */ true>(
           std::move(dictDataStream), dictVInts, elementWidth),
       dictionaryWidth,
       dictionarySize](velox::memory::MemoryPool* pool) mutable {
        return VELOX_WIDTH_DISPATCH(
            dictionaryWidth, readDict, dictReader.get(), dictionarySize, pool);
      });
  return [&dictCache = *stripeDictionaryCache_, dictEncodingKey]() {
    // If this is not flat map or if dictionary is not shared, return as is
    return dictCache.getIntDictionary(dictEncodingKey);
  };
}

void StripeStreamsImpl::loadStreams() {
  const auto& stripeFooter = *readState_->stripeMetadata->footer;

  if (selector_) {
    // HACK!!!
    //
    // Column selector filters based on requested schema (ie, table schema),
    // while we need filter based on file schema. As a result we cannot call
    // shouldReadNode directly. Instead, build projected nodes set based on node
    // id from file schema. Column selector should really be fixed to handle
    // file schema properly.
    VELOX_CHECK_NULL(projectedNodes_);
    projectedNodes_ = std::make_shared<BitSet>(0);
    auto expected = selector_->getSchemaWithId();
    auto actual = readState_->readerBase->schemaWithId();
    findProjectedNodes(
        *projectedNodes_, *expected, *actual, [&](uint32_t node) {
          return selector_->shouldReadNode(node);
        });
  }

  const auto addStreamDwrf = [&](const proto::Stream& stream, auto& offset) {
    if (stream.has_offset()) {
      offset = stream.offset();
    }
    if (projectedNodes_->contains(stream.node())) {
      streams_.insert_or_assign(stream, StreamInformationImpl{offset, stream});
    }

    offset += stream.length();
  };

  const auto addStreamOrc = [&](const proto::orc::Stream& stream,
                                auto& offset) {
    if (projectedNodes_->contains(stream.column())) {
      streams_.insert_or_assign(stream, StreamInformationImpl{offset, stream});
    }

    offset += stream.length();
  };

  uint64_t streamOffset{0};
  for (int i = 0; i < stripeFooter.streamsSize(); i++) {
    if (stripeFooter.format() == DwrfFormat::kDwrf) {
      addStreamDwrf(stripeFooter.streamDwrf(i), streamOffset);
    } else {
      addStreamOrc(stripeFooter.streamOrc(i), streamOffset);
    }
  }

  // update column encoding for each stream
  for (uint32_t i = 0; i < stripeFooter.columnEncodingSize(); ++i) {
    if (stripeFooter.format() == DwrfFormat::kDwrf) {
      const auto& e = stripeFooter.columnEncodingDwrf(i);
      const auto node = e.has_node() ? e.node() : i;
      if (projectedNodes_->contains(node)) {
        encodings_[{node, e.has_sequence() ? e.sequence() : 0}] = i;
      }
    } else {
      // kOrc
      if (projectedNodes_->contains(i)) {
        encodings_[{i, 0}] = i;
      }
    }
  }

  // handle encrypted columns, only supported for dwrf
  if (stripeFooter.format() == DwrfFormat::kDwrf) {
    const auto& decryptionHandler =
        *readState_->stripeMetadata->decryptionHandler;
    if (decryptionHandler.isEncrypted()) {
      VELOX_CHECK_EQ(
          decryptionHandler.getEncryptionGroupCount(),
          stripeFooter.encryptiongroupsSize());
      folly::F14FastSet<uint32_t> groupIndices;
      bits::forEachSetBit(
          projectedNodes_->bits(),
          0,
          projectedNodes_->max() + 1,
          [&](uint32_t node) {
            if (decryptionHandler.isEncrypted(node)) {
              groupIndices.insert(
                  decryptionHandler.getEncryptionGroupIndex(node));
            }
          });

      // decrypt encryption groups
      for (auto index : groupIndices) {
        const auto& group = stripeFooter.encryptiongroupsDwrf(index);
        const auto groupProto =
            readState_->readerBase
                ->readProtoFromString<proto::StripeEncryptionGroup>(
                    group,
                    std::addressof(
                        decryptionHandler.getEncryptionProviderByIndex(index)));
        streamOffset = 0;
        for (auto& stream : groupProto->streams()) {
          addStreamDwrf(stream, streamOffset);
        }
        for (auto& encoding : groupProto->encoding()) {
          VELOX_CHECK(encoding.has_node(), "node is required");
          const auto node = encoding.node();
          if (projectedNodes_->contains(node)) {
            decryptedEncodings_[{
                node, encoding.has_sequence() ? encoding.sequence() : 0}] =
                encoding;
          }
        }
      }
    }
  }
}

std::unique_ptr<dwio::common::SeekableInputStream>
StripeStreamsImpl::getCompressedStream(
    const DwrfStreamIdentifier& si,
    std::string_view label) const {
  const auto& info = getStreamInfo(si);

  std::unique_ptr<dwio::common::SeekableInputStream> streamInput;
  if (isIndexStream(si.kind())) {
    streamInput = getIndexStreamFromCache(info);
  }

  if (!streamInput) {
    streamInput = readState_->stripeMetadata->stripeInput->enqueue(
        {info.getOffset() + stripeStart_, info.getLength(), label}, &si);
  }

  VELOX_CHECK_NOT_NULL(streamInput, " Stream can't be read", si.toString());
  return streamInput;
}

folly::F14FastMap<uint32_t, std::vector<uint32_t>>
StripeStreamsImpl::getEncodingKeys() const {
  VELOX_CHECK_EQ(
      decryptedEncodings_.size(),
      0,
      "Not supported for reader with encryption");

  folly::F14FastMap<uint32_t, std::vector<uint32_t>> encodingKeys;
  for (const auto& kv : encodings_) {
    const auto ek = kv.first;
    encodingKeys[ek.node()].push_back(ek.sequence());
  }

  return encodingKeys;
}

folly::F14FastMap<uint32_t, std::vector<DwrfStreamIdentifier>>
StripeStreamsImpl::getStreamIdentifiers() const {
  folly::F14FastMap<uint32_t, std::vector<DwrfStreamIdentifier>>
      nodeToStreamIdMap;

  for (const auto& item : streams_) {
    nodeToStreamIdMap[item.first.encodingKey().node()].push_back(item.first);
  }
  return nodeToStreamIdMap;
}

std::unique_ptr<dwio::common::SeekableInputStream> StripeStreamsImpl::getStream(
    const DwrfStreamIdentifier& si,
    std::string_view label,
    bool /*throwIfNotFound*/) const {
  // If not found, return an empty {}
  const auto& info = getStreamInfo(si, /*throwIfNotFound=*/false);
  if (!info.valid()) {
    // Stream not found.
    return {};
  }

  std::unique_ptr<dwio::common::SeekableInputStream> streamInput;
  if (isIndexStream(si.kind())) {
    streamInput = getIndexStreamFromCache(info);
  }

  if (!streamInput) {
    streamInput = readState_->stripeMetadata->stripeInput->enqueue(
        {info.getOffset() + stripeStart_, info.getLength(), label}, &si);
  }

  if (!streamInput) {
    return streamInput;
  }

  const auto streamDebugInfo =
      fmt::format("Stripe {} Stream {}", stripeIndex_, si.toString());
  return readState_->readerBase->createDecompressedStream(
      std::move(streamInput),
      streamDebugInfo,
      getDecrypter(si.encodingKey().node()));
}

uint32_t StripeStreamsImpl::visitStreamsOfNode(
    uint32_t node,
    std::function<void(const StreamInformation&)> visitor) const {
  uint32_t count = 0;
  for (auto& item : streams_) {
    if (item.first.encodingKey().node() == node) {
      visitor(item.second);
      ++count;
    }
  }
  return count;
}

bool StripeStreamsImpl::getUseVInts(const DwrfStreamIdentifier& si) const {
  const auto& info = getStreamInfo(si, false);
  if (!info.valid()) {
    return true;
  }

  return info.getUseVInts();
}

std::unique_ptr<dwio::common::SeekableInputStream>
StripeStreamsImpl::getIndexStreamFromCache(
    const StreamInformation& info) const {
  auto& metadataCache = readState_->readerBase->metadataCache();
  if (!metadataCache) {
    return nullptr;
  }

  auto indexBase = metadataCache->get(StripeCacheMode::INDEX, stripeIndex_);
  if (!indexBase) {
    return nullptr;
  }

  const auto offset = info.getOffset();
  const auto length = info.getLength();
  if (auto* cacheInput =
          dynamic_cast<dwio::common::CacheInputStream*>(indexBase.get())) {
    cacheInput->SkipInt64(offset);
    cacheInput->setRemainingBytes(length);
    return indexBase;
  }

  const void* start;
  {
    int32_t ignored;
    const bool ret = indexBase->Next(&start, &ignored);
    VELOX_CHECK(ret, "Failed to read index");
  }
  return std::make_unique<dwio::common::SeekableArrayInputStream>(
      static_cast<const char*>(start) + offset, length);
}

void StripeStreamsImpl::loadReadPlan() {
  VELOX_CHECK(!readPlanLoaded_, "only load read plan once!");
  SCOPE_EXIT {
    readPlanLoaded_ = true;
  };

  auto* input = readState_->stripeMetadata->stripeInput;
  input->load(LogType::STREAM_BUNDLE);
}

} // namespace facebook::velox::dwrf
