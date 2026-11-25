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
#include "velox/experimental/ucx-exchange/UcxChannelState.h"

#include <string_view>
#include <utility>

#include "folly/container/F14Map.h"
#include "velox/common/EnumDefine.h"

namespace facebook::velox::ucx_exchange {

namespace {
const folly::F14FastMap<OutcomeKind, std::string_view>& outcomeKindNames() {
  static const folly::F14FastMap<OutcomeKind, std::string_view> kNames = {
      {OutcomeKind::kClean, "CLEAN"},
      {OutcomeKind::kProducerError, "PRODUCER_ERROR"},
      {OutcomeKind::kTransportError, "TRANSPORT_ERROR"},
      {OutcomeKind::kVersionMismatch, "VERSION_MISMATCH"},
  };
  return kNames;
}

const folly::F14FastMap<DestinationState, std::string_view>&
destinationStateNames() {
  static const folly::F14FastMap<DestinationState, std::string_view> kNames = {
      {DestinationState::kUnpublished, "UNPUBLISHED"},
      {DestinationState::kOpen, "OPEN"},
      {DestinationState::kEnded, "ENDED"},
      {DestinationState::kReleased, "RELEASED"},
  };
  return kNames;
}

const folly::F14FastMap<DestinationEvent, std::string_view>&
destinationEventNames() {
  static const folly::F14FastMap<DestinationEvent, std::string_view> kNames = {
      {DestinationEvent::kPublished, "PUBLISHED"},
      {DestinationEvent::kEndMarkerTaken, "END_MARKER_TAKEN"},
      {DestinationEvent::kRemoved, "REMOVED"},
      {DestinationEvent::kResultsDeleted, "RESULTS_DELETED"},
      {DestinationEvent::kConsumerGone, "CONSUMER_GONE"},
      {DestinationEvent::kSendFailed, "SEND_FAILED"},
  };
  return kNames;
}

const folly::F14FastMap<SourceState, std::string_view>& sourceStateNames() {
  static const folly::F14FastMap<SourceState, std::string_view> kNames = {
      {SourceState::kRegistered, "REGISTERED"},
      {SourceState::kOpen, "OPEN"},
      {SourceState::kEnded, "ENDED"},
      {SourceState::kReleased, "RELEASED"},
  };
  return kNames;
}

const folly::F14FastMap<SourceEvent, std::string_view>& sourceEventNames() {
  static const folly::F14FastMap<SourceEvent, std::string_view> kNames = {
      {SourceEvent::kResponseAccepted, "RESPONSE_ACCEPTED"},
      {SourceEvent::kEnded, "ENDED"},
      {SourceEvent::kClientClosed, "CLIENT_CLOSED"},
  };
  return kNames;
}
} // namespace

VELOX_DEFINE_ENUM_NAME(OutcomeKind, outcomeKindNames);
VELOX_DEFINE_ENUM_NAME(DestinationState, destinationStateNames);
VELOX_DEFINE_ENUM_NAME(DestinationEvent, destinationEventNames);
VELOX_DEFINE_ENUM_NAME(SourceState, sourceStateNames);
VELOX_DEFINE_ENUM_NAME(SourceEvent, sourceEventNames);

ChannelOutcome ChannelOutcome::clean() {
  return {};
}

ChannelOutcome ChannelOutcome::producerError(std::string error) {
  return {OutcomeKind::kProducerError, std::move(error)};
}

ChannelOutcome ChannelOutcome::transportError(std::string error) {
  return {OutcomeKind::kTransportError, std::move(error)};
}

ChannelOutcome ChannelOutcome::versionMismatch(std::string error) {
  return {OutcomeKind::kVersionMismatch, std::move(error)};
}

bool ChannelOutcome::isClean() const {
  return kind == OutcomeKind::kClean;
}

DestinationDecision decideDestination(
    DestinationState current,
    DestinationEvent event,
    const ChannelOutcome& candidate) {
  const bool unpublished = current == DestinationState::kUnpublished;
  const bool open = current == DestinationState::kOpen;
  const bool ended = current == DestinationState::kEnded;

  switch (event) {
    case DestinationEvent::kPublished:
      if (unpublished) {
        return {DestinationState::kOpen};
      }
      break;
    case DestinationEvent::kEndMarkerTaken:
      if (open) {
        return {DestinationState::kEnded, ChannelOutcome::clean()};
      }
      break;
    case DestinationEvent::kRemoved:
      if (unpublished || open) {
        return {
            DestinationState::kEnded,
            candidate,
            /*freeChunks=*/true};
      }
      break;
    case DestinationEvent::kResultsDeleted:
      if (unpublished) {
        return {
            DestinationState::kReleased,
            candidate,
            /*freeChunks=*/true};
      }
      if (open) {
        return {
            DestinationState::kEnded,
            candidate,
            /*freeChunks=*/true};
      }
      if (ended) {
        return {
            DestinationState::kReleased,
            std::nullopt,
            /*freeChunks=*/true};
      }
      break;
    case DestinationEvent::kConsumerGone:
      if (unpublished || open || ended) {
        return {
            DestinationState::kReleased,
            std::nullopt,
            /*freeChunks=*/true};
      }
      break;
    case DestinationEvent::kSendFailed:
      if (open || ended) {
        return {
            DestinationState::kReleased,
            std::nullopt,
            /*freeChunks=*/true,
            /*failTask=*/true};
      }
      break;
  }
  return {current};
}

SourceDecision decideSource(
    SourceState current,
    SourceEvent event,
    const ChannelOutcome& outcome) {
  const bool registered = current == SourceState::kRegistered;
  const bool open = current == SourceState::kOpen;
  const bool ended = current == SourceState::kEnded;

  switch (event) {
    case SourceEvent::kResponseAccepted:
      if (registered) {
        return {SourceState::kOpen};
      }
      break;
    case SourceEvent::kEnded:
      if (registered || open) {
        return {SourceState::kEnded, outcome};
      }
      break;
    case SourceEvent::kClientClosed:
      if (registered || open) {
        return {
            SourceState::kReleased,
            std::nullopt,
            /*tellServer=*/true};
      }
      if (ended) {
        return {SourceState::kReleased};
      }
      break;
  }
  return {current};
}

std::string truncateProducerError(std::string error) {
  static constexpr std::string_view kTruncated{"...(truncated)"};
  if (error.size() > kMaxProducerErrorSize) {
    auto end = kMaxProducerErrorSize - kTruncated.size();
    // Back up to the start of a UTF-8 character.
    while (end > 0 && (static_cast<unsigned char>(error[end]) & 0xC0) == 0x80) {
      --end;
    }
    error.resize(end);
    error.append(kTruncated);
  }
  return error;
}

SendFailureCause classifySendFailure(
    bool cancelledByServer,
    bool endpointAlive) {
  if (cancelledByServer) {
    return SendFailureCause::kNone;
  }
  return endpointAlive ? SendFailureCause::kSendFailed
                       : SendFailureCause::kConsumerGone;
}

} // namespace facebook::velox::ucx_exchange
