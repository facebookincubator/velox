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

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "velox/common/EnumDeclare.h"

namespace facebook::velox::ucx_exchange {

/// How a channel ended.
enum class OutcomeKind : uint8_t {
  /// The consumer received every chunk and the end marker.
  kClean,
  /// The producer ended early and says why.
  kProducerError,
  /// The connection, a send or a receive failed, or a message could not be
  /// decoded.
  kTransportError,
  /// The peers run different protocol versions.
  kVersionMismatch,
};

VELOX_DECLARE_ENUM_NAME(OutcomeKind);

/// The decided end of a channel half. The first decision for a channel wins.
struct ChannelOutcome {
  OutcomeKind kind{OutcomeKind::kClean};

  /// The error message. Empty if and only if 'kind' is kClean.
  std::string error;

  static ChannelOutcome clean();
  static ChannelOutcome producerError(std::string error);
  static ChannelOutcome transportError(std::string error);
  static ChannelOutcome versionMismatch(std::string error);

  bool isClean() const;

  bool operator==(const ChannelOutcome&) const = default;
};

/// The state of one destination in the producer's output queue.
enum class DestinationState : uint8_t {
  /// A reader waits, but the destination is not announced yet.
  kUnpublished,
  /// Chunks flow to its reader.
  kOpen,
  /// The end marker, or the error, is decided and queued.
  kEnded,
  /// Its results are deleted and nothing more is sent.
  kReleased,
};

VELOX_DECLARE_ENUM_NAME(DestinationState);

/// An event that moves a destination through its states.
enum class DestinationEvent : uint8_t {
  /// The destination is announced to its reader.
  kPublished,
  /// The server took the end marker from the queue.
  kEndMarkerTaken,
  /// The producer's task was removed.
  kRemoved,
  /// The results of the destination were deleted.
  kResultsDeleted,
  /// The consumer sent the close message, or its endpoint closed.
  kConsumerGone,
  /// A send failed while the endpoint stayed open.
  kSendFailed,
};

VELOX_DECLARE_ENUM_NAME(DestinationEvent);

/// What an event does to a destination.
struct DestinationDecision {
  /// The state after the event.
  DestinationState next;

  /// Set only when this event decides the destination's outcome.
  std::optional<ChannelOutcome> outcome;

  /// True if the destination's queued chunks are freed.
  bool freeChunks{false};

  /// True if the producer task is failed with the send's error.
  bool failTask{false};
};

/// Applies 'event' to a destination in state 'current'. A pair that the outcome
/// table does not list leaves the state unchanged with no outcome, so the first
/// decision wins. 'candidate' is the outcome that kRemoved and kResultsDeleted
/// propose; the other events ignore it.
DestinationDecision decideDestination(
    DestinationState current,
    DestinationEvent event,
    const ChannelOutcome& candidate);

/// The state of one source in the consumer's exchange queue.
enum class SourceState : uint8_t {
  /// The client added the source.
  kRegistered,
  /// The handshake response arrived.
  kOpen,
  /// The source reported an end: clean, the producer's error or a transport
  /// error.
  kEnded,
  /// The client closed.
  kReleased,
};

VELOX_DECLARE_ENUM_NAME(SourceState);

/// An event that moves a source through its states.
enum class SourceEvent : uint8_t {
  /// The handshake response arrived and carries the same protocol version.
  kResponseAccepted,
  /// The source reports its end, with the outcome the call carries.
  kEnded,
  /// The client closed the source.
  kClientClosed,
};

VELOX_DECLARE_ENUM_NAME(SourceEvent);

/// What an event does to a source.
struct SourceDecision {
  /// The state after the event.
  SourceState next;

  /// Set only when this event decides the source's outcome.
  std::optional<ChannelOutcome> outcome;

  /// True if the producer's server must be told with the close message.
  bool tellServer{false};
};

/// Applies 'event' to a source in state 'current'. A pair that the outcome
/// table does not list leaves the state unchanged with no outcome, so the first
/// decision wins. 'outcome' is the outcome that kEnded reports; the other
/// events ignore it.
SourceDecision decideSource(
    SourceState current,
    SourceEvent event,
    const ChannelOutcome& outcome);

/// The longest producer error, in bytes, that travels to a consumer.
inline constexpr size_t kMaxProducerErrorSize{4'096};

/// Returns 'error' unchanged if it is at most kMaxProducerErrorSize bytes.
/// Otherwise cuts it at the start of a UTF-8 character and appends
/// "...(truncated)", so that the result is at most kMaxProducerErrorSize bytes
/// and holds whole characters.
std::string truncateProducerError(std::string error);

/// How a server reads a send that did not complete with UCS_OK.
enum class SendFailureCause : uint8_t {
  /// The server cancelled the send itself.
  kNone,
  /// The consumer's endpoint is gone.
  kConsumerGone,
  /// The send failed while the endpoint is alive.
  kSendFailed,
};

/// Classifies a failed send. UCXX completes every in-flight send with
/// UCS_ERR_CANCELED, both when the endpoint is closed locally and when the peer
/// fails, so the status cannot tell the causes apart.
/// @param cancelledByServer True if the server closed or released itself, so it
///        started the cancel.
/// @param endpointAlive True if the endpoint has not reported an error or a
///        close.
SendFailureCause classifySendFailure(
    bool cancelledByServer,
    bool endpointAlive);

} // namespace facebook::velox::ucx_exchange

VELOX_ENUM_FORMATTER(facebook::velox::ucx_exchange::OutcomeKind);
VELOX_ENUM_FORMATTER(facebook::velox::ucx_exchange::DestinationState);
VELOX_ENUM_FORMATTER(facebook::velox::ucx_exchange::DestinationEvent);
VELOX_ENUM_FORMATTER(facebook::velox::ucx_exchange::SourceState);
VELOX_ENUM_FORMATTER(facebook::velox::ucx_exchange::SourceEvent);
