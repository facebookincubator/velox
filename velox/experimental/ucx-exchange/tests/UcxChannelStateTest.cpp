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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <utility>

namespace facebook::velox::ucx_exchange {
namespace {

constexpr DestinationEvent kAllDestinationEvents[] = {
    DestinationEvent::kPublished,
    DestinationEvent::kEndMarkerTaken,
    DestinationEvent::kRemoved,
    DestinationEvent::kResultsDeleted,
    DestinationEvent::kConsumerGone,
    DestinationEvent::kSendFailed,
};

TEST(UcxChannelStateTest, publishOpensUnpublished) {
  const auto decision = decideDestination(
      DestinationState::kUnpublished,
      DestinationEvent::kPublished,
      ChannelOutcome::clean());
  EXPECT_EQ(decision.next, DestinationState::kOpen);
  EXPECT_FALSE(decision.outcome.has_value());
  EXPECT_FALSE(decision.freeChunks);
  EXPECT_FALSE(decision.failTask);
}

TEST(UcxChannelStateTest, publishKeepsReleased) {
  const auto decision = decideDestination(
      DestinationState::kReleased,
      DestinationEvent::kPublished,
      ChannelOutcome::clean());
  EXPECT_EQ(decision.next, DestinationState::kReleased);
  EXPECT_FALSE(decision.outcome.has_value());
  EXPECT_FALSE(decision.freeChunks);
  EXPECT_FALSE(decision.failTask);
}

TEST(UcxChannelStateTest, endMarkerEndsCleanly) {
  const auto decision = decideDestination(
      DestinationState::kOpen,
      DestinationEvent::kEndMarkerTaken,
      ChannelOutcome::clean());
  EXPECT_EQ(decision.next, DestinationState::kEnded);
  ASSERT_TRUE(decision.outcome.has_value());
  EXPECT_EQ(*decision.outcome, ChannelOutcome::clean());
  EXPECT_FALSE(decision.freeChunks);
  EXPECT_FALSE(decision.failTask);
}

TEST(UcxChannelStateTest, removalDecidesCandidate) {
  for (const auto current :
       {DestinationState::kUnpublished, DestinationState::kOpen}) {
    SCOPED_TRACE(fmt::format("state {}", current));
    const auto decision = decideDestination(
        current,
        DestinationEvent::kRemoved,
        ChannelOutcome::producerError("x"));
    EXPECT_EQ(decision.next, DestinationState::kEnded);
    ASSERT_TRUE(decision.outcome.has_value());
    EXPECT_EQ(decision.outcome->kind, OutcomeKind::kProducerError);
    EXPECT_EQ(decision.outcome->error, "x");
    EXPECT_TRUE(decision.freeChunks);
    EXPECT_FALSE(decision.failTask);
  }
}

TEST(UcxChannelStateTest, removalKeepsEarlierDecision) {
  for (const auto current :
       {DestinationState::kEnded, DestinationState::kReleased}) {
    SCOPED_TRACE(fmt::format("state {}", current));
    const auto decision = decideDestination(
        current,
        DestinationEvent::kRemoved,
        ChannelOutcome::producerError("x"));
    EXPECT_EQ(decision.next, current);
    EXPECT_FALSE(decision.outcome.has_value());
    EXPECT_FALSE(decision.freeChunks);
    EXPECT_FALSE(decision.failTask);
  }
}

TEST(UcxChannelStateTest, deleteResultsWhileOpenDecidesCandidate) {
  const auto candidate = ChannelOutcome::producerError("discarded");
  {
    const auto decision = decideDestination(
        DestinationState::kOpen, DestinationEvent::kResultsDeleted, candidate);
    EXPECT_EQ(decision.next, DestinationState::kEnded);
    EXPECT_EQ(decision.outcome, candidate);
    EXPECT_TRUE(decision.freeChunks);
    EXPECT_FALSE(decision.failTask);
  }
  {
    const auto decision = decideDestination(
        DestinationState::kUnpublished,
        DestinationEvent::kResultsDeleted,
        candidate);
    EXPECT_EQ(decision.next, DestinationState::kReleased);
    EXPECT_EQ(decision.outcome, candidate);
    EXPECT_TRUE(decision.freeChunks);
    EXPECT_FALSE(decision.failTask);
  }
}

TEST(UcxChannelStateTest, deleteResultsAfterEndReleases) {
  const auto decision = decideDestination(
      DestinationState::kEnded,
      DestinationEvent::kResultsDeleted,
      ChannelOutcome::producerError("discarded"));
  EXPECT_EQ(decision.next, DestinationState::kReleased);
  EXPECT_FALSE(decision.outcome.has_value());
  EXPECT_TRUE(decision.freeChunks);
  EXPECT_FALSE(decision.failTask);
}

TEST(UcxChannelStateTest, consumerGoneReleasesWithoutOutcome) {
  for (const auto current :
       {DestinationState::kUnpublished,
        DestinationState::kOpen,
        DestinationState::kEnded}) {
    SCOPED_TRACE(fmt::format("state {}", current));
    const auto decision = decideDestination(
        current, DestinationEvent::kConsumerGone, ChannelOutcome::clean());
    EXPECT_EQ(decision.next, DestinationState::kReleased);
    EXPECT_FALSE(decision.outcome.has_value());
    EXPECT_TRUE(decision.freeChunks);
    EXPECT_FALSE(decision.failTask);
  }
}

TEST(UcxChannelStateTest, sendFailedFailsTask) {
  for (const auto current :
       {DestinationState::kOpen, DestinationState::kEnded}) {
    SCOPED_TRACE(fmt::format("state {}", current));
    const auto decision = decideDestination(
        current, DestinationEvent::kSendFailed, ChannelOutcome::clean());
    EXPECT_EQ(decision.next, DestinationState::kReleased);
    EXPECT_FALSE(decision.outcome.has_value());
    EXPECT_TRUE(decision.freeChunks);
    EXPECT_TRUE(decision.failTask);
  }
  {
    const auto decision = decideDestination(
        DestinationState::kReleased,
        DestinationEvent::kSendFailed,
        ChannelOutcome::clean());
    EXPECT_EQ(decision.next, DestinationState::kReleased);
    EXPECT_FALSE(decision.failTask);
  }
}

TEST(UcxChannelStateTest, releasedIgnoresEveryEvent) {
  for (const auto event : kAllDestinationEvents) {
    SCOPED_TRACE(fmt::format("event {}", event));
    const auto decision = decideDestination(
        DestinationState::kReleased, event, ChannelOutcome::producerError("x"));
    EXPECT_EQ(decision.next, DestinationState::kReleased);
    EXPECT_FALSE(decision.outcome.has_value());
    EXPECT_FALSE(decision.freeChunks);
    EXPECT_FALSE(decision.failTask);
  }
}

TEST(UcxChannelStateTest, responseOpensSource) {
  const auto decision = decideSource(
      SourceState::kRegistered,
      SourceEvent::kResponseAccepted,
      ChannelOutcome::clean());
  EXPECT_EQ(decision.next, SourceState::kOpen);
  EXPECT_FALSE(decision.outcome.has_value());
  EXPECT_FALSE(decision.tellServer);
}

TEST(UcxChannelStateTest, endDecidesOnce) {
  const auto first = ChannelOutcome::producerError("first");
  for (const auto current : {SourceState::kRegistered, SourceState::kOpen}) {
    SCOPED_TRACE(fmt::format("state {}", current));
    const auto decision = decideSource(current, SourceEvent::kEnded, first);
    EXPECT_EQ(decision.next, SourceState::kEnded);
    EXPECT_EQ(decision.outcome, first);
    EXPECT_FALSE(decision.tellServer);
  }

  const auto second = decideSource(
      SourceState::kEnded,
      SourceEvent::kEnded,
      ChannelOutcome::transportError("second"));
  EXPECT_EQ(second.next, SourceState::kEnded);
  EXPECT_FALSE(second.outcome.has_value());
  EXPECT_FALSE(second.tellServer);
}

TEST(UcxChannelStateTest, errorAfterCloseIgnored) {
  const auto decision = decideSource(
      SourceState::kReleased,
      SourceEvent::kEnded,
      ChannelOutcome::transportError("late"));
  EXPECT_EQ(decision.next, SourceState::kReleased);
  EXPECT_FALSE(decision.outcome.has_value());
  EXPECT_FALSE(decision.tellServer);
}

TEST(UcxChannelStateTest, closeOfLiveSourceTellsServer) {
  for (const auto current : {SourceState::kRegistered, SourceState::kOpen}) {
    SCOPED_TRACE(fmt::format("state {}", current));
    const auto decision = decideSource(
        current, SourceEvent::kClientClosed, ChannelOutcome::clean());
    EXPECT_EQ(decision.next, SourceState::kReleased);
    EXPECT_FALSE(decision.outcome.has_value());
    EXPECT_TRUE(decision.tellServer);
  }
}

TEST(UcxChannelStateTest, closeOfEndedSourceDoesNotTellServer) {
  const auto decision = decideSource(
      SourceState::kEnded, SourceEvent::kClientClosed, ChannelOutcome::clean());
  EXPECT_EQ(decision.next, SourceState::kReleased);
  EXPECT_FALSE(decision.outcome.has_value());
  EXPECT_FALSE(decision.tellServer);
}

TEST(UcxChannelStateTest, classifySendFailureByCause) {
  EXPECT_EQ(classifySendFailure(true, true), SendFailureCause::kNone);
  EXPECT_EQ(classifySendFailure(true, false), SendFailureCause::kNone);
  EXPECT_EQ(classifySendFailure(false, false), SendFailureCause::kConsumerGone);
  EXPECT_EQ(classifySendFailure(false, true), SendFailureCause::kSendFailed);
}

TEST(UcxChannelStateTest, outcomeConstructors) {
  EXPECT_TRUE(ChannelOutcome::clean().isClean());
  EXPECT_TRUE(ChannelOutcome::clean().error.empty());
  EXPECT_FALSE(ChannelOutcome::producerError("x").isClean());
  EXPECT_EQ(
      ChannelOutcome::transportError("x").kind, OutcomeKind::kTransportError);
  EXPECT_EQ(
      ChannelOutcome::versionMismatch("x").kind, OutcomeKind::kVersionMismatch);
  EXPECT_NE(ChannelOutcome::producerError("x"), ChannelOutcome::clean());
  EXPECT_NE(
      ChannelOutcome::producerError("x"), ChannelOutcome::transportError("x"));
}

TEST(UcxChannelStateTest, enumsFormatAsNames) {
  EXPECT_EQ(fmt::format("{}", DestinationState::kUnpublished), "UNPUBLISHED");
  EXPECT_EQ(fmt::format("{}", DestinationEvent::kSendFailed), "SEND_FAILED");
  EXPECT_EQ(fmt::format("{}", SourceState::kRegistered), "REGISTERED");
  EXPECT_EQ(fmt::format("{}", SourceEvent::kClientClosed), "CLIENT_CLOSED");
  EXPECT_EQ(
      fmt::format("{}", OutcomeKind::kVersionMismatch), "VERSION_MISMATCH");
}

TEST(UcxChannelStateTest, truncateProducerErrorKeepsShortErrors) {
  const std::string error(kMaxProducerErrorSize, 'a');
  EXPECT_EQ(truncateProducerError(error), error);
}

TEST(UcxChannelStateTest, truncateProducerErrorKeepsWholeCharacters) {
  // Returns whether 'text' consists of whole UTF-8 characters.
  const auto isWholeUtf8 = [](std::string_view text) {
    size_t i = 0;
    while (i < text.size()) {
      const auto lead = static_cast<unsigned char>(text[i]);
      const size_t length = lead < 0x80 ? 1
          : (lead >> 5) == 0x06         ? 2
          : (lead >> 4) == 0x0E         ? 3
          : (lead >> 3) == 0x1E         ? 4
                                        : 0;
      if (length == 0 || i + length > text.size()) {
        return false;
      }
      for (size_t j = 1; j < length; ++j) {
        if ((static_cast<unsigned char>(text[i + j]) & 0xC0) != 0x80) {
          return false;
        }
      }
      i += length;
    }
    return true;
  };

  // A run of two-byte characters, once at an even and once at an odd offset,
  // so that one case cuts inside a character whatever the text before it.
  for (const std::string lead : {"", "a"}) {
    SCOPED_TRACE(fmt::format("lead of {} bytes", lead.size()));
    std::string error = lead;
    for (int i = 0; i < 3'000; ++i) {
      error += "\xC3\xA9";
    }
    const auto truncated = truncateProducerError(std::move(error));
    EXPECT_THAT(truncated, testing::EndsWith("...(truncated)"));
    EXPECT_LE(truncated.size(), kMaxProducerErrorSize);
    EXPECT_TRUE(isWholeUtf8(truncated))
        << "Ends with: " << truncated.substr(truncated.size() - 20);
  }
}

} // namespace
} // namespace facebook::velox::ucx_exchange
