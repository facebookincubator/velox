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

/// Unit tests for the backend error summary shared by every whole-batch
/// failure fan-out (the RPC operator, the MetaGen batch client, and the
/// embedding batch path).
///
/// A backend whose thrift method declares no `throws` clause reports a
/// rejection as an untyped TApplicationException whose message is one useful
/// sentence followed by the server's stack trace. Copying that into every row
/// of a failed batch puts thousands of characters of trace in the output
/// column and allocates one copy per row.

#include "velox/exec/rpc/BackendErrorSummary.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox::exec::rpc {
namespace {

// The prefix a caller puts in front of the summarized backend text.
const std::string kErrorPrefix = "[RPC_BATCH] batch error: ";

// The one useful sentence a rejected MetaGen batch produces. 107 characters,
// matching the first line of the payload measured on the fleet.
const std::string kBackendSentence =
    "apache::thrift::TApplicationException: you must supply a metagen key in "
    "the auth token field of the request";

// Builds a payload shaped like the measured backend rejection: the sentence
// followed by 50 stack frames, each starting at the beginning of its own line
// (51 lines in total). The frames are NOT separated from the message by a
// space, so a " #0 " marker never appears — the newline is the only boundary.
std::string makeServerTracePayload() {
  std::string payload = kBackendSentence;
  for (int frame = 0; frame < 50; ++frame) {
    payload += "\n#" + std::to_string(frame) +
        " /www/flib/gen_ai/metagen/"
        "TMetaGenAsyncHandlerBatchDialogCompletion.php(" +
        std::to_string(100 + frame) + "): batchDialogCompletion()";
  }
  return payload;
}

TEST(BackendErrorSummaryTest, dropsServerTraceFromEveryRow) {
  const std::string rawError = makeServerTracePayload();
  // The fixture is the shape the fan-out has to survive: one sentence, then
  // thousands of characters of trace.
  ASSERT_GT(rawError.size(), 5'000u);

  const auto responses = makeBatchErrorResponses(4, kErrorPrefix, rawError);

  ASSERT_THAT(responses, testing::SizeIs(4));
  for (const auto& response : responses) {
    ASSERT_TRUE(response.hasError());
    EXPECT_EQ(response.errorKind, velox::rpc::RPCErrorKind::kBackendError);
    // The sentence survives; not one frame of the trace does.
    EXPECT_THAT(*response.error, testing::HasSubstr(kBackendSentence));
    EXPECT_THAT(*response.error, testing::Not(testing::HasSubstr(".php")));
    EXPECT_THAT(*response.error, testing::Not(testing::HasSubstr("#0")));
    EXPECT_LE(
        response.error->size() - kErrorPrefix.size(), kMaxBackendErrorBytes);
  }
  // Every row carries the same summarized text.
  EXPECT_EQ(*responses.front().error, *responses.back().error);
}

TEST(BackendErrorSummaryTest, shortErrorPassedThroughUnchanged) {
  const std::string rawError = "connection reset by peer";
  const auto responses = makeBatchErrorResponses(1, kErrorPrefix, rawError);

  ASSERT_THAT(responses, testing::SizeIs(1));
  ASSERT_TRUE(responses[0].hasError());
  EXPECT_EQ(*responses[0].error, kErrorPrefix + rawError);
}

TEST(BackendErrorSummaryTest, trailingNewlineIsNotCalledTruncation) {
  // Nothing was dropped but the newline the backend ended its message with.
  // Marking the row truncated would send a reader to the log for text that is
  // not there, which is what makes the marker worth trusting elsewhere.
  const auto responses =
      makeBatchErrorResponses(1, kErrorPrefix, "connection reset by peer\n");

  ASSERT_THAT(responses, testing::SizeIs(1));
  ASSERT_TRUE(responses[0].hasError());
  EXPECT_EQ(*responses[0].error, kErrorPrefix + "connection reset by peer");
}

TEST(BackendErrorSummaryTest, leadingNewlineKeepsTheFirstRealLine) {
  // A message that opens with a newline has an empty first line, so cutting at
  // it would leave the row carrying the prefix and the marker and no
  // diagnostic text at all.
  const auto responses = makeBatchErrorResponses(
      1,
      kErrorPrefix,
      "\n" + kBackendSentence + "\n#0 handler.php(100): call()");

  ASSERT_THAT(responses, testing::SizeIs(1));
  ASSERT_TRUE(responses[0].hasError());
  EXPECT_THAT(*responses[0].error, testing::HasSubstr(kBackendSentence));
  EXPECT_THAT(*responses[0].error, testing::Not(testing::HasSubstr(".php")));
}

TEST(BackendErrorSummaryTest, truncationDoesNotSplitAMultibyteCharacter) {
  // A backend that echoes the rejected prompt back is not restricted to ASCII,
  // and the row lands in a UTF-8 VARCHAR column. The cap below falls between
  // the two bytes of the 'e' with an acute accent, which a byte-index cut
  // would leave half of in the column.
  const std::string rawError = "prompt rejected: café is not a model";
  ASSERT_EQ(rawError.find("é"), 20u);
  ASSERT_GT(rawError.size(), 36u);

  // 36 - 15 marker bytes cuts at byte 21, the trailing byte of the accent.
  EXPECT_EQ(
      summarizeBackendError(rawError, 36),
      "prompt rejected: caf... (truncated)");
}

TEST(BackendErrorSummaryTest, resultNeverExceedsTheRequestedCap) {
  // The elision marker is appended after the cut, so the kept text has to
  // leave room for it. Computing the cut against the full cap instead reads
  // as correct and overruns by the length of the marker on every truncated
  // row, which is the one thing this function exists to bound.
  for (const size_t maxLength : {16UL, 32UL, 64UL, 256UL}) {
    EXPECT_LE(
        summarizeBackendError(std::string(4'000, 'x'), maxLength).size(),
        maxLength);
  }
}

TEST(BackendErrorSummaryTest, capsSingleLineErrorWithNoNewline) {
  // No newline to cut at, so the hard cap is the only thing that can bound it.
  const std::string rawError(4'000, 'x');
  const auto responses = makeBatchErrorResponses(1, kErrorPrefix, rawError);

  ASSERT_THAT(responses, testing::SizeIs(1));
  ASSERT_TRUE(responses[0].hasError());
  const std::string& error = *responses[0].error;
  ASSERT_THAT(error, testing::StartsWith(kErrorPrefix));
  EXPECT_THAT(error, testing::HasSubstr("truncated"));
  EXPECT_LE(error.size() - kErrorPrefix.size(), kMaxBackendErrorBytes);
}

TEST(BackendErrorSummaryTest, capsTraceAppendedToTheSameLine) {
  // A backend that appends its frames to the message instead of starting a new
  // line leaves no newline to cut at. Nothing tries to recognize a frame
  // marker; the cap alone has to bound the row.
  std::string rawError = kBackendSentence;
  for (int frame = 0; frame < 50; ++frame) {
    rawError += " #" + std::to_string(frame) + " handler.php(100): call()";
  }
  const auto responses = makeBatchErrorResponses(1, kErrorPrefix, rawError);

  ASSERT_THAT(responses, testing::SizeIs(1));
  ASSERT_TRUE(responses[0].hasError());
  EXPECT_LE(
      responses[0].error->size() - kErrorPrefix.size(), kMaxBackendErrorBytes);
}

TEST(BackendErrorSummaryTest, capIsTheCallerSuppliedLength) {
  // Both inputs are longer than the cap, so both go through the cap rather
  // than the unchanged-passthrough branch, and the result fills the caller's
  // budget exactly.
  const std::string rawError(200, 'x');

  EXPECT_EQ(summarizeBackendError(rawError, 32).size(), 32u);
  EXPECT_EQ(summarizeBackendError(rawError, 100).size(), 100u);
}

TEST(BackendErrorSummaryTest, OLD_FORM_capIsTheCallerSuppliedLength) {
  const std::string rawError(100, 'x');

  EXPECT_EQ(summarizeBackendError(rawError, 32).size(), 32u);
  EXPECT_EQ(summarizeBackendError(rawError, 100).size(), 100u);
}

} // namespace
} // namespace facebook::velox::exec::rpc
