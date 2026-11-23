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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iterator>

#include "velox/dwio/dwrf/common/Encryption.h"
#include "velox/dwio/dwrf/writer/Writer.h"

using namespace ::testing;

namespace facebook::velox::dwrf {
TEST(TestEncodingIter, ctor) {
  auto collectNodes =
      [](const proto::StripeFooter& footer,
         const std::vector<proto::StripeEncryptionGroup>& encryptionGroups) {
        std::vector<uint32_t> nodes;
        for (auto iter = EncodingIter::begin(footer, encryptionGroups),
                  end = EncodingIter::end(footer, encryptionGroups);
             iter != end;
             ++iter) {
          nodes.push_back(iter->node());
        }
        return nodes;
      };

  proto::StripeFooter footer;
  std::vector<proto::StripeEncryptionGroup> encryptionGroups;
  // footer []
  // encryption groups []
  EXPECT_THAT(collectNodes(footer, encryptionGroups), IsEmpty());

  footer.add_encoding()->set_node(1);
  // footer [e(node=1)]
  // encryption groups []
  EXPECT_THAT(collectNodes(footer, encryptionGroups), ElementsAre(1));

  encryptionGroups.resize(2);
  // footer [e(node=1)]
  // encryption groups [[], []]
  EXPECT_THAT(collectNodes(footer, encryptionGroups), ElementsAre(1));

  encryptionGroups[1].add_encoding()->set_node(2);
  // footer [e(node=1)]
  // encryption groups [[], [e(node=2)]]
  EXPECT_THAT(collectNodes(footer, encryptionGroups), ElementsAre(1, 2));

  footer.clear_encoding();
  // footer []
  // encryption groups [[], [e(node=2)]]
  EXPECT_THAT(collectNodes(footer, encryptionGroups), ElementsAre(2));

  encryptionGroups[1].clear_encoding();
  // footer []
  // encryption groups [[], []]
  EXPECT_THAT(collectNodes(footer, encryptionGroups), IsEmpty());
}

TEST(TestEncodingIter, encodingIterBeginAndEnd) {
  proto::StripeFooter footer;
  footer.add_encoding()->set_node(1);
  std::vector<proto::StripeEncryptionGroup> encryptionGroups(2);
  encryptionGroups[0].add_encoding()->set_node(2);
  encryptionGroups[1].add_encoding()->set_node(3);

  auto iter = EncodingIter::begin(footer, encryptionGroups);
  const auto end = EncodingIter::end(footer, encryptionGroups);
  std::vector<uint32_t> nodes;
  for (; iter != end; ++iter) {
    nodes.push_back(iter->node());
  }
  EXPECT_THAT(nodes, ElementsAre(1, 2, 3));
  EXPECT_EQ(iter, end);
}

namespace {
void testEncodingIter(
    const std::vector<std::pair<uint32_t, uint32_t>>& footerEncodingNodes,
    const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>&
        encryptionGroupNodes) {
  proto::StripeFooter footer;
  std::vector<proto::StripeEncryptionGroup> encryptionGroups;
  std::vector<std::pair<uint32_t, uint32_t>> allEncoding;
  for (const auto& pair : footerEncodingNodes) {
    auto encoding = footer.add_encoding();
    encoding->set_node(pair.first);
    encoding->set_sequence(pair.second);
    allEncoding.push_back(pair);
  }

  for (const auto& groupNodes : encryptionGroupNodes) {
    proto::StripeEncryptionGroup group;
    for (const auto& pair : groupNodes) {
      auto encoding = group.add_encoding();
      encoding->set_node(pair.first);
      encoding->set_sequence(pair.second);
      allEncoding.push_back(pair);
    }
    encryptionGroups.push_back(group);
  }

  auto iter = EncodingIter::begin(footer, encryptionGroups);
  auto end = EncodingIter::end(footer, encryptionGroups);
  ASSERT_NE(iter, end);
  std::vector<std::pair<uint32_t, uint32_t>> iteratedEncodings;
  for (; iter != end; ++iter) {
    iteratedEncodings.push_back({iter->node(), iter->sequence()});
  }

  EXPECT_THAT(
      iteratedEncodings,
      ElementsAreArray(allEncoding.data(), allEncoding.size()));
}
} // namespace

TEST(TestEncodingManager, encodingIter) {
  testEncodingIter({{1, 0}}, {});
  testEncodingIter({}, {{{1, 0}}});
  testEncodingIter({{1, 0}}, {{{2, 1}, {2, 3}}});
  testEncodingIter({{2, 1}, {2, 3}}, {{{1, 0}}});
  testEncodingIter(
      {{1, 0}}, {{{2, 1}, {2, 3}}, {{3, 0}, {4, 0}, {5, 1}, {5, 2}, {5, 4}}});
  testEncodingIter(
      {{2, 1}, {2, 3}, {3, 0}, {4, 0}, {5, 1}, {5, 2}, {5, 4}}, {{{1, 0}}});
}
} // namespace facebook::velox::dwrf
