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

// Golden digests of the options-off fixture files (OptionsOffFixtures.h), as
// written by the writer before the typed-lanes options existed. The files carry
// no writer metadata (fixtureOptions() leaves WriterOptions::metadata empty),
// so the digests don't depend on the host or the binary that writes them.
//
// How they are made: builds of that earlier writer and of this one, in
// mode/opt and in mode/dev, write every fixture and print fileDigest() of each
// file. The four lists must be identical; that list is pasted here. Regenerate
// them the same way only if a deliberate format change alters the bytes of
// files written with every typed-lanes option off.

#include <array>
#include <cstdint>
#include <string_view>

namespace facebook::nimble::test::options_off {

/// The digest of one fixture file.
struct FixtureGolden {
  /// Fixture::name.
  std::string_view fixture;

  /// fileDigest() of the file.
  uint64_t digest;
};

/// False: the digests below are generated. The tests that compare against them
/// run, and a missing entry fails.
inline constexpr bool kGoldensPending{false};

/// One entry per fixture in allFixtures(), written with fixtureOptions() and no
/// typed-lanes option touched.
inline constexpr std::array<FixtureGolden, 28> kFixtureGoldens{{
    {"freshDefaultFactors.uncompressed", 0xaa84b4e1116e0949},
    {"freshFixedBitWidth.uncompressed", 0x829bdc0f8003cea2},
    {"selectionCache.uncompressed", 0xd203a5d141291ca6},
    {"layoutTree.uncompressed", 0x829bdc0f8003cea2},
    {"replayFallback.uncompressed", 0x829bdc0f8003cea2},
    {"sharedDictionary.uncompressed", 0xa104903489ac1769},
    {"pinnedEncodings.uncompressed", 0xd28b1687e9b3d5f8},
    {"freshDefaultFactors.uncompressed.exactBits", 0x3291fbf082c07721},
    {"freshFixedBitWidth.uncompressed.exactBits", 0x2c054ba3ca62c9fd},
    {"selectionCache.uncompressed.exactBits", 0x589852bd00bdf699},
    {"layoutTree.uncompressed.exactBits", 0x2c054ba3ca62c9fd},
    {"replayFallback.uncompressed.exactBits", 0x2c054ba3ca62c9fd},
    {"sharedDictionary.uncompressed.exactBits", 0x4a622d8a10493289},
    {"pinnedEncodings.uncompressed.exactBits", 0x432b6cc454f753f5},
    {"freshDefaultFactors.zstdBelowMinSize", 0xaa84b4e1116e0949},
    {"freshFixedBitWidth.zstdBelowMinSize", 0x829bdc0f8003cea2},
    {"selectionCache.zstdBelowMinSize", 0xd203a5d141291ca6},
    {"layoutTree.zstdBelowMinSize", 0x829bdc0f8003cea2},
    {"replayFallback.zstdBelowMinSize", 0x829bdc0f8003cea2},
    {"sharedDictionary.zstdBelowMinSize", 0xa104903489ac1769},
    {"pinnedEncodings.zstdBelowMinSize", 0xd28b1687e9b3d5f8},
    {"freshDefaultFactors.zstdBelowMinSize.exactBits", 0x3291fbf082c07721},
    {"freshFixedBitWidth.zstdBelowMinSize.exactBits", 0x2c054ba3ca62c9fd},
    {"selectionCache.zstdBelowMinSize.exactBits", 0x589852bd00bdf699},
    {"layoutTree.zstdBelowMinSize.exactBits", 0x2c054ba3ca62c9fd},
    {"replayFallback.zstdBelowMinSize.exactBits", 0x2c054ba3ca62c9fd},
    {"sharedDictionary.zstdBelowMinSize.exactBits", 0x4a622d8a10493289},
    {"pinnedEncodings.zstdBelowMinSize.exactBits", 0x432b6cc454f753f5},
}};

} // namespace facebook::nimble::test::options_off
