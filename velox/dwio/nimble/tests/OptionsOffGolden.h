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

// The golden digests of the options-off fixture files (OptionsOffFixtures.h):
// fileDigest() of each file as written by the writer before the typed-lanes
// options existed. The digests and the tests that compare with them are
// internal-only (fb/OptionsOffFbTest.cpp). The files carry no writer metadata
// (fixtureOptions() leaves WriterOptions::metadata empty), so the digests don't
// depend on the host. They do depend on the build: a digest covers the whole
// file, whose bytes depend on library versions as well as on the writer.
// FlatBuffers, for one, serializes the footer, the stripe groups, the schema
// and the statistics. So the digests hold only for the internal build, and
// OptionsOffGoldenTest.cpp compares files within one build instead.
//
// How they are made: builds of that earlier writer and of this one, in
// mode/opt and in mode/dev, write every fixture and print fileDigest() of each
// file. The four lists must be identical; that list is pasted into the test.
// Regenerate them the same way only if a deliberate format change alters the
// bytes of files written with every typed-lanes option off.

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

/// False: the digests in fb/OptionsOffFbTest.cpp are generated. The tests that
/// compare against them run, and a missing entry fails.
inline constexpr bool kGoldensPending{false};

} // namespace facebook::nimble::test::options_off
