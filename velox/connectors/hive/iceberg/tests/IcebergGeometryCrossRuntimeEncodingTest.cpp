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

// Cross-runtime compatibility check for the internal geometry encoding.
//
// A geometry value read by the native worker crosses exchange, spill and
// coordinator boundaries and may be interpreted by Presto's Java worker, so
// this runtime's common::geospatial::GeometrySerializer output and the Java
// worker's EsriGeometrySerde output must be the same bytes for the same WKB
// input. Testing each side against WKT independently does not establish that.
//
// examples/geometry_internal_encoding_golden.tsv pins those bytes. The same
// file, with the same SHA-256 drift guard, is checked into prestodb/presto at
// presto-iceberg/src/test/resources/iceberg_v3/geometry_internal_encoding_golden.tsv,
// where TestIcebergGeometryCrossRuntimeEncoding asserts against it. If either
// implementation changes its encoding, exactly one of the two suites starts
// failing.

#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

#include <fstream>
#include <sstream>

#include <folly/Singleton.h>
#include <folly/String.h>
#include <openssl/sha.h>

#include "velox/common/geospatial/GeometrySerde.h"
#include "velox/connectors/hive/iceberg/IcebergGeometryConverter.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/functions/prestosql/types/GeometryRegistration.h"
#include "velox/functions/prestosql/types/GeometryType.h"

#define USE_UNSTABLE_GEOS_CPP_API 1
#include <geos/io/WKBWriter.h>
#include <geos/io/WKTReader.h>

namespace facebook::velox::connector::hive::iceberg {
namespace {

class IcebergGeometryCrossRuntimeEncodingTest : public test::IcebergTestBase {
 protected:
  void SetUp() override {
    test::IcebergTestBase::SetUp();
    folly::SingletonVault::singleton()->registrationComplete();
    registerGeometryType();
  }

  // ISO WKB, as an Iceberg `geometry` column stores it on disk.
  static std::string toWkb(const std::string& wkt) {
    geos::io::WKTReader wktReader;
    geos::io::WKBWriter wkbWriter;
    std::ostringstream out;
    wkbWriter.write(*wktReader.read(wkt), out);
    return out.str();
  }

  VectorPtr makeVarbinaryVector(
      const std::vector<std::optional<std::string>>& values) {
    return makeFlatVector<StringView>(
        values.size(),
        [&](vector_size_t i) {
          return values[i].has_value() ? StringView(*values[i]) : StringView();
        },
        [&](vector_size_t i) { return !values[i].has_value(); },
        VARBINARY());
  }

  static std::string fromHex(const std::string& hex) {
    VELOX_CHECK_EQ(hex.size() % 2, 0);
    std::string out;
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
      out.push_back(
          static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    return out;
  }

  static std::string toHex(StringView bytes) {
    static const char* kDigits = "0123456789ABCDEF";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (size_t i = 0; i < bytes.size(); ++i) {
      const auto byte = static_cast<unsigned char>(bytes.data()[i]);
      out.push_back(kDigits[byte >> 4]);
      out.push_back(kDigits[byte & 0xF]);
    }
    return out;
  }

  // SHA-256 of the record lines of the golden fixture, mirrored in the file's
  // own drift-guard header and in prestodb/presto's
  // TestIcebergGeometryCrossRuntimeEncoding. Updating it means changing an
  // internal encoding, which needs a deliberate, explained change in both
  // repositories.
  static constexpr const char* kGoldenRecordsSha256 =
      "9452293bacd3a49b73a75b819896a7e0d39e57f2e91cf3cda19fd37e6485f6ed";

  static std::string goldenPath() {
    return facebook::velox::test::getDataFilePath(
        "velox/connectors/hive/iceberg/tests",
        "examples/geometry_internal_encoding_golden.tsv");
  }

  // The record lines only: every line that is neither blank nor a comment,
  // joined with a newline and terminated by one. Must match how the hash in
  // the file header was computed.
  static std::string readGoldenRecordPayload() {
    std::ifstream file(goldenPath());
    VELOX_CHECK(file.is_open(), "cannot open golden file {}", goldenPath());
    std::string payload;
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      payload.append(line).push_back('\n');
    }
    return payload;
  }

  static std::string sha256Hex(const std::string& bytes) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(
        reinterpret_cast<const unsigned char*>(bytes.data()),
        bytes.size(),
        digest);
    std::string hex;
    hex.reserve(sizeof(digest) * 2);
    static const char* kDigits = "0123456789abcdef";
    for (unsigned char byte : digest) {
      hex.push_back(kDigits[byte >> 4]);
      hex.push_back(kDigits[byte & 0xF]);
    }
    return hex;
  }

  struct GoldenRecord {
    std::string wkt;
    std::string wkb;
    std::string internalHex;
  };

  // Reads the cross-runtime golden table. The same file is checked into
  // prestodb/presto at
  // presto-iceberg/src/test/resources/iceberg_v3/geometry_internal_encoding_golden.tsv.
  static std::vector<GoldenRecord> loadGolden() {
    const auto path = goldenPath();
    std::ifstream file(path);
    VELOX_CHECK(file.is_open(), "cannot open golden file {}", path);
    std::vector<GoldenRecord> records;
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      const auto first = line.find('\t');
      const auto second = line.find('\t', first + 1);
      VELOX_CHECK_NE(
          first, std::string::npos, "malformed golden record: {}", line);
      VELOX_CHECK_NE(
          second, std::string::npos, "malformed golden record: {}", line);
      records.push_back(
          {line.substr(0, first),
           fromHex(line.substr(first + 1, second - first - 1)),
           line.substr(second + 1)});
    }
    return records;
  }
};

TEST_F(IcebergGeometryCrossRuntimeEncodingTest, goldenFileHasNotDrifted) {
  // Fails on any accidental edit to this copy, and on any edit that was not
  // mirrored into the prestodb/presto copy, which asserts the same constant.
  EXPECT_EQ(sha256Hex(readGoldenRecordPayload()), kGoldenRecordsSha256)
      << "the cross-runtime golden fixture changed; see the drift-guard header in "
      << goldenPath();
}

TEST_F(IcebergGeometryCrossRuntimeEncodingTest, internalEncodingMatchesGolden) {
  // Pins the bytes a GEOMETRY vector must hold, so that a value produced here
  // and a value produced by Presto's Java worker (EsriGeometrySerde) are
  // interchangeable across an exchange.
  const auto records = loadGolden();
  ASSERT_GE(records.size(), 17);

  std::vector<std::string> mismatches;
  for (const auto& record : records) {
    auto input = makeVarbinaryVector({record.wkb});
    auto converted = convertIcebergGeometry(input, GEOMETRY(), pool(), "geom");
    ASSERT_TRUE(isGeometryType(converted->type()));
    const auto actual =
        toHex(converted->asFlatVector<StringView>()->valueAt(0));
    if (actual != record.internalHex) {
      mismatches.push_back(
          fmt::format(
              "{}\n  golden: {}\n  actual: {}",
              record.wkt,
              record.internalHex,
              actual));
    }
  }
  EXPECT_TRUE(mismatches.empty())
      << mismatches.size() << " of " << records.size()
      << " values do not match the cross-runtime golden encoding:\n"
      << folly::join("\n", mismatches);
}

TEST_F(IcebergGeometryCrossRuntimeEncodingTest, goldenWkbMatchesGeosOutput) {
  // Guards the golden file itself: its WKB column must be what a WKB writer
  // produces for its WKT, so a corrupted golden file cannot silently weaken the
  // test above.
  for (const auto& record : loadGolden()) {
    const auto written = toWkb(record.wkt);
    EXPECT_EQ(toHex(StringView(written)), toHex(StringView(record.wkb)))
        << "golden WKB does not round-trip for " << record.wkt;
  }
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
