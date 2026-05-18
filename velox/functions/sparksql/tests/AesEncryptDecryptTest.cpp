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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/encode/Base64.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class AesEncryptDecryptTest : public SparkFunctionBaseTest {
 protected:
  // Helper to encrypt with given mode/padding and optional IV/AAD.
  // When iv/aad is nullopt the SQL argument is a real NULL literal
  // (not empty bytes), so the C++ callNullable path receives nullptr.
  std::optional<std::string> encrypt(
      const std::optional<std::string>& input,
      const std::optional<std::string>& key,
      const std::string& mode = "GCM",
      const std::string& padding = "DEFAULT",
      const std::optional<std::string>& iv = std::nullopt,
      const std::optional<std::string>& aad = std::nullopt) {
    const std::string ivExpr =
        iv.has_value() ? "cast(c2 as varbinary)" : "cast(null as varbinary)";
    const std::string aadExpr =
        aad.has_value() ? "cast(c3 as varbinary)" : "cast(null as varbinary)";
    return evaluateOnce<
        std::string,
        std::string,
        std::string,
        std::string,
        std::string>(
        fmt::format(
            "aes_encrypt(cast(c0 as varbinary), cast(c1 as varbinary), "
            "'{}', '{}', {}, {})",
            mode,
            padding,
            ivExpr,
            aadExpr),
        input,
        key,
        iv.value_or(""),
        aad.value_or(""));
  }

  // Helper to decrypt. Same nullopt-as-NULL convention as encrypt().
  std::optional<std::string> decrypt(
      const std::optional<std::string>& input,
      const std::optional<std::string>& key,
      const std::string& mode = "GCM",
      const std::string& padding = "DEFAULT",
      const std::optional<std::string>& iv = std::nullopt,
      const std::optional<std::string>& aad = std::nullopt) {
    const std::string ivExpr =
        iv.has_value() ? "cast(c2 as varbinary)" : "cast(null as varbinary)";
    const std::string aadExpr =
        aad.has_value() ? "cast(c3 as varbinary)" : "cast(null as varbinary)";
    return evaluateOnce<
        std::string,
        std::string,
        std::string,
        std::string,
        std::string>(
        fmt::format(
            "aes_decrypt(cast(c0 as varbinary), cast(c1 as varbinary), "
            "'{}', '{}', {}, {})",
            mode,
            padding,
            ivExpr,
            aadExpr),
        input,
        key,
        iv.value_or(""),
        aad.value_or(""));
  }

  // Round-trip test: encrypt then decrypt should return original.
  void testRoundTrip(
      const std::string& input,
      const std::string& key,
      const std::string& mode,
      const std::string& padding) {
    auto encrypted = encrypt(input, key, mode, padding);
    ASSERT_TRUE(encrypted.has_value());
    auto decrypted = decrypt(encrypted.value(), key, mode, padding);
    ASSERT_TRUE(decrypted.has_value());
    SCOPED_TRACE(fmt::format("mode={} padding={}", mode, padding));
    EXPECT_EQ(decrypted.value(), input);
  }
};

// ECB mode tests — deterministic (no IV).
TEST_F(AesEncryptDecryptTest, ecbMode) {
  const std::string key16 = "abcdefghijklmnop"; // 128-bit
  const std::string key24 = "abcdefghijklmnop12345678"; // 192-bit
  const std::string key32 = "abcdefghijklmnop12345678ABCDEFGH"; // 256-bit

  // ECB is deterministic — same input+key always produces same output.
  testRoundTrip("Spark", key16, "ECB", "PKCS");
  testRoundTrip("Spark SQL", key16, "ECB", "PKCS");
  testRoundTrip("", key16, "ECB", "PKCS");
  testRoundTrip("Spark", key24, "ECB", "PKCS");
  testRoundTrip("Spark", key32, "ECB", "PKCS");

  // ECB with DEFAULT padding (same as PKCS for ECB).
  testRoundTrip("Spark", key16, "ECB", "DEFAULT");

  // Verify exact ciphertext matches Spark's DataFrameFunctionsSuite output.
  auto ecb16 = encrypt("Spark", key16, "ECB", "PKCS");
  ASSERT_TRUE(ecb16.has_value());
  EXPECT_EQ(
      encoding::Base64::encode(ecb16.value()), "4Hv0UKCx6nfUeAoPZo1z+w==");
  EXPECT_EQ(decrypt(ecb16.value(), key16, "ECB", "PKCS").value(), "Spark");

  auto ecbEmpty = encrypt("", key16, "ECB", "PKCS");
  ASSERT_TRUE(ecbEmpty.has_value());
  EXPECT_EQ(
      encoding::Base64::encode(ecbEmpty.value()), "jmTOhz8XTbskI/zYFFgOFQ==");
  EXPECT_EQ(decrypt(ecbEmpty.value(), key16, "ECB", "PKCS").value(), "");

  auto ecb24 = encrypt("Spark", key24, "ECB", "PKCS");
  ASSERT_TRUE(ecb24.has_value());
  EXPECT_EQ(
      encoding::Base64::encode(ecb24.value()), "NeTYNgA+PCQBN50DA//O2w==");
  EXPECT_EQ(decrypt(ecb24.value(), key24, "ECB", "PKCS").value(), "Spark");

  auto ecb32 = encrypt("Spark", key32, "ECB", "PKCS");
  ASSERT_TRUE(ecb32.has_value());
  EXPECT_EQ(
      encoding::Base64::encode(ecb32.value()), "9J3iZbIxnmaG+OIA9Amd+A==");
  EXPECT_EQ(decrypt(ecb32.value(), key32, "ECB", "PKCS").value(), "Spark");
}

// CBC mode tests.
TEST_F(AesEncryptDecryptTest, cbcMode) {
  const std::string key16 = "1234567890abcdef";
  const std::string key32 = "abcdefghijklmnop12345678ABCDEFGH";

  // CBC with auto-generated IV (round-trip).
  testRoundTrip("Spark", key16, "CBC", "DEFAULT");
  testRoundTrip("Apache Spark", key16, "CBC", "PKCS");
  testRoundTrip("", key16, "CBC", "DEFAULT");

  // CBC with explicit IV (16 zero bytes).
  std::string iv16(16, '\0');
  auto encrypted = encrypt("Spark", key32, "CBC", "DEFAULT", iv16);
  ASSERT_TRUE(encrypted.has_value());
  auto decrypted = decrypt(encrypted.value(), key32, "CBC", "DEFAULT");
  ASSERT_TRUE(decrypted.has_value());
  EXPECT_EQ(decrypted.value(), "Spark");
}

// GCM mode tests (default mode).
TEST_F(AesEncryptDecryptTest, gcmMode) {
  const std::string key16 = "0000111122223333";
  const std::string key32 = "abcdefghijklmnop12345678ABCDEFGH";

  // GCM with auto-generated IV (round-trip).
  testRoundTrip("Spark", key16, "GCM", "DEFAULT");
  testRoundTrip("Spark SQL", key16, "GCM", "NONE");
  testRoundTrip("", key16, "GCM", "DEFAULT");

  // GCM with explicit IV (12 zero bytes).
  std::string iv12(12, '\0');
  auto encrypted = encrypt("Spark", key32, "GCM", "DEFAULT", iv12);
  ASSERT_TRUE(encrypted.has_value());
  auto decrypted = decrypt(encrypted.value(), key32, "GCM", "DEFAULT");
  ASSERT_TRUE(decrypted.has_value());
  EXPECT_EQ(decrypted.value(), "Spark");
}

// GCM with AAD.
TEST_F(AesEncryptDecryptTest, gcmWithAad) {
  const std::string key32 = "abcdefghijklmnop12345678ABCDEFGH";
  std::string iv12(12, '\0');
  std::string aad = "This is an AAD mixed into the input";

  // Encrypt with explicit IV and AAD.
  auto encrypted = encrypt("Spark", key32, "GCM", "DEFAULT", iv12, aad);
  ASSERT_TRUE(encrypted.has_value());

  // Decrypt: IV auto-extracted from prefix, AAD passed explicitly.
  auto decrypted =
      decrypt(encrypted.value(), key32, "GCM", "DEFAULT", std::string(""), aad);
  ASSERT_TRUE(decrypted.has_value());
  EXPECT_EQ(decrypted.value(), "Spark");
}

// All three key sizes.
TEST_F(AesEncryptDecryptTest, keySizes) {
  testRoundTrip("Spark", "0000111122223333", "ECB", "PKCS"); // 128
  testRoundTrip("Spark", "000011112222333344445555", "ECB", "PKCS"); // 192
  testRoundTrip(
      "Spark", "00001111222233334444555566667777", "ECB", "PKCS"); // 256
}

// Case-insensitive mode and padding.
TEST_F(AesEncryptDecryptTest, caseInsensitiveMode) {
  const std::string key16 = "abcdefghijklmnop";

  testRoundTrip("Spark", key16, "ecb", "pkcs");
  testRoundTrip("Spark", key16, "Ecb", "Pkcs");

  auto upper = encrypt("Spark", key16, "ECB", "PKCS");
  auto lower = encrypt("Spark", key16, "ecb", "pkcs");
  ASSERT_TRUE(upper.has_value());
  ASSERT_TRUE(lower.has_value());
  EXPECT_EQ(upper.value(), lower.value());
}

// Null IV and AAD should not produce null output.
TEST_F(AesEncryptDecryptTest, nullIvAndAad) {
  const std::string key16 = "0000111122223333";

  // Null IV → auto-generate IV (GCM round-trip).
  auto encrypted = encrypt("Spark", key16, "GCM", "DEFAULT", std::nullopt);
  ASSERT_TRUE(encrypted.has_value());
  auto decrypted = decrypt(encrypted.value(), key16, "GCM", "DEFAULT");
  ASSERT_TRUE(decrypted.has_value());
  EXPECT_EQ(decrypted.value(), "Spark");

  // Null AAD → no AAD (GCM round-trip).
  std::string iv12(12, '\0');
  encrypted = encrypt("Spark", key16, "GCM", "DEFAULT", iv12, std::nullopt);
  ASSERT_TRUE(encrypted.has_value());
  decrypted = decrypt(encrypted.value(), key16, "GCM", "DEFAULT");
  ASSERT_TRUE(decrypted.has_value());
  EXPECT_EQ(decrypted.value(), "Spark");

  // Both null IV and AAD.
  encrypted =
      encrypt("Spark", key16, "GCM", "DEFAULT", std::nullopt, std::nullopt);
  ASSERT_TRUE(encrypted.has_value());
  decrypted = decrypt(encrypted.value(), key16, "GCM", "DEFAULT");
  ASSERT_TRUE(decrypted.has_value());
  EXPECT_EQ(decrypted.value(), "Spark");
}

// NONE padding with block-aligned input.
TEST_F(AesEncryptDecryptTest, nonePaddingBlockAligned) {
  std::string key16 = "0000111122223333";
  testRoundTrip("0123456789abcdef", key16, "ECB", "NONE");
}

// --- Failure tests ---

// Invalid key length.
TEST_F(AesEncryptDecryptTest, invalidKeyLength) {
  VELOX_ASSERT_THROW(
      encrypt("Spark", "short", "ECB", "PKCS"), "Invalid AES key length");

  VELOX_ASSERT_THROW(
      encrypt("Spark", "12345678901234567", "ECB", "PKCS"), // 17 bytes
      "Invalid AES key length");
}

// Invalid mode.
TEST_F(AesEncryptDecryptTest, invalidMode) {
  VELOX_ASSERT_THROW(
      encrypt("Spark", "0000111122223333", "CTR", "DEFAULT"),
      "Unsupported AES mode");
}

// Invalid padding.
TEST_F(AesEncryptDecryptTest, invalidPadding) {
  VELOX_ASSERT_THROW(
      encrypt("Spark", "0000111122223333", "ECB", "INVALID"),
      "Unsupported AES padding");

  VELOX_ASSERT_THROW(
      encrypt("Spark", "0000111122223333", "GCM", "PKCS"),
      "PKCS padding is not supported for GCM mode");
}

// Invalid IV length.
TEST_F(AesEncryptDecryptTest, invalidIvLength) {
  std::string key16 = "0000111122223333";

  VELOX_ASSERT_THROW(
      encrypt("Spark", key16, "CBC", "DEFAULT", std::string(8, '\0')),
      "Invalid IV length");

  VELOX_ASSERT_THROW(
      encrypt("Spark", key16, "GCM", "DEFAULT", std::string(16, '\0')),
      "Invalid IV length");

  VELOX_ASSERT_THROW(
      encrypt("Spark", key16, "ECB", "PKCS", std::string(16, '\0')),
      "IV is not supported for ECB mode");
}

// AAD only supported for GCM.
TEST_F(AesEncryptDecryptTest, aadUnsupportedMode) {
  std::string key16 = "0000111122223333";
  std::string iv16(16, '\0');

  VELOX_ASSERT_THROW(
      encrypt("Spark", key16, "CBC", "DEFAULT", iv16, std::string("aad")),
      "AAD is not supported for CBC mode");

  VELOX_ASSERT_THROW(
      encrypt("Spark", key16, "ECB", "PKCS", std::nullopt, std::string("aad")),
      "AAD is not supported for ECB mode");
}

// NONE padding with non-block-aligned input.
TEST_F(AesEncryptDecryptTest, nonePaddingBlockAlignment) {
  std::string key16 = "0000111122223333";

  VELOX_ASSERT_THROW(
      encrypt("Spark", key16, "ECB", "NONE"), "multiple of 16 bytes");
  VELOX_ASSERT_THROW(
      encrypt("Spark", key16, "CBC", "NONE"), "multiple of 16 bytes");
}

// GCM authentication: tampered ciphertext or wrong AAD must fail.
TEST_F(AesEncryptDecryptTest, gcmAuthFailure) {
  std::string key32 = "abcdefghijklmnop12345678ABCDEFGH";
  std::string iv12(12, '\0');
  std::string aad = "correct AAD";

  auto encrypted = encrypt("Spark", key32, "GCM", "DEFAULT", iv12, aad);
  ASSERT_TRUE(encrypted.has_value());

  VELOX_ASSERT_THROW(
      decrypt(
          encrypted.value(),
          key32,
          "GCM",
          "DEFAULT",
          iv12,
          std::string("wrong")),
      "AES decryption failed");

  auto tampered = encrypted.value();
  tampered.back() ^= 0xFF;
  VELOX_ASSERT_THROW(
      decrypt(tampered, key32, "GCM", "DEFAULT"), "AES decryption failed");
}

// Decrypt with the wrong key must fail. Distinct from tampered-ciphertext
// (covered in gcmAuthFailure) and from short-input (covered in
// decryptInputTooShort).
TEST_F(AesEncryptDecryptTest, decryptWrongKey) {
  const std::string keyA = "0000111122223333";
  const std::string keyB = "ffffeeeeddddcccc";

  // CBC/PKCS: wrong key produces invalid PKCS padding on Final_ex.
  auto encryptedCbc = encrypt("Spark", keyA, "CBC", "DEFAULT");
  ASSERT_TRUE(encryptedCbc.has_value());
  VELOX_ASSERT_THROW(
      decrypt(encryptedCbc.value(), keyB, "CBC", "DEFAULT"),
      "AES decryption failed");

  // GCM: wrong key fails the authentication tag check.
  std::string iv12(12, '\0');
  auto encryptedGcm = encrypt("Spark", keyA, "GCM", "DEFAULT", iv12);
  ASSERT_TRUE(encryptedGcm.has_value());
  VELOX_ASSERT_THROW(
      decrypt(encryptedGcm.value(), keyB, "GCM", "DEFAULT"),
      "AES decryption failed");
}

// Per-row mode/padding (column reference, not literal). Spark vanilla
// evaluates mode/padding per row via RuntimeReplaceable + StaticInvoke
// (ExpressionImplUtils.aesEncrypt reads all 6 args per row), and Velox now
// matches: `initialize` only caches the config when mode AND padding are
// literal; otherwise `callNullable` parses per row. Null mode/padding rows
// produce null output (Spark vanilla behavior).
TEST_F(AesEncryptDecryptTest, nonConstantModeAndPadding) {
  using facebook::velox::test::assertEqualVectors;
  // 4 rows: legal combos + 1 row with null mode (expect null output).
  auto inputs = makeFlatVector<std::string>(
      {"alpha", "beta", "gamma", "delta"}, VARBINARY());
  auto keys = makeFlatVector<std::string>(
      {"0000111122223333",
       "0000111122223333",
       "0000111122223333",
       "0000111122223333"},
      VARBINARY());
  auto modes = makeNullableFlatVector<std::string>(
      {"GCM", "CBC", "ECB", std::nullopt});
  auto paddings = makeFlatVector<std::string>(
      {"DEFAULT", "PKCS", "PKCS", "DEFAULT"});
  auto rowVector = makeRowVector({inputs, keys, modes, paddings});

  // Encrypt with column-ref mode/padding (no iv, no aad).
  auto encrypted = evaluate<FlatVector<StringView>>(
      "aes_encrypt(c0, c1, c2, c3, cast(null as varbinary), "
      "cast(null as varbinary))",
      rowVector);
  ASSERT_FALSE(encrypted->isNullAt(0));
  ASSERT_FALSE(encrypted->isNullAt(1));
  ASSERT_FALSE(encrypted->isNullAt(2));
  ASSERT_TRUE(encrypted->isNullAt(3)); // null mode -> null output

  // Round-trip: decrypt with the same column-ref mode/padding.
  auto encStrs = makeFlatVector<std::string>(
      {std::string(encrypted->valueAt(0).data(), encrypted->valueAt(0).size()),
       std::string(encrypted->valueAt(1).data(), encrypted->valueAt(1).size()),
       std::string(encrypted->valueAt(2).data(), encrypted->valueAt(2).size()),
       std::string("")},
      VARBINARY());
  auto rtRowVector = makeRowVector({encStrs, keys, modes, paddings});
  auto decrypted = evaluate<FlatVector<StringView>>(
      "aes_decrypt(c0, c1, c2, c3, cast(null as varbinary), "
      "cast(null as varbinary))",
      rtRowVector);
  EXPECT_EQ(
      std::string(
          decrypted->valueAt(0).data(), decrypted->valueAt(0).size()),
      "alpha");
  EXPECT_EQ(
      std::string(
          decrypted->valueAt(1).data(), decrypted->valueAt(1).size()),
      "beta");
  EXPECT_EQ(
      std::string(
          decrypted->valueAt(2).data(), decrypted->valueAt(2).size()),
      "gamma");
  EXPECT_TRUE(decrypted->isNullAt(3)); // null mode -> null output
}

// Per-row mode/padding parse error: bad mode in one row throws (Spark
// vanilla also throws per row; we don't silently return null).
TEST_F(AesEncryptDecryptTest, nonConstantModeBadValueThrows) {
  auto inputs = makeFlatVector<std::string>({"alpha", "beta"}, VARBINARY());
  auto keys = makeFlatVector<std::string>(
      {"0000111122223333", "0000111122223333"}, VARBINARY());
  auto modes = makeFlatVector<std::string>({"GCM", "BAD"});
  auto paddings = makeFlatVector<std::string>({"DEFAULT", "DEFAULT"});
  auto rowVector = makeRowVector({inputs, keys, modes, paddings});
  VELOX_ASSERT_THROW(
      evaluate<FlatVector<StringView>>(
          "aes_encrypt(c0, c1, c2, c3, cast(null as varbinary), "
          "cast(null as varbinary))",
          rowVector),
      "Unsupported AES mode: BAD");
}

// Decrypt with input too short.
TEST_F(AesEncryptDecryptTest, decryptInputTooShort) {
  const std::string key16 = "0000111122223333";

  VELOX_ASSERT_THROW(
      decrypt(std::string(8, '\0'), key16, "CBC", "DEFAULT"),
      "Input too short");

  VELOX_ASSERT_THROW(
      decrypt(std::string(20, '\0'), key16, "GCM", "DEFAULT"),
      "Input too short");

  std::string iv12(12, '\0');
  VELOX_ASSERT_THROW(
      decrypt(std::string(10, '\0'), key16, "GCM", "DEFAULT", iv12),
      "Input too short");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
