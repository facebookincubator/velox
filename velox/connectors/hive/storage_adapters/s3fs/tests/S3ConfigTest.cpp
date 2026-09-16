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

#include "velox/connectors/hive/storage_adapters/s3fs/S3Config.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/config/Config.h"

#include <gtest/gtest.h>

namespace facebook::velox::filesystems {
namespace {

TEST(S3ConfigTest, defaultConfig) {
  auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>());
  auto s3Config = S3Config("", config);
  ASSERT_EQ(s3Config.useVirtualAddressing(), true);
  ASSERT_EQ(s3Config.useSSL(), true);
  ASSERT_EQ(s3Config.useInstanceCredentials(), false);
  ASSERT_EQ(s3Config.endpoint(), std::nullopt);
  ASSERT_EQ(s3Config.endpointRegion(), std::nullopt);
  ASSERT_EQ(s3Config.accessKey(), std::nullopt);
  ASSERT_EQ(s3Config.secretKey(), std::nullopt);
  ASSERT_EQ(s3Config.iamRole(), std::nullopt);
  ASSERT_EQ(s3Config.iamRoleSessionName(), "velox-session");
  ASSERT_EQ(s3Config.payloadSigningPolicy(), "Never");
  ASSERT_EQ(s3Config.cacheKey("foo", config), "foo");
  ASSERT_EQ(s3Config.bucket(), "");
  ASSERT_EQ(s3Config.useIMDS(), true);
  ASSERT_EQ(s3Config.minPartSize(), 10485760);
}

TEST(S3ConfigTest, overrideConfig) {
  std::unordered_map<std::string, std::string> configFromFile = {
      {S3Config::baseConfigKey(S3Config::Keys::kPathStyleAccess), "true"},
      {S3Config::baseConfigKey(S3Config::Keys::kSSLEnabled), "false"},
      {S3Config::baseConfigKey(S3Config::Keys::kUseInstanceCredentials),
       "true"},
      {S3Config::kS3PayloadSigningPolicy, "RequestDependent"},
      {S3Config::baseConfigKey(S3Config::Keys::kEndpoint), "endpoint"},
      {S3Config::baseConfigKey(S3Config::Keys::kEndpointRegion), "region"},
      {S3Config::baseConfigKey(S3Config::Keys::kAccessKey), "access"},
      {S3Config::baseConfigKey(S3Config::Keys::kSecretKey), "secret"},
      {S3Config::baseConfigKey(S3Config::Keys::kIamRole), "iam"},
      {S3Config::baseConfigKey(S3Config::Keys::kIamRoleSessionName), "velox"},
      {S3Config::baseConfigKey(S3Config::Keys::kCredentialsProvider),
       "my-credentials-provider"},
      {S3Config::baseConfigKey(S3Config::Keys::kIMDSEnabled), "false"},
      {S3Config::baseConfigKey(S3Config::Keys::kMultipartMinPartSize), "20MB"}};
  auto configBase =
      std::make_shared<config::ConfigBase>(std::move(configFromFile));
  auto s3Config = S3Config("bucket", configBase);
  ASSERT_EQ(s3Config.useVirtualAddressing(), false);
  ASSERT_EQ(s3Config.useSSL(), false);
  ASSERT_EQ(s3Config.useInstanceCredentials(), true);
  ASSERT_EQ(s3Config.endpoint(), "endpoint");
  ASSERT_EQ(s3Config.endpointRegion(), "region");
  ASSERT_EQ(s3Config.accessKey(), std::optional("access"));
  ASSERT_EQ(s3Config.secretKey(), std::optional("secret"));
  ASSERT_EQ(s3Config.iamRole(), std::optional("iam"));
  ASSERT_EQ(s3Config.iamRoleSessionName(), "velox");
  ASSERT_EQ(s3Config.payloadSigningPolicy(), "RequestDependent");
  ASSERT_EQ(s3Config.cacheKey("foo", configBase), "endpoint-foo");
  ASSERT_EQ(s3Config.cacheKey("bar", configBase), "endpoint-bar");
  ASSERT_EQ(s3Config.bucket(), "bucket");
  ASSERT_EQ(s3Config.credentialsProvider(), "my-credentials-provider");
  ASSERT_EQ(s3Config.useIMDS(), false);
  ASSERT_EQ(s3Config.minPartSize(), 20971520);
}

TEST(S3ConfigTest, overrideBucketConfig) {
  std::string_view bucket = "bucket";
  std::unordered_map<std::string, std::string> bucketConfigFromFile = {
      {S3Config::baseConfigKey(S3Config::Keys::kPathStyleAccess), "true"},
      {S3Config::baseConfigKey(S3Config::Keys::kSSLEnabled), "false"},
      {S3Config::baseConfigKey(S3Config::Keys::kUseInstanceCredentials),
       "true"},
      {S3Config::baseConfigKey(S3Config::Keys::kEndpoint), "endpoint"},
      {S3Config::bucketConfigKey(S3Config::Keys::kEndpoint, bucket),
       "bucket.s3-region.amazonaws.com"},
      {S3Config::baseConfigKey(S3Config::Keys::kAccessKey), "access"},
      {S3Config::bucketConfigKey(S3Config::Keys::kAccessKey, bucket),
       "bucket-access"},
      {S3Config::kS3PayloadSigningPolicy, "Always"},
      {S3Config::baseConfigKey(S3Config::Keys::kSecretKey), "secret"},
      {S3Config::bucketConfigKey(S3Config::Keys::kSecretKey, bucket),
       "bucket-secret"},
      {S3Config::baseConfigKey(S3Config::Keys::kIamRole), "iam"},
      {S3Config::baseConfigKey(S3Config::Keys::kIamRoleSessionName), "velox"},
      {S3Config::baseConfigKey(S3Config::Keys::kCredentialsProvider),
       "my-credentials-provider"},
      {S3Config::bucketConfigKey(S3Config::Keys::kCredentialsProvider, bucket),
       "override-credentials-provider"},
      {S3Config::baseConfigKey(S3Config::Keys::kIMDSEnabled), "false"},
      {S3Config::baseConfigKey(S3Config::Keys::kMultipartMinPartSize), "20MB"}};
  auto configBase =
      std::make_shared<config::ConfigBase>(std::move(bucketConfigFromFile));
  auto s3Config = S3Config(bucket, configBase);
  ASSERT_EQ(s3Config.useVirtualAddressing(), false);
  ASSERT_EQ(s3Config.useSSL(), false);
  ASSERT_EQ(s3Config.useInstanceCredentials(), true);
  ASSERT_EQ(s3Config.endpoint(), "bucket.s3-region.amazonaws.com");
  // Inferred from the endpoint.
  ASSERT_EQ(s3Config.endpointRegion(), "region");
  ASSERT_EQ(s3Config.accessKey(), std::optional("bucket-access"));
  ASSERT_EQ(s3Config.secretKey(), std::optional("bucket-secret"));
  ASSERT_EQ(s3Config.iamRole(), std::optional("iam"));
  ASSERT_EQ(s3Config.iamRoleSessionName(), "velox");
  ASSERT_EQ(s3Config.payloadSigningPolicy(), "Always");
  ASSERT_EQ(
      s3Config.cacheKey(bucket, configBase),
      "bucket.s3-region.amazonaws.com-bucket");
  ASSERT_EQ(s3Config.cacheKey("foo", configBase), "endpoint-foo");
  ASSERT_EQ(s3Config.credentialsProvider(), "override-credentials-provider");
  ASSERT_EQ(s3Config.useIMDS(), false);
  ASSERT_EQ(s3Config.minPartSize(), 20971520);
}

TEST(S3ConfigTest, deprecatedPrefixFallback) {
  std::string_view bucket = "bucket";
  // Configure entirely through the deprecated "hive.s3." prefix.
  std::unordered_map<std::string, std::string> configFromFile = {
      {"hive.s3.endpoint", "endpoint"},
      {"hive.s3.aws-access-key", "access"},
      {"hive.s3.aws-secret-key", "secret"},
      {"hive.s3.bucket.bucket.aws-access-key", "bucket-access"},
      {"hive.s3.payload-signing-policy", "Always"},
      {"hive.s3.log-level", "Info"},
      {"hive.s3.log-location", "/tmp/logs"},
  };
  auto configBase =
      std::make_shared<config::ConfigBase>(std::move(configFromFile));
  auto s3Config = S3Config(bucket, configBase);
  ASSERT_EQ(s3Config.endpoint(), std::optional("endpoint"));
  ASSERT_EQ(s3Config.accessKey(), std::optional("bucket-access"));
  ASSERT_EQ(s3Config.secretKey(), std::optional("secret"));
  ASSERT_EQ(s3Config.payloadSigningPolicy(), "Always");
  // cacheKey also honors the deprecated endpoint key.
  ASSERT_EQ(s3Config.cacheKey(bucket, configBase), "endpoint-bucket");
  // The log settings that RegisterS3FileSystem reads honor the fallback too.
  ASSERT_EQ(
      S3Config::configValue(*configBase, S3Config::kS3LogLevel),
      std::optional("Info"));
  ASSERT_EQ(
      S3Config::configValue(*configBase, S3Config::kS3LogLocation),
      std::optional("/tmp/logs"));
}

TEST(S3ConfigTest, canonicalPrefixWins) {
  // When both prefixes are set, the canonical "s3." value takes precedence.
  std::unordered_map<std::string, std::string> configFromFile = {
      {S3Config::baseConfigKey(S3Config::Keys::kEndpoint), "canonical"},
      {"hive.s3.endpoint", "deprecated"},
  };
  auto configBase =
      std::make_shared<config::ConfigBase>(std::move(configFromFile));
  auto s3Config = S3Config("bucket", configBase);
  ASSERT_EQ(s3Config.endpoint(), std::optional("canonical"));
  ASSERT_EQ(s3Config.cacheKey("bucket", configBase), "canonical-bucket");

  ASSERT_EQ(
      S3Config::configValue(
          *configBase, S3Config::baseConfigKey(S3Config::Keys::kEndpoint)),
      std::optional("canonical"));
}

TEST(S3ConfigTest, minPartSizeValidation) {
  // Test that setting min-part-size below 5MB throws an error.
  std::unordered_map<std::string, std::string> configFromFile = {
      {S3Config::baseConfigKey(S3Config::Keys::kMultipartMinPartSize), "4MB"}};
  auto configBase =
      std::make_shared<config::ConfigBase>(std::move(configFromFile));

  VELOX_ASSERT_THROW(
      S3Config("bucket", configBase),
      "The min-part-size S3 configuration must exceed 5MB");

  configFromFile = {
      {S3Config::baseConfigKey(S3Config::Keys::kMultipartMinPartSize), "10GB"}};
  configBase = std::make_shared<config::ConfigBase>(std::move(configFromFile));
  VELOX_ASSERT_THROW(
      S3Config("bucket", configBase),
      "The min-part-size S3 configuration must not exceed 5GB");
}

TEST(S3ConfigTest, minPartSizeValidationBucketConfig) {
  // Test that setting bucket-specific min-part-size below 5MB throws an error.
  std::string_view bucket = "testbucket";
  std::unordered_map<std::string, std::string> configFromFile = {
      {S3Config::bucketConfigKey(S3Config::Keys::kMultipartMinPartSize, bucket),
       "3MB"}};
  auto configBase =
      std::make_shared<config::ConfigBase>(std::move(configFromFile));

  VELOX_ASSERT_THROW(
      S3Config(bucket, configBase),
      "The min-part-size S3 configuration must exceed 5MB");

  configFromFile = {
      {S3Config::bucketConfigKey(S3Config::Keys::kMultipartMinPartSize, bucket),
       "10GB"}};
  configBase = std::make_shared<config::ConfigBase>(std::move(configFromFile));

  VELOX_ASSERT_THROW(
      S3Config(bucket, configBase),
      "The min-part-size S3 configuration must not exceed 5GB");
}

} // namespace
} // namespace facebook::velox::filesystems
