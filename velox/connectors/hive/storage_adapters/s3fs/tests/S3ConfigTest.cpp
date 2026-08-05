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

std::string hiveS3ConfigKey(S3Config::Keys key) {
  return "hive." + S3Config::baseConfigKey(key);
}

std::string hiveS3BucketConfigKey(S3Config::Keys key, std::string_view bucket) {
  return "hive." + S3Config::bucketConfigKey(key, bucket);
}

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
      {hiveS3ConfigKey(S3Config::Keys::kPathStyleAccess), "true"},
      {hiveS3ConfigKey(S3Config::Keys::kSSLEnabled), "false"},
      {hiveS3ConfigKey(S3Config::Keys::kUseInstanceCredentials), "true"},
      {"hive.s3.payload-signing-policy", "RequestDependent"},
      {hiveS3ConfigKey(S3Config::Keys::kEndpoint), "endpoint"},
      {hiveS3ConfigKey(S3Config::Keys::kEndpointRegion), "region"},
      {hiveS3ConfigKey(S3Config::Keys::kAccessKey), "access"},
      {hiveS3ConfigKey(S3Config::Keys::kSecretKey), "secret"},
      {hiveS3ConfigKey(S3Config::Keys::kIamRole), "iam"},
      {hiveS3ConfigKey(S3Config::Keys::kIamRoleSessionName), "velox"},
      {hiveS3ConfigKey(S3Config::Keys::kCredentialsProvider),
       "my-credentials-provider"},
      {hiveS3ConfigKey(S3Config::Keys::kIMDSEnabled), "false"},
      {hiveS3ConfigKey(S3Config::Keys::kMultipartMinPartSize), "20MB"}};
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

TEST(S3ConfigTest, normalizeConnectorScopedConfig) {
  std::string_view bucket = "bucket";
  std::unordered_map<std::string, std::string> configFromFile = {
      {"iceberg.s3.endpoint", "iceberg-endpoint"},
      {"iceberg.s3.aws-access-key", "iceberg-access"},
      {"iceberg.s3.min-part-size", "20MB"},
      {"iceberg.s3.payload-signing-policy", "Always"},
      {"iceberg.s3.bucket.bucket.endpoint", "bucket.s3-region.amazonaws.com"}};
  auto scopedConfig =
      std::make_shared<config::ConfigBase>(std::move(configFromFile));
  auto configBase = S3Config::normalizeConfig(scopedConfig);
  auto s3Config = S3Config(bucket, scopedConfig);

  ASSERT_EQ(
      configBase->get<std::string>(
          S3Config::baseConfigKey(S3Config::Keys::kEndpoint)),
      std::optional("iceberg-endpoint"));
  ASSERT_EQ(s3Config.endpoint(), "bucket.s3-region.amazonaws.com");
  // Inferred from the endpoint.
  ASSERT_EQ(s3Config.endpointRegion(), "region");
  ASSERT_EQ(s3Config.accessKey(), std::optional("iceberg-access"));
  ASSERT_EQ(s3Config.payloadSigningPolicy(), "Always");
  ASSERT_EQ(
      S3Config::cacheKey(bucket, scopedConfig),
      "bucket.s3-region.amazonaws.com-bucket");
  ASSERT_EQ(s3Config.minPartSize(), 20971520);
}

TEST(S3ConfigTest, normalizeMixedConnectorScopedConfig) {
  auto configBase = S3Config::normalizeConfig(
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{
              {"hive.s3.endpoint", "shared-endpoint"},
              {"iceberg.s3.endpoint", "shared-endpoint"}}));
  auto s3Config = S3Config("bucket", configBase);

  ASSERT_EQ(s3Config.endpoint(), "shared-endpoint");

  VELOX_ASSERT_THROW(
      S3Config::normalizeConfig(
          std::make_shared<config::ConfigBase>(
              std::unordered_map<std::string, std::string>{
                  {"hive.s3.endpoint", "hive-endpoint"},
                  {"iceberg.s3.endpoint", "iceberg-endpoint"}})),
      "Multiple connector-scoped S3 configs map to 's3.endpoint' with "
      "different values. Pass a connector-specific config object to "
      "disambiguate.");
}

TEST(S3ConfigTest, overrideBucketConfig) {
  std::string_view bucket = "bucket";
  std::unordered_map<std::string, std::string> bucketConfigFromFile = {
      {hiveS3ConfigKey(S3Config::Keys::kPathStyleAccess), "true"},
      {hiveS3ConfigKey(S3Config::Keys::kSSLEnabled), "false"},
      {hiveS3ConfigKey(S3Config::Keys::kUseInstanceCredentials), "true"},
      {hiveS3ConfigKey(S3Config::Keys::kEndpoint), "endpoint"},
      {hiveS3BucketConfigKey(S3Config::Keys::kEndpoint, bucket),
       "bucket.s3-region.amazonaws.com"},
      {hiveS3ConfigKey(S3Config::Keys::kAccessKey), "access"},
      {hiveS3BucketConfigKey(S3Config::Keys::kAccessKey, bucket),
       "bucket-access"},
      {"hive.s3.payload-signing-policy", "Always"},
      {hiveS3ConfigKey(S3Config::Keys::kSecretKey), "secret"},
      {hiveS3BucketConfigKey(S3Config::Keys::kSecretKey, bucket),
       "bucket-secret"},
      {hiveS3ConfigKey(S3Config::Keys::kIamRole), "iam"},
      {hiveS3ConfigKey(S3Config::Keys::kIamRoleSessionName), "velox"},
      {hiveS3ConfigKey(S3Config::Keys::kCredentialsProvider),
       "my-credentials-provider"},
      {hiveS3BucketConfigKey(S3Config::Keys::kCredentialsProvider, bucket),
       "override-credentials-provider"},
      {hiveS3ConfigKey(S3Config::Keys::kIMDSEnabled), "false"},
      {hiveS3ConfigKey(S3Config::Keys::kMultipartMinPartSize), "20MB"}};
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

TEST(S3ConfigTest, minPartSizeValidation) {
  // Test that setting min-part-size below 5MB throws an error.
  std::unordered_map<std::string, std::string> configFromFile = {
      {hiveS3ConfigKey(S3Config::Keys::kMultipartMinPartSize), "4MB"}};
  auto configBase =
      std::make_shared<config::ConfigBase>(std::move(configFromFile));

  VELOX_ASSERT_THROW(
      S3Config("bucket", configBase),
      "The min-part-size S3 configuration must exceed 5MB");

  configFromFile = {
      {hiveS3ConfigKey(S3Config::Keys::kMultipartMinPartSize), "10GB"}};
  configBase = std::make_shared<config::ConfigBase>(std::move(configFromFile));
  VELOX_ASSERT_THROW(
      S3Config("bucket", configBase),
      "The min-part-size S3 configuration must not exceed 5GB");
}

TEST(S3ConfigTest, minPartSizeValidationBucketConfig) {
  // Test that setting bucket-specific min-part-size below 5MB throws an error.
  std::string_view bucket = "testbucket";
  std::unordered_map<std::string, std::string> configFromFile = {
      {hiveS3BucketConfigKey(S3Config::Keys::kMultipartMinPartSize, bucket),
       "3MB"}};
  auto configBase =
      std::make_shared<config::ConfigBase>(std::move(configFromFile));

  VELOX_ASSERT_THROW(
      S3Config(bucket, configBase),
      "The min-part-size S3 configuration must exceed 5MB");

  configFromFile = {
      {hiveS3BucketConfigKey(S3Config::Keys::kMultipartMinPartSize, bucket),
       "10GB"}};
  configBase = std::make_shared<config::ConfigBase>(std::move(configFromFile));

  VELOX_ASSERT_THROW(
      S3Config(bucket, configBase),
      "The min-part-size S3 configuration must not exceed 5GB");
}

} // namespace
} // namespace facebook::velox::filesystems
