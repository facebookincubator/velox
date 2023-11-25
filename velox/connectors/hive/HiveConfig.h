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

#include <folly/Conv.h>
#include <glog/logging.h>
#include <iostream>
#include <optional>
#include <string>

namespace facebook::velox {
class Config;
}

namespace facebook::velox::connector::hive {

enum class InsertExistingPartitionsBehavior {
  kError,
  kOverwrite,
};

std::string insertExistingPartitionsBehaviorString(
    InsertExistingPartitionsBehavior behavior);

/// Hive connector configs.
class HiveConfig {
 public:
  HiveConfig(
      const std::unordered_map<std::string, std::string>& connectorConf) {
    for (auto config = connectorConf.begin(); config != connectorConf.end();
         config++) {
      if (configSetters.count(config->first) != 0) {
        configSetters.at(config->first)(this, config->second);
      } else {
        LOG(ERROR) << "Invalid hive config:" << config->first << std::endl;
        ;
      }
    }
  };

  InsertExistingPartitionsBehavior insertExistingPartitionsBehavior() const;

  uint32_t maxPartitionsPerWriters() const;

  bool immutablePartitions() const;

  bool s3UseVirtualAddressing() const;

  std::string s3GetLogLevel() const;

  bool s3UseSSL() const;

  bool s3UseInstanceCredentials() const;

  std::string s3Endpoint() const;

  std::optional<std::string> s3AccessKey() const;

  std::optional<std::string> s3SecretKey() const;

  std::optional<std::string> s3IAMRole() const;

  std::string s3IAMRoleSessionName() const;

  std::string gcsEndpoint() const;

  std::string gcsScheme() const;

  std::string gcsCredentials() const;

  bool isOrcUseColumnNames() const;

  bool isFileColumnNamesReadAsLowerCase() const;

  int64_t maxCoalescedBytes() const;

  int32_t maxCoalescedDistanceBytes() const;

  int32_t numCacheFileHandles() const;

  bool isFileHandleCacheEnabled() const;

  uint64_t fileWriterFlushThresholdBytes() const;

  uint32_t sortWriterMaxOutputRows() const;

  uint64_t sortWriterMaxOutputBytes() const;

  uint64_t getOrcWriterMaxStripeSize() const;

  uint64_t getOrcWriterMaxDictionaryMemory() const;

  /// Behavior on insert into existing partitions.
  static constexpr const char* kInsertExistingPartitionsBehavior =
      "insert-existing-partitions-behavior";

  /// Maximum number of (bucketed) partitions per a single table writer
  /// instance.
  static constexpr const char* kMaxPartitionsPerWriters =
      "max-partitions-per-writers";

  /// Whether new data can be inserted into an unpartition table.
  /// Velox currently does not support appending data to existing partitions.
  static constexpr const char* kImmutablePartitions =
      "hive.immutable-partitions";

  /// Virtual addressing is used for AWS S3 and is the default
  /// (path-style-access is false). Path access style is used for some on-prem
  /// systems like Minio.
  static constexpr const char* kS3PathStyleAccess = "hive.s3.path-style-access";

  /// Log granularity of AWS C++ SDK.
  static constexpr const char* kS3LogLevel = "hive.s3.log-level";

  /// Use HTTPS to communicate with the S3 API.
  static constexpr const char* kS3SSLEnabled = "hive.s3.ssl.enabled";

  /// Use the EC2 metadata service to retrieve API credentials.
  static constexpr const char* kS3UseInstanceCredentials =
      "hive.s3.use-instance-credentials";

  /// The S3 storage endpoint server. This can be used to connect to an
  /// S3-compatible storage system instead of AWS.
  static constexpr const char* kS3Endpoint = "hive.s3.endpoint";

  /// Default AWS access key to use.
  static constexpr const char* kS3AwsAccessKey = "hive.s3.aws-access-key";

  /// Default AWS secret key to use.
  static constexpr const char* kS3AwsSecretKey = "hive.s3.aws-secret-key";

  /// IAM role to assume.
  static constexpr const char* kS3IamRole = "hive.s3.iam-role";

  /// Session name associated with the IAM role.
  static constexpr const char* kS3IamRoleSessionName =
      "hive.s3.iam-role-session-name";

  /// The GCS storage endpoint server.
  static constexpr const char* kGCSEndpoint = "hive.gcs.endpoint";

  /// The GCS storage scheme, https for default credentials.
  static constexpr const char* kGCSScheme = "hive.gcs.scheme";

  /// The GCS service account configuration as json string
  static constexpr const char* kGCSCredentials = "hive.gcs.credentials";

  /// Maps table field names to file field names using names, not indices.
  static constexpr const char* kOrcUseColumnNames = "hive.orc.use-column-names";

  /// Reads the source file column name as lower case.
  static constexpr const char* kFileColumnNamesReadAsLowerCase =
      "file_column_names_read_as_lower_case";

  /// Sets the max coalesce bytes for a request.
  static constexpr const char* kMaxCoalescedBytes = "max-coalesced-bytes";

  /// Sets the max coalesce distance bytes for combining requests.
  static constexpr const char* kMaxCoalescedDistanceBytes =
      "max-coalesced-distance-bytes";

  /// Maximum number of entries in the file handle cache.
  static constexpr const char* kNumCacheFileHandles = "num-cached-file-handles";

  /// Enable file handle cache.
  static constexpr const char* kEnableFileHandleCache =
      "file-handle-cache-enabled";

  static constexpr const char* kOrcWriterMaxStripeSize =
      "hive.orc.writer.stripe-max-size";

  static constexpr const char* kOrcWriterMaxDictionaryMemory =
      "hive.orc.writer.dictionary-max-memory";

  static constexpr const char* kSortWriterMaxOutputRows =
      "sort-writer-max-output-rows";
  static constexpr const char* kSortWriterMaxOutputBytes =
      "sort-writer-max-output-bytes";

 private:
  static void setInsertExistingPartitionsBehavior(
      HiveConfig* hiveConfig,
      std::string insertExistingPartitionsBehavior);

  static void setMaxPartitionsPerWriters(
      HiveConfig* hiveConfig,
      std::string maxPartitionsPerWriters);

  static void setImmutablePartitions(
      HiveConfig* hiveConfig,
      std::string immutablePartitions);

  static void setS3PathStyleAccess(
      HiveConfig* hiveConfig,
      std::string s3PathStyleAccess);

  static void setS3LogLevel(HiveConfig* hiveConfig, std::string s3LogLevel);

  static void setS3SSLEnabled(HiveConfig* hiveConfig, std::string s3SSLEnabled);

  static void setS3UseInstanceCredentials(
      HiveConfig* hiveConfig,
      std::string s3UseInstanceCredentials);

  static void setS3Endpoint(HiveConfig* hiveConfig, std::string s3Endpoint);

  static void setS3AwsAccessKey(
      HiveConfig* hiveConfig,
      std::string s3AwsAccessKey);

  static void setS3AwsSecretKey(
      HiveConfig* hiveConfig,
      std::string s3AwsSecretKey);

  static void setS3IamRole(HiveConfig* hiveConfig, std::string s3IamRole);

  static void setS3IamRoleSessionName(
      HiveConfig* hiveConfig,
      std::string s3IamRoleSessionName);

  static void setGCSEndpoint(HiveConfig* hiveConfig, std::string GCSEndpoint);

  static void setGCSScheme(HiveConfig* hiveConfig, std::string GCSScheme);

  static void setGCSCredentials(
      HiveConfig* hiveConfig,
      std::string GCSCredentials);

  static void setOrcUseColumnNames(
      HiveConfig* hiveConfig,
      std::string orcUseColumnNames);

  static void setFileColumnNamesReadAsLowerCase(
      HiveConfig* hiveConfig,
      std::string fileColumnNamesReadAsLowerCase);

  static void setMaxCoalescedBytes(
      HiveConfig* hiveConfig,
      std::string maxCoalescedBytes);

  static void setMaxCoalescedDistanceBytes(
      HiveConfig* hiveConfig,
      std::string maxCoalescedDistanceBytes);

  static void setNumCacheFileHandles(
      HiveConfig* hiveConfig,
      std::string numCacheFileHandles);

  // static
  static void setEnableFileHandleCache(
      HiveConfig* hiveConfig,
      std::string enableFileHandleCache);

  static void setSortWriterMaxOutputRows(
      HiveConfig* hiveConfig,
      std::string sortWriterMaxOutputRows);

  static void setSortWriterMaxOutputBytes(
      HiveConfig* hiveConfig,
      std::string sortWriterMaxOutputBytes);

  // static
  static void setOrcWriterMaxStripeSize(
      HiveConfig* hiveConfig,
      std::string orcWriterMaxStripeSize);

  static void setOrcWriterMaxDictionaryMemory(
      HiveConfig* hiveConfig,
      std::string orcWriterMaxDictionaryMemory);

  InsertExistingPartitionsBehavior insertExistingPartitionsBehavior_ =
      InsertExistingPartitionsBehavior::kError;
  uint32_t maxPartitionsPerWriters_ = 100;
  bool immutablePartitions_ = false;
  bool s3PathStyleAccess_ = false;
  std::string s3LogLevel_ = "FATAL";
  bool s3SSLEnabled_ = true;
  bool s3UseInstanceCredentials_ = false;
  std::string s3Endpoint_ = "";
  std::optional<std::string> s3AwsAccessKey_{};
  std::optional<std::string> s3AwsSecretKey_{};
  std::optional<std::string> s3IamRole_{};
  std::string s3IamRoleSessionName_ = "velox-session";
  std::string GCSEndpoint_ = "";
  std::string GCSScheme_ = "https";
  std::string GCSCredentials_ = "";
  bool orcUseColumnNames_ = false;
  bool fileColumnNamesReadAsLowerCase_ = false;
  int64_t maxCoalescedBytes_ = 128 << 20;
  int32_t maxCoalescedDistanceBytes_ = 512 << 10;
  int32_t numCacheFileHandles_ = 20'000;
  bool enableFileHandleCache_ = true;
  int32_t sortWriterMaxOutputRows_ = 1024;
  uint64_t sortWriterMaxOutputBytes_ = 10UL << 20;
  uint64_t orcWriterMaxStripeSize_ = 64L * 1024L * 1024L;
  uint64_t orcWriterMaxDictionaryMemory_ = 16L * 1024L * 1024L;

  std::unordered_map<std::string, std::function<void(HiveConfig*, std::string)>>
      configSetters = {
          {kInsertExistingPartitionsBehavior,
           setInsertExistingPartitionsBehavior},
          {kMaxPartitionsPerWriters, setMaxPartitionsPerWriters},
          {kImmutablePartitions, setImmutablePartitions},
          {kS3PathStyleAccess, setS3PathStyleAccess},
          {kS3LogLevel, setS3LogLevel},
          {kS3SSLEnabled, setS3SSLEnabled},
          {kS3UseInstanceCredentials, setS3UseInstanceCredentials},
          {kS3Endpoint, setS3Endpoint},
          {kS3AwsAccessKey, setS3AwsAccessKey},
          {kS3AwsSecretKey, setS3AwsSecretKey},
          {kS3IamRole, setS3IamRole},
          {kS3IamRoleSessionName, setS3IamRoleSessionName},
          {kGCSEndpoint, setGCSEndpoint},
          {kGCSScheme, setGCSScheme},
          {kGCSCredentials, setGCSCredentials},
          {kOrcUseColumnNames, setOrcUseColumnNames},
          {kFileColumnNamesReadAsLowerCase, setFileColumnNamesReadAsLowerCase},
          {kMaxCoalescedBytes, setMaxCoalescedBytes},
          {kMaxCoalescedDistanceBytes, setMaxCoalescedDistanceBytes},
          {kNumCacheFileHandles, setNumCacheFileHandles},
          {kEnableFileHandleCache, setEnableFileHandleCache},
          {kOrcWriterMaxStripeSize, setOrcWriterMaxStripeSize},
          {kOrcWriterMaxDictionaryMemory, setOrcWriterMaxDictionaryMemory},
          {kSortWriterMaxOutputRows, setSortWriterMaxOutputRows},
          {kSortWriterMaxOutputBytes, setSortWriterMaxOutputBytes},
  };
};

} // namespace facebook::velox::connector::hive
