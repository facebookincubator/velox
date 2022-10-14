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

#include "velox/connectors/WriteProtocol.h"

namespace facebook::velox::connector::hive {

/// Parameters for Hive writers.
class HiveWriterParameters : public WriterParameters {
 public:
  enum class UpdateMode {
    kNew, // Write files to a new directory.
    kAppend, // Append files to an existing directory.
    kOverwrite, // Overwrite an existing directory.
  };

  HiveWriterParameters(
      UpdateMode updateMode,
      const std::string& targetFileName,
      const std::string& targetDirectory,
      const std::optional<std::string>& writeFileName = std::nullopt,
      const std::optional<std::string>& writeDirectory = std::nullopt)
      : WriterParameters(),
        updateMode_(updateMode),
        targetFileName_(targetFileName),
        targetDirectory_(targetDirectory),
        writeFileName_(writeFileName.value_or(targetFileName)),
        writeDirectory_(writeDirectory.value_or(targetDirectory)) {}

  UpdateMode updateMode() const {
    return updateMode_;
  }

  static std::string updateModeToString(UpdateMode updateMode) {
    switch (updateMode) {
      case UpdateMode::kNew:
        return "NEW";
      case UpdateMode::kAppend:
        return "APPEND";
      case UpdateMode::kOverwrite:
        return "OVERWRITE";
      default:
        VELOX_UNSUPPORTED("Unsupported update mode.");
    }
  }

  const std::string& targetFileName() const {
    return targetFileName_;
  }

  const std::string& writeFileName() const {
    return writeFileName_;
  }

  const std::string& targetDirectory() const {
    return targetDirectory_;
  }

  const std::string& writeDirectory() const {
    return writeDirectory_;
  }

 private:
  const UpdateMode updateMode_;
  /// The final name of a file after committing.
  const std::string targetFileName_;
  /// The final directory that a file should be in after committing.
  const std::string targetDirectory_;
  /// The temporary name of the file that a running writer writes to.
  const std::string writeFileName_;
  /// The temporary directory that a running writer writes to.
  const std::string writeDirectory_;
};

/// WriteProtocol base implementation for Hive writes. WriterParameters have the
/// write file name the same as the target file name, so no commit is needed for
/// file move.
class HiveNoCommitWriteProtocol : public WriteProtocol {
 public:
  ~HiveNoCommitWriteProtocol() override {}

  CommitStrategy getCommitStrategy() const override {
    return CommitStrategy::kNoCommit;
  }

  std::shared_ptr<const WriterParameters> getWriterParameters(
      const std::shared_ptr<const velox::connector::ConnectorInsertTableHandle>&
          tableHandle,
      const velox::connector::ConnectorQueryCtx* FOLLY_NONNULL
          connectorQueryCtx) const override;

  static bool registerProtocol() {
    return registerWriteProtocol(CommitStrategy::kNoCommit, []() {
      return std::make_shared<HiveNoCommitWriteProtocol>();
    });
  }
};

/// WriteProtocol implementation for Hive writes. WriterParameters have the
/// write file name different from the target file name. So commit is needed to
/// move write file to the target location.
class HiveTaskCommitWriteProtocol : public WriteProtocol {
 public:
  ~HiveTaskCommitWriteProtocol() override {}

  CommitStrategy getCommitStrategy() const override {
    return CommitStrategy::kTaskCommit;
  }

  std::shared_ptr<const WriterParameters> getWriterParameters(
      const std::shared_ptr<const velox::connector::ConnectorInsertTableHandle>&
          tableHandle,
      const velox::connector::ConnectorQueryCtx* FOLLY_NONNULL
          connectorQueryCtx) const override;

  static bool registerProtocol() {
    return registerWriteProtocol(CommitStrategy::kTaskCommit, []() {
      return std::make_shared<HiveTaskCommitWriteProtocol>();
    });
  }
};

} // namespace facebook::velox::connector::hive
