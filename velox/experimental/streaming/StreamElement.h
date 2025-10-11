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

#include "velox/vector/ComplexVector.h"

namespace facebook::velox::streaming {

/**
 * Streaming operators may be RowVector or Watermark, and it may need
 * some additional info to tell the caller who generate the result.
 */
class StreamElement {
 public:
  StreamElement(std::string nodeId) : nodeId_(std::move(nodeId)) {}

  virtual bool isWatermark() = 0;

  virtual bool isRecord() = 0;

  const std::string nodeId() const {
    return nodeId_;
  }

 private:
  // Node ID of the operator that generates this element.
  const std::string nodeId_;
};

using StreamElementPtr = std::shared_ptr<StreamElement>;

class Watermark :  public StreamElement {
 public:
  Watermark(std::string nodeId, long timestamp)
      : StreamElement(nodeId), timestamp_(timestamp) {}

  long timestamp() const {
    return timestamp_;
  }

  bool isWatermark() override {
    return true;
  }

  bool isRecord() override {
    return false;
  }

 private:
  const long timestamp_;
};

class StreamRecord :  public StreamElement {
 public:
  StreamRecord(std::string nodeId, RowVectorPtr record)
      : StreamElement(nodeId),
        record_(std::move(record)),
        timestamp_(-1),
        hasTimestamp_(false),
        key_(-1) {}

  StreamRecord(std::string nodeId, RowVectorPtr record, long timestamp)
      : StreamElement(nodeId),
        record_(std::move(record)),
        timestamp_(timestamp),
        hasTimestamp_(true),
        key_(-1) {}

  StreamRecord(std::string nodeId, int key, RowVectorPtr record)
      : StreamElement(nodeId),
        record_(std::move(record)),
        timestamp_(-1),
        hasTimestamp_(false),
        key_(key) {}

  const RowVectorPtr& record() const {
    return record_;
  }

  long timestamp() const {
    return timestamp_;
  }

  int key() const {
    return key_;
  }

  bool isWatermark() override {
    return false;
  }

  bool isRecord() override {
    return true;
  }

  bool hasTimestamp() const {
    return hasTimestamp_;
  }

 private:
  const RowVectorPtr record_;
  const long timestamp_;
  bool hasTimestamp_ = false;
  const int key_;
};
} // namespace facebook::velox::streaming
