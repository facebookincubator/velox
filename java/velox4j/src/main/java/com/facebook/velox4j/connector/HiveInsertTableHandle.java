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
package com.facebook.velox4j.connector;

import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonProperty;

// TODO: `dwio::common::WriterOptions` has not serde from Velox so not included
//  in the Java class.
public class HiveInsertTableHandle extends ConnectorInsertTableHandle {
  private final List<HiveColumnHandle> inputColumns;
  private final LocationHandle locationHandle;
  private final FileFormat storageFormat;
  private final HiveBucketProperty bucketProperty;
  private final CompressionKind compressionKind;
  private final Map<String, String> serdeParameters;
  private final boolean ensureFiles;
  private final FileNameGenerator fileNameGenerator;

  @JsonCreator
  public HiveInsertTableHandle(
      @JsonProperty("inputColumns") List<HiveColumnHandle> inputColumns,
      @JsonProperty("locationHandle") LocationHandle locationHandle,
      @JsonProperty("tableStorageFormat") FileFormat storageFormat,
      @JsonProperty("bucketProperty") HiveBucketProperty bucketProperty,
      @JsonProperty("compressionKind") CompressionKind compressionKind,
      @JsonProperty("serdeParameters") Map<String, String> serdeParameters,
      @JsonProperty("ensureFiles") boolean ensureFiles,
      @JsonProperty("fileNameGenerator") FileNameGenerator fileNameGenerator) {
    this.inputColumns = inputColumns;
    this.locationHandle = locationHandle;
    this.storageFormat = storageFormat;
    this.bucketProperty = bucketProperty;
    this.compressionKind = compressionKind;
    this.serdeParameters = serdeParameters;
    this.ensureFiles = ensureFiles;
    this.fileNameGenerator = fileNameGenerator;
  }

  @JsonGetter("inputColumns")
  public List<HiveColumnHandle> getInputColumns() {
    return inputColumns;
  }

  @JsonGetter("locationHandle")
  public LocationHandle getLocationHandle() {
    return locationHandle;
  }

  @JsonGetter("tableStorageFormat")
  public FileFormat getStorageFormat() {
    return storageFormat;
  }

  @JsonGetter("bucketProperty")
  public HiveBucketProperty getBucketProperty() {
    return bucketProperty;
  }

  @JsonGetter("compressionKind")
  public CompressionKind getCompressionKind() {
    return compressionKind;
  }

  @JsonGetter("serdeParameters")
  public Map<String, String> getSerdeParameters() {
    return serdeParameters;
  }

  @JsonGetter("ensureFiles")
  public boolean ensureFiles() {
    return ensureFiles;
  }

  @JsonGetter("fileNameGenerator")
  public FileNameGenerator getFileNameGenerator() {
    return fileNameGenerator;
  }

  @Override
  public boolean supportsMultiThreading() {
    return true;
  }
}
