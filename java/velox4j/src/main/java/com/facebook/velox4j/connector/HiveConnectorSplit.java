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

import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;

import com.fasterxml.jackson.annotation.*;
import com.google.common.base.Preconditions;

// TODO: Add a builder for this class.
public class HiveConnectorSplit extends ConnectorSplit {
  private final String filePath;
  private final FileFormat fileFormat;
  private final long start;
  private final long length;
  private final Map<String, Optional<String>> partitionKeys;
  private final OptionalInt tableBucketNumber;
  private final Optional<HiveBucketConversion> bucketConversion;
  private final Map<String, String> customSplitInfo;
  private final Optional<String> extraFileInfo;
  private final Map<String, String> serdeParameters;
  private final Map<String, String> infoColumns;
  private final Optional<FileProperties> properties;
  private final Optional<RowIdProperties> rowIdProperties;

  @JsonCreator
  public HiveConnectorSplit(
      @JsonProperty("connectorId") String connectorId,
      @JsonProperty("splitWeight") long splitWeight,
      @JsonProperty("cacheable") boolean cacheable,
      @JsonProperty("filePath") String filePath,
      @JsonProperty("fileFormat") FileFormat fileFormat,
      @JsonProperty("start") long start,
      @JsonProperty("length") long length,
      @JsonProperty("partitionKeys") Map<String, Optional<String>> partitionKeys,
      @JsonProperty("tableBucketNumber") OptionalInt tableBucketNumber,
      @JsonProperty("bucketConversion") Optional<HiveBucketConversion> bucketConversion,
      @JsonProperty("customSplitInfo") Map<String, String> customSplitInfo,
      @JsonProperty("extraFileInfo") Optional<String> extraFileInfo,
      @JsonProperty("serdeParameters") Map<String, String> serdeParameters,
      @JsonProperty("infoColumns") Map<String, String> infoColumns,
      @JsonProperty("properties") Optional<FileProperties> properties,
      @JsonProperty("rowIdProperties") Optional<RowIdProperties> rowIdProperties) {
    super(connectorId, splitWeight, cacheable);
    this.filePath = Preconditions.checkNotNull(filePath);
    this.fileFormat = Preconditions.checkNotNull(fileFormat);
    this.start = start;
    this.length = length;
    this.partitionKeys = Preconditions.checkNotNull(partitionKeys);
    this.tableBucketNumber = tableBucketNumber;
    this.bucketConversion = bucketConversion;
    this.customSplitInfo = Preconditions.checkNotNull(customSplitInfo);
    this.extraFileInfo = extraFileInfo;
    this.serdeParameters = Preconditions.checkNotNull(serdeParameters);
    this.infoColumns = Preconditions.checkNotNull(infoColumns);
    this.properties = properties;
    this.rowIdProperties = rowIdProperties;
  }

  @JsonGetter("splitWeight")
  public long getSplitWeight() {
    return super.getSplitWeight();
  }

  @JsonGetter("cacheable")
  public boolean isCacheable() {
    return super.isCacheable();
  }

  @JsonGetter("filePath")
  public String getFilePath() {
    return filePath;
  }

  @JsonGetter("fileFormat")
  public FileFormat getFileFormat() {
    return fileFormat;
  }

  @JsonGetter("start")
  public long getStart() {
    return start;
  }

  @JsonGetter("length")
  public long getLength() {
    return length;
  }

  @JsonGetter("partitionKeys")
  public Map<String, Optional<String>> getPartitionKeys() {
    return partitionKeys;
  }

  @JsonGetter("tableBucketNumber")
  public OptionalInt getTableBucketNumber() {
    return tableBucketNumber;
  }

  @JsonGetter("bucketConversion")
  public Optional<HiveBucketConversion> getBucketConversion() {
    return bucketConversion;
  }

  @JsonGetter("customSplitInfo")
  public Map<String, String> getCustomSplitInfo() {
    return customSplitInfo;
  }

  @JsonGetter("extraFileInfo")
  public Optional<String> getExtraFileInfo() {
    return extraFileInfo;
  }

  @JsonGetter("serdeParameters")
  public Map<String, String> getSerdeParameters() {
    return serdeParameters;
  }

  @JsonGetter("infoColumns")
  public Map<String, String> getInfoColumns() {
    return infoColumns;
  }

  @JsonGetter("properties")
  public Optional<FileProperties> getProperties() {
    return properties;
  }

  @JsonGetter("rowIdProperties")
  public Optional<RowIdProperties> getRowIdProperties() {
    return rowIdProperties;
  }
}
