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

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

public class HiveBucketConversion {
  private final int tableBucketCount;
  private final int partitionBucketCount;
  private final List<HiveColumnHandle> bucketColumnHandles;

  @JsonCreator
  public HiveBucketConversion(
      @JsonProperty("tableBucketCount") int tableBucketCount,
      @JsonProperty("partitionBucketCount") int partitionBucketCount,
      @JsonProperty("bucketColumnHandles") List<HiveColumnHandle> bucketColumnHandles) {
    this.tableBucketCount = tableBucketCount;
    this.partitionBucketCount = partitionBucketCount;
    this.bucketColumnHandles = bucketColumnHandles;
  }

  @JsonProperty("tableBucketCount")
  public int getTableBucketCount() {
    return tableBucketCount;
  }

  @JsonProperty("partitionBucketCount")
  public int getPartitionBucketCount() {
    return partitionBucketCount;
  }

  @JsonProperty("bucketColumnHandles")
  public List<HiveColumnHandle> getBucketColumnHandles() {
    return bucketColumnHandles;
  }
}
