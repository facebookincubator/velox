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
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonValue;

import com.facebook.velox4j.serializable.ISerializable;
import com.facebook.velox4j.type.Type;

public class HiveBucketProperty extends ISerializable {
  private final Kind kind;
  private final int bucketCount;
  private final List<String> bucketedBy;
  private final List<Type> bucketedTypes;
  private final List<HiveSortingColumn> sortedBy;

  @JsonCreator
  public HiveBucketProperty(
      @JsonProperty("kind") Kind kind,
      @JsonProperty("bucketCount") int bucketCount,
      @JsonProperty("bucketedBy") List<String> bucketedBy,
      @JsonProperty("bucketedTypes") List<Type> bucketedTypes,
      @JsonProperty("sortedBy") List<HiveSortingColumn> sortedBy) {
    this.kind = kind;
    this.bucketCount = bucketCount;
    this.bucketedBy = bucketedBy;
    this.bucketedTypes = bucketedTypes;
    this.sortedBy = sortedBy;
  }

  @JsonGetter("kind")
  public Kind getKind() {
    return kind;
  }

  @JsonGetter("bucketCount")
  public int getBucketCount() {
    return bucketCount;
  }

  @JsonGetter("bucketedBy")
  public List<String> getBucketedBy() {
    return bucketedBy;
  }

  @JsonGetter("bucketedTypes")
  public List<Type> getBucketedTypes() {
    return bucketedTypes;
  }

  @JsonGetter("sortedBy")
  public List<HiveSortingColumn> getSortedBy() {
    return sortedBy;
  }

  public enum Kind {
    HIVE_COMPATIBLE(0),
    PRESTO_NATIVE(1);

    private final int value;

    Kind(int value) {
      this.value = value;
    }

    @JsonValue
    public int toValue() {
      return value;
    }
  }
}
