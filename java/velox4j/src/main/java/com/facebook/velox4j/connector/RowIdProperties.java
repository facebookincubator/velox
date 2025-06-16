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

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.base.Preconditions;

public class RowIdProperties {
  private final long metadataVersion;
  private final long partitionId;
  private final String tableGuid;

  @JsonCreator
  public RowIdProperties(
      @JsonProperty("metadataVersion") long metadataVersion,
      @JsonProperty("partitionId") long partitionId,
      @JsonProperty("tableGuid") String tableGuid) {
    this.metadataVersion = metadataVersion;
    this.partitionId = partitionId;
    this.tableGuid = Preconditions.checkNotNull(tableGuid);
  }

  @JsonGetter("metadataVersion")
  public long getMetadataVersion() {
    return metadataVersion;
  }

  @JsonGetter("partitionId")
  public long getPartitionId() {
    return partitionId;
  }

  @JsonGetter("tableGuid")
  public String getTableGuid() {
    return tableGuid;
  }
}
