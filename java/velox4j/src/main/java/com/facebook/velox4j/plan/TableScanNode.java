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
package com.facebook.velox4j.plan;

import java.util.Collections;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonProperty;

import com.facebook.velox4j.connector.Assignment;
import com.facebook.velox4j.connector.ConnectorTableHandle;
import com.facebook.velox4j.type.Type;

public class TableScanNode extends PlanNode {
  private final Type outputType;
  private final ConnectorTableHandle tableHandle;
  private final List<Assignment> assignments;

  @JsonCreator
  public TableScanNode(
      @JsonProperty("id") String id,
      @JsonProperty("outputType") Type outputType,
      @JsonProperty("tableHandle") ConnectorTableHandle tableHandle,
      @JsonProperty("assignments") List<Assignment> assignments) {
    super(id);
    this.outputType = outputType;
    this.tableHandle = tableHandle;
    this.assignments = assignments;
  }

  @JsonGetter("outputType")
  public Type getOutputType() {
    return outputType;
  }

  @JsonGetter("tableHandle")
  public ConnectorTableHandle getTableHandle() {
    return tableHandle;
  }

  @JsonGetter("assignments")
  public List<Assignment> getAssignments() {
    return assignments;
  }

  @Override
  protected List<PlanNode> getSources() {
    return Collections.emptyList();
  }
}
