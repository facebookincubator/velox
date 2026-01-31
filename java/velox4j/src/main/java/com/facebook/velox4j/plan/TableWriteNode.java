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

import java.util.List;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonProperty;

import com.facebook.velox4j.connector.CommitStrategy;
import com.facebook.velox4j.connector.ConnectorInsertTableHandle;
import com.facebook.velox4j.connector.InsertTableHandle;
import com.facebook.velox4j.type.RowType;

public class TableWriteNode extends PlanNode {
  private final RowType columns;
  private final List<String> columnNames;
  private final AggregationNode aggregationNode;
  private final InsertTableHandle insertTableHandle;
  private final boolean hasPartitioningScheme;
  private final RowType outputType;
  private final CommitStrategy commitStrategy;
  private final List<PlanNode> sources;

  @JsonCreator
  public TableWriteNode(
      @JsonProperty("id") String id,
      @JsonProperty("columns") RowType columns,
      @JsonProperty("columnNames") List<String> columnNames,
      @JsonProperty("aggregationNode") AggregationNode aggregationNode,
      @JsonProperty("connectorId") String connectorId,
      @JsonProperty("connectorInsertTableHandle") ConnectorInsertTableHandle insertTableHandle,
      @JsonProperty("hasPartitioningScheme") boolean hasPartitioningScheme,
      @JsonProperty("outputType") RowType outputType,
      @JsonProperty("commitStrategy") CommitStrategy commitStrategy,
      @JsonProperty("sources") List<PlanNode> sources) {
    super(id);
    this.columns = columns;
    this.columnNames = columnNames;
    this.aggregationNode = aggregationNode;
    this.insertTableHandle = new InsertTableHandle(connectorId, insertTableHandle);
    this.hasPartitioningScheme = hasPartitioningScheme;
    this.outputType = outputType;
    this.commitStrategy = commitStrategy;
    this.sources = sources;
  }

  @JsonGetter("columns")
  public RowType getColumns() {
    return columns;
  }

  @JsonGetter("columnNames")
  public List<String> getColumnNames() {
    return columnNames;
  }

  @JsonGetter("aggregationNode")
  public AggregationNode getAggregationNode() {
    return aggregationNode;
  }

  @JsonGetter("connectorId")
  public String getConnectorId() {
    return insertTableHandle.getConnectorId();
  }

  @JsonGetter("connectorInsertTableHandle")
  public ConnectorInsertTableHandle getInsertTableHandle() {
    return insertTableHandle.connectorInsertTableHandle();
  }

  @JsonGetter("hasPartitioningScheme")
  public boolean hasPartitioningScheme() {
    return hasPartitioningScheme;
  }

  @JsonGetter("outputType")
  public RowType getOutputType() {
    return outputType;
  }

  @JsonGetter("commitStrategy")
  public CommitStrategy getCommitStrategy() {
    return commitStrategy;
  }

  @Override
  protected List<PlanNode> getSources() {
    return sources;
  }
}
