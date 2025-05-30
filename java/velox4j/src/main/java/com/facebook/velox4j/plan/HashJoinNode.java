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
import com.google.common.base.Preconditions;

import com.facebook.velox4j.expression.FieldAccessTypedExpr;
import com.facebook.velox4j.expression.TypedExpr;
import com.facebook.velox4j.join.JoinType;
import com.facebook.velox4j.type.RowType;

public class HashJoinNode extends AbstractJoinNode {
  private final boolean nullAware;

  public HashJoinNode(
      String id,
      JoinType joinType,
      List<FieldAccessTypedExpr> leftKeys,
      List<FieldAccessTypedExpr> rightKeys,
      TypedExpr filter,
      PlanNode left,
      PlanNode right,
      RowType outputType,
      boolean nullAware) {
    super(id, joinType, leftKeys, rightKeys, filter, left, right, outputType);
    this.nullAware = nullAware;
  }

  @JsonCreator
  private static HashJoinNode create(
      @JsonProperty("id") String id,
      @JsonProperty("joinType") JoinType joinType,
      @JsonProperty("leftKeys") List<FieldAccessTypedExpr> leftKeys,
      @JsonProperty("rightKeys") List<FieldAccessTypedExpr> rightKeys,
      @JsonProperty("filter") TypedExpr filter,
      @JsonProperty("sources") List<PlanNode> sources,
      @JsonProperty("outputType") RowType outputType,
      @JsonProperty("nullAware") boolean nullAware) {
    Preconditions.checkArgument(
        sources.size() == 2, "HashJoinNode should have 2 sources, but has %s", sources.size());
    return new HashJoinNode(
        id,
        joinType,
        leftKeys,
        rightKeys,
        filter,
        sources.get(0),
        sources.get(1),
        outputType,
        nullAware);
  }

  @JsonGetter("nullAware")
  public boolean isNullAware() {
    return nullAware;
  }
}
