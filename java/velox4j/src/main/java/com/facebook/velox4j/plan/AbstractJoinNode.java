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

import com.fasterxml.jackson.annotation.JsonGetter;

import com.facebook.velox4j.expression.FieldAccessTypedExpr;
import com.facebook.velox4j.expression.TypedExpr;
import com.facebook.velox4j.join.JoinType;
import com.facebook.velox4j.type.RowType;

public abstract class AbstractJoinNode extends PlanNode {
  private final List<PlanNode> sources;
  private final JoinType joinType;
  private final List<FieldAccessTypedExpr> leftKeys;
  private final List<FieldAccessTypedExpr> rightKeys;
  private final TypedExpr filter;
  private final RowType outputType;

  public AbstractJoinNode(
      String id,
      JoinType joinType,
      List<FieldAccessTypedExpr> leftKeys,
      List<FieldAccessTypedExpr> rightKeys,
      TypedExpr filter,
      PlanNode left,
      PlanNode right,
      RowType outputType) {
    super(id);
    this.sources = List.of(left, right);
    this.joinType = joinType;
    this.leftKeys = leftKeys;
    this.rightKeys = rightKeys;
    this.filter = filter;
    this.outputType = outputType;
  }

  @Override
  protected List<PlanNode> getSources() {
    return sources;
  }

  @JsonGetter("joinType")
  public JoinType getJoinType() {
    return joinType;
  }

  @JsonGetter("leftKeys")
  public List<FieldAccessTypedExpr> getLeftKeys() {
    return leftKeys;
  }

  @JsonGetter("rightKeys")
  public List<FieldAccessTypedExpr> getRightKeys() {
    return rightKeys;
  }

  @JsonGetter("filter")
  public TypedExpr getFilter() {
    return filter;
  }

  @JsonGetter("outputType")
  public RowType getOutputType() {
    return outputType;
  }
}
