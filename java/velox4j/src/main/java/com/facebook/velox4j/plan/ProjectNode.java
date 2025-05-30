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

import com.facebook.velox4j.expression.TypedExpr;

public class ProjectNode extends PlanNode {
  private final List<PlanNode> sources;
  private final List<String> names;
  private final List<TypedExpr> projections;

  @JsonCreator
  public ProjectNode(
      @JsonProperty("id") String id,
      @JsonProperty("sources") List<PlanNode> sources,
      @JsonProperty("names") List<String> names,
      @JsonProperty("projections") List<TypedExpr> projections) {
    super(id);
    this.sources = sources;
    this.names = names;
    this.projections = projections;
  }

  @Override
  protected List<PlanNode> getSources() {
    return sources;
  }

  @JsonGetter("names")
  public List<String> getNames() {
    return names;
  }

  @JsonGetter("projections")
  public List<TypedExpr> getProjections() {
    return projections;
  }
}
