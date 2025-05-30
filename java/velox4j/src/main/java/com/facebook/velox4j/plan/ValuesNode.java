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

import com.facebook.velox4j.data.BaseVectors;
import com.facebook.velox4j.data.RowVector;

public class ValuesNode extends PlanNode {
  private final String serializedRowVectors;
  private final boolean parallelizable;
  private final int repeatTimes;

  @JsonCreator
  public ValuesNode(
      @JsonProperty("id") String id,
      @JsonProperty("data") String serializedRowVectors,
      @JsonProperty("parallelizable") boolean parallelizable,
      @JsonProperty("repeatTimes") int repeatTimes) {
    super(id);
    this.serializedRowVectors = Preconditions.checkNotNull(serializedRowVectors);
    this.parallelizable = parallelizable;
    this.repeatTimes = repeatTimes;
  }

  public static ValuesNode create(
      String id, List<RowVector> vectors, boolean parallelizable, int repeatTimes) {
    return new ValuesNode(id, BaseVectors.serializeAll(vectors), parallelizable, repeatTimes);
  }

  @JsonGetter("data")
  public String getSerializedRowVectors() {
    return serializedRowVectors;
  }

  @JsonGetter("parallelizable")
  public boolean isParallelizable() {
    return parallelizable;
  }

  @JsonGetter("repeatTimes")
  public int getRepeatTimes() {
    return repeatTimes;
  }

  @Override
  protected List<PlanNode> getSources() {
    return List.of();
  }
}
