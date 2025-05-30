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
import com.fasterxml.jackson.annotation.JsonInclude;

import com.facebook.velox4j.serializable.ISerializable;

public abstract class PlanNode extends ISerializable {
  private final String id;

  protected PlanNode(String id) {
    this.id = id;
  }

  @JsonGetter("id")
  public String getId() {
    return id;
  }

  @JsonGetter("sources")
  @JsonInclude(JsonInclude.Include.NON_EMPTY)
  protected abstract List<PlanNode> getSources();
}
