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

public class LimitNode extends PlanNode {
  private final List<PlanNode> sources;
  private final long offset;
  private final long count;
  private final boolean partial;

  @JsonCreator
  public LimitNode(
      @JsonProperty("id") String id,
      @JsonProperty("sources") List<PlanNode> sources,
      @JsonProperty("offset") long offset,
      @JsonProperty("count") long count,
      @JsonProperty("partial") boolean partial) {
    super(id);
    this.sources = sources;
    this.offset = offset;
    this.count = count;
    this.partial = partial;
  }

  @Override
  protected List<PlanNode> getSources() {
    return sources;
  }

  @JsonGetter("offset")
  public long getOffset() {
    return offset;
  }

  @JsonGetter("count")
  public long getCount() {
    return count;
  }

  @JsonGetter("partial")
  public boolean isPartial() {
    return partial;
  }
}
