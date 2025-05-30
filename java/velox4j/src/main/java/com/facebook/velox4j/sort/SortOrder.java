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
package com.facebook.velox4j.sort;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

public class SortOrder {
  private final boolean ascending;
  private final boolean nullsFirst;

  @JsonCreator
  public SortOrder(
      @JsonProperty("ascending") boolean ascending,
      @JsonProperty("nullsFirst") boolean nullsFirst) {
    this.ascending = ascending;
    this.nullsFirst = nullsFirst;
  }

  @JsonProperty("ascending")
  public boolean isAscending() {
    return ascending;
  }

  @JsonProperty("nullsFirst")
  public boolean isNullsFirst() {
    return nullsFirst;
  }
}
