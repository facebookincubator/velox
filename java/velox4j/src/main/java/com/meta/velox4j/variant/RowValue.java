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
package com.meta.velox4j.variant;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonProperty;

public class RowValue extends Variant {
  private final List<Variant> row;

  @JsonCreator
  public RowValue(@JsonProperty("value") List<Variant> row) {
    this.row = Collections.unmodifiableList(row);
  }

  @JsonGetter("value")
  public List<Variant> getRow() {
    return row;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    RowValue rowValue = (RowValue) o;
    return Objects.equals(row, rowValue.row);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(row);
  }

  @Override
  public String toString() {
    return "RowValue{" + "row=" + row + '}';
  }
}
