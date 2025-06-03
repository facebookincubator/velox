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
package com.facebook.velox4j.variant;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

public class ArrayValue extends Variant {
  private final List<Variant> array;

  @JsonCreator
  public ArrayValue(@JsonProperty("value") List<Variant> array) {
    if (array == null) {
      this.array = null;
      return;
    }
    Variants.checkSameType(array);
    this.array = Collections.unmodifiableList(array);
  }

  @JsonGetter("value")
  @JsonInclude(JsonInclude.Include.ALWAYS)
  public List<Variant> getArray() {
    return array;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    ArrayValue that = (ArrayValue) o;
    return Objects.equals(array, that.array);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(array);
  }

  @Override
  public String toString() {
    return "ArrayValue{" + "array=" + array + '}';
  }
}
