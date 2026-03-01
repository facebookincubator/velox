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
package com.facebook.velox4j.type;

import java.util.Collections;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.base.Preconditions;

public class ArrayType extends Type {
  private final List<Type> children;

  @JsonCreator
  private ArrayType(@JsonProperty("cTypes") List<Type> children) {
    Preconditions.checkArgument(
        children.size() == 1, "ArrayType should have 1 child, but has %s", children.size());
    this.children = children;
  }

  public static ArrayType create(Type child) {
    return new ArrayType(Collections.singletonList(child));
  }

  @JsonGetter("cTypes")
  public List<Type> getChildren() {
    return children;
  }
}
