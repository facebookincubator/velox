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

import java.util.List;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.base.Preconditions;

import com.facebook.velox4j.exception.VeloxException;

public class RowType extends Type {
  private final List<String> names;
  private final List<Type> children;

  @JsonCreator
  public RowType(
      @JsonProperty("names") List<String> names, @JsonProperty("cTypes") List<Type> children) {
    Preconditions.checkArgument(
        names.size() == children.size(), "RowType should have same number of names and children");
    this.names = names;
    this.children = children;
  }

  @JsonProperty("names")
  public List<String> getNames() {
    return names;
  }

  @JsonProperty("cTypes")
  public List<Type> getChildren() {
    return children;
  }

  public int size() {
    return names.size();
  }

  public Type findChild(String name) {
    int index = names.indexOf(name);
    if (index == -1) {
      throw new VeloxException("Field " + name + " not found in RowType");
    }
    return children.get(index);
  }
}
