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
package com.facebook.velox4j.aggregate;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonProperty;

import com.facebook.velox4j.expression.CallTypedExpr;
import com.facebook.velox4j.expression.FieldAccessTypedExpr;
import com.facebook.velox4j.sort.SortOrder;
import com.facebook.velox4j.type.Type;

public class Aggregate {
  public final CallTypedExpr call;
  public final List<Type> rawInputTypes;
  public final FieldAccessTypedExpr mask;
  public final List<FieldAccessTypedExpr> sortingKeys;
  public final List<SortOrder> sortingOrders;
  public final boolean distinct;

  @JsonCreator
  public Aggregate(
      @JsonProperty("call") CallTypedExpr call,
      @JsonProperty("rawInputTypes") List<Type> rawInputTypes,
      @JsonProperty("mask") FieldAccessTypedExpr mask,
      @JsonProperty("sortingKeys") List<FieldAccessTypedExpr> sortingKeys,
      @JsonProperty("sortingOrders") List<SortOrder> sortingOrders,
      @JsonProperty("distinct") boolean distinct) {
    this.call = call;
    this.rawInputTypes = rawInputTypes;
    this.mask = mask;
    this.sortingKeys = sortingKeys;
    this.sortingOrders = sortingOrders;
    this.distinct = distinct;
  }

  @JsonGetter("call")
  public CallTypedExpr getCall() {
    return call;
  }

  @JsonGetter("rawInputTypes")
  public List<Type> getRawInputTypes() {
    return rawInputTypes;
  }

  @JsonGetter("mask")
  public FieldAccessTypedExpr getMask() {
    return mask;
  }

  @JsonGetter("sortingKeys")
  public List<FieldAccessTypedExpr> getSortingKeys() {
    return sortingKeys;
  }

  @JsonGetter("sortingOrders")
  public List<SortOrder> getSortingOrders() {
    return sortingOrders;
  }

  @JsonGetter("distinct")
  public boolean isDistinct() {
    return distinct;
  }
}
