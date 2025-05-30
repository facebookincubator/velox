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
package com.facebook.velox4j.connector;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonProperty;

import com.facebook.velox4j.expression.TypedExpr;
import com.facebook.velox4j.type.RowType;

public class HiveTableHandle extends ConnectorTableHandle {
  private final String tableName;
  private final boolean filterPushdownEnabled;
  private final List<SubfieldFilter> subfieldFilters;
  private final TypedExpr remainingFilter;
  private final RowType dataColumns;
  private final Map<String, String> tableParameters;

  @JsonCreator
  public HiveTableHandle(
      @JsonProperty("connectorId") String connectorId,
      @JsonProperty("tableName") String tableName,
      @JsonProperty("filterPushdownEnabled") boolean filterPushdownEnabled,
      @JsonProperty("subfieldFilters") List<SubfieldFilter> subfieldFilters,
      @JsonProperty("remainingFilter") TypedExpr remainingFilter,
      @JsonProperty("dataColumns") RowType dataColumns,
      @JsonProperty("tableParameters") Map<String, String> tableParameters) {
    super(connectorId);
    this.tableName = tableName;
    this.filterPushdownEnabled = filterPushdownEnabled;
    this.subfieldFilters = subfieldFilters;
    this.remainingFilter = remainingFilter;
    this.dataColumns = dataColumns;
    // Use of tree map guarantees the key serialization order.
    this.tableParameters =
        tableParameters == null
            ? Collections.emptyMap()
            : Collections.unmodifiableSortedMap(new TreeMap<>(tableParameters));
  }

  @JsonGetter("tableName")
  public String getTableName() {
    return tableName;
  }

  @JsonGetter("filterPushdownEnabled")
  public boolean isFilterPushdownEnabled() {
    return filterPushdownEnabled;
  }

  @JsonGetter("subfieldFilters")
  public List<SubfieldFilter> getSubfieldFilters() {
    return subfieldFilters;
  }

  @JsonGetter("remainingFilter")
  public TypedExpr getRemainingFilter() {
    return remainingFilter;
  }

  @JsonGetter("dataColumns")
  public RowType getDataColumns() {
    return dataColumns;
  }

  @JsonGetter("tableParameters")
  public Map<String, String> getTableParameters() {
    return tableParameters;
  }
}
