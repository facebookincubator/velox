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

package com.facebook.velox4j.test.dataset.tpch;

import java.util.List;

import com.facebook.velox4j.type.BigIntType;
import com.facebook.velox4j.type.DecimalType;
import com.facebook.velox4j.type.RowType;
import com.facebook.velox4j.type.VarCharType;

public enum TpchTableName {
  REGION(
      "region/region.parquet",
      new RowType(
          List.of("r_regionkey", "r_name", "r_comment"),
          List.of(new BigIntType(), new VarCharType(), new VarCharType()))),

  CUSTOMER(
      "customer/customer.parquet",
      new RowType(
          List.of(
              "s_suppkey",
              "s_name",
              "s_address",
              "s_nationkey",
              "s_phone",
              "s_acctbal",
              "s_comment"),
          List.of(
              new BigIntType(),
              new VarCharType(),
              new VarCharType(),
              new BigIntType(),
              new VarCharType(),
              new DecimalType(12, 2),
              new VarCharType()))),

  NATION(
      "nation/nation.parquet",
      new RowType(
          List.of("n_nationkey", "n_name", "n_regionkey", "n_comment"),
          List.of(new BigIntType(), new VarCharType(), new BigIntType(), new VarCharType())));

  private final String relativePath;
  private final RowType schema;

  TpchTableName(String relativePath, RowType schema) {
    this.relativePath = relativePath;
    this.schema = schema;
  }

  public RowType schema() {
    return schema;
  }

  public String relativePath() {
    return relativePath;
  }
}
