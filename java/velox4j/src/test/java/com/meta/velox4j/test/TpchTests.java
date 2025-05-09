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
package com.meta.velox4j.test;

import java.io.File;
import java.util.List;

import com.meta.velox4j.type.BigIntType;
import com.meta.velox4j.type.DecimalType;
import com.meta.velox4j.type.RowType;
import com.meta.velox4j.type.VarcharType;

public final class TpchTests {
  private static final String DATA_DIRECTORY = "data/tpch-sf0.1";

  public enum Table {
    REGION(
        "region/region.parquet",
        new RowType(
            List.of("r_regionkey", "r_name", "r_comment"),
            List.of(new BigIntType(), new VarcharType(), new VarcharType()))),

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
                new VarcharType(),
                new VarcharType(),
                new BigIntType(),
                new VarcharType(),
                new DecimalType(12, 2),
                new VarcharType()))),

    NATION(
        "nation/nation.parquet",
        new RowType(
            List.of("n_nationkey", "n_name", "n_regionkey", "n_comment"),
            List.of(new BigIntType(), new VarcharType(), new BigIntType(), new VarcharType())));

    private final RowType schema;
    private final File file;

    Table(String fileName, RowType schema) {
      this.schema = schema;
      this.file = ResourceTests.copyResourceToTmp(String.format("%s/%s", DATA_DIRECTORY, fileName));
    }

    public RowType schema() {
      return schema;
    }

    public File file() {
      return file;
    }
  }
}
