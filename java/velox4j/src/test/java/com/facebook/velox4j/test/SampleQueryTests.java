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
package com.facebook.velox4j.test;

import java.util.List;

import com.facebook.velox4j.iterator.UpIterator;
import com.facebook.velox4j.type.BigIntType;
import com.facebook.velox4j.type.RowType;

public final class SampleQueryTests {
  private static final String SAMPLE_QUERY_PATH = "query/example-1.json";
  private static final String SAMPLE_QUERY_OUTPUT_PATH = "query-output/example-1.tsv";
  private static final RowType SAMPLE_QUERY_TYPE =
      new RowType(
          List.of("c0", "a0", "a1"), List.of(new BigIntType(), new BigIntType(), new BigIntType()));

  public static RowType getSchema() {
    return SAMPLE_QUERY_TYPE;
  }

  public static String readQueryJson() {
    return ResourceTests.readResourceAsString(SAMPLE_QUERY_PATH);
  }

  public static void assertIterator(UpIterator itr) {
    assertIterator(itr, 1);
  }

  public static void assertIterator(UpIterator itr, int repeatTimes) {
    UpIteratorTests.IteratorAssertionBuilder builder =
        UpIteratorTests.assertIterator(itr).assertNumRowVectors(repeatTimes);
    for (int i = 0; i < repeatTimes; i++) {
      builder =
          builder.assertRowVectorToString(
              i, ResourceTests.readResourceAsString(SAMPLE_QUERY_OUTPUT_PATH));
    }
    builder.run();
  }
}
