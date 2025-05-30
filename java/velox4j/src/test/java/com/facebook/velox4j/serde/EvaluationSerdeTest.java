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
package com.facebook.velox4j.serde;

import java.util.Collections;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import com.facebook.velox4j.conf.Config;
import com.facebook.velox4j.conf.ConnectorConfig;
import com.facebook.velox4j.eval.Evaluation;
import com.facebook.velox4j.expression.CallTypedExpr;
import com.facebook.velox4j.memory.AllocationListener;
import com.facebook.velox4j.memory.MemoryManager;
import com.facebook.velox4j.test.Velox4jTests;
import com.facebook.velox4j.type.IntegerType;

public class EvaluationSerdeTest {
  private static MemoryManager memoryManager;

  @BeforeClass
  public static void beforeClass() throws Exception {
    Velox4jTests.ensureInitialized();
    memoryManager = MemoryManager.create(AllocationListener.NOOP);
  }

  @AfterClass
  public static void afterClass() throws Exception {
    memoryManager.close();
  }

  @Test
  public void testExpression() {
    final CallTypedExpr expr =
        new CallTypedExpr(new IntegerType(), Collections.emptyList(), "random_int");
    final Evaluation evaluation = new Evaluation(expr, Config.empty(), ConnectorConfig.empty());
    SerdeTests.testISerializableRoundTrip(evaluation);
  }
}
