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
package com.facebook.velox4j.eval;

import java.util.List;

import org.junit.*;

import com.facebook.velox4j.Velox4j;
import com.facebook.velox4j.conf.Config;
import com.facebook.velox4j.conf.ConnectorConfig;
import com.facebook.velox4j.data.BaseVector;
import com.facebook.velox4j.data.BaseVectorTests;
import com.facebook.velox4j.data.RowVector;
import com.facebook.velox4j.data.SelectivityVector;
import com.facebook.velox4j.expression.CallTypedExpr;
import com.facebook.velox4j.expression.FieldAccessTypedExpr;
import com.facebook.velox4j.memory.AllocationListener;
import com.facebook.velox4j.memory.MemoryManager;
import com.facebook.velox4j.session.Session;
import com.facebook.velox4j.test.ResourceTests;
import com.facebook.velox4j.test.Velox4jTests;
import com.facebook.velox4j.type.BigIntType;

public class EvaluationTest {
  private static MemoryManager memoryManager;
  private static Session session;

  @BeforeClass
  public static void beforeClass() throws Exception {
    Velox4jTests.ensureInitialized();
    memoryManager = MemoryManager.create(AllocationListener.NOOP);
  }

  @AfterClass
  public static void afterClass() throws Exception {
    memoryManager.close();
  }

  @Before
  public void setUp() throws Exception {
    session = Velox4j.newSession(memoryManager);
  }

  @After
  public void tearDown() throws Exception {
    session.close();
  }

  @Test
  public void testFieldAccess() {
    final RowVector input = BaseVectorTests.newSampleRowVector(session);
    final int size = input.getSize();
    final SelectivityVector sv = session.selectivityVectorOperations().create(size);
    final Evaluation expr =
        new Evaluation(
            FieldAccessTypedExpr.create(new BigIntType(), "c0"),
            Config.empty(),
            ConnectorConfig.empty());
    final Evaluator evaluator = session.evaluationOperations().createEvaluator(expr);
    final BaseVector out = evaluator.eval(sv, input);
    final String outString = out.toString();
    Assert.assertEquals(
        ResourceTests.readResourceAsString("eval-output/field-access-1.arrow").stripTrailing(),
        outString);
  }

  @Test
  public void testMultipleEvalCalls() {
    final RowVector input = BaseVectorTests.newSampleRowVector(session);
    final int size = input.getSize();
    final SelectivityVector sv = session.selectivityVectorOperations().create(size);
    final Evaluation expr =
        new Evaluation(
            FieldAccessTypedExpr.create(new BigIntType(), "c0"),
            Config.empty(),
            ConnectorConfig.empty());
    final Evaluator evaluator = session.evaluationOperations().createEvaluator(expr);
    final String expected = ResourceTests.readResourceAsString("eval-output/field-access-1.arrow");
    for (int i = 0; i < 10; i++) {
      final BaseVector out = evaluator.eval(sv, input);
      final String outString = out.toString();
      Assert.assertEquals(expected.stripTrailing(), outString);
    }
  }

  @Test
  public void testMultiply() {
    final RowVector input = BaseVectorTests.newSampleRowVector(session);
    final int size = input.getSize();
    final SelectivityVector sv = session.selectivityVectorOperations().create(size);
    final Evaluation expr =
        new Evaluation(
            new CallTypedExpr(
                new BigIntType(),
                List.of(
                    FieldAccessTypedExpr.create(new BigIntType(), "c0"),
                    FieldAccessTypedExpr.create(new BigIntType(), "a1")),
                "multiply"),
            Config.empty(),
            ConnectorConfig.empty());
    final Evaluator evaluator = session.evaluationOperations().createEvaluator(expr);
    final BaseVector out = evaluator.eval(sv, input);
    final String outString = out.toString();
    Assert.assertEquals(
        ResourceTests.readResourceAsString("eval-output/multiply-1.arrow").stripTrailing(),
        outString);
  }
}
