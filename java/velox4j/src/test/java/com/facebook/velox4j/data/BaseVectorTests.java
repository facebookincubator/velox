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
package com.facebook.velox4j.data;

import java.util.List;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.junit.Assert;

import com.facebook.velox4j.serde.Serde;
import com.facebook.velox4j.session.Session;
import com.facebook.velox4j.test.ResourceTests;
import com.facebook.velox4j.type.Type;

public final class BaseVectorTests {
  private BaseVectorTests() {}

  public static void assertEquals(BaseVector expected, BaseVector actual) {
    final Type typeExpected = expected.getType();
    final Type typeActual = actual.getType();
    Assert.assertEquals(Serde.toPrettyJson(typeExpected), Serde.toPrettyJson(typeActual));
    Assert.assertEquals(expected.toString(), actual.toString());
  }

  public static void assertEquals(
      List<? extends BaseVector> expected, List<? extends BaseVector> actual) {
    Assert.assertEquals(expected.size(), actual.size());
    for (int i = 0; i < expected.size(); i++) {
      assertEquals(expected.get(i), actual.get(i));
    }
  }

  public static BaseVector newSampleIntVector(Session session) {
    final BufferAllocator alloc = new RootAllocator();
    final IntVector arrowVector = new IntVector("foo", alloc);
    arrowVector.setValueCount(1);
    arrowVector.set(0, 15);
    final BaseVector baseVector = session.arrowOperations().fromArrowVector(alloc, arrowVector);
    arrowVector.close();
    return baseVector;
  }

  public static RowVector newSampleRowVector(Session session) {
    final String serialized =
        ResourceTests.readResourceAsString("vector/rowvector-1.b64").stripTrailing();
    final BaseVector deserialized = session.baseVectorOperations().deserializeOne(serialized);
    return deserialized.asRowVector();
  }
}
