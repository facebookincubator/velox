package com.meta.velox4j.data;

import java.util.List;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.junit.Assert;

import com.meta.velox4j.serde.Serde;
import com.meta.velox4j.session.Session;
import com.meta.velox4j.test.ResourceTests;
import com.meta.velox4j.type.Type;

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
    final BaseVector baseVector = session.arrowOps().fromArrowVector(alloc, arrowVector);
    arrowVector.close();
    return baseVector;
  }

  public static RowVector newSampleRowVector(Session session) {
    final String serialized = ResourceTests.readResourceAsString("vector/rowvector-1.b64");
    final BaseVector deserialized = session.baseVectorOps().deserializeOne(serialized);
    return deserialized.asRowVector();
  }
}
