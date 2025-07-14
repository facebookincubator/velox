package com.meta.velox4j.data;

import java.util.List;

import org.junit.*;

import com.meta.velox4j.Velox4j;
import com.meta.velox4j.memory.AllocationListener;
import com.meta.velox4j.memory.MemoryManager;
import com.meta.velox4j.serde.Serde;
import com.meta.velox4j.session.Session;
import com.meta.velox4j.test.ResourceTests;
import com.meta.velox4j.test.Velox4jTests;
import com.meta.velox4j.type.IntegerType;
import com.meta.velox4j.type.RealType;
import com.meta.velox4j.type.RowType;
import com.meta.velox4j.type.Type;

public class BaseVectorTest {
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
  public void testCreateEmpty1() {
    final Type type = new RealType();
    final BaseVector vector = session.baseVectorOps().createEmpty(type);
    Assert.assertEquals(Serde.toPrettyJson(type), Serde.toPrettyJson(vector.getType()));
    Assert.assertEquals(0, vector.getSize());
  }

  @Test
  public void testCreateEmpty2() {
    final Type type =
        new RowType(List.of("foo2", "bar2"), List.of(new IntegerType(), new IntegerType()));
    final BaseVector vector = session.baseVectorOps().createEmpty(type);
    Assert.assertEquals(Serde.toPrettyJson(type), Serde.toPrettyJson(vector.getType()));
    Assert.assertEquals(0, vector.getSize());
  }

  @Test
  public void testToString() {
    final RowVector input = BaseVectorTests.newSampleRowVector(session);
    Assert.assertEquals(
        ResourceTests.readResourceAsString("vector-output/to-string-1.tsv"), input.toString());
  }

  @Test
  public void testSlice() {
    final RowVector input = BaseVectorTests.newSampleRowVector(session);
    Assert.assertEquals(3, input.getSize());
    final RowVector sliced1 = input.slice(0, 2).asRowVector();
    final RowVector sliced2 = input.slice(2, 1).asRowVector();
    Assert.assertEquals(
        ResourceTests.readResourceAsString("vector-output/slice-1.tsv"), sliced1.toString());
    Assert.assertEquals(
        ResourceTests.readResourceAsString("vector-output/slice-2.tsv"), sliced2.toString());
  }

  @Test
  public void testAppend() {
    final RowVector input1 = BaseVectorTests.newSampleRowVector(session);
    final RowVector input2 = BaseVectorTests.newSampleRowVector(session);
    Assert.assertEquals(3, input1.getSize());
    Assert.assertEquals(3, input2.getSize());
    input1.append(input2);
    Assert.assertEquals(6, input1.getSize());
    Assert.assertEquals(3, input2.getSize());
    Assert.assertEquals(
        ResourceTests.readResourceAsString("vector-output/append-1.tsv"), input1.toString());
    input2.append(input1);
    Assert.assertEquals(6, input1.getSize());
    Assert.assertEquals(9, input2.getSize());
    Assert.assertEquals(
        ResourceTests.readResourceAsString("vector-output/append-2.tsv"), input2.toString());
  }
}
