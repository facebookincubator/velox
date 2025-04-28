package com.meta.velox4j.data;

import org.junit.*;

import com.meta.velox4j.Velox4j;
import com.meta.velox4j.memory.AllocationListener;
import com.meta.velox4j.memory.MemoryManager;
import com.meta.velox4j.session.Session;
import com.meta.velox4j.test.Velox4jTests;

public class SelectivityVectorTest {
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
  public void testIsValid() {
    final int length = 10;
    final SelectivityVector sv = session.selectivityVectorOps().create(length);
    for (int i = 0; i < length; i++) {
      Assert.assertTrue(sv.isValid(i));
    }
  }
}
