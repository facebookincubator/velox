package com.facebook.velox4j.data;

import com.facebook.velox4j.Velox4j;
import com.facebook.velox4j.memory.AllocationListener;
import com.facebook.velox4j.memory.MemoryManager;
import com.facebook.velox4j.session.Session;
import com.facebook.velox4j.test.Velox4jTests;
import org.junit.*;

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
