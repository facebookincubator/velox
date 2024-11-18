package velox.jni;

import org.junit.jupiter.api.BeforeAll;

abstract class NativeTest {

  @BeforeAll
  public static void init(){
    VeloxNative.init(null);
  }
}
