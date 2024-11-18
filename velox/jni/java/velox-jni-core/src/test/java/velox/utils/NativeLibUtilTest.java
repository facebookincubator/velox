package velox.utils;


import org.junit.jupiter.api.Test;

public class NativeLibUtilTest {


  @Test
  public void testLoadLib() {
    NativeLibUtil.loadLibrary("libjni.dylib");
  }

}