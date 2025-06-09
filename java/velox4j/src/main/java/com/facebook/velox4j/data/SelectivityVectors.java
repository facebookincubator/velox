package com.facebook.velox4j.data;

import com.facebook.velox4j.jni.JniApi;

public class SelectivityVectors {
  private final JniApi jniApi;

  public SelectivityVectors(JniApi jniApi) {
    this.jniApi = jniApi;
  }

  /** Creates an empty selectivity vector with the given length where all bits are available. */
  public SelectivityVector create(int length) {
    return jniApi.createSelectivityVector(length);
  }
}
