package com.meta.velox4j.data;

import com.meta.velox4j.jni.JniApi;

public class SelectivityVectors {
  private final JniApi jniApi;

  public SelectivityVectors(JniApi jniApi) {
    this.jniApi = jniApi;
  }

  public SelectivityVector create(int length) {
    return jniApi.createSelectivityVector(length);
  }
}
