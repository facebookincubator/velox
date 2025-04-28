package com.meta.velox4j.data;

import com.meta.velox4j.jni.CppObject;
import com.meta.velox4j.jni.StaticJniApi;

public class SelectivityVector implements CppObject {
  private final long id;

  public SelectivityVector(long id) {
    this.id = id;
  }

  @Override
  public long id() {
    return id;
  }

  public boolean isValid(int idx) {
    return StaticJniApi.get().selectivityVectorIsValid(this, idx);
  }
}
