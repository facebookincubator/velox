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

package velox.jni;

import org.apache.spark.sql.types.StructType;

public class NativePlanBuilder extends NativeClass {
  public NativePlanBuilder() {
    setHandle(nativeCreate());
  }

  private native void nativeJavaScan(String schema);

  private native void nativeFilter(String condition);

  private native void nativeProject(String[] projections);


  private native String nativeBuilder();

  protected native long nativeCreate();

  private native void nativeRelease();
  private native void nativeLimit(int offset, int limit);

  @Override
  protected void releaseInternal() {
    nativeRelease();
  }


  // for native call , don't delete
  public NativePlanBuilder project(String[] projections) {
    nativeProject(projections);
    return this;
  }

  public NativePlanBuilder scan(StructType schema) {
    nativeJavaScan(schema.catalogString());
    return this;
  }

  public NativePlanBuilder filter(String condition) {
    nativeFilter(condition);
    return this;
  }

  public NativePlanBuilder limit(int offset, int limit) {
    nativeLimit(offset, limit);
    return this;
  }

  public String builder() {
    return nativeBuilder();
  }


}
