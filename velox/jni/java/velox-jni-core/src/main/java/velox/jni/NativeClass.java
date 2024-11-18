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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

public abstract class NativeClass implements AutoCloseable {
  long handle = 0;

  static final boolean IS_DEBUG = Boolean.parseBoolean(System.getProperty("NativeDebug", "false"));

  public static ConcurrentHashMap<Long, NativeClass> refCache = new ConcurrentHashMap<>();
  public static ConcurrentHashMap<Long, StackTraceElement[]> stackCache = new ConcurrentHashMap<>();

  public void setHandle(long handle) {
    this.handle = handle;
    this.isRelease = false;
    if (IS_DEBUG) {
      refCache.put(handle, this);
      StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
      stackCache.put(handle, stackTrace);
    }
  }

  boolean isRelease = true;

  public static List<NativeClass> allUnRelease() {
    ArrayList<NativeClass> nativeClasses = new ArrayList<>();
    for (NativeClass n : refCache.values()) {
      if (!n.isRelease) {
        nativeClasses.add(n);
      }
    }
    return nativeClasses;
  }

  public long nativePTR() {
    return this.handle;
  }

  @NativeCall
  public void releaseHandle() {
    this.handle = 0;
  }

  protected abstract void releaseInternal();

  public final void Release() {
    releaseInternal();
  }

  @Override
  public synchronized void close() {
    if (!isRelease) {
      Release();
      isRelease = true;
    }
  }

}

