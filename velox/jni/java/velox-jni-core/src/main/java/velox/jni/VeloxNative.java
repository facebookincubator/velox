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

import velox.utils.NativeLibUtil;

public class VeloxNative {
  private static volatile boolean isLoaded = false;

  static final String osName = System.getProperty("os.name").toLowerCase();

  static final String osArch;
  protected static final boolean isMacOs;
  protected static final boolean isLinuxOs;
  protected static final boolean isX86_64;
  protected static final boolean isAarch64;

  static {
    isMacOs = osName.startsWith("mac");
    isLinuxOs = osName.startsWith("linux");
    osArch = System.getProperty("os.arch").toLowerCase();
    isX86_64 = osArch.startsWith("amd64") || osArch.startsWith("x86_64");
    isAarch64 = osArch.startsWith("aarch64");
  }

  public static String getFileExtension() {
    return isMacOs ? ".dylib" : ".so";
  }

  public static void init(String conf) {
    // Early exit if already loaded
    if (isLoaded) {
      return;
    }

    synchronized (VeloxNative.class) {
      // Double-checked locking pattern
      if (isLoaded) {
        return;
      }

      boolean loaded = false;
      try {
        String libraryName = "libjni" + getFileExtension();
        loaded = tryLoadLibrary(libraryName, conf);
      } catch (RuntimeException e) {
        throw new IllegalStateException("Failed to load native libraries.", e);
      }
      if (!loaded) {
        throw new IllegalStateException("Native libraries could not be loaded.");
      }
      isLoaded = true;
    }
  }

  private static boolean tryLoadLibrary(String libraryName, String conf) {
    try {
      NativeLibUtil.loadLibrary(libraryName);
      if (conf != null && !conf.isEmpty()) {
        NativeLibUtil.init(conf);
      }
      return true;
    } catch (RuntimeException e) {
      // Log or handle the exception if needed
      return false;
    }
  }
}
