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
package com.facebook.velox4j.test;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;

import com.google.common.base.Preconditions;

import com.facebook.velox4j.jni.JniWorkspace;
import com.facebook.velox4j.resource.Resources;

public final class ResourceTests {
  public static String readResourceAsString(String path) {
    final ClassLoader classloader = Thread.currentThread().getContextClassLoader();
    try (final InputStream is = classloader.getResourceAsStream(path)) {
      Preconditions.checkArgument(is != null, "Resource %s not found", path);
      final ByteArrayOutputStream o = new ByteArrayOutputStream();
      while (true) {
        int b = is.read();
        if (b == -1) {
          break;
        }
        o.write(b);
      }
      final byte[] bytes = o.toByteArray();
      o.close();
      return new String(bytes, Charset.defaultCharset());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static File copyResourceToTmp(String path) {
    try {
      final File folder = JniWorkspace.getDefault().getSubDir("test");
      final File tmp = File.createTempFile("velox4j-test-", ".tmp", folder);
      Resources.copyResource(path, tmp);
      return tmp;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
