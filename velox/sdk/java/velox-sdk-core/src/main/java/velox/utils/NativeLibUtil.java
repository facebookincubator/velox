// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

package velox.utils;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sparkproject.guava.base.Joiner;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

public class NativeLibUtil {
  private final static Logger LOG = LoggerFactory.getLogger(NativeLibUtil.class);

  public static void loadLibrary(String libFileName) {
    // dev model
    Optional<File> libFileOptional = findLibraryInClasspath(libFileName);
    if (libFileOptional.isPresent()) {
      System.load(libFileOptional.get().getAbsolutePath());
      return;
    }
    // Try to load library from classpath
    boolean loaded = false;
    try {
      // Find the path to the JAR file that contains the class
      String jarPath = NativeLibUtil.class.getProtectionDomain()
          .getCodeSource().getLocation().toURI().getPath();

      // Check if the JAR file contains the library
      if (jarPath.endsWith("velox-sdk-core.jar")) {
        // Create a temporary directory to extract the .so file
        Path tempDir = Files.createTempDirectory("nativeLibs");

        try (JarFile jar = new JarFile(jarPath)) {
          JarEntry entry = jar.getJarEntry(libFileName);
          if (entry != null) {
            // Extract the .so file to the temporary directory
            File libFile = new File(tempDir.toFile(), libFileName);
            try (InputStream is = jar.getInputStream(entry);
                 OutputStream os = new FileOutputStream(libFile)) {
              byte[] buffer = new byte[1024];
              int bytesRead;
              while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
              }
            }
            // Load the .so file
            System.load(libFile.getAbsolutePath());
            loaded = true;
          }
        }
      }
    } catch (Exception e) {
      LOG.warn("Unable to extract and load " + libFileName, e);
    }
    // If not loaded, fallback to original loading logic
    if (!loaded) {
      List<String> candidates = new ArrayList<>(Arrays.asList(
          System.getProperty("java.library.path").split(":")));
      // Fall back to automatically finding the library in test environments.
      // This makes it easier to run tests from Eclipse without specially configuring
      // the Run Configurations.
      try {
        String myPath = NativeLibUtil.class.getProtectionDomain()
            .getCodeSource().getLocation().getPath();
        if (myPath.toString().endsWith("sdk-core/target/classes/") ||
            myPath.toString().endsWith("sdk-core/target/eclipse-classes/")) {
          candidates.add(myPath + "../../../../../../_build/velox/sdk");
        }
      } catch (Exception e) {
        LOG.warn("Unable to get path for NativeLibUtil class", e);
      }

      for (String path : candidates) {
        File libFile = new File(path + File.separator + libFileName);
        if (libFile.exists()) {
          System.load(libFile.getPath());
          return;
        }
      }

      throw new RuntimeException("Failed to load " + libFileName + " from any " +
          "candidate location:\n" + Joiner.on("\n").join(candidates));
    }
  }

  private static Optional<File> findLibraryInClasspath(String libFileName) {
    try {
      URL resource = Thread.currentThread().getContextClassLoader().getResource(libFileName);
      if (resource != null) {
        File libFile = new File(resource.getFile());
        return Optional.of(libFile);
      }
      return Optional.empty();
    } catch (Exception e) {
      return Optional.empty();
    }
  }


  native public static void init(String conf);
}
