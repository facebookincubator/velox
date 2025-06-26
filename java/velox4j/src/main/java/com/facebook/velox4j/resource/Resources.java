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
package com.facebook.velox4j.resource;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import com.google.common.base.Preconditions;

import com.facebook.velox4j.exception.VeloxException;

public class Resources {

  /**
   * Get a collection of resource paths by the input RegEx pattern in a certain container folder.
   *
   * <p>Code is copied from <a
   * href="https://stackoverflow.com/questions/3923129/get-a-list-of-resources-from-classpath-directory">here</a>
   * and then modified for Velox4J's use.
   *
   * @param container The container folder. E.g., `META-INF`. Should not be left empty, because
   *     Classloader requires for at a meaningful file name to search inside the loaded jar files.
   * @param pattern The pattern to match on the file names.
   * @return The relative resource paths in the order they are found.
   */
  public static List<ResourceFile> getResources(final String container, final Pattern pattern) {
    Preconditions.checkArgument(
        !container.isEmpty(),
        "Resource search should only be used under a certain container folder");
    Preconditions.checkArgument(
        !container.startsWith(File.separator) && !container.endsWith(File.separator),
        "Resource container should not start or end with\"/\"");
    final List<ResourceFile> buffer = new ArrayList<>();
    final Enumeration<URL> containerUrls;
    try {
      containerUrls = Thread.currentThread().getContextClassLoader().getResources(container);
    } catch (IOException e) {
      throw new VeloxException(e);
    }
    while (containerUrls.hasMoreElements()) {
      final URL containerUrl = containerUrls.nextElement();
      getResources(container, containerUrl, pattern, buffer);
    }
    return Collections.unmodifiableList(buffer);
  }

  private static void getResources(
      final String container,
      final URL containerUrl,
      final Pattern pattern,
      final List<ResourceFile> buffer) {
    final String protocol = containerUrl.getProtocol();
    switch (protocol) {
      case "file":
        final File fileContainer = new File(containerUrl.getPath());
        Preconditions.checkState(
            fileContainer.exists() && fileContainer.isDirectory(),
            "Specified file container " + containerUrl + " is not a directory or not a file");
        getResourcesFromDirectory(container, fileContainer, fileContainer, pattern, buffer);
        break;
      case "jar":
        final String jarContainerPath = containerUrl.getPath();
        final Pattern jarContainerPattern = Pattern.compile("file:([^!]+)!/(.+)");
        final Matcher m = jarContainerPattern.matcher(jarContainerPath);
        if (!m.matches()) {
          throw new VeloxException("Illegal Jar container URL: " + containerUrl);
        }
        final String jarPath = m.group(1);
        final File jarFile = new File(jarPath);
        Preconditions.checkState(
            jarFile.exists() && jarFile.isFile(),
            "Specified Jar container " + containerUrl + " is not a Jar file");
        final String dir = m.group(2);
        getResourcesFromJarFile(container, jarFile, dir, pattern, buffer);
        break;
      default:
        throw new VeloxException("Unrecognizable resource protocol: " + protocol);
    }
  }

  private static void getResourcesFromJarFile(
      final String container,
      final File jarFile,
      final String dir,
      final Pattern pattern,
      final List<ResourceFile> buffer) {
    final ZipFile zf;
    try {
      zf = new ZipFile(jarFile);
    } catch (final IOException e) {
      throw new RuntimeException(e);
    }
    final Enumeration<? extends ZipEntry> entry = zf.entries();
    while (entry.hasMoreElements()) {
      final ZipEntry ze = entry.nextElement();
      final String fileName = ze.getName();
      if (!fileName.startsWith(dir)) {
        continue;
      }
      final String relativeFileName =
          new File(dir).toURI().relativize(new File(fileName).toURI()).getPath();
      final boolean accept = pattern.matcher(relativeFileName).matches();
      if (accept) {
        buffer.add(new PlainResourceFile(container, relativeFileName));
      }
    }
    try {
      zf.close();
    } catch (final IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static void getResourcesFromDirectory(
      final String container,
      final File root,
      final File directory,
      final Pattern pattern,
      final List<ResourceFile> buffer) {
    final File[] fileList = Preconditions.checkNotNull(directory.listFiles());
    for (final File file : fileList) {
      if (file.isDirectory()) {
        getResourcesFromDirectory(container, root, file, pattern, buffer);
      } else {
        final String relative = root.toURI().relativize(file.toURI()).getPath();
        final boolean accept = pattern.matcher(relative).matches();
        if (accept) {
          buffer.add(new PlainResourceFile(container, relative));
        }
      }
    }
  }

  public static void copyResource(String fromPath, File toFile) {
    Preconditions.checkArgument(!toFile.isDirectory(), "File %s is not a file", toFile);
    final ClassLoader classloader = Thread.currentThread().getContextClassLoader();
    try (final InputStream is =
        new BufferedInputStream(
            Preconditions.checkNotNull(
                classloader.getResourceAsStream(fromPath), "Resource %s not found", fromPath))) {
      final BufferedOutputStream o = new BufferedOutputStream(new FileOutputStream(toFile));
      while (true) {
        int b = is.read();
        if (b == -1) {
          break;
        }
        o.write(b);
      }
      o.flush();
      o.close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
