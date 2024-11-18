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

