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

  private native String nativeNodeId();

  //
//
//  public native void nativeTestString(String test);
//
//
//  private native String nativeWindowFunction(String functionCallJson,
//                                             String frameJson,
//                                             boolean ignoreNulls);
//
//  native void nativeWindow(String[] partitionKeys,
//                           String[] sortingKeys,
//                           String[] sortingOrders,
//                           String[] windowColumnNames,
//                           String[] windowFunctions,
//                           boolean inputsSorted);
//
//  native void nativeSort(
//      String[] sortingKeys,
//      String[] sortingOrders,
//      boolean isPartial);

  private native void nativeUnnest(String[] replicateVariables,
                                   String[] unnestVariables,
                                   String[] unnestNames,
                                   String ordinalityName);

  private native void nativeLimit(int offset, int limit);

  private native void nativePartitionedOutput(String[] offset, int numPartitions);


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

  public void unnest(String[] replicateVariables,
                     String[] unnestVariables,
                     String[] unnestNames,
                     String ordinalityName) {
    nativeUnnest(replicateVariables, unnestVariables, unnestNames, ordinalityName);
  }


  public void partitionedOutput(String[] keys, int numPartitions) {
    nativePartitionedOutput(keys, numPartitions);
  }


  public String builderAndRelease() {
    String s = nativeBuilder();
    close();
    return s;
  }

  public String nodeId() {
    return nativeNodeId();
  }

  public String builder() {
    return nativeBuilder();
  }


}
