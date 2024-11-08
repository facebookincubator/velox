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

//
//  public NativePlanBuilder expand(String[][] project, String[] alias) {
//    nativeExpand(project, alias);
//    return this;
//  }
//
//  public NativePlanBuilder hashJoin(String joinType, boolean nullAware, String[] leftKeys, String[] rightKeys, String filter, String rightPlan, String output) {
//    nativeShuffledHashJoin(joinType, nullAware, leftKeys, rightKeys, filter, rightPlan, output);
//    return this;
//  }
//
//  public NativePlanBuilder MergeJoin(String joinType, String[] leftKeys, String[] rightKeys, String filter, String rightPlan, String output) {
//    nativeMergeJoin(joinType, leftKeys, rightKeys, filter, rightPlan, output);
//    return this;
//  }
//
//  public void aggregate(String step,
//                        String[] group,
//                        String[] aggNames,
//                        String[] agg,
//                        boolean ignoreNullKey) {
//
//    nativeAggregation(step, group, aggNames, agg, ignoreNullKey);
//  }
//
//
//  public String windowFunction(String functionCallJson,
//                               String frameJson,
//                               boolean ignoreNulls) {
//    return nativeWindowFunction(functionCallJson, frameJson, ignoreNulls);
//  }
//
//
//  public void window(String[] partitionKeys,
//                     String[] sortingKeys,
//                     SortOrder[] sortingOrders,
//                     String[] windowColumnNames,
//                     String[] windowFunctions,
//                     boolean inputsSorted) {
//    String[] sortingOrders1 = PlanUtils.jsonChildren(sortingOrders);
//    nativeWindow(partitionKeys, sortingKeys, sortingOrders1, windowColumnNames, windowFunctions, inputsSorted);
//  }
//
//  public void sort(String[] sortingKeys,
//                   SortOrder[] sortingOrders,
//                   boolean isPartial) {
//    String[] sortingOrders1 = PlanUtils.jsonChildren(sortingOrders);
//    nativeSort(sortingKeys, sortingOrders1, isPartial);
//  }

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
