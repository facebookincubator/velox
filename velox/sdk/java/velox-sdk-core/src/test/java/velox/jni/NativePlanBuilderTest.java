package velox.jni;

import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class NativePlanBuilderTest extends NativeTest {


  @Test
  public void testScan() {
    try (NativePlanBuilder scan = new NativePlanBuilder()
        .scan(StructType.fromDDL("a int, B int"));) {
      String builder = scan
          .builder();
      Assertions.assertEquals("{\"assignments\":[{\"assign\":\"a\",\"columnHandle\":{\"columnType\":\"Regular\",\"dataType\":{\"name\":\"Type\",\"type\":\"INTEGER\"},\"hiveColumnHandleName\":\"a\",\"hiveType\":{\"name\":\"Type\",\"type\":\"INTEGER\"},\"name\":\"HiveColumnHandle\",\"requiredSubfields\":[]}},{\"assign\":\"B\",\"columnHandle\":{\"columnType\":\"Regular\",\"dataType\":{\"name\":\"Type\",\"type\":\"INTEGER\"},\"hiveColumnHandleName\":\"B\",\"hiveType\":{\"name\":\"Type\",\"type\":\"INTEGER\"},\"name\":\"HiveColumnHandle\",\"requiredSubfields\":[]}}],\"id\":\"0\",\"name\":\"TableScanNode\",\"outputType\":{\"cTypes\":[{\"name\":\"Type\",\"type\":\"INTEGER\"},{\"name\":\"Type\",\"type\":\"INTEGER\"}],\"name\":\"Type\",\"names\":[\"a\",\"B\"],\"type\":\"ROW\"},\"tableHandle\":{\"connectorId\":\"test-hive\",\"filterPushdownEnabled\":true,\"name\":\"HiveTableHandle\",\"subfieldFilters\":[],\"tableName\":\"hive_table\"}}", builder);
    }
  }


  @Test
  public void testScanFilter() {
    try (NativePlanBuilder scan = new NativePlanBuilder()
        .scan(StructType.fromDDL("a int, B int"));) {

      scan.filter("a > 10");
      String builder = scan
          .builder();
      Assertions.assertEquals("{\"filter\":{\"functionName\":\"gt\",\"inputs\":[{\"inputs\":[{\"fieldName\":\"a\",\"inputs\":[{\"name\":\"InputTypedExpr\",\"type\":{\"cTypes\":[{\"name\":\"Type\",\"type\":\"INTEGER\"},{\"name\":\"Type\",\"type\":\"INTEGER\"}],\"name\":\"Type\",\"names\":[\"a\",\"B\"],\"type\":\"ROW\"}}],\"name\":\"FieldAccessTypedExpr\",\"type\":{\"name\":\"Type\",\"type\":\"INTEGER\"}}],\"name\":\"CastTypedExpr\",\"nullOnFailure\":false,\"type\":{\"name\":\"Type\",\"type\":\"BIGINT\"}},{\"name\":\"ConstantTypedExpr\",\"type\":{\"name\":\"Type\",\"type\":\"BIGINT\"},\"value\":{\"type\":\"BIGINT\",\"value\":10}}],\"name\":\"CallTypedExpr\",\"type\":{\"name\":\"Type\",\"type\":\"BOOLEAN\"}},\"id\":\"1\",\"name\":\"FilterNode\",\"sources\":[{\"assignments\":[{\"assign\":\"a\",\"columnHandle\":{\"columnType\":\"Regular\",\"dataType\":{\"name\":\"Type\",\"type\":\"INTEGER\"},\"hiveColumnHandleName\":\"a\",\"hiveType\":{\"name\":\"Type\",\"type\":\"INTEGER\"},\"name\":\"HiveColumnHandle\",\"requiredSubfields\":[]}},{\"assign\":\"B\",\"columnHandle\":{\"columnType\":\"Regular\",\"dataType\":{\"name\":\"Type\",\"type\":\"INTEGER\"},\"hiveColumnHandleName\":\"B\",\"hiveType\":{\"name\":\"Type\",\"type\":\"INTEGER\"},\"name\":\"HiveColumnHandle\",\"requiredSubfields\":[]}}],\"id\":\"0\",\"name\":\"TableScanNode\",\"outputType\":{\"cTypes\":[{\"name\":\"Type\",\"type\":\"INTEGER\"},{\"name\":\"Type\",\"type\":\"INTEGER\"}],\"name\":\"Type\",\"names\":[\"a\",\"B\"],\"type\":\"ROW\"},\"tableHandle\":{\"connectorId\":\"test-hive\",\"filterPushdownEnabled\":true,\"name\":\"HiveTableHandle\",\"subfieldFilters\":[],\"tableName\":\"hive_table\"}}]}", builder);
    }
  }


  @Test
  public void testScanProject() {
    try (NativePlanBuilder scan = new NativePlanBuilder()
        .scan(StructType.fromDDL("a int, B int"));) {

      String[] projections = {"a"};
      scan.project(projections);
      String builder = scan
          .builder();
      Assertions.assertEquals("{\"id\":\"1\",\"name\":\"ProjectNode\",\"names\":[\"a\"],\"projections\":[{\"fieldName\":\"a\",\"inputs\":[{\"name\":\"InputTypedExpr\",\"type\":{\"cTypes\":[{\"name\":\"Type\",\"type\":\"INTEGER\"},{\"name\":\"Type\",\"type\":\"INTEGER\"}],\"name\":\"Type\",\"names\":[\"a\",\"B\"],\"type\":\"ROW\"}}],\"name\":\"FieldAccessTypedExpr\",\"type\":{\"name\":\"Type\",\"type\":\"INTEGER\"}}],\"sources\":[{\"assignments\":[{\"assign\":\"a\",\"columnHandle\":{\"columnType\":\"Regular\",\"dataType\":{\"name\":\"Type\",\"type\":\"INTEGER\"},\"hiveColumnHandleName\":\"a\",\"hiveType\":{\"name\":\"Type\",\"type\":\"INTEGER\"},\"name\":\"HiveColumnHandle\",\"requiredSubfields\":[]}},{\"assign\":\"B\",\"columnHandle\":{\"columnType\":\"Regular\",\"dataType\":{\"name\":\"Type\",\"type\":\"INTEGER\"},\"hiveColumnHandleName\":\"B\",\"hiveType\":{\"name\":\"Type\",\"type\":\"INTEGER\"},\"name\":\"HiveColumnHandle\",\"requiredSubfields\":[]}}],\"id\":\"0\",\"name\":\"TableScanNode\",\"outputType\":{\"cTypes\":[{\"name\":\"Type\",\"type\":\"INTEGER\"},{\"name\":\"Type\",\"type\":\"INTEGER\"}],\"name\":\"Type\",\"names\":[\"a\",\"B\"],\"type\":\"ROW\"},\"tableHandle\":{\"connectorId\":\"test-hive\",\"filterPushdownEnabled\":true,\"name\":\"HiveTableHandle\",\"subfieldFilters\":[],\"tableName\":\"hive_table\"}}]}", builder);
    }
  }

}