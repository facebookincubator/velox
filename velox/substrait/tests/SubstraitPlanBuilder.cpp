#include "velox/substrait/tests/SubstraitPlanBuilder.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/substrait/SubstraitToVeloxPlan.h"
#include "velox/substrait/VeloxToSubstraitExpr.h"

#include "velox/type/Type.h"

namespace facebook::velox::substrait {

using facebook::velox::exec::test::AssertQueryBuilder;

std::unique_ptr<SubstraitVirtualTable> MakeVirtualTable(
    const SubstraitExtensionCollectorPtr& extensionCollector,
    google::protobuf::Arena& arena,
    std::vector<RowVectorPtr> data) {
  size_t numVectors = data.size();
  std::unique_ptr<SubstraitVirtualTable> virtualTable =
      std::make_unique<SubstraitVirtualTable>();
  // Construct the expression converter.
  auto exprConvertor =
      std::make_shared<VeloxToSubstraitExprConvertor>(extensionCollector);

  for (int64_t row = 0; row < numVectors; row++) {
    // The row data.
    ::substrait::Expression_Literal_Struct* litValue =
        virtualTable->add_values();

    const auto& rowVector = data.at(row);
    // The column number of the row data.
    int64_t numColumns = rowVector->childrenSize();

    for (int64_t column = 0; column < numColumns; column++) {
      std::unique_ptr<SubstraitExprLit> substraitField =
          std::make_unique<SubstraitExprLit>();

      const VectorPtr& child = rowVector->childAt(column);

      substraitField->MergeFrom(exprConvertor->toSubstraitExpr(
          arena, std::make_shared<core::ConstantTypedExpr>(child), litValue));
    }
  }
  return virtualTable;
}

std::unique_ptr<SubstraitReadRel> MakeReadRel(
    const SubstraitExtensionCollectorPtr& extensionCollector,
    google::protobuf::Arena& arena,
    std::vector<RowVectorPtr> data) {
  std::shared_ptr<VeloxToSubstraitTypeConvertor> typeConvertor;
  std::unique_ptr<SubstraitReadRel> readRel =
      std::make_unique<SubstraitReadRel>();
  // create Virtual Table
  size_t numVectors = data.size();
  std::unique_ptr<SubstraitVirtualTable> virtualTable =
      std::make_unique<SubstraitVirtualTable>();
  // Construct the expression converter.
  auto exprConvertor =
      std::make_shared<VeloxToSubstraitExprConvertor>(extensionCollector);
  VELOX_CHECK(numVectors > 0, "No data provided");

  // here we process the required information to create a RowTypePtr to
  // update the ReadRel schema

  // first vector
  const auto& zeroRowVector = data.at(0);
  // The column number of the row data.
  int64_t numColumns = zeroRowVector->childrenSize();
  std::vector<std::string> names(numColumns);
  std::vector<TypePtr> types(numColumns);
  for (int64_t row = 0; row < numVectors; row++) {
    // The row data.
    ::substrait::Expression_Literal_Struct* litValue =
        virtualTable->add_values();

    const auto& rowVector = data.at(row);

    for (int64_t column = 0; column < numColumns; column++) {
      std::unique_ptr<SubstraitExprLit> substraitField =
          std::make_unique<SubstraitExprLit>();

      const VectorPtr& child = rowVector->childAt(column);
      if (row == 0) {
        names[column] = "f" + std::to_string(row);
        types[column] = child->type();
      }
      substraitField->MergeFrom(exprConvertor->toSubstraitExpr(
          arena, std::make_shared<core::ConstantTypedExpr>(child), litValue));
    }
  }

  RowTypePtr rowPtr = ROW(std::move(names), std::move(types));
  readRel->mutable_base_schema()->MergeFrom(
      typeConvertor->toSubstraitNamedStruct(arena, std::move(rowPtr)));
  readRel->mutable_virtual_table()->MergeFrom(*virtualTable.release());
  return readRel;
}

std::unique_ptr<::substrait::Rel> MakeSimpleReadRelation(
    const SubstraitExtensionCollectorPtr& extensionCollector,
    google::protobuf::Arena& arena,
    std::vector<RowVectorPtr> data) {
  std::unique_ptr<::substrait::Rel> rel = std::make_unique<::substrait::Rel>();
  auto readRel = MakeReadRel(extensionCollector, arena, data);
  rel->mutable_read()->MergeFrom(*readRel);
  return rel;
}

void SubstraitPlanBuilder::Sample1() {
  auto pool = memory::getDefaultMemoryPool();
  auto a = makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6});
  auto b = makeFlatVector<int64_t>({0, 5, 10, 15, 20, 25, 30});

  auto data = makeRowVector({"a", "b"}, {a, b});
  google::protobuf::Arena arena;
  std::shared_ptr<SubstraitExtensionCollector> extensionCollector =
      std::make_shared<SubstraitExtensionCollector>();
  std::shared_ptr<SubstraitVeloxPlanConverter> planConverter =
      std::make_shared<SubstraitVeloxPlanConverter>(pool.get());

  std::unique_ptr<SubstraitPlanRel> planRel =
      std::make_unique<SubstraitPlanRel>();
  auto rel = MakeSimpleReadRelation(extensionCollector, arena, {data});
  planRel->mutable_rel()->MergeFrom(*rel);
  std::unique_ptr<SubstraitPlan> substraitPlan =
      std::make_unique<SubstraitPlan>();
  substraitPlan->mutable_relations()->AddAllocated(planRel.release());
  extensionCollector->addExtensionsToPlan(substraitPlan.get());
  auto veloxPlan = planConverter->toVeloxPlan(*substraitPlan);

  auto res = AssertQueryBuilder(veloxPlan).copyResults(pool.get());
  std::cout << std::endl << "> RES" << res->toString() << std::endl;
  std::cout << res->toString(0, res->size()) << std::endl;
}

void MakeAndRunVeloxPlan() {
  SubstraitPlanBuilder builder;
  builder.Sample1();
}

} // namespace facebook::velox::substrait
