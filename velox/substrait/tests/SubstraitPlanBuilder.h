#include "velox/substrait/SubstraitExtensionCollector.h"
#include "velox/substrait/VeloxToSubstraitType.h"
#include "velox/substrait/proto/substrait/algebra.pb.h"
#include "velox/substrait/proto/substrait/plan.pb.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using SubstraitVirtualTable = ::substrait::ReadRel::VirtualTable;
using SubstraitExprLitStruct = ::substrait::Expression::Literal::Struct;
using SubstraitExprLit = ::substrait::Expression::Literal;

using SubstraitReadRel = ::substrait::ReadRel;
using SubstraitPlanRel = ::substrait::PlanRel;
using SubstraitPlan = ::substrait::Plan;

using facebook::velox::test::VectorTestBase;

namespace facebook::velox::substrait {
using SubstraitExtensionCollectorPtr =
    std::shared_ptr<SubstraitExtensionCollector>;
using VeloxToSubstraitTypeConvertorPtr =
    std::shared_ptr<VeloxToSubstraitTypeConvertor>;
void MakeAndRunVeloxPlan();

class SubstraitPlanBuilder : public VectorTestBase {
 public:
  SubstraitPlanBuilder(){};
  void Sample1();
};
} // namespace facebook::velox::substrait