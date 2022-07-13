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

#include "velox/tpch/gen/DBGenIterator.h"

#include <folly/Singleton.h>
#include "velox/common/base/Exceptions.h"
#include "velox/external/duckdb/tpch/dbgen/include/dbgen/dbgen_gunk.hpp"

namespace facebook::velox::tpch {

namespace {

DBGenContext globalCtx;

// DBGenLease is a singleton that controls access to the DBGEN C functions. It
// handles initialization and cleanup of dbgen gunk structures, and set/unset of
// global variables used by DBGEN.
//
// Only acquire instances of this class using folly::Singleton.
class DBGenLease {
 public:
  DBGenLease() {
    // load_dists()/cleanup_dists() need to be called to ensure the global
    // variables required by dbgen are populated.
    load_dists(10 * 1024 * 1024, &globalCtx); // 10 MB buffer size for text generation.
  }
  ~DBGenLease() {
    cleanup_dists();
  }

  // Get a lease, or a permission to safely call internal dbgen functions.
  std::unique_lock<std::mutex> getLease(size_t scaleFactor) {
    auto lock = std::unique_lock<std::mutex>{mutex_};

    // DBGEN takes the scale factor through this global variable.
    globalCtx.scale_factor = scaleFactor;


    // This is tricky: dbgen code initializes seeds using hard-coded literals in
    // the C codebase, which are updated every time a record is generated. In
    // order to make these functions reproducible, before we make the first
    // invocation we need to make a copy (a backup) of the initial state of
    // these seeds. For subsequent leases, we restore that backed up state to
    // ensure results are reproducible.
    if (firstCall_) {
      // Store the initial random seed.
      memcpy(seedBackup_, seed_, sizeof(seed_t) * MAX_STREAM + 1);
      firstCall_ = false;
    } else {
      // Restore random seeds from backup.
      memcpy(seed_, seedBackup_, sizeof(seed_t) * MAX_STREAM + 1);
    }
    return lock;
  }

 private:
  std::mutex mutex_;

  seed_t* seed_{globalCtx.Seed};
  seed_t seedBackup_[MAX_STREAM + 1];
  bool firstCall_{true};
};

// Make the object above a singleton.
static folly::Singleton<DBGenLease> DBGenLeaseSingleton;

} // namespace

DBGenIterator DBGenIterator::create(size_t scaleFactor) {
  auto dbGenLease = DBGenLeaseSingleton.try_get();
  VELOX_CHECK_NOT_NULL(dbGenLease);
  return DBGenIterator(dbGenLease->getLease(scaleFactor));
}

void DBGenIterator::initNation(size_t offset) {
  sd_nation(NATION, offset, &ctx);
}

void DBGenIterator::initRegion(size_t offset) {
  sd_region(REGION, offset, &ctx);
}

void DBGenIterator::initOrder(size_t offset) {
  sd_order(ORDER, offset, &ctx);
  sd_line(LINE, offset, &ctx);
}

void DBGenIterator::initSupplier(size_t offset) {
  sd_supp(SUPP, offset, &ctx);
}

void DBGenIterator::initPart(size_t offset) {
  sd_part(PART, offset, &ctx);
  sd_psupp(PSUPP, offset, &ctx);
}

void DBGenIterator::initCustomer(size_t offset) {
  sd_cust(CUST, offset, &ctx);
}

void DBGenIterator::genNation(size_t index, code_t& code) {
  row_start(NATION, &ctx);
  mk_nation(index, &code, &ctx);
  row_stop_h(NATION, &ctx);
}

void DBGenIterator::genRegion(size_t index, code_t& code) {
  row_start(REGION, &ctx);
  mk_region(index, &code, &ctx);
  row_stop_h(REGION, &ctx);
}

void DBGenIterator::genOrder(size_t index, order_t& order) {
  row_start(ORDER, &ctx);
  mk_order(index, &order, &ctx, /*update-num=*/0);
  row_stop_h(ORDER, &ctx);
}

void DBGenIterator::genSupplier(size_t index, supplier_t& supplier) {
  row_start(SUPP, &ctx);
  mk_supp(index, &supplier, &ctx);
  row_stop_h(SUPP, &ctx);
}

void DBGenIterator::genPart(size_t index, part_t& part) {
  row_start(PART, &ctx);
  mk_part(index, &part, &ctx);
  row_stop_h(PART, &ctx);
}

void DBGenIterator::genCustomer(size_t index, customer_t& customer) {
  row_start(CUST, &ctx);
  mk_cust(index, &customer, &ctx);
  row_stop_h(CUST, &ctx);
}

} // namespace facebook::velox::tpch
