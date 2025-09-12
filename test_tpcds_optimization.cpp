#include "velox/tpcds/gen/TpcdsGen.h"
#include <iostream>
#include <chrono>

using namespace facebook::velox::tpcds;

int main() {
    // Test basic functionality
    std::cout << "Testing TPC-DS table name optimization...\n";
    
    // Test all table names
    const auto tables = {
        Table::TBL_CALL_CENTER,
        Table::TBL_CATALOG_PAGE,
        Table::TBL_CATALOG_RETURNS,
        Table::TBL_CATALOG_SALES,
        Table::TBL_CUSTOMER,
        Table::TBL_CUSTOMER_ADDRESS,
        Table::TBL_CUSTOMER_DEMOGRAPHICS,
        Table::TBL_DATE_DIM,
        Table::TBL_HOUSEHOLD_DEMOGRAPHICS,
        Table::TBL_INCOME_BAND,
        Table::TBL_INVENTORY,
        Table::TBL_ITEM,
        Table::TBL_PROMOTION,
        Table::TBL_REASON,
        Table::TBL_SHIP_MODE,
        Table::TBL_STORE,
        Table::TBL_STORE_RETURNS,
        Table::TBL_STORE_SALES,
        Table::TBL_TIME_DIM,
        Table::TBL_WAREHOUSE,
        Table::TBL_WEB_PAGE,
        Table::TBL_WEB_RETURNS,
        Table::TBL_WEB_SALES,
        Table::TBL_WEB_SITE
    };
    
    // Test correctness
    std::cout << "Testing correctness:\n";
    for (auto table : tables) {
        const std::string& name = toTableName(table);
        std::cout << "Table " << static_cast<int>(table) << " -> " << name << std::endl;
        
        // Test round-trip conversion
        Table backToTable = fromTableName(name);
        if (backToTable != table) {
            std::cerr << "ERROR: Round-trip conversion failed for " << name << std::endl;
            return 1;
        }
    }
    
    // Test performance - measure time for many calls
    std::cout << "\nTesting performance:\n";
    const int iterations = 100000;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        for (auto table : tables) {
            const std::string& name = toTableName(table);
            (void)name; // Suppress unused variable warning
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time for " << iterations << " iterations: " << duration.count() << " microseconds\n";
    std::cout << "Average time per call: " << (double)duration.count() / (iterations * tables.size()) << " microseconds\n";
    
    // Test using TableName directly (the new enum-based approach)
    std::cout << "\nTesting TableName enum interface:\n";
    std::cout << "call_center: " << TableName::toName(Table::TBL_CALL_CENTER) << std::endl;
    
    auto maybeTable = TableName::tryToTable("customer");
    if (maybeTable) {
        std::cout << "customer -> Table enum value: " << static_cast<int>(*maybeTable) << std::endl;
    }
    
    std::cout << "\nAll tests passed! ✓\n";
    return 0;
}
