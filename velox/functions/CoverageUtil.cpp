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

#include "velox/functions/CoverageUtil.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include "velox/exec/Aggregate.h"
#include "velox/exec/WindowFunction.h"
#include "velox/expression/SimpleFunctionRegistry.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/FunctionRegistry.h"

namespace facebook::velox::functions {

class TablePrinter {
 public:
  TablePrinter(
      size_t numScalarColumns,
      size_t columnSize,
      std::string indent,
      std::ostream& out)
      : numScalarColumns_{numScalarColumns},
        columnSize_{columnSize},
        indent_{std::move(indent)},
        out_{out} {}

  void header() {
    std::string line(columnSize_, '=');
    out_ << indent_ << line;
    for (int i = 1; i < numScalarColumns_; i++) {
      out_ << "  " << line;
    }
    out_ << "  ==  " << line << "  ==  " << line << std::endl;

    auto scalarFunctionsColumnWidth =
        columnSize_ * numScalarColumns_ + 2 * (numScalarColumns_ - 1);

    out_ << indent_ << std::left << std::setw(scalarFunctionsColumnWidth)
         << "Scalar Functions"
         << "      " << std::setw(columnSize_) << "Aggregate Functions"
         << "      "
         << "Window Functions" << std::endl;
    out_ << indent_ << std::string(scalarFunctionsColumnWidth, '=')
         << "  ==  " << line << "  ==  " << line << std::endl;
  }

  void header_new(std::string functionName) {
    std::string line(functionName.length() + 2, '=');
    out_ << line << std::endl;
    out_ << " " << functionName << std::endl;
    out_ << line << std::endl;
  }

  void startRow() {
    out_ << indent_;
    currentColumn_ = 0;
  }

  void addColumn(const std::string& text) {
    addColumn(text, columnSize_);
  }

  void addEmptyColumn() {
    if (currentColumn_ == numScalarColumns_ ||
        currentColumn_ == numScalarColumns_ + 2) {
      // If the current column is after the Scalar Functions columns or
      // the column next to Aggregate Functions column.
      addColumn("", 2);
    } else {
      addColumn("", columnSize_);
    }
  }

  void endRow() {
    out_ << std::endl;
  }

  void footer() {
    std::string line(columnSize_, '=');
    out_ << indent_ << line;
    for (int i = 1; i < numScalarColumns_; i++) {
      out_ << "  " << line;
    }
    out_ << "  ==  " << line << "  ==  " << line << std::endl;
  }

 private:
  void addColumn(const std::string& text, size_t columnSize) {
    if (currentColumn_) {
      out_ << "  ";
    }
    out_ << std::setw(columnSize) << text;
    ++currentColumn_;
  }

  const size_t numScalarColumns_;
  const size_t columnSize_;
  const std::string indent_;
  std::ostream& out_;

  size_t currentColumn_{0};
};

class TableCellTracker {
 public:
  // Takes zero-based row and column numbers.
  void add(int row, int column) {
    cells_.emplace_back(row, column);
  }

  const std::vector<std::pair<int, int>>& cells() const {
    return cells_;
  }

 private:
  std::vector<std::pair<int, int>> cells_;
};

std::vector<std::string> readFunctionNamesFromFile(
    const std::string& filePath) {
  std::ifstream functions("data/" + filePath);

  std::vector<std::string> names;
  std::string name;
  while (getline(functions, name)) {
    names.emplace_back(name);
  }

  functions.close();
  return names;
}

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t");
    if (first == std::string::npos) {
        return ""; // Empty or all whitespace
    }
    size_t last = str.find_last_not_of(" \t");
    return str.substr(first, (last - first + 1));
}

std::pair<std::string, std::string> parseLine(const std::string& line) {
    size_t pos = line.find(':');
    if (pos != std::string::npos) {
        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));
        return std::make_pair(key, value);
    }
    return std::make_pair("", "");
}

std::tuple<std::unordered_map<std::string, std::vector<std::string>>, std::unordered_map<std::string, std::vector<std::string>>, std::unordered_map<std::string, std::vector<std::string>>> readPrestoFunctionNamesAndSignaturesFromFile() {
  std::ifstream functions("data/all_functions_with_signatures.txt");

  std::unordered_map<std::string, std::vector<std::string>> scalar, aggregate, window;
  std::string currentSection;

  std::string line;
  while (std::getline(functions, line)) {
      if (!line.empty() && line.back() == ':') {
          currentSection = line.substr(0, line.size() - 1);
      } else {
          std::pair<std::string, std::string> keyValue = parseLine(line);
          if (!keyValue.first.empty()) {
              if (currentSection == "scalar") {
                  scalar[keyValue.first].push_back(keyValue.second);
              } else if (currentSection == "aggregate") {
                  aggregate[keyValue.first].push_back(keyValue.second);
              } else if (currentSection == "window") {
                  window[keyValue.first].push_back(keyValue.second);
              }
          }
      }
  }
  return std::make_tuple(scalar, aggregate, window);
}

std::string toFuncLink(
    const std::string& name,
    const std::string& domain = "") {
  return fmt::format("{}:func:`{}`", domain, name);
}

std::string toFuncLinkNew(
    const std::string& name,
    const std::string& domain = "",
    const std::string& functiontype = "") {
  if (functiontype == "signature") {
    return fmt::format("\t - {}", name);
  } else {
    return fmt::format("- {}:func:`{}`", domain, name);
  }
}

int maxLength(const std::vector<std::string>& names) {
  int maxLength = 0;

  for (const auto& name : names) {
    auto len = name.size();
    if (len > maxLength) {
      maxLength = len;
    }
  }
  return maxLength;
}

/// Prints out CSS rules to
/// - add lightblue background to table header;
/// - add lightblue background to an empty column that separates scalar,
/// aggregate, and window functions;
/// - highlight cells identified by TableCellTracker.
void printTableCss(
    size_t numScalarColumns,
    const TableCellTracker& cellTracker,
    std::ostream& out) {
  out << "    div.body {max-width: 1300px;}" << std::endl;
  out << "    table.coverage th {background-color: lightblue; text-align: center;}"
      << std::endl;
  out << "    table.coverage "
      << "td:nth-child(" << numScalarColumns + 1 << ") "
      << "{background-color: lightblue;}" << std::endl;
  out << "    table.coverage "
      << "td:nth-child(" << numScalarColumns + 3 << ") "
      << "{background-color: lightblue;}" << std::endl;

  for (const auto& entry : cellTracker.cells()) {
    out << "    table.coverage "
        << "tr:nth-child(" << entry.first + 1 << ") "
        << "td:nth-child(" << entry.second + 1 << ") "
        << "{background-color: #6BA81E;}" << std::endl;
  }
}

std::string formatSignature(const std::string& signature) {
    std::string modifiedSignature = signature; // Make a copy to modify.
    size_t first_arrow_pos = modifiedSignature.find("->");
    if (first_arrow_pos != std::string::npos) {
        size_t second_arrow_pos = modifiedSignature.rfind("->");
        if (second_arrow_pos != std::string::npos && second_arrow_pos != first_arrow_pos) {
            modifiedSignature.replace(first_arrow_pos, second_arrow_pos - first_arrow_pos + 2, "->");
        }
    }
    return modifiedSignature;
}

/// Returns alphabetically sorted list of scalar functions available in Velox,
/// excluding companion functions.
std::pair<std::unordered_map<std::string, std::vector<std::string>>, int>
getScalarSignatureMap(std::vector<std::string> names) {
  std::unordered_map<std::string, std::vector<std::string>> signatureMap;
  int maxScalarLength = 0;

for (const auto& scalarName : names) {
    auto vectorFunctionSignatures = exec::getVectorFunctionSignatures(scalarName);
    auto simpleFunctionSignatures = exec::simpleFunctions().getFunctionSignatures(scalarName);
    std::vector<std::string> signatures;

    if (vectorFunctionSignatures.has_value()) {
        for (const auto& signature : vectorFunctionSignatures.value()) {
            signatures.push_back(formatSignature(fmt::format("{}", signature->toString())));
        }
    }

    for (const auto& signature : simpleFunctionSignatures) {
        signatures.push_back(formatSignature(fmt::format("{}", signature->toString())));
    }
    signatureMap[scalarName] = signatures;
}

  return std::make_pair(signatureMap, maxScalarLength);
}

/// Returns alphabetically sorted list of window functions available in Velox,
/// excluding companion functions.
std::pair<std::unordered_map<std::string, std::vector<std::string>>, int>
getWindowSignatureMap(std::vector<std::string> names) {
  std::unordered_map<std::string, std::vector<std::string>> signatureMap;
  int maxWindowLength = 0;

  for (const auto& name : names) {
    auto windowFunctionSignatures = exec::getWindowFunctionSignatures(name);
    std::vector<std::string> signatures;
    for (const auto& signature : windowFunctionSignatures.value()) {
      signatures.push_back(formatSignature(fmt::format("{}", signature->toString())));
    }
    signatureMap[name] = signatures;
  }

  return std::make_pair(signatureMap, maxWindowLength);
}

/// Returns alphabetically sorted list of aggregate functions available in
/// Velox, excluding compaion functions.
std::pair<std::unordered_map<std::string, std::vector<std::string>>, int>
getAggregateSignatureMap(std::vector<std::string> names) {
  std::unordered_map<std::string, std::vector<std::string>> signatureMap;
  int maxAggregateLength = 0;

  for (const auto& name : names) {
    auto aggregateFunctionSignatures =
        exec::getAggregateFunctionSignatures(name);
    std::vector<std::string> signatures;
    for (const auto& signature : aggregateFunctionSignatures.value()) {
      signatures.push_back(formatSignature(fmt::format("{}", signature->toString())));
    }
    signatureMap[name] = signatures;
  }

  return std::make_pair(signatureMap, maxAggregateLength);
}

void printCoverageMap(
    const std::vector<std::string>& scalarNames,
    const std::vector<std::string>& aggNames,
    const std::vector<std::string>& windowNames,
    const std::unordered_set<std::string>& veloxNames,
    const std::unordered_set<std::string>& veloxAggNames,
    const std::unordered_set<std::string>& veloxWindowNames,
    const std::string& domain) {
  const auto scalarCnt = scalarNames.size();
  const auto aggCnt = aggNames.size();
  const auto windowCnt = windowNames.size();

  // Make sure there is enough space for the longest function name + :func:
  // syntax that turns function name into a link to function's description.
  const int columnSize = std::max(
                             {maxLength(scalarNames),
                              maxLength(aggNames),
                              maxLength(windowNames)}) +
      toFuncLink("", domain).size();

  const std::string indent(4, ' ');

  const int numScalarColumns = 5;

  // Split scalar functions into 'numScalarColumns' columns. Put all aggregate
  // functions into one column.
  auto numRows = std::max(
      {(size_t)std::ceil((double)scalarCnt / numScalarColumns),
       aggCnt,
       windowCnt});

  // Keep track of cells which contain functions available in Velox. These cells
  // need to be highlighted using CSS rules.
  TableCellTracker veloxCellTracker;

  auto printName = [&](int row,
                       int column,
                       const std::string& name,
                       const std::unordered_set<std::string>& veloxNames) {
    if (veloxNames.count(name)) {
      veloxCellTracker.add(row, column);
      return toFuncLink(name, domain);
    } else {
      return name;
    }
  };

  std::ostringstream out;
  TablePrinter printer(numScalarColumns, columnSize, indent, out);
  printer.header();
  for (int i = 0; i < numRows; i++) {
    printer.startRow();

    // N columns of scalar functions.
    for (int j = 0; j < numScalarColumns; j++) {
      auto n = i + numRows * j;
      n < scalarCnt
          ? printer.addColumn(printName(i, j, scalarNames[n], veloxNames))
          : printer.addEmptyColumn();
    }

    // 1 empty column.
    printer.addEmptyColumn();

    // 1 column of aggregate functions.
    i < aggCnt ? printer.addColumn(printName(
                     i, numScalarColumns + 1, aggNames[i], veloxAggNames))
               : printer.addEmptyColumn();

    // 1 empty column.
    printer.addEmptyColumn();

    // 1 column of window functions.
    i < windowCnt
        ? printer.addColumn(printName(
              i, numScalarColumns + 3, windowNames[i], veloxWindowNames))
        : printer.addEmptyColumn();

    printer.endRow();
  }
  printer.footer();

  std::cout << ".. raw:: html" << std::endl << std::endl;
  std::cout << "    <style>" << std::endl;
  printTableCss(numScalarColumns, veloxCellTracker, std::cout);
  std::cout << "    </style>" << std::endl << std::endl;

  std::cout << ".. table::" << std::endl;
  std::cout << "    :widths: auto" << std::endl;
  std::cout << "    :class: coverage" << std::endl << std::endl;
  std::cout << out.str() << std::endl;
}

std::string toUpperCase(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

// Function to compare two strings after converting them to uppercase
bool ComparePrestoAndVeloxSignatures(const std::string& signature, const std::string& functionName,
                            std::unordered_map<std::string, std::vector<std::string>> signatureMap) {
   auto funcName = signatureMap.find(functionName);
    if (funcName != signatureMap.end()) {
        const std::vector<std::string>& values = funcName->second;
        for (const std::string& value : values) {
          if(toUpperCase(signature) == toUpperCase(value)){
            return true;
          }
        }
    }
    return false;
}

void printCoverageMapWithSignatures(
    const std::vector<std::string>& scalarNames, // presto names
    const std::vector<std::string>& aggNames, // presto names
    const std::vector<std::string>& windowNames, // presto names
    const std::unordered_set<std::string>& veloxNamesSet, // velox names
    const std::unordered_set<std::string>& veloxAggNamesSet, // velox names
    const std::unordered_set<std::string>& veloxWindowNamesSet, // velox names
    const std::string& domain) {


  const auto scalarCnt = scalarNames.size();
  const auto aggCnt = aggNames.size();
  const auto windowCnt = windowNames.size();

  std::vector<std::string> veloxNames(
      veloxNamesSet.begin(), veloxNamesSet.end());
  std::vector<std::string> veloxAggNames(
      veloxAggNamesSet.begin(), veloxAggNamesSet.end());
  std::vector<std::string> veloxWindowNames(
      veloxWindowNamesSet.begin(), veloxWindowNamesSet.end());

  auto scalarSignatureMap = getScalarSignatureMap(veloxNames);
  auto aggSignatureMap = getAggregateSignatureMap(veloxAggNames);
  auto windowSignatureMap = getWindowSignatureMap(veloxWindowNames);

  const auto prestoSignatureMap = readPrestoFunctionNamesAndSignaturesFromFile();
  auto scalarPrestoSignatureMap = std::get<0>(prestoSignatureMap);
  auto aggPrestoSignatureMap = std::get<1>(prestoSignatureMap);
  auto windowPrestoSignatureMap = std::get<2>(prestoSignatureMap);

  const auto scalarPrestoCnt = scalarPrestoSignatureMap.size();
  const auto aggPrestoCnt = aggPrestoSignatureMap.size();
  const auto windowPrestoCnt = windowPrestoSignatureMap.size();
  
  std::cout << scalarPrestoCnt <<std::endl;
  std::cout << aggPrestoCnt <<std::endl;
  std::cout << windowPrestoCnt <<std::endl;

  const std::string indent(4, ' ');

  TableCellTracker veloxCellTracker;

  auto printName = [&](int row,
                       int column,
                       const std::string& name,
                       const std::vector<std::string>& veloxNames
                       ) {
    // check if the passed presto function exists in velox, 
    auto veloxFunctionExists = std::find(veloxNames.begin(), veloxNames.end(), name);
    if (veloxFunctionExists != veloxNames.end()) { // if the passed presto function exists in velox
      veloxCellTracker.add(row, column); // highlight the function name
      return toFuncLinkNew(name, domain, ""); // print the function name with - :func
    } else { //if the passed presto function doesn't exist in velox
      return fmt::format("- {}", name);
    }
  };


  auto printSignature = [&](int row,
                            int column, 
                            const std::string& signature,
                            const std::string& functionName,
                            std::unordered_map<std::string, std::vector<std::string>> signatureMap
                            ) {
  if (ComparePrestoAndVeloxSignatures(signature, functionName, signatureMap) == true) { // if the passed presto signature exists in velox
      veloxCellTracker.add(row, column); //highlight the signature
      return toFuncLinkNew( fmt::format("-*- {}", signature), domain, "signature");
    } else {
      return toFuncLinkNew(signature, domain, "signature");
    }
  };

  std::ostringstream out;
  TablePrinter printer(3, 0, indent, out);
  int numRows = 0; 
  printer.header_new("Scalar Functions");

  for (const auto& scalarName : scalarNames) { // loop through all the presto scalar names.
    numRows = numRows + 1;
    printer.startRow();
    printer.addColumn( // add the function name first ( check if the function exists in velox, if yes, highlight it and add :func else dont)
        printName(numRows, 0, scalarName, veloxNames));
    printer.endRow();

    // check if the function name exists in the presto signature map
    auto scalarExists = scalarPrestoSignatureMap.find(scalarName);

    if (scalarExists != scalarPrestoSignatureMap.end()){
      // if it exists, then loop through the values of signatures for that specific function name
      for (const std::string& signature : scalarPrestoSignatureMap[scalarName]){
        numRows = numRows + 1;
        printer.addColumn(printSignature( // add the signature ( check whether it exists in velox also, if it exists in velox, highlight it, else dont)
            numRows, 0, signature, scalarName, scalarSignatureMap.first));
        printer.endRow();
      }
    } else {
      std::cout << "scalar doesnt exist " << scalarName <<std::endl;
    }
    printer.startRow();
    printer.endRow();
  }
  printer.header_new("Aggregate Functions");

  for (const auto& aggName : aggNames) {
    numRows = numRows + 1;
    printer.startRow();
    printer.addColumn(
        printName(numRows, 0, aggName, veloxAggNames));
    printer.endRow();
    auto aggExists = aggPrestoSignatureMap.find(aggName);
    if (aggExists != aggPrestoSignatureMap.end()) {
      for (const std::string& signature : aggPrestoSignatureMap[aggName]) {
        numRows = numRows + 1;
        printer.startRow();
        printer.addColumn(printSignature(
            numRows, 0, signature, aggName, aggSignatureMap.first));
        printer.endRow();
      }
    } else {
      std::cout << "agg doesnt exist " << aggName <<std::endl;
    }
    printer.startRow();
    printer.endRow();
  }

  printer.header_new("Window Functions");
  for (const auto& windowName : windowNames) {
    numRows = numRows + 1;
    printer.startRow();
    printer.addColumn(
        printName(numRows, 0, windowName, veloxWindowNames));
    printer.endRow();
    auto windowExists = windowPrestoSignatureMap.find(windowName);
    if (windowExists != windowPrestoSignatureMap.end()) {
      for (const std::string& signature :
           windowPrestoSignatureMap[windowName]) {
        numRows = numRows + 1;
        printer.startRow();
        printer.addColumn(printSignature(
            numRows, 0, signature, windowName, windowSignatureMap.first));
        printer.endRow();
      }
    } else {
      std::cout << "window doesnt exist " << windowName <<std::endl;
    }
    printer.startRow();
    printer.endRow();
  }

  printer.footer();

  std::cout << ".. raw:: html" << std::endl << std::endl;
  std::cout << "    <style>" << std::endl;
  printTableCss(0, veloxCellTracker, std::cout);
  std::cout << "    </style>" << std::endl << std::endl;

  std::cout << ".. table::" << std::endl;
  std::cout << "    :widths: auto" << std::endl;
  std::cout << "    :class: coverage" << std::endl << std::endl;
  std::cout << out.str() << std::endl;
}

// A function name is a companion function's if the name is an existing
// aggregation functio name followed by a specific suffixes.
bool isCompanionFunctionName(
    const std::string& name,
    const std::unordered_map<std::string, exec::AggregateFunctionEntry>&
        aggregateFunctions) {
  auto suffixOffset = name.rfind("_partial");
  if (suffixOffset == std::string::npos) {
    suffixOffset = name.rfind("_merge_extract");
  }
  if (suffixOffset == std::string::npos) {
    suffixOffset = name.rfind("_merge");
  }
  if (suffixOffset == std::string::npos) {
    suffixOffset = name.rfind("_extract");
  }
  if (suffixOffset == std::string::npos) {
    return false;
  }
  return aggregateFunctions.count(name.substr(0, suffixOffset)) > 0;
}

/// Returns alphabetically sorted list of scalar functions available in Velox,
/// excluding companion functions.
std::vector<std::string> getSortedScalarNames() {
  // Do not print "internal" functions.
  static const std::unordered_set<std::string> kBlockList = {"row_constructor"};

  auto functions = getFunctionSignatures();

  std::vector<std::string> names;
  names.reserve(functions.size());
  exec::aggregateFunctions().withRLock([&](const auto& aggregateFunctions) {
    for (const auto& func : functions) {
      const auto& name = func.first;
      if (!isCompanionFunctionName(name, aggregateFunctions) &&
          kBlockList.count(name) == 0) {
        names.emplace_back(name);
      }
    }
  });
  std::sort(names.begin(), names.end());
  return names;
}

/// Returns alphabetically sorted list of aggregate functions available in
/// Velox, excluding compaion functions.
std::vector<std::string> getSortedAggregateNames() {
  std::vector<std::string> names;
  exec::aggregateFunctions().withRLock([&](const auto& functions) {
    names.reserve(functions.size());
    for (const auto& entry : functions) {
      if (!isCompanionFunctionName(entry.first, functions)) {
        names.push_back(entry.first);
      }
    }
  });
  std::sort(names.begin(), names.end());
  return names;
}

/// Returns alphabetically sorted list of window functions available in Velox,
/// excluding companion functions.
std::vector<std::string> getSortedWindowNames() {
  const auto& functions = exec::windowFunctions();

  std::vector<std::string> names;
  names.reserve(functions.size());
  exec::aggregateFunctions().withRLock([&](const auto& aggregateFunctions) {
    for (const auto& entry : functions) {
      if (!isCompanionFunctionName(entry.first, aggregateFunctions) &&
          aggregateFunctions.count(entry.first) == 0) {
        names.emplace_back(entry.first);
      }
    }
  });
  std::sort(names.begin(), names.end());
  return names;
}

/// Takes a super-set of simple, vector and aggregate function names and prints
/// coverage map showing which of these functions are available in Velox.
/// Companion functions are excluded.
void printCoverageMap(
    const std::vector<std::string>& scalarNames,
    const std::vector<std::string>& aggNames,
    const std::vector<std::string>& windowNames,
    const std::string& domain = "",
    const std::string& typeOfOutput = "") {
  auto veloxFunctions = getFunctionSignatures();

  std::unordered_set<std::string> veloxNames;
  veloxNames.reserve(veloxFunctions.size());
  for (const auto& func : veloxFunctions) {
    veloxNames.emplace(func.first);
  }

  std::unordered_set<std::string> veloxAggNames;
  std::unordered_set<std::string> veloxWindowNames;
  const auto& veloxWindowFunctions = exec::windowFunctions();

  exec::aggregateFunctions().withRLock(
      [&](const auto& veloxAggregateFunctions) {
        for (const auto& entry : veloxAggregateFunctions) {
          if (!isCompanionFunctionName(entry.first, veloxAggregateFunctions)) {
            veloxAggNames.emplace(entry.first);
          }
        }
        for (const auto& entry : veloxWindowFunctions) {
          if (!isCompanionFunctionName(entry.first, veloxAggregateFunctions) &&
              veloxAggregateFunctions.count(entry.first) == 0) {
            veloxWindowNames.emplace(entry.first);
          }
        }
      });

  if (typeOfOutput == "signatures") {
    printCoverageMapWithSignatures(
        scalarNames,
        aggNames,
        windowNames,
        veloxNames,
        veloxAggNames,
        veloxWindowNames,
        domain);
  } else {
    printCoverageMap(
        scalarNames,
        aggNames,
        windowNames,
        veloxNames,
        veloxAggNames,
        veloxWindowNames,
        domain);
  }
}

void printCoverageMapForAll(
    const std::string& domain,
    const std::string& typeOfOutput) {
  auto scalarNames = readFunctionNamesFromFile("all_scalar_functions.txt");
  std::sort(scalarNames.begin(), scalarNames.end());

  auto aggNames = readFunctionNamesFromFile("all_aggregate_functions.txt");
  std::sort(aggNames.begin(), aggNames.end());

  auto windowNames = readFunctionNamesFromFile("all_window_functions.txt");
  std::sort(windowNames.begin(), windowNames.end());

  printCoverageMap(scalarNames, aggNames, windowNames, domain, typeOfOutput);
}

void printVeloxFunctions(
    const std::unordered_set<std::string>& linkBlockList,
    const std::string& domain) {
  auto scalarNames = getSortedScalarNames();
  auto aggNames = getSortedAggregateNames();
  auto windowNames = getSortedWindowNames();

  const int columnSize = std::max(
                             {maxLength(scalarNames),
                              maxLength(aggNames),
                              maxLength(windowNames)}) +
      toFuncLink("", domain).size();

  const std::string indent(4, ' ');

  auto scalarCnt = scalarNames.size();
  auto aggCnt = aggNames.size();
  auto windowCnt = windowNames.size();
  auto numRows =
      std::max({(size_t)std::ceil(scalarCnt / 3.0), aggCnt, windowCnt});

  auto printName = [&](const std::string& name) {
    return linkBlockList.count(name) == 0 ? toFuncLink(name, domain) : name;
  };

  TablePrinter printer(3, columnSize, indent, std::cout);
  printer.header();
  for (int i = 0; i < numRows; i++) {
    printer.startRow();

    // 3 columns of scalar functions.
    for (int j = 0; j < 3; j++) {
      auto n = i + numRows * j;
      n < scalarCnt ? printer.addColumn(printName(scalarNames[n]))
                    : printer.addEmptyColumn();
    }

    // 1 empty column.
    printer.addEmptyColumn();

    // 1 column of aggregate functions.
    i < aggCnt ? printer.addColumn(printName(aggNames[i]))
               : printer.addEmptyColumn();

    // 1 empty column.
    printer.addEmptyColumn();

    // 1 column of window functions.
    i < windowCnt ? printer.addColumn(printName(windowNames[i]))
                  : printer.addEmptyColumn();

    printer.endRow();
  }
  printer.footer();
}

void printVeloxFunctionsWithSignatures(
    const std::unordered_set<std::string>& linkBlockList,
    const std::string& domain) {
  auto scalarNames = getSortedScalarNames();
  auto aggNames = getSortedAggregateNames();
  auto windowNames = getSortedWindowNames();

  auto scalarSignatureMap = getScalarSignatureMap(scalarNames);
  auto aggSignatureMap = getAggregateSignatureMap(aggNames);
  auto windowSignatureMap = getWindowSignatureMap(windowNames);

  // const int columnSize = std::max(
  //                            {maxLength(scalarNames),
  //                             maxLength(aggNames),
  //                             maxLength(windowNames)}) +
  //     toFuncLink("", domain).size();

  const std::string indent(4, ' ');

  auto scalarCnt = scalarNames.size();
  auto aggCnt = aggNames.size();
  auto windowCnt = windowNames.size();
  // auto numRows =
  //     std::max({(size_t)std::ceil(scalarCnt / 3.0), aggCnt, windowCnt});

  auto printName = [&](const std::string& name, const std::string& functype) {
    return linkBlockList.count(name) == 0
        ? toFuncLinkNew(name, domain, functype)
        : name;
  };

  TablePrinter printer(3, 0, indent, std::cout);
  printer.header_new("Scalar Functions");
  for (const auto& scalarName : scalarNames) {
    printer.startRow();
    printer.addColumn(printName(scalarName, ""));
    printer.endRow();
    for (const std::string& signature : scalarSignatureMap.first[scalarName]) {
      printer.startRow();
      printer.addColumn(printName(signature, "signature"));
      printer.endRow();
    }
  }

  printer.header_new("Aggregate Functions");

  for (const auto& aggName : aggNames) {
    printer.startRow();
    printer.addColumn(printName(aggName, ""));
    printer.endRow();
    for (const std::string& signature : aggSignatureMap.first[aggName]) {
      printer.startRow();
      printer.addColumn(printName(signature, "signature"));
      printer.endRow();
    }
  }

  printer.header_new("Window Functions");

  for (const auto& windowName : windowNames) {
    printer.startRow();
    printer.addColumn(printName(windowName, ""));
    printer.endRow();
    for (const std::string& signature : windowSignatureMap.first[windowName]) {
      printer.startRow();
      printer.addColumn(printName(signature, "signature"));
      printer.endRow();
    }
  }

  printer.footer();
}

void printCoverageMapForMostUsed(
    const std::string& domain,
    const std::string& typeOfOutput) {
  auto scalarNameList = readFunctionNamesFromFile("all_scalar_functions.txt");
  std::unordered_set<std::string> scalarNames(
      scalarNameList.begin(), scalarNameList.end());

  auto aggNameList = readFunctionNamesFromFile("all_aggregate_functions.txt");
  std::unordered_set<std::string> aggNames(
      aggNameList.begin(), aggNameList.end());

  auto windowNameList = readFunctionNamesFromFile("all_window_functions.txt");
  std::unordered_set<std::string> windowNames(
      windowNameList.begin(), windowNameList.end());

  auto allMostUsed = readFunctionNamesFromFile("most_used_functions.txt");
  std::vector<std::string> scalarMostUsed;
  std::copy_if(
      allMostUsed.begin(),
      allMostUsed.end(),
      std::back_inserter(scalarMostUsed),
      [&](auto name) { return scalarNames.count(name) > 0; });

  std::vector<std::string> aggMostUsed;
  std::copy_if(
      allMostUsed.begin(),
      allMostUsed.end(),
      std::back_inserter(aggMostUsed),
      [&](auto name) { return aggNames.count(name) > 0; });

  std::vector<std::string> windowMostUsed;
  std::copy_if(
      allMostUsed.begin(),
      allMostUsed.end(),
      std::back_inserter(windowMostUsed),
      [&](auto name) { return windowNames.count(name) > 0; });

  printCoverageMap(
      scalarMostUsed, aggMostUsed, windowMostUsed, domain, typeOfOutput);
}

} // namespace facebook::velox::functions
