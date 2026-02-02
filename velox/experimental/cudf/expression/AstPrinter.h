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
/*
 * A standalone AST expression tree printer for cudf::ast::expression.
 * Can be copied to any project that uses cudf AST expressions.
 *
 * Usage:
 *   cudf::ast::expression const& expr = ...;
 *   cudf::ast::print_expression(expr);
 *   // or
 *   std::string str = cudf::ast::expression_to_string(expr);
 */
#pragma once

#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/expressions.hpp>

#include <iostream>
#include <sstream>
#include <string>

namespace cudf {
namespace ast {

/**
 * @brief Get the string representation of an ast_operator
 */
inline std::string ast_operator_to_string(ast_operator op) {
  switch (op) {
    // Binary operators
    case ast_operator::ADD:
      return "ADD (+)";
    case ast_operator::SUB:
      return "SUB (-)";
    case ast_operator::MUL:
      return "MUL (*)";
    case ast_operator::DIV:
      return "DIV (/)";
    case ast_operator::TRUE_DIV:
      return "TRUE_DIV";
    case ast_operator::FLOOR_DIV:
      return "FLOOR_DIV";
    case ast_operator::MOD:
      return "MOD (%)";
    case ast_operator::PYMOD:
      return "PYMOD";
    case ast_operator::POW:
      return "POW (^)";
    case ast_operator::EQUAL:
      return "EQUAL (==)";
    case ast_operator::NULL_EQUAL:
      return "NULL_EQUAL";
    case ast_operator::NOT_EQUAL:
      return "NOT_EQUAL (!=)";
    case ast_operator::LESS:
      return "LESS (<)";
    case ast_operator::GREATER:
      return "GREATER (>)";
    case ast_operator::LESS_EQUAL:
      return "LESS_EQUAL (<=)";
    case ast_operator::GREATER_EQUAL:
      return "GREATER_EQUAL (>=)";
    case ast_operator::BITWISE_AND:
      return "BITWISE_AND (&)";
    case ast_operator::BITWISE_OR:
      return "BITWISE_OR (|)";
    case ast_operator::BITWISE_XOR:
      return "BITWISE_XOR (^)";
    case ast_operator::LOGICAL_AND:
      return "LOGICAL_AND (&&)";
    case ast_operator::NULL_LOGICAL_AND:
      return "NULL_LOGICAL_AND";
    case ast_operator::LOGICAL_OR:
      return "LOGICAL_OR (||)";
    case ast_operator::NULL_LOGICAL_OR:
      return "NULL_LOGICAL_OR";
    // Unary operators
    case ast_operator::IDENTITY:
      return "IDENTITY";
    case ast_operator::IS_NULL:
      return "IS_NULL";
    case ast_operator::SIN:
      return "SIN";
    case ast_operator::COS:
      return "COS";
    case ast_operator::TAN:
      return "TAN";
    case ast_operator::ARCSIN:
      return "ARCSIN";
    case ast_operator::ARCCOS:
      return "ARCCOS";
    case ast_operator::ARCTAN:
      return "ARCTAN";
    case ast_operator::SINH:
      return "SINH";
    case ast_operator::COSH:
      return "COSH";
    case ast_operator::TANH:
      return "TANH";
    case ast_operator::ARCSINH:
      return "ARCSINH";
    case ast_operator::ARCCOSH:
      return "ARCCOSH";
    case ast_operator::ARCTANH:
      return "ARCTANH";
    case ast_operator::EXP:
      return "EXP";
    case ast_operator::LOG:
      return "LOG";
    case ast_operator::SQRT:
      return "SQRT";
    case ast_operator::CBRT:
      return "CBRT";
    case ast_operator::CEIL:
      return "CEIL";
    case ast_operator::FLOOR:
      return "FLOOR";
    case ast_operator::ABS:
      return "ABS";
    case ast_operator::RINT:
      return "RINT";
    case ast_operator::BIT_INVERT:
      return "BIT_INVERT (~)";
    case ast_operator::NOT:
      return "NOT (!)";
    case ast_operator::CAST_TO_INT64:
      return "CAST_TO_INT64";
    case ast_operator::CAST_TO_UINT64:
      return "CAST_TO_UINT64";
    case ast_operator::CAST_TO_FLOAT64:
      return "CAST_TO_FLOAT64";
    default:
      return "UNKNOWN_OPERATOR";
  }
}

/**
 * @brief Get the string representation of a table_reference
 */
inline std::string table_reference_to_string(table_reference ref) {
  switch (ref) {
    case table_reference::LEFT:
      return "LEFT";
    case table_reference::RIGHT:
      return "RIGHT";
    case table_reference::OUTPUT:
      return "OUTPUT";
    default:
      return "UNKNOWN";
  }
}

/**
 * @brief Get the string representation of a cudf::type_id
 */
inline std::string type_id_to_string(cudf::type_id id) {
  switch (id) {
    case cudf::type_id::EMPTY:
      return "EMPTY";
    case cudf::type_id::INT8:
      return "INT8";
    case cudf::type_id::INT16:
      return "INT16";
    case cudf::type_id::INT32:
      return "INT32";
    case cudf::type_id::INT64:
      return "INT64";
    case cudf::type_id::UINT8:
      return "UINT8";
    case cudf::type_id::UINT16:
      return "UINT16";
    case cudf::type_id::UINT32:
      return "UINT32";
    case cudf::type_id::UINT64:
      return "UINT64";
    case cudf::type_id::FLOAT32:
      return "FLOAT32";
    case cudf::type_id::FLOAT64:
      return "FLOAT64";
    case cudf::type_id::BOOL8:
      return "BOOL8";
    case cudf::type_id::TIMESTAMP_DAYS:
      return "TIMESTAMP_DAYS";
    case cudf::type_id::TIMESTAMP_SECONDS:
      return "TIMESTAMP_SECONDS";
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return "TIMESTAMP_MILLISECONDS";
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return "TIMESTAMP_MICROSECONDS";
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return "TIMESTAMP_NANOSECONDS";
    case cudf::type_id::DURATION_DAYS:
      return "DURATION_DAYS";
    case cudf::type_id::DURATION_SECONDS:
      return "DURATION_SECONDS";
    case cudf::type_id::DURATION_MILLISECONDS:
      return "DURATION_MILLISECONDS";
    case cudf::type_id::DURATION_MICROSECONDS:
      return "DURATION_MICROSECONDS";
    case cudf::type_id::DURATION_NANOSECONDS:
      return "DURATION_NANOSECONDS";
    case cudf::type_id::DICTIONARY32:
      return "DICTIONARY32";
    case cudf::type_id::STRING:
      return "STRING";
    case cudf::type_id::LIST:
      return "LIST";
    case cudf::type_id::DECIMAL32:
      return "DECIMAL32";
    case cudf::type_id::DECIMAL64:
      return "DECIMAL64";
    case cudf::type_id::DECIMAL128:
      return "DECIMAL128";
    case cudf::type_id::STRUCT:
      return "STRUCT";
    default:
      return "UNKNOWN_TYPE";
  }
}

/**
 * @brief A visitor class that prints an AST expression tree
 *
 * Uses dynamic_cast (RTTI) to identify expression types since modifying
 * the base expression class is not possible for a standalone file.
 */
class expression_printer {
 public:
  /**
   * @brief Construct a new expression printer
   *
   * @param os Output stream to write to
   * @param indent_str String to use for each level of indentation
   */
  explicit expression_printer(std::ostream& os, std::string indent_str = "  ")
      : _os(os), _indent_str(std::move(indent_str)) {}

  /**
   * @brief Visit and print an expression tree
   *
   * @param expr The expression to print
   * @param depth Current depth in the tree (for indentation)
   * @return The index of this node in the visitation order
   */
  int visit(expression const& expr, int depth = 0) {
    int current_index = _node_index++;

    // Try each expression type using dynamic_cast
    if (auto const* lit = dynamic_cast<literal const*>(&expr)) {
      visit_literal(*lit, current_index, depth);
    } else if (
        auto const* col_ref = dynamic_cast<column_reference const*>(&expr)) {
      visit_column_reference(*col_ref, current_index, depth);
    } else if (auto const* op = dynamic_cast<operation const*>(&expr)) {
      visit_operation(*op, current_index, depth);
    } else if (
        auto const* col_name =
            dynamic_cast<column_name_reference const*>(&expr)) {
      visit_column_name_reference(*col_name, current_index, depth);
    } else {
      print_indent(depth);
      _os << "[" << current_index << "] UNKNOWN_EXPRESSION_TYPE\n";
    }

    return current_index;
  }

  /**
   * @brief Reset the node index counter
   */
  void reset() {
    _node_index = 0;
  }

  /**
   * @brief Get the current node count
   */
  [[nodiscard]] int node_count() const {
    return _node_index;
  }

 private:
  std::ostream& _os;
  std::string _indent_str;
  int _node_index = 0;

  void print_indent(int depth) {
    for (int i = 0; i < depth; ++i) {
      _os << _indent_str;
    }
  }

  void visit_literal(literal const& expr, int index, int depth) {
    print_indent(depth);
    _os << "[" << index << "] LITERAL\n";

    print_indent(depth + 1);
    _os << "data_type: " << type_id_to_string(expr.get_data_type().id())
        << "\n";

    // Note: Cannot print the actual value without knowing the type at compile
    // time and having device memory access. The scalar is on the device.
    print_indent(depth + 1);
    _os << "(value is stored on device, type-specific access required)\n";
  }

  void
  visit_column_reference(column_reference const& expr, int index, int depth) {
    print_indent(depth);
    _os << "[" << index << "] COLUMN_REFERENCE\n";

    print_indent(depth + 1);
    _os << "column_index: " << expr.get_column_index() << "\n";

    print_indent(depth + 1);
    _os << "table_source: "
        << table_reference_to_string(expr.get_table_source()) << "\n";
  }

  void visit_operation(operation const& expr, int index, int depth) {
    print_indent(depth);
    _os << "[" << index << "] OPERATION\n";

    print_indent(depth + 1);
    _os << "operator: " << ast_operator_to_string(expr.get_operator()) << "\n";

    auto const& operands = expr.get_operands();
    print_indent(depth + 1);
    _os << "arity: " << operands.size() << "\n";

    if (!operands.empty()) {
      print_indent(depth + 1);
      _os << "operands:\n";
      for (size_t i = 0; i < operands.size(); ++i) {
        print_indent(depth + 2);
        _os << "operand[" << i << "]:\n";
        visit(operands[i].get(), depth + 3);
      }
    }
  }

  void visit_column_name_reference(
      column_name_reference const& expr,
      int index,
      int depth) {
    print_indent(depth);
    _os << "[" << index << "] COLUMN_NAME_REFERENCE\n";

    print_indent(depth + 1);
    _os << "column_name: \"" << expr.get_column_name() << "\"\n";
  }
};

/**
 * @brief Print an AST expression tree to an output stream
 *
 * @param expr The expression to print
 * @param os Output stream (defaults to std::cout)
 * @param indent_str Indentation string for each level
 */
inline void print_expression(
    expression const& expr,
    std::ostream& os = std::cout,
    std::string const& indent_str = "  ") {
  os << "=== AST Expression Tree ===\n";
  expression_printer printer(os, indent_str);
  printer.visit(expr);
  os << "=== Total nodes: " << printer.node_count() << " ===\n";
}

/**
 * @brief Convert an AST expression tree to a string
 *
 * @param expr The expression to convert
 * @param indent_str Indentation string for each level
 * @return String representation of the expression tree
 */
inline std::string expression_to_string(
    expression const& expr,
    std::string const& indent_str = "  ") {
  std::ostringstream oss;
  print_expression(expr, oss, indent_str);
  return oss.str();
}

/**
 * @brief A compact one-line representation of an expression
 *
 * Creates a more compact representation suitable for logging/debugging
 */
class compact_expression_printer {
 public:
  /**
   * @brief Get a compact string representation of an expression
   *
   * @param expr The expression to represent
   * @return Compact string representation
   */
  static std::string to_string(expression const& expr) {
    return visit_impl(expr);
  }

 private:
  static std::string visit_impl(expression const& expr) {
    if (auto const* lit = dynamic_cast<literal const*>(&expr)) {
      return "Literal(" + type_id_to_string(lit->get_data_type().id()) + ")";
    }

    if (auto const* col_ref = dynamic_cast<column_reference const*>(&expr)) {
      return "Col(" + table_reference_to_string(col_ref->get_table_source()) +
          "[" + std::to_string(col_ref->get_column_index()) + "])";
    }

    if (auto const* op = dynamic_cast<operation const*>(&expr)) {
      auto const& operands = op->get_operands();
      std::string result = ast_operator_to_string(op->get_operator()) + "(";
      for (size_t i = 0; i < operands.size(); ++i) {
        if (i > 0)
          result += ", ";
        result += visit_impl(operands[i].get());
      }
      result += ")";
      return result;
    }

    if (auto const* col_name =
            dynamic_cast<column_name_reference const*>(&expr)) {
      return "ColName(\"" + col_name->get_column_name() + "\")";
    }

    return "Unknown";
  }
};

/**
 * @brief Get a compact one-line string representation of an expression
 *
 * @param expr The expression to represent
 * @return Compact string representation
 */
inline std::string expression_to_compact_string(expression const& expr) {
  return compact_expression_printer::to_string(expr);
}

/**
 * @brief Print a compact one-line representation of an expression
 *
 * @param expr The expression to print
 * @param os Output stream (defaults to std::cout)
 */
inline void print_expression_compact(
    expression const& expr,
    std::ostream& os = std::cout) {
  os << expression_to_compact_string(expr) << "\n";
}

} // namespace ast
} // namespace cudf
