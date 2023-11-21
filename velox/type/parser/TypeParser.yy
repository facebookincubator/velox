%{
#include <FlexLexer.h>
#include "velox/common/base/Exceptions.h"
#include "velox/type/Type.h"
%}
%require "3.0.4"
%language "C++"

%define parser_class_name {Parser}
%define api.namespace {facebook::velox::type}
%define api.value.type variant
%parse-param {Scanner* scanner}
%define parse.error verbose

%code requires
{
    namespace facebook::velox::type {
        class Scanner;
    } // namespace facebook::velox::type
    namespace facebook::velox {
        class Type;
    } // namespace facebook::velox
    struct RowArguments {
       std::vector<std::string> names;
       std::vector<std::shared_ptr<const facebook::velox::Type>> types;
    };
} // %code requires

%code
{
    #include <velox/type/parser/Scanner.h>
    #define yylex(x) scanner->lex(x)
    using namespace facebook::velox;
    TypePtr typeFromString(const std::string& type, bool failIfNotRegistered = true) {
        auto upper = type;
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
        if (upper == "INT") {
            upper = "INTEGER";
        } else if (upper == "DOUBLE PRECISION") {
            upper = "DOUBLE";
        }
        auto inferredType = getType(upper, {});
        if (failIfNotRegistered) {
            VELOX_CHECK(inferredType, "Failed to parse type [{}]. Type not registered.", type);
        }
        return inferredType;
    }

    std::pair<std::string, std::shared_ptr<const Type>> inferTypeWithSpaces(std::vector<std::string>& words,
                                                                            bool cannotHaveFieldName = false) {
        // First check if all the words form a type.
        // Then check if the first word is a field name and the remaining form a type.
        // If cannotHaveFieldName = true, then all words must form a type.
        VELOX_CHECK_GE(words.size(), 2);
        std::string fieldName = words[0];
        std::string typeName = words[1];
        for (int i = 2; i < words.size(); ++i) {
           typeName = fmt::format("{} {}", typeName, words[i]);
        }
        auto allWords = fmt::format("{} {}", fieldName, typeName);
        // Fail if cannotHaveFieldName = true.
        auto type = typeFromString(allWords, cannotHaveFieldName);
        if (type) {
            return std::make_pair("", type);
        }
        return std::make_pair(fieldName, typeFromString(typeName));
    }
}

%token               LPAREN RPAREN COMMA ARRAY MAP ROW FUNCTION DECIMAL
%token <std::string> WORD VARIABLE QUOTED_ID
%token <long long>   NUMBER
%token YYEOF         0

%nterm <std::shared_ptr<const Type>> special_type function_type decimal_type row_type array_type map_type variable_type
%nterm <std::shared_ptr<const Type>> type
%nterm <RowArguments> type_list_opt_names
%nterm <std::vector<std::shared_ptr<const Type>>> type_list
%nterm <std::pair<std::string, std::shared_ptr<const Type>>> named_type
%nterm <std::vector<std::string>> type_with_spaces

%%

type_spec : named_type           { scanner->setType($1.second); }
          | type                 { scanner->setType($1); }
          | error                { yyerrok; }
          ;

named_type : QUOTED_ID type       { $1.erase(0, 1); $1.pop_back(); $$ = std::make_pair($1, $2); }  // Remove the quotes.
           | QUOTED_ID type_with_spaces { $1.erase(0, 1); $1.pop_back(); auto type = inferTypeWithSpaces($2, true); $$ = std::make_pair($1, type.second); }  // Remove the quotes.
           | WORD special_type    { $$ = std::make_pair($1, $2); }
           | type_with_spaces     { $$ = inferTypeWithSpaces($1); }
           ;

special_type : array_type                  { $$ = $1; }
             | map_type                    { $$ = $1; }
             | row_type                    { $$ = $1; }
             | function_type               { $$ = $1; }
             | variable_type               { $$ = $1; }
             | decimal_type                { $$ = $1; }
             ;

type : special_type    { $$ = $1; }
     | WORD            { $$ = typeFromString($1); }

type_with_spaces : type_with_spaces WORD { $1.push_back($2); $$ = std::move($1); }
                 | WORD WORD             { $$.push_back($1); $$.push_back($2); }
                 ;

variable_type : VARIABLE LPAREN NUMBER RPAREN  { $$ = typeFromString($1); }
              | VARIABLE                       { $$ = typeFromString($1); }
              ;

array_type : ARRAY LPAREN type RPAREN { $$ = ARRAY($3); }
           ;

decimal_type : DECIMAL LPAREN NUMBER COMMA NUMBER RPAREN { $$ = DECIMAL($3, $5); }
             ;

type_list : type                   { $$.push_back($1); }
          | type_list COMMA type   { $1.push_back($3); $$ = std::move($1); }
          ;

type_list_opt_names : type_list_opt_names COMMA named_type { $1.names.push_back($3.first); $1.types.push_back($3.second);
                                                             $$.names = std::move($1.names); $$.types = std::move($1.types); }
                    | named_type                           { $$.names.push_back($1.first); $$.types.push_back($1.second); }
                    | type_list_opt_names COMMA type       { $1.names.push_back(""); $1.types.push_back($3);
                                                             $$.names = std::move($1.names); $$.types = std::move($1.types); }
                    | type                                 { $$.names.push_back(""); $$.types.push_back($1); }
                    ;

row_type : ROW LPAREN type_list_opt_names RPAREN  { $$ = ROW(std::move($3.names), std::move($3.types)); }
         ;

map_type : MAP LPAREN type COMMA type RPAREN { $$ = MAP($3, $5); }
         ;

function_type : FUNCTION LPAREN type_list RPAREN { auto returnType = $3.back(); $3.pop_back();
                                                   $$ = FUNCTION(std::move($3), returnType); }

%%

void facebook::velox::type::Parser::error(const std::string& msg) {
    VELOX_FAIL("Failed to parse type [{}]. {}", scanner->input(), msg);
}
