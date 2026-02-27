%{
#include <FlexLexer.h>
#include "velox/common/base/Exceptions.h"
#include "velox/expression/TypeSignature.h"
#include "velox/expression/signature_parser/ParseUtil.h"
%}
%require "3.0.4"
%language "C++"

%define parser_class_name {Parser}
%define api.namespace {facebook::velox::exec}
%define api.value.type variant
%parse-param {Scanner* scanner}
%define parse.error verbose

%code requires
{
    namespace facebook::velox::exec {
        class Scanner;
        class TypeSignature;
    } // namespace facebook::velox::exec
} // %code requires

%code
{
    #include <velox/expression/signature_parser/Scanner.h>
    #define yylex(x) scanner->lex(x)
}

%token               LPAREN RPAREN COMMA ARRAY MAP ROW FUNCTION ELLIPSIS
%token <std::string> WORD VARIABLE QUOTED_ID DECIMAL
%token YYEOF         0

%expect 0

%nterm <std::shared_ptr<exec::TypeSignature>> special_type function_type decimal_type row_type array_type map_type
%nterm <std::shared_ptr<exec::TypeSignature>> type named_type row_field
%nterm <std::vector<exec::TypeSignature>> type_list row_field_list
%nterm <std::vector<std::string>> type_with_spaces

%%

type_spec : type                 { scanner->setTypeSignature($1); }
          | type_with_spaces     { scanner->setTypeSignature(inferTypeWithSpaces($1)); }
          | error                { yyerrok; }
          ;

type : special_type   { $$ = $1; }
     | WORD           { $$ = std::make_shared<exec::TypeSignature>(exec::TypeSignature($1, {})); }
     | WORD LPAREN type_list RPAREN { $$ = std::make_shared<exec::TypeSignature>(exec::TypeSignature($1, std::move($3))); }
     ;

special_type : array_type                  { $$ = $1; }
             | map_type                    { $$ = $1; }
             | row_type                    { $$ = $1; }
             | function_type               { $$ = $1; }
             | decimal_type                { $$ = $1; }
             ;

named_type : QUOTED_ID type          { $1.erase(0, 1); $1.pop_back(); $$ = std::make_shared<exec::TypeSignature>(exec::TypeSignature($2->baseName(), $2->parameters(), $1)); }  // Remove the quotes.
           | WORD special_type       { $$ = std::make_shared<exec::TypeSignature>(exec::TypeSignature($2->baseName(), $2->parameters(), $1)); }
           | type_with_spaces        { $$ = inferTypeWithSpaces($1, true); }
           ;

type_with_spaces : type_with_spaces WORD { $1.push_back($2); $$ = std::move($1); }
                 | WORD WORD             { $$.push_back($1); $$.push_back($2); }
                 ;

decimal_type : DECIMAL LPAREN WORD COMMA WORD RPAREN { $$ = std::make_shared<exec::TypeSignature>(exec::TypeSignature($1, { exec::TypeSignature($3, {}), exec::TypeSignature($5, {}) })); }
             ;

type_list : type                   { $$.push_back(*($1)); }
          | type_list COMMA type   { $1.push_back(*($3)); $$ = std::move($1); }
          ;

row_field : named_type { $$ = $1; }
          | type       { $$ = $1; }
          ;

row_field_list : row_field { $$.push_back(*($1)); }
               | row_field_list COMMA row_field { $1.push_back(*($3)); $$ = std::move($1); }
               ;

row_type
    : ROW LPAREN row_field RPAREN
        { std::vector<exec::TypeSignature> params; params.push_back(*($3)); $$ = std::make_shared<exec::TypeSignature>("row", params); }
    | ROW LPAREN row_field COMMA ELLIPSIS RPAREN
        { if ($3->rowFieldName().has_value()) { VELOX_FAIL("Homogeneous row cannot have a field name"); } std::vector<exec::TypeSignature> params; params.push_back(*($3)); $$ = std::make_shared<exec::TypeSignature>("row", params, std::nullopt, true); }
    | ROW LPAREN row_field COMMA row_field_list RPAREN
        { std::vector<exec::TypeSignature> params; params.push_back(*($3)); for (auto& f : $5) { params.push_back(f); } $$ = std::make_shared<exec::TypeSignature>("row", params); }
    ;

// Homogeneous rows use only the explicit pattern: row(T, ...)

array_type : ARRAY LPAREN type RPAREN             { $$ = std::make_shared<exec::TypeSignature>(exec::TypeSignature("array", { *($3) })); }
           | ARRAY LPAREN type_with_spaces RPAREN { $$ = std::make_shared<exec::TypeSignature>(exec::TypeSignature("array", { *inferTypeWithSpaces($3) })); }
           ;

map_type : MAP LPAREN type COMMA type RPAREN                         { $$ = std::make_shared<exec::TypeSignature>(exec::TypeSignature("map", {*($3), *($5)})); }
         | MAP LPAREN type COMMA type_with_spaces RPAREN             { $$ = std::make_shared<exec::TypeSignature>(exec::TypeSignature("map", {*($3), *inferTypeWithSpaces($5)})); }
         | MAP LPAREN type_with_spaces COMMA type RPAREN             { $$ = std::make_shared<exec::TypeSignature>(exec::TypeSignature("map", {*inferTypeWithSpaces($3), *($5)})); }
         | MAP LPAREN type_with_spaces COMMA type_with_spaces RPAREN { $$ = std::make_shared<exec::TypeSignature>(exec::TypeSignature("map", {*inferTypeWithSpaces($3), *inferTypeWithSpaces($5)})); }
         ;

function_type : FUNCTION LPAREN type_list RPAREN { $$ = std::make_shared<exec::TypeSignature>(exec::TypeSignature("function", {$3})); }

%%

void facebook::velox::exec::Parser::error(const std::string& msg) {
    VELOX_FAIL("Failed to parse type signature [{}]: {}", scanner->input(), msg);
}

#ifdef VELOX_ENABLE_ASAN_UBSAN_SANITIZERS
// When using sanitizers (address/undefined behavior) the compiler
// wants the typeinfo for yyFlexLexer in the mono library.
// But in the normal compilation this is not required in the mono library.
// Therefore, this adds a dummy implementation to force creation of the
// typeinfo to avoid link issues.
// There are two other parsers: The old TypeParser (to be removed) and
// new (moved) TypeParser for prestosql.
// The dummy class needs to be added only once and implementes the
// virtual functions of yyFlexLexer.
#ifdef yylex
#undef yylex
#endif
yyFlexLexer::~yyFlexLexer() {}
void yyFlexLexer::yy_switch_to_buffer(yy_buffer_state*) {
  throw std::runtime_error("Bad call to yyFlexLexer::yy_switch_to_buffer()");
}
yy_buffer_state* yyFlexLexer::yy_create_buffer( std::istream* s, int size ) {
  throw std::runtime_error("Bad call to yyFlexLexer::yy_create_buffer()");
}
yy_buffer_state* yyFlexLexer::yy_create_buffer( std::istream& s, int size ) {
  throw std::runtime_error("Bad call to yyFlexLexer::yy_create_buffer()");
}
void yyFlexLexer::yy_delete_buffer( yy_buffer_state* b ) {
  throw std::runtime_error("Bad call to yyFlexLexer::yy_delete_buffer()");
}
void yyFlexLexer::yyrestart( std::istream* s ) {
  throw std::runtime_error("Bad call to yyFlexLexer::yyrestart()");
}
void yyFlexLexer::yyrestart( std::istream& s ) {
  throw std::runtime_error("Bad call to yyFlexLexer::yyrestart()");
}
void yyFlexLexer::yypush_buffer_state( yy_buffer_state* new_buffer ) {
  throw std::runtime_error("Bad call to yyFlexLexer::yypush_buffer_state()");
}
void yyFlexLexer::yypop_buffer_state() {
  throw std::runtime_error("Bad call to yyFlexLexer::yypop_buffer_state()");
}
int yyFlexLexer::yylex() {
  throw std::runtime_error("Bad call to yyFlexLexer::yylex() - dummy");
}
void yyFlexLexer::switch_streams( std::istream& new_in, std::ostream& new_out ) {
  throw std::runtime_error("Bad call to yyFlexLexer::switch_streams()");
}
void yyFlexLexer::switch_streams( std::istream* new_in, std::ostream* new_out ) {
  throw std::runtime_error("Bad call to yyFlexLexer::switch_streams()");
}
int yyFlexLexer::yywrap() {
  throw std::runtime_error("Bad call to yyFlexLexer::yywrap()");
}
int yyFlexLexer::LexerInput( char* buf, int max_size ) {
  throw std::runtime_error("Bad call to yyFlexLexer::LexerInput()");
}
void yyFlexLexer::LexerOutput( const char* buf, int size ) {
  throw std::runtime_error("Bad call to yyFlexLexer::LexerOutput()");
}
void yyFlexLexer::LexerError( const char* msg ) {
  throw std::runtime_error("Bad call to yyFlexLexer::LexerError()");
}
#endif // VELOX_ENABLE_ASAN_UBSAN_SANITIZERS
