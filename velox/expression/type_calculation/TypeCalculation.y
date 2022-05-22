%{
#include <FlexLexer.h>
#include <velox/common/base/Exceptions.h>
%}
%require "3.7.4"
%language "C++"
 
%define api.parser.class {Parser}
%define api.namespace {facebook::velox::expression::calculate}
%define api.value.type variant
%parse-param {Scanner* scanner}
%define parse.error detailed
 
%code requires
{
    namespace facebook::velox::expression::calculate {
        class Scanner;
    } // namespace facebook::velox::expression::calculate
} // %code requires
 
%code
{
    #include "Scanner.h"
    #define yylex(x) scanner->lex(x)
}

%token              EOL LPAREN RPAREN COMMA MIN MAX
%token <long long>  INT
%token <std::string> VAR

%nterm <long long>  iexp
 
%nonassoc           ASSIGN
%left               PLUS MINUS
%left               MULTIPLY DIVIDE MODULO
%precedence         UMINUS
 
%%
 
lines   : %empty
        | lines line
        ;
 
line    : EOL                       { std::cerr << "Read an empty line.\n"; }
        | VAR ASSIGN iexp EOL       { scanner->setValue($1, $3); }
        | error EOL                 { yyerrok; }
        ;
 
iexp    : INT                       { $$ = $1; }
        | iexp PLUS iexp            { $$ = $1 + $3; }
        | iexp MINUS iexp           { $$ = $1 - $3; }
        | iexp MULTIPLY iexp        { $$ = $1 * $3; }
        | iexp DIVIDE iexp          { $$ = $1 / $3; }
        | iexp MODULO iexp          { $$ = $1 % $3; }
        | MINUS iexp %prec UMINUS   { $$ = -$2; }
        | LPAREN iexp RPAREN        { $$ = $2; }
        | MAX LPAREN iexp COMMA iexp RPAREN { $$ = std::max($3, $5); }
        | MIN LPAREN iexp COMMA iexp RPAREN { $$ = std::min($3, $5); }
        | VAR                       { $$ = scanner->getValue($1); }
        ;
 
%%
 
void facebook::velox::expression::calculate::Parser::error(const std::string& msg) {
    VELOX_FAIL(msg);
}
