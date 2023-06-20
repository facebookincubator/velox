/*
Copyright (c) 2018  zbq.

License for use and distribution: Eclipse Public License

CMake language grammar reference:
https://cmake.org/cmake/help/v3.12/manual/cmake-language.7.html

*/

grammar CMake;

options 
{

language=Python3;
caseInsensitive = true;
}

file_:  (target_command | command )* EOF;

target_command: Target_command '(' target (keyword)? (single_argument)* ')';

command:  Identifier '(' (single_argument|compound_argument)* ')';

single_argument:  Identifier | Unquoted_argument | Bracket_argument | Quoted_argument;

compound_argument:  '(' (single_argument|compound_argument)* ')';

Target_command: 'add_library';

target:  Identifier | Identifier_namespace; 

Identifier_namespace: Identifier '::' Identifier;
 
// Target:  Identifier;

// with each keyword as a parser rule we can do .public() instead of having to resort to string cmp
keyword:  public | private | interface | alias;

public: 'PUBLIC';

private: 'PRIVATE';

interface: 'INTERFACE';

alias: 'ALIAS';

// this should probably be a fragment that is used to construct more specific tokens like command/target/header/source...
Identifier:  [a-z_][a-z0-9_]*;

Unquoted_argument:  (~[ \t\r\n()#"\\] | Escape_sequence)+;

Escape_sequence:  Escape_identity | Escape_encoded | Escape_semicolon;

fragment
Escape_identity:  '\\' ~[a-z0-9;];

fragment
Escape_encoded:  '\\t' | '\\r' | '\\n';

fragment
Escape_semicolon:  '\\;';

Quoted_argument:  '"' (~[\\"] | Escape_sequence | Quoted_cont)* '"';

fragment
Quoted_cont:  '\\' ('\r' '\n'? | '\n');

Bracket_argument:  '[' Bracket_arg_nested ']';

fragment
Bracket_arg_nested:  '=' Bracket_arg_nested '='
	| '[' .*? ']';

Bracket_comment:  '#[' Bracket_arg_nested ']'
	-> channel(HIDDEN);

Line_comment:  '#' (  // #
	  	  | '[' '='*   // #[==
		  | '[' '='* ~('=' | '[' | '\r' | '\n') ~('\r' | '\n')*  // #[==xx
		  | ~('[' | '\r' | '\n') ~('\r' | '\n')*  // #xx
		  ) ('\r' '\n'? | '\n' | EOF)
    -> channel(HIDDEN);

Newline:  ('\r' '\n'? | '\n')+
	-> channel(HIDDEN);

Space:  [ \t]+
	-> channel(HIDDEN);
