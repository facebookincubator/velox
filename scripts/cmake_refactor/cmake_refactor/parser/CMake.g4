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

file_:  (add_library | add_alias | add_interface | link_libraries | include_directories | command )* EOF;

add_library: Add_library '(' target library_type? source_file+ Exclude? ')';

add_alias: Add_library '(' target alias target ')';

// There are use cases for interface libraries with source though velox does not use them
add_interface: Add_library '(' target interface ')';

link_libraries: 'target_link_libraries' '(' target keyword? link_targets additonal_targets? ')';

include_directories: 'target_include_directories' '(' target keyword? single_argument+ ')';

link_targets: target+;

additonal_targets: keyword link_targets;

source_file: Header | Source | Variable;

command: Name '(' (single_argument|compound_argument)* ')';

single_argument:  Name | Variable | Path | source_file | Identifier_namespace | keyword | Unquoted_argument | Bracket_argument | Quoted_argument;

compound_argument:  '(' (single_argument|compound_argument)* ')';

target:  Name | Identifier_namespace | Variable; 

Variable: '${'  Name '}';

Identifier_namespace: Identifier '::' Identifier;
 
// Target:  Identifier;

library_type: 'STATIC' | 'SHARED' | 'MODULE' | 'OBJECT';

// with each keyword as a parser rule we can do .public() instead of having to resort to string cmp
keyword:  public | private | interface;

public: 'PUBLIC';

private: 'PRIVATE';

interface: 'INTERFACE';

alias: 'ALIAS';

Exclude: 'EXCLUDE_FROM_ALL';

Add_library: 'add_library';

Header: Path'.' 'h' ('pp')?;

// We should unify endings across the repo probably?
Source: Path'.' 'c' ('c' | 'pp' | '++' | 'xx')?;

Name: Identifier;
// this should probably be a fragment that is used to construct more specific tokens like command/target/header/source...
fragment
Identifier:  '${'?[a-z_.+0-9-]([-/a-z0-9_]|'${' | '+' | '}' | '.')*;


Unquoted_argument:  (~[ \t\r\n()#"\\] | Escape_sequence)+;

Escape_sequence:  Escape_identity | Escape_encoded | Escape_semicolon;

Path: Variable? [.a-z0-9/_-]+;
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
