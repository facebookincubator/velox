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
#include <FlexLexer.h>

#include "velox/common/base/Exceptions.h"

// Provide a dedicated translation unit for yyFlexLexer so the linker has a
// single, always-linked home for its vtable and typeinfo. Some builds,
// especially mono-library builds with sanitizers, can otherwise fail to
// resolve the base-class RTTI when individual parser targets only provide
// derived scanner implementations.
#ifdef yylex
#undef yylex
#endif

yyFlexLexer::~yyFlexLexer() {}

int yyFlexLexer::yylex() {
  VELOX_FAIL("Bad call to yyFlexLexer::yylex()");
}

void yyFlexLexer::yy_switch_to_buffer(yy_buffer_state*) {
  VELOX_FAIL("Bad call to yyFlexLexer::yy_switch_to_buffer()");
}
yy_buffer_state* yyFlexLexer::yy_create_buffer(std::istream* s, int size) {
  VELOX_FAIL("Bad call to yyFlexLexer::yy_create_buffer()");
}
yy_buffer_state* yyFlexLexer::yy_create_buffer(std::istream& s, int size) {
  VELOX_FAIL("Bad call to yyFlexLexer::yy_create_buffer()");
}
void yyFlexLexer::yy_delete_buffer(yy_buffer_state* b) {
  VELOX_FAIL("Bad call to yyFlexLexer::yy_delete_buffer()");
}
void yyFlexLexer::yyrestart(std::istream* s) {
  VELOX_FAIL("Bad call to yyFlexLexer::yyrestart()");
}
void yyFlexLexer::yyrestart(std::istream& s) {
  VELOX_FAIL("Bad call to yyFlexLexer::yyrestart()");
}
void yyFlexLexer::yypush_buffer_state(yy_buffer_state* new_buffer) {
  VELOX_FAIL("Bad call to yyFlexLexer::yypush_buffer_state()");
}
void yyFlexLexer::yypop_buffer_state() {
  VELOX_FAIL("Bad call to yyFlexLexer::yypop_buffer_state()");
}
void yyFlexLexer::switch_streams(std::istream& new_in, std::ostream& new_out) {
  VELOX_FAIL("Bad call to yyFlexLexer::switch_streams()");
}
void yyFlexLexer::switch_streams(std::istream* new_in, std::ostream* new_out) {
  VELOX_FAIL("Bad call to yyFlexLexer::switch_streams()");
}
int yyFlexLexer::yywrap() {
  VELOX_FAIL("Bad call to yyFlexLexer::yywrap()");
}
int yyFlexLexer::LexerInput(char* buf, int max_size) {
  VELOX_FAIL("Bad call to yyFlexLexer::LexerInput()");
}
void yyFlexLexer::LexerOutput(const char* buf, int size) {
  VELOX_FAIL("Bad call to yyFlexLexer::LexerOutput()");
}
void yyFlexLexer::LexerError(const char* msg) {
  VELOX_FAIL("Bad call to yyFlexLexer::LexerError()");
}
