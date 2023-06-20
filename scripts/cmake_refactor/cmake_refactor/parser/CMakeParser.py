# Generated from CMake.g4 by ANTLR 4.13.0
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,17,85,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,1,0,1,0,5,0,25,8,0,10,0,12,0,
        28,9,0,1,0,1,0,1,1,1,1,1,1,1,1,3,1,36,8,1,1,1,5,1,39,8,1,10,1,12,
        1,42,9,1,1,1,1,1,1,2,1,2,1,2,1,2,5,2,50,8,2,10,2,12,2,53,9,2,1,2,
        1,2,1,3,1,3,1,4,1,4,1,4,5,4,62,8,4,10,4,12,4,65,9,4,1,4,1,4,1,5,
        1,5,1,6,1,6,1,6,1,6,3,6,75,8,6,1,7,1,7,1,8,1,8,1,9,1,9,1,10,1,10,
        1,10,0,0,11,0,2,4,6,8,10,12,14,16,18,20,0,2,2,0,9,10,12,13,1,0,8,
        9,84,0,26,1,0,0,0,2,31,1,0,0,0,4,45,1,0,0,0,6,56,1,0,0,0,8,58,1,
        0,0,0,10,68,1,0,0,0,12,74,1,0,0,0,14,76,1,0,0,0,16,78,1,0,0,0,18,
        80,1,0,0,0,20,82,1,0,0,0,22,25,3,2,1,0,23,25,3,4,2,0,24,22,1,0,0,
        0,24,23,1,0,0,0,25,28,1,0,0,0,26,24,1,0,0,0,26,27,1,0,0,0,27,29,
        1,0,0,0,28,26,1,0,0,0,29,30,5,0,0,1,30,1,1,0,0,0,31,32,5,7,0,0,32,
        33,5,1,0,0,33,35,3,10,5,0,34,36,3,12,6,0,35,34,1,0,0,0,35,36,1,0,
        0,0,36,40,1,0,0,0,37,39,3,6,3,0,38,37,1,0,0,0,39,42,1,0,0,0,40,38,
        1,0,0,0,40,41,1,0,0,0,41,43,1,0,0,0,42,40,1,0,0,0,43,44,5,2,0,0,
        44,3,1,0,0,0,45,46,5,9,0,0,46,51,5,1,0,0,47,50,3,6,3,0,48,50,3,8,
        4,0,49,47,1,0,0,0,49,48,1,0,0,0,50,53,1,0,0,0,51,49,1,0,0,0,51,52,
        1,0,0,0,52,54,1,0,0,0,53,51,1,0,0,0,54,55,5,2,0,0,55,5,1,0,0,0,56,
        57,7,0,0,0,57,7,1,0,0,0,58,63,5,1,0,0,59,62,3,6,3,0,60,62,3,8,4,
        0,61,59,1,0,0,0,61,60,1,0,0,0,62,65,1,0,0,0,63,61,1,0,0,0,63,64,
        1,0,0,0,64,66,1,0,0,0,65,63,1,0,0,0,66,67,5,2,0,0,67,9,1,0,0,0,68,
        69,7,1,0,0,69,11,1,0,0,0,70,75,3,14,7,0,71,75,3,16,8,0,72,75,3,18,
        9,0,73,75,3,20,10,0,74,70,1,0,0,0,74,71,1,0,0,0,74,72,1,0,0,0,74,
        73,1,0,0,0,75,13,1,0,0,0,76,77,5,3,0,0,77,15,1,0,0,0,78,79,5,4,0,
        0,79,17,1,0,0,0,80,81,5,5,0,0,81,19,1,0,0,0,82,83,5,6,0,0,83,21,
        1,0,0,0,9,24,26,35,40,49,51,61,63,74
    ]

class CMakeParser ( Parser ):

    grammarFileName = "CMake.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'('", "')'", "'PUBLIC'", "'PRIVATE'", 
                     "'INTERFACE'", "'ALIAS'", "'add_library'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "Target_command", 
                      "Identifier_namespace", "Identifier", "Unquoted_argument", 
                      "Escape_sequence", "Quoted_argument", "Bracket_argument", 
                      "Bracket_comment", "Line_comment", "Newline", "Space" ]

    RULE_file_ = 0
    RULE_target_command = 1
    RULE_command = 2
    RULE_single_argument = 3
    RULE_compound_argument = 4
    RULE_target = 5
    RULE_keyword = 6
    RULE_public = 7
    RULE_private = 8
    RULE_interface = 9
    RULE_alias = 10

    ruleNames =  [ "file_", "target_command", "command", "single_argument", 
                   "compound_argument", "target", "keyword", "public", "private", 
                   "interface", "alias" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    Target_command=7
    Identifier_namespace=8
    Identifier=9
    Unquoted_argument=10
    Escape_sequence=11
    Quoted_argument=12
    Bracket_argument=13
    Bracket_comment=14
    Line_comment=15
    Newline=16
    Space=17

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.0")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class File_Context(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOF(self):
            return self.getToken(CMakeParser.EOF, 0)

        def target_command(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(CMakeParser.Target_commandContext)
            else:
                return self.getTypedRuleContext(CMakeParser.Target_commandContext,i)


        def command(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(CMakeParser.CommandContext)
            else:
                return self.getTypedRuleContext(CMakeParser.CommandContext,i)


        def getRuleIndex(self):
            return CMakeParser.RULE_file_

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFile_" ):
                listener.enterFile_(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFile_" ):
                listener.exitFile_(self)




    def file_(self):

        localctx = CMakeParser.File_Context(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_file_)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 26
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==7 or _la==9:
                self.state = 24
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [7]:
                    self.state = 22
                    self.target_command()
                    pass
                elif token in [9]:
                    self.state = 23
                    self.command()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 28
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 29
            self.match(CMakeParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Target_commandContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Target_command(self):
            return self.getToken(CMakeParser.Target_command, 0)

        def target(self):
            return self.getTypedRuleContext(CMakeParser.TargetContext,0)


        def keyword(self):
            return self.getTypedRuleContext(CMakeParser.KeywordContext,0)


        def single_argument(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(CMakeParser.Single_argumentContext)
            else:
                return self.getTypedRuleContext(CMakeParser.Single_argumentContext,i)


        def getRuleIndex(self):
            return CMakeParser.RULE_target_command

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTarget_command" ):
                listener.enterTarget_command(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTarget_command" ):
                listener.exitTarget_command(self)




    def target_command(self):

        localctx = CMakeParser.Target_commandContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_target_command)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 31
            self.match(CMakeParser.Target_command)
            self.state = 32
            self.match(CMakeParser.T__0)
            self.state = 33
            self.target()
            self.state = 35
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & 120) != 0):
                self.state = 34
                self.keyword()


            self.state = 40
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while (((_la) & ~0x3f) == 0 and ((1 << _la) & 13824) != 0):
                self.state = 37
                self.single_argument()
                self.state = 42
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 43
            self.match(CMakeParser.T__1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CommandContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Identifier(self):
            return self.getToken(CMakeParser.Identifier, 0)

        def single_argument(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(CMakeParser.Single_argumentContext)
            else:
                return self.getTypedRuleContext(CMakeParser.Single_argumentContext,i)


        def compound_argument(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(CMakeParser.Compound_argumentContext)
            else:
                return self.getTypedRuleContext(CMakeParser.Compound_argumentContext,i)


        def getRuleIndex(self):
            return CMakeParser.RULE_command

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCommand" ):
                listener.enterCommand(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCommand" ):
                listener.exitCommand(self)




    def command(self):

        localctx = CMakeParser.CommandContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_command)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 45
            self.match(CMakeParser.Identifier)
            self.state = 46
            self.match(CMakeParser.T__0)
            self.state = 51
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while (((_la) & ~0x3f) == 0 and ((1 << _la) & 13826) != 0):
                self.state = 49
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [9, 10, 12, 13]:
                    self.state = 47
                    self.single_argument()
                    pass
                elif token in [1]:
                    self.state = 48
                    self.compound_argument()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 53
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 54
            self.match(CMakeParser.T__1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Single_argumentContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Identifier(self):
            return self.getToken(CMakeParser.Identifier, 0)

        def Unquoted_argument(self):
            return self.getToken(CMakeParser.Unquoted_argument, 0)

        def Bracket_argument(self):
            return self.getToken(CMakeParser.Bracket_argument, 0)

        def Quoted_argument(self):
            return self.getToken(CMakeParser.Quoted_argument, 0)

        def getRuleIndex(self):
            return CMakeParser.RULE_single_argument

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSingle_argument" ):
                listener.enterSingle_argument(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSingle_argument" ):
                listener.exitSingle_argument(self)




    def single_argument(self):

        localctx = CMakeParser.Single_argumentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_single_argument)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 56
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 13824) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Compound_argumentContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def single_argument(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(CMakeParser.Single_argumentContext)
            else:
                return self.getTypedRuleContext(CMakeParser.Single_argumentContext,i)


        def compound_argument(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(CMakeParser.Compound_argumentContext)
            else:
                return self.getTypedRuleContext(CMakeParser.Compound_argumentContext,i)


        def getRuleIndex(self):
            return CMakeParser.RULE_compound_argument

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCompound_argument" ):
                listener.enterCompound_argument(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCompound_argument" ):
                listener.exitCompound_argument(self)




    def compound_argument(self):

        localctx = CMakeParser.Compound_argumentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_compound_argument)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 58
            self.match(CMakeParser.T__0)
            self.state = 63
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while (((_la) & ~0x3f) == 0 and ((1 << _la) & 13826) != 0):
                self.state = 61
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [9, 10, 12, 13]:
                    self.state = 59
                    self.single_argument()
                    pass
                elif token in [1]:
                    self.state = 60
                    self.compound_argument()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 65
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 66
            self.match(CMakeParser.T__1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TargetContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Identifier(self):
            return self.getToken(CMakeParser.Identifier, 0)

        def Identifier_namespace(self):
            return self.getToken(CMakeParser.Identifier_namespace, 0)

        def getRuleIndex(self):
            return CMakeParser.RULE_target

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTarget" ):
                listener.enterTarget(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTarget" ):
                listener.exitTarget(self)




    def target(self):

        localctx = CMakeParser.TargetContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_target)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 68
            _la = self._input.LA(1)
            if not(_la==8 or _la==9):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class KeywordContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def public(self):
            return self.getTypedRuleContext(CMakeParser.PublicContext,0)


        def private(self):
            return self.getTypedRuleContext(CMakeParser.PrivateContext,0)


        def interface(self):
            return self.getTypedRuleContext(CMakeParser.InterfaceContext,0)


        def alias(self):
            return self.getTypedRuleContext(CMakeParser.AliasContext,0)


        def getRuleIndex(self):
            return CMakeParser.RULE_keyword

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterKeyword" ):
                listener.enterKeyword(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitKeyword" ):
                listener.exitKeyword(self)




    def keyword(self):

        localctx = CMakeParser.KeywordContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_keyword)
        try:
            self.state = 74
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [3]:
                self.enterOuterAlt(localctx, 1)
                self.state = 70
                self.public()
                pass
            elif token in [4]:
                self.enterOuterAlt(localctx, 2)
                self.state = 71
                self.private()
                pass
            elif token in [5]:
                self.enterOuterAlt(localctx, 3)
                self.state = 72
                self.interface()
                pass
            elif token in [6]:
                self.enterOuterAlt(localctx, 4)
                self.state = 73
                self.alias()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PublicContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return CMakeParser.RULE_public

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPublic" ):
                listener.enterPublic(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPublic" ):
                listener.exitPublic(self)




    def public(self):

        localctx = CMakeParser.PublicContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_public)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 76
            self.match(CMakeParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PrivateContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return CMakeParser.RULE_private

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPrivate" ):
                listener.enterPrivate(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPrivate" ):
                listener.exitPrivate(self)




    def private(self):

        localctx = CMakeParser.PrivateContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_private)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 78
            self.match(CMakeParser.T__3)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class InterfaceContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return CMakeParser.RULE_interface

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInterface" ):
                listener.enterInterface(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInterface" ):
                listener.exitInterface(self)




    def interface(self):

        localctx = CMakeParser.InterfaceContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_interface)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 80
            self.match(CMakeParser.T__4)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AliasContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return CMakeParser.RULE_alias

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAlias" ):
                listener.enterAlias(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAlias" ):
                listener.exitAlias(self)




    def alias(self):

        localctx = CMakeParser.AliasContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_alias)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 82
            self.match(CMakeParser.T__5)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





