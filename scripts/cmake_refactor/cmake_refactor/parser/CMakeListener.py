# Generated from CMake.g4 by ANTLR 4.13.0
from antlr4 import *
if "." in __name__:
    from .CMakeParser import CMakeParser
else:
    from CMakeParser import CMakeParser

# This class defines a complete listener for a parse tree produced by CMakeParser.
class CMakeListener(ParseTreeListener):

    # Enter a parse tree produced by CMakeParser#file_.
    def enterFile_(self, ctx:CMakeParser.File_Context):
        pass

    # Exit a parse tree produced by CMakeParser#file_.
    def exitFile_(self, ctx:CMakeParser.File_Context):
        pass


    # Enter a parse tree produced by CMakeParser#target_command.
    def enterTarget_command(self, ctx:CMakeParser.Target_commandContext):
        pass

    # Exit a parse tree produced by CMakeParser#target_command.
    def exitTarget_command(self, ctx:CMakeParser.Target_commandContext):
        pass


    # Enter a parse tree produced by CMakeParser#command.
    def enterCommand(self, ctx:CMakeParser.CommandContext):
        pass

    # Exit a parse tree produced by CMakeParser#command.
    def exitCommand(self, ctx:CMakeParser.CommandContext):
        pass


    # Enter a parse tree produced by CMakeParser#single_argument.
    def enterSingle_argument(self, ctx:CMakeParser.Single_argumentContext):
        pass

    # Exit a parse tree produced by CMakeParser#single_argument.
    def exitSingle_argument(self, ctx:CMakeParser.Single_argumentContext):
        pass


    # Enter a parse tree produced by CMakeParser#compound_argument.
    def enterCompound_argument(self, ctx:CMakeParser.Compound_argumentContext):
        pass

    # Exit a parse tree produced by CMakeParser#compound_argument.
    def exitCompound_argument(self, ctx:CMakeParser.Compound_argumentContext):
        pass


    # Enter a parse tree produced by CMakeParser#target.
    def enterTarget(self, ctx:CMakeParser.TargetContext):
        pass

    # Exit a parse tree produced by CMakeParser#target.
    def exitTarget(self, ctx:CMakeParser.TargetContext):
        pass


    # Enter a parse tree produced by CMakeParser#keyword.
    def enterKeyword(self, ctx:CMakeParser.KeywordContext):
        pass

    # Exit a parse tree produced by CMakeParser#keyword.
    def exitKeyword(self, ctx:CMakeParser.KeywordContext):
        pass


    # Enter a parse tree produced by CMakeParser#public.
    def enterPublic(self, ctx:CMakeParser.PublicContext):
        pass

    # Exit a parse tree produced by CMakeParser#public.
    def exitPublic(self, ctx:CMakeParser.PublicContext):
        pass


    # Enter a parse tree produced by CMakeParser#private.
    def enterPrivate(self, ctx:CMakeParser.PrivateContext):
        pass

    # Exit a parse tree produced by CMakeParser#private.
    def exitPrivate(self, ctx:CMakeParser.PrivateContext):
        pass


    # Enter a parse tree produced by CMakeParser#interface.
    def enterInterface(self, ctx:CMakeParser.InterfaceContext):
        pass

    # Exit a parse tree produced by CMakeParser#interface.
    def exitInterface(self, ctx:CMakeParser.InterfaceContext):
        pass


    # Enter a parse tree produced by CMakeParser#alias.
    def enterAlias(self, ctx:CMakeParser.AliasContext):
        pass

    # Exit a parse tree produced by CMakeParser#alias.
    def exitAlias(self, ctx:CMakeParser.AliasContext):
        pass



del CMakeParser