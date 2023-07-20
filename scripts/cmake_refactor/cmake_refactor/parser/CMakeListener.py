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


    # Enter a parse tree produced by CMakeParser#add_library.
    def enterAdd_library(self, ctx:CMakeParser.Add_libraryContext):
        pass

    # Exit a parse tree produced by CMakeParser#add_library.
    def exitAdd_library(self, ctx:CMakeParser.Add_libraryContext):
        pass


    # Enter a parse tree produced by CMakeParser#add_alias.
    def enterAdd_alias(self, ctx:CMakeParser.Add_aliasContext):
        pass

    # Exit a parse tree produced by CMakeParser#add_alias.
    def exitAdd_alias(self, ctx:CMakeParser.Add_aliasContext):
        pass


    # Enter a parse tree produced by CMakeParser#add_interface.
    def enterAdd_interface(self, ctx:CMakeParser.Add_interfaceContext):
        pass

    # Exit a parse tree produced by CMakeParser#add_interface.
    def exitAdd_interface(self, ctx:CMakeParser.Add_interfaceContext):
        pass


    # Enter a parse tree produced by CMakeParser#link_libraries.
    def enterLink_libraries(self, ctx:CMakeParser.Link_librariesContext):
        pass

    # Exit a parse tree produced by CMakeParser#link_libraries.
    def exitLink_libraries(self, ctx:CMakeParser.Link_librariesContext):
        pass


    # Enter a parse tree produced by CMakeParser#include_directories.
    def enterInclude_directories(self, ctx:CMakeParser.Include_directoriesContext):
        pass

    # Exit a parse tree produced by CMakeParser#include_directories.
    def exitInclude_directories(self, ctx:CMakeParser.Include_directoriesContext):
        pass


    # Enter a parse tree produced by CMakeParser#link_targets.
    def enterLink_targets(self, ctx:CMakeParser.Link_targetsContext):
        pass

    # Exit a parse tree produced by CMakeParser#link_targets.
    def exitLink_targets(self, ctx:CMakeParser.Link_targetsContext):
        pass


    # Enter a parse tree produced by CMakeParser#additonal_targets.
    def enterAdditonal_targets(self, ctx:CMakeParser.Additonal_targetsContext):
        pass

    # Exit a parse tree produced by CMakeParser#additonal_targets.
    def exitAdditonal_targets(self, ctx:CMakeParser.Additonal_targetsContext):
        pass


    # Enter a parse tree produced by CMakeParser#source_file.
    def enterSource_file(self, ctx:CMakeParser.Source_fileContext):
        pass

    # Exit a parse tree produced by CMakeParser#source_file.
    def exitSource_file(self, ctx:CMakeParser.Source_fileContext):
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


    # Enter a parse tree produced by CMakeParser#library_type.
    def enterLibrary_type(self, ctx:CMakeParser.Library_typeContext):
        pass

    # Exit a parse tree produced by CMakeParser#library_type.
    def exitLibrary_type(self, ctx:CMakeParser.Library_typeContext):
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