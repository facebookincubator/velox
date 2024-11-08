# - Find JNI java libraries.
# This module finds if Java is installed and determines where the
# include files and libraries are. It also determines what the name of
# the library is. This code sets the following variables:
#
#  JNI_INCLUDE_DIRS      = the include dirs to use
#  JNI_LIBRARIES         = the libraries to use
#  JAVA_AWT_LIBRARY      = the path to the jawt library
#  JAVA_JVM_LIBRARY      = the path to the jvm library
#  JAVA_INCLUDE_PATH     = the include path to jni.h
#  JAVA_INCLUDE_PATH2    = the include path to jni_md.h
#  JAVA_AWT_INCLUDE_PATH = the include path to jawt.h
#

#=============================================================================
# Copyright 2001-2009 Kitware, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the names of Kitware, Inc., the Insight Software Consortium,
#   nor the names of their contributors may be used to endorse or promote
#   products derived from this software without specific prior written
#   permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

# Expand {libarch} occurences to java_libarch subdirectory(-ies) and set ${_var}
macro(JAVA_APPEND_LIBRARY_DIRECTORIES _var)
    # Determine java arch-specific library subdir
    if (CMAKE_SYSTEM_NAME MATCHES "Linux")
        # Based on openjdk/jdk/make/common/shared/Platform.gmk as of 6b16
        # and kaffe as of 1.1.8 which uses the first part of the
        # GNU config.guess platform triplet.
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "^i[3-9]86$")
            set(_java_libarch "i386")
        elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
            set(_java_libarch "amd64" "x86_64")
        elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc")
            set(_java_libarch "ppc" "powerpc" "ppc64")
        elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^sparc")
            set(_java_libarch "sparc" "sparcv9")
        else (CMAKE_SYSTEM_PROCESSOR MATCHES "^i[3-9]86$")
            set(_java_libarch "${CMAKE_SYSTEM_PROCESSOR}")
        endif(CMAKE_SYSTEM_PROCESSOR MATCHES "^i[3-9]86$")
    else (CMAKE_SYSTEM_NAME MATCHES "Linux")
        set(_java_libarch "i386" "amd64" "ppc") # previous default
    endif (CMAKE_SYSTEM_NAME MATCHES "Linux")

    foreach(_path ${ARGN})
        if (_path MATCHES "{libarch}")
            foreach(_libarch ${_java_libarch})
                string(REPLACE "{libarch}" "${_libarch}" _newpath "${_path}")
                list(APPEND ${_var} "${_newpath}")
            endforeach(_libarch)
        else (_path MATCHES "{libarch}")
            list(APPEND ${_var} "${_path}")
        endif (_path MATCHES "{libarch}")
    endforeach(_path)
endmacro(JAVA_APPEND_LIBRARY_DIRECTORIES)

file(TO_CMAKE_PATH "$ENV{JAVA_HOME}" _JAVA_HOME)

get_filename_component(java_install_version
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\JavaSoft\\Java Development Kit;CurrentVersion]" NAME
        )

set(JAVA_AWT_LIBRARY_DIRECTORIES
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\JavaSoft\\Java Development Kit\\1.4;JavaHome]/lib"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\JavaSoft\\Java Development Kit\\1.3;JavaHome]/lib"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\JavaSoft\\Java Development Kit\\${java_install_version};JavaHome]/lib"
        )


java_append_library_directories(JAVA_AWT_LIBRARY_DIRECTORIES
        ${_JAVA_HOME}/jre/lib/{libarch}
        ${_JAVA_HOME}/jre/lib
        ${_JAVA_HOME}/lib/{libarch}
        ${_JAVA_HOME}/lib
        ${_JAVA_HOME}
        /usr/lib
        /usr/local/lib
        /usr/lib/jvm/java/lib
        /usr/lib/java/jre/lib/{libarch}
        /usr/local/lib/java/jre/lib/{libarch}
        /usr/local/share/java/jre/lib/{libarch}
        /usr/lib/j2sdk1.4-sun/jre/lib/{libarch}
        /usr/lib/j2sdk1.5-sun/jre/lib/{libarch}
        /opt/sun-jdk-1.5.0.04/jre/lib/{libarch}
        /usr/lib/jvm/java-6-sun/jre/lib/{libarch}
        /usr/lib/jvm/java-1.5.0-sun/jre/lib/{libarch}
        /usr/lib/jvm/java-6-sun-1.6.0.00/jre/lib/{libarch}       # can this one be removed according to #8821 ? Alex
        /usr/lib/jvm/java-openjdk/jre/lib/{libarch}
        /usr/lib/jvm/java-6-openjdk/jre/lib/{libarch}
        /usr/lib/jvm/java-openjdk/jre/lib/{libarch}
        # Debian specific paths for default JVM
        /usr/lib/jvm/default-java/jre/lib/{libarch}
        /usr/lib/jvm/default-java/jre/lib
        /usr/lib/jvm/default-java/lib
        )

set(JAVA_JVM_LIBRARY_DIRECTORIES)
foreach (dir ${JAVA_AWT_LIBRARY_DIRECTORIES})
    set(JAVA_JVM_LIBRARY_DIRECTORIES
            ${JAVA_JVM_LIBRARY_DIRECTORIES}
            "${dir}"
            "${dir}/client"
            "${dir}/server"
            )
endforeach (dir)


set(JAVA_AWT_INCLUDE_DIRECTORIES
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\JavaSoft\\Java Development Kit\\1.4;JavaHome]/include"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\JavaSoft\\Java Development Kit\\1.3;JavaHome]/include"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\JavaSoft\\Java Development Kit\\${java_install_version};JavaHome]/include"
        ${_JAVA_HOME}/include
        /usr/include
        /usr/local/include
        /usr/lib/java/include
        /usr/local/lib/java/include
        /usr/lib/jvm/java/include
        /usr/lib/jvm/java-6-sun/include
        /usr/lib/jvm/java-1.5.0-sun/include
        /usr/lib/jvm/java-6-sun-1.6.0.00/include       # can this one be removed according to #8821 ? Alex
        /usr/lib/jvm/java-6-openjdk/include
        /usr/local/share/java/include
        /usr/lib/j2sdk1.4-sun/include
        /usr/lib/j2sdk1.5-sun/include
        /opt/sun-jdk-1.5.0.04/include
        # Debian specific path for default JVM
        /usr/lib/jvm/default-java/include
        )

foreach(JAVA_PROG "${JAVA_RUNTIME}" "${JAVA_COMPILE}" "${JAVA_ARCHIVE}" "${JAVA_HEADER}")
    get_filename_component(jpath "${JAVA_PROG}" PATH)
    foreach (JAVA_INC_PATH ../include ../java/include ../share/java/include)
        if (EXISTS ${jpath}/${JAVA_INC_PATH})
            set(JAVA_AWT_INCLUDE_DIRECTORIES
                    ${JAVA_AWT_INCLUDE_DIRECTORIES}
                    "${jpath}/${JAVA_INC_PATH}"
                    )
        endif (EXISTS ${jpath}/${JAVA_INC_PATH})
    endforeach (JAVA_INC_PATH)

    foreach (JAVA_LIB_PATH
            ../lib ../jre/lib ../jre/lib/i386
            ../java/lib ../java/jre/lib ../java/jre/lib/i386
            ../share/java/lib ../share/java/jre/lib ../share/java/jre/lib/i386)
        if (EXISTS ${jpath}/${JAVA_LIB_PATH})
            set(JAVA_AWT_LIBRARY_DIRECTORIES
                    ${JAVA_AWT_LIBRARY_DIRECTORIES}
                    "${jpath}/${JAVA_LIB_PATH}"
                    )
        endif (EXISTS ${jpath}/${JAVA_LIB_PATH})
    endforeach (JAVA_LIB_PATH)
endforeach (JAVA_PROG)

if (APPLE)
    if (EXISTS ~/Library/Frameworks/JavaVM.framework)
        set(JAVA_HAVE_FRAMEWORK 1)
    endif (EXISTS ~/Library/Frameworks/JavaVM.framework)

    if (EXISTS /Library/Frameworks/JavaVM.framework)
        SET(JAVA_HAVE_FRAMEWORK 1)
    endif (EXISTS /Library/Frameworks/JavaVM.framework)

    if (EXISTS /System/Library/Frameworks/JavaVM.framework)
        set(JAVA_HAVE_FRAMEWORK 1)
    endif(EXISTS /System/Library/Frameworks/JavaVM.framework)

    if (JAVA_HAVE_FRAMEWORK)
        if(NOT JAVA_AWT_LIBRARY)
            set(JAVA_AWT_LIBRARY "-framework JavaVM" CACHE FILEPATH "Java Frameworks" FORCE)
        endif(NOT JAVA_AWT_LIBRARY)

        if (NOT JAVA_JVM_LIBRARY)
            set (JAVA_JVM_LIBRARY "-framework JavaVM" CACHE FILEPATH "Java Frameworks" FORCE)
        endif (NOT JAVA_JVM_LIBRARY)

        if (NOT JAVA_AWT_INCLUDE_PATH)
            if (EXISTS /System/Library/Frameworks/JavaVM.framework/Headers/jawt.h)
                set(JAVA_AWT_INCLUDE_PATH "/System/Library/Frameworks/JavaVM.framework/Headers" CACHE FILEPATH "jawt.h location" FORCE)
            endif(EXISTS /System/Library/Frameworks/JavaVM.framework/Headers/jawt.h)
        endif (NOT JAVA_AWT_INCLUDE_PATH)

        #
        # If using "-framework JavaVM", prefer its headers *before* the others in
        # JAVA_AWT_INCLUDE_DIRECTORIES... (*prepend* to the list here)
        #
        set(JAVA_AWT_INCLUDE_DIRECTORIES
                ~/Library/Frameworks/JavaVM.framework/Headers
                /Library/Frameworks/JavaVM.framework/Headers
                /System/Library/Frameworks/JavaVM.framework/Headers
                ${JAVA_AWT_INCLUDE_DIRECTORIES}
                )
    endif(JAVA_HAVE_FRAMEWORK)
else (APPLE)
    find_library(JAVA_AWT_LIBRARY
            NAMES
            jawt
            PATHS
            ${JAVA_AWT_LIBRARY_DIRECTORIES}
            )

    find_library(JAVA_JSIG_LIBRARY
            NAMES
            jsig
            PATHS
            ${JAVA_JVM_LIBRARY_DIRECTORIES}
            )
    find_library(JAVA_JVM_LIBRARY
            NAMES
            jvm
            JavaVM
            PATHS
            ${JAVA_JVM_LIBRARY_DIRECTORIES}
            )
endif (APPLE)

find_library(JAVA_AWT_LIBRARY
        NAMES
        jawt
        PATHS
        ${JAVA_AWT_LIBRARY_DIRECTORIES}
        )

find_library(JAVA_JSIG_LIBRARY
        NAMES
        jsig
        PATHS
        ${JAVA_JVM_LIBRARY_DIRECTORIES}
        )
find_library(JAVA_JVM_LIBRARY
        NAMES
        jvm
        JavaVM
        PATHS
        ${JAVA_JVM_LIBRARY_DIRECTORIES}
        )
# add in the include path
find_path(JAVA_INCLUDE_PATH
        NAMES
        jni.h
        PATHS
        ${JAVA_AWT_INCLUDE_DIRECTORIES}
        )

find_path(JAVA_INCLUDE_PATH2
        NAMES
        jni_md.h
        PATHS
        ${JAVA_INCLUDE_PATH}
        ${JAVA_INCLUDE_PATH}/win32
        ${JAVA_INCLUDE_PATH}/linux
        ${JAVA_INCLUDE_PATH}/freebsd
        ${JAVA_INCLUDE_PATH}/solaris
        ${JAVA_INCLUDE_PATH}/darwin
        )

find_path(JAVA_AWT_INCLUDE_PATH
        NAMES
        jawt.h
        PATHS
        ${JAVA_INCLUDE_PATH}
        )

set(JNI_LIBRARIES
        ${JAVA_AWT_LIBRARY}
        ${JAVA_JSIG_LIBRARY}
        ${JAVA_JVM_LIBRARY}
        )

set(JNI_INCLUDE_DIRS
        ${JAVA_INCLUDE_PATH}
        ${JAVA_INCLUDE_PATH2}
        ${JAVA_AWT_INCLUDE_PATH}
        )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JNI DEFAULT_MSG JNI_LIBRARIES JNI_INCLUDE_DIRS JAVA_AWT_LIBRARY JAVA_JSIG_LIBRARY JAVA_JVM_LIBRARY)

mark_as_advanced(JNI_LIBRARIES JNI_INCLUDE_DIRS JAVA_AWT_LIBRARY JAVA_JSIG_LIBRARY JAVA_JVM_LIBRARY)
