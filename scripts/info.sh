set -e

version='0.0.1'
info=$(mktemp)
cmake --system-information >> $info 

e() { echo "$*"
}

ext() {
  grep -oP '\".+\"$' $1 | tr -d '\"' 
}

e '```'
e "Velox System Info v${version}"
e Commit: $(git rev-parse HEAD 2> /dev/null || echo "Not in a git repo.")
e CMake Version: $(cmake --version | grep -oP '\d+\.\d+\.\d+')
e System: $(grep -aP 'CMAKE_SYSTEM \"' $info | ext )
e Arch: $(grep -aP 'CMAKE_SYSTEM_PROCESSOR' $info | ext)
e C++ Compiler: $(grep -aP 'CMAKE_CXX_COMPILER ==' $info | ext)
e C++ Compiler Version: $(grep -aP 'CMAKE_CXX_COMPILER_VERSION' $info | ext)
e C Compiler: $(grep -aP 'CMAKE_C_COMPILER ==' $info | ext)
e C Compiler Version: $(grep -aP 'CMAKE_C_COMPILER_VERSION' $info | ext)
e CMake Prefix Path: $(grep -aP '_PREFIX_PATH ' $info | ext)
e 
echo '```'

rm $info
