#!/bin/bash

export PACKENVPATH=$(which python)
PACKENVPATH=$(echo "$PACKENVPATH" | rev | cut -d'/' -f3- | rev)
swig -c++ -python -o Stab_jac_wrap.cpp Stab_jac.i
g++ -c Stab_jac.cpp Stab_jac_wrap.cpp -I$HOME/eigen -I$PACKENVPATH/lib/python3.8/site-packages/numpy/core/include -I$PACKENVPATH/include/python3.8 -fPIC -std=c++14 -O2
g++ -shared Stab_jac.o Stab_jac_wrap.o -o _Stab_jac.so -fPIC
python -c "import _Stab_jac"

