#!/bin/bash

export PACKENVPATH=$(which python)
PACKENVPATH=$(echo "$PACKENVPATH" | rev | cut -d'/' -f3- | rev)
swig -c++ -python -o Shootc_wrap.cpp Shootc.i
g++ -c Shootc.cpp Shootc_wrap.cpp -I$HOME/eigen -I$PACKENVPATH/lib/python3.8/site-packages/numpy/core/include -I$PACKENVPATH/include/python3.8 -fPIC -std=c++14 -O2
g++ -shared Shootc.o Shootc_wrap.o -o _Shootc.so -fPIC
python -c "import _Shootc"

