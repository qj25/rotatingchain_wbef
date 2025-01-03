%module Stab_jac
%{
#define SWIG_FILE_WITH_INIT
// #include <iostream>
#include "Stab_jac.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (int DIM1, double* IN_ARRAY1) {
    (int dim_stabjac, double* out_stabjac),
    (int dim_y, double* in_y)
};
// %apply (int DIM1, double* ARGOUT_ARRAY1) {(int dim_nf, double* node_force)};


%include "Stab_jac.h"
// %include "Der_obj.h"
// %include "Der_utils.h"


// class DER_iso2
// {
// public:
//     DER_iso2(
//         int dim_np,
//         double *node_pos,
//         int dim_bf0,
//         double *bf0sim,
//         const double theta_n,
//         const double overall_rot
//     );

//     ~DER_iso2();

//     void updateVars(
//         int dim_np,
//         double *node_pos,
//         int dim_bf0,
//         double *bf0sim
//     );  //

//     void calculateCenterlineF2(int dim_nf, double *node_force);
// };

// %include "Der_obj.h"
// %include "Der_utils.h"


// #include <Eigen/Dense>
// #include <Eigen/Core>

// extern DER_iso2(
//     int dim_np,
//     double *node_pos,
//     int dim_bf0,
//     double *bf0sim,
//     const double theta_n,
//     const double overall_rot
// );

// extern DER_iso2::updateVars(
//         int dim_np,
//         double *node_pos,
//         int dim_bf0,
//         double *bf0sim
//     );  //

// extern DER_iso2::calculateCenterlineF2(int dim_nf, double *node_force);

/*
    python3-config --cflags
    swig -c++ -python -o Der_iso2_wrap.cpp Der_iso2.i

    g++ -c -fpic Der_iso2.cpp Der_utils.cpp -std=c++14
    g++ -c -fpic Der_iso2_wrap.cpp -I/home/qj/anaconda3/envs/rlenv/include/python3.6m -std=c++14
    g++ -shared Der_iso2.o Der_utils.o Der_iso2_wrap.o _Der_iso2.so

    g++ -c Der_iso2.cpp Der_iso2_wrap.cpp -I/home/qj/anaconda3/envs/rlenv/include/python3.6m -fPIC -std=c++14
    ld -shared Der_iso2.o Der_iso2_wrap.o -o _Der_iso2.so -fPIC

    g++ -Wl,--gc-sections -fPIC -shared -lstdc++ Der_iso2.o Der_iso2_wrap.o -o _Der_iso2.so
*/