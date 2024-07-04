#ifndef SHOOTC_H
#define SHOOTC_H

#include "Shootc.h"
#include <vector>
#include "Eigen/Core"
#include "Eigen/Geometry"
// #include <thread>

/*
DER_iso2:
1. calculates forces on each node in an arrangement of equality constrained 
discrete cylinders.
2. refer to paper: Simulation and Manipulation of a Deformable Linear Object
*/

class Shootc
{
public:
    Shootc(
        const double in_v_rot,
        const double in_l,
        const double in_mu,
        const int in_n_steps,
        const double in_g,
        const double in_k,
        const double in_srl,
        bool in_error_verbose
    );

    bool error_verbose;

    double split_ratio_latter;

    double g;
    double k;
    double v_rot;
    double rbar;

    double l;
    double lbar;
    double mu;
    double m;

    int n_steps;
    std::vector<double> sbar;
    std::vector<double> sbar_actual;
    double sbar_step;

    double rho_1_0;
    double grad_ratio;
    double u_1_0;

    double fbar;
    std::vector<double> rho_1;
    std::vector<double> u_1;
    std::vector<double> u;
    double u_1_desired;

    double get_info(
        int dim_u1,
        double *out_u_1,
        int dim_sbar,
        double *out_sbar,
        int dim_rho1,
        double *out_rho_1
    );
    double bang(double fbg, double in_rho_1_0);

private:
    void set_fbar(double fbg);
    double func_fixed2(double v_u, double v_sbar);
    double calc_u_0();
    double get_f();
};

#endif
