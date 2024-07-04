#include "Shootc.h"
#include <iostream>
#include <cmath>
// #include <iostream>
// #include <cmath>
// #include <vector>
// #include <chrono>
// #include <thread>
// #include <mutex>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"

/*
To-do:
- Make getthetan public

A_cpp(node_pos, bf0sim) --> (bfend)
A_py(bfend) --> (theta_n)
B_cpp(theta_n) --> (force_nodes)
*/

// std::mutex mtx;

Shootc::Shootc(
    const double in_v_rot,
    const double in_l,
    const double in_mu,
    const int in_n_steps,
    const double in_g,
    const double in_k,
    const double in_srl,
    bool in_error_verbose
)
{
    error_verbose = in_error_verbose;

    split_ratio_latter = in_srl;

    g = in_g;
    k = in_k;

    v_rot = in_v_rot;

    n_steps = in_n_steps;
    l = in_l;
    lbar = l*v_rot*v_rot/g;
    mu = in_mu;
    m = mu * l / n_steps;


    sbar_step = lbar/(n_steps-1);
    for (int i = 0; i < n_steps; i++) {
        sbar.push_back(i*sbar_step);
        sbar_actual.push_back(0.0);
        rho_1.push_back(0.0);
        u_1.push_back(0.0);
        u.push_back(0.0);
    }

    u_1_0 = 0.0;
}

double Shootc::bang(double fbg, double in_rho_1_0)
{
    // fbar is (F[0]*z_1[0]) * v_rot * v_rot / (g*g*mu)
    // z'(0) is z_1[0] = std::sqrt(1 - rho_1[0]*rho_1[0])

    double u_2;
    double next_bsa;
    double next_sbar_actual;
    std::vector<double> fnet{0.0, 0.0, 0.0};

    double z_1_0;
    std::vector<double> f_0{0.0, 0.0, 0.0};

    for (int i = 0; i < n_steps; i++) {
        rho_1[i] = 0.0;
        u_1[i] = 0.0;
        u[i] = 0.0;
    }
    u_1[0] = u_1_0;
    set_fbar(fbg);
    fbg = fbar;

    rho_1_0 = in_rho_1_0;
    grad_ratio = rho_1_0/std::sqrt(1-rho_1_0*rho_1_0);

    u[0] = calc_u_0();
    // std::cout << 'h' << std::endl;
    // std::cout << u[0] << std::endl;
    rho_1[0] = rho_1_0;
    u_2 = - rho_1[0];

    // get F[0]
    z_1_0 = std::sqrt(1 - rho_1_0*rho_1_0);
    f_0[2] = get_f();
    f_0[0] = f_0[2] * rho_1_0 / z_1_0;

    next_bsa = 0.0;
    next_sbar_actual = 0.0;
    sbar_actual[0] = 0.0;
    for (int i = 1; i < n_steps; i++) {
        double barstep_actual;
        double u_feed;
        double sbar_feed;
        double fnet_norm;
        double tmp1;

        // getting barstep_actual
        if (i>1) {
            barstep_actual = next_bsa;
            sbar_actual[i] = next_sbar_actual;
        }
        else {
            std::vector<double> f_grav_cent{-u_1[i-1]*m*g, 0.0, -m*g};
            // fnet_x += -(u_1[i-1]*m*g);
            fnet_norm = 0.0;
            for (int vec_i=0;vec_i<fnet.size();vec_i++)
            {
                fnet[vec_i] += f_grav_cent[vec_i];
                fnet[vec_i] += f_0[vec_i];
                fnet_norm += fnet[vec_i]*fnet[vec_i];
            }
            fnet_norm = std::sqrt(fnet_norm);
            // std::cout << fnet_norm << std::endl;
            // std::cout << 'L' << std::endl;
            barstep_actual = sbar_step + (fnet_norm/k)*v_rot*v_rot/g;
            sbar_actual[1] = sbar_actual[0] + barstep_actual;
            // sbar_actual[1] = sbar_actual[0] + sbar_step;
        }

        u[i] = u[i-1] + u_1[i-1] * barstep_actual;
        u_1[i] = u_1[i-1] + u_2 * barstep_actual;

        // u[i] = u[i-1] + u_1[i-1] * sbar_step;
        // u_1[i] = u_1[i-1] + u_2 * sbar_step;

        // u[i] = u[i-1] + u_1[i-1] * sbar_step;
        // u_1[i] = u_1[i-1] + u_2 * sbar_step;

        // getting next barstep_actual
        std::vector<double> f_grav_cent{-u_1[i]*m*g, 0.0, -m*g};
        fnet_norm = 0.0;
        for (int vec_i=0;vec_i<fnet.size();vec_i++)
        {
            fnet[vec_i] += f_grav_cent[vec_i];
            fnet_norm += fnet[vec_i]*fnet[vec_i];
        }
        fnet_norm = std::sqrt(fnet_norm);
        next_bsa = sbar_step + (fnet_norm/k)*v_rot*v_rot/g;
        next_sbar_actual = sbar_actual[i] + next_bsa;

        // next u_2 with feed split
        u_feed = (u[i] + split_ratio_latter*u_1[i]*next_bsa);
        sbar_feed = (
            (1-split_ratio_latter)*sbar_actual[i]
            + split_ratio_latter*next_sbar_actual
        );
        u_2 = func_fixed2(u_feed, sbar_feed);
        rho_1[i] = -u_2;

        // std::cout << ' ' << std::endl;
        // std::cout << i << std::endl;
        // std::cout << u[i] << std::endl;
        // std::cout << rho_1[i-1] << std::endl;

    }
    if (error_verbose) {std::cout << u_1[n_steps-1] << std::endl;}

    return u_1[n_steps-1];
}

double Shootc::get_info(
    int dim_u1,
    double *out_u_1,
    int dim_sbar,
    double *out_sbar,
    int dim_rho1,
    double *out_rho_1
)
{
    for (int i = 0; i < n_steps; i++) {
        out_rho_1[i] = rho_1[i];
        out_u_1[i] = u_1[i];
        out_sbar[i] = sbar_actual[i];
    }

    rbar = u_1[n_steps-1];
    return rbar;
}

// Extra funcs
void Shootc::set_fbar(double fbg)
{
    if (abs(fbg) < 1e-10) {
        if (error_verbose) {std::cout << 'd' << std::endl;}
        fbg = 1e-10;
    }
    fbar = fbg;
}

double Shootc::func_fixed2(
    double v_u,
    double v_sbar
)
{
    return -v_u/std::sqrt(v_u*v_u+pow((v_sbar+fbar),2));
}

double Shootc::calc_u_0() {return grad_ratio * fbar;}

double Shootc::get_f() {return fbar*g*g*mu/(v_rot*v_rot);}