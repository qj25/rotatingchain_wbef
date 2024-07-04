#ifndef STAB_JAC_H
#define STAB_JAC_H

#include "Stab_jac.h"
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

class Stab_jac
{
public:
    Stab_jac(
        const double in_rho_a,
        const double in_d,
        const double in_c_f,
        const double in_c_n,
        const double in_k,
        bool in_error_verbose
    );

    bool error_verbose;

    double rho_a;
    double d;
    double c_f;
    double c_n;
    double k;

    Eigen::Matrix3d i3;


    int n;
    int n_i;
    double l_total;
    double l;
    double m;
    Eigen::Vector3d v_rot;
    Eigen::Matrix3d vrot_skew;
    std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > x;
    std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > x_dot;

    std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > l_n;
    std::vector<double> ln_norm;

    double l_n_avg;

    std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vt;
    std::vector<double> vt_norm;

    std::vector<std::vector <Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> >> jac_mat;

    void calc_stab(
        int dim_stabjac,
        double *out_stabjac,
        int dim_y,
        double *in_y,
        const double in_mu,
        const double in_l,
        const double in_v_rot
    );

private:
    void populate_jac2();

    void update_all_funcs();

    Eigen::Matrix3d func_ff_x();
    Eigen::Matrix3d func_ff_xdot();

    Eigen::Matrix3d func_fc_1x(int i_node);
    Eigen::Matrix3d func_fc_x(int i_node);
    Eigen::Matrix3d func_fc0_x(int i_node);
    Eigen::Matrix3d func_fc_x1(int i_node);

    double func_sinxi(int i_node);
    Eigen::Vector3d func_sinxi_1x(int i_node);
    Eigen::Vector3d func_sinxi_x(int i_node);
    Eigen::Vector3d func_sinxi_xdot(int i_node);

    double func_cosxi(int i_node);
    Eigen::Vector3d func_cosxi_1x(int i_node);
    Eigen::Vector3d func_cosxi_x(int i_node);
    Eigen::Vector3d func_cosxi_xdot(int i_node);

    double func_cd(int i_node);
    Eigen::Vector3d func_cd_1x(int i_node);
    Eigen::Vector3d func_cd_x(int i_node);
    Eigen::Vector3d func_cd_xdot(int i_node);

    double func_cl(int i_node);
    Eigen::Vector3d func_cl_1x(int i_node);
    Eigen::Vector3d func_cl_x(int i_node);
    Eigen::Vector3d func_cl_xdot(int i_node);

    Eigen::Vector3d func_ed(int i_node);
    Eigen::Matrix3d func_ed_x(int i_node);
    Eigen::Matrix3d func_ed_xdot(int i_node);

    Eigen::Vector3d func_el(int i_node);
    Eigen::Matrix3d func_el_1x(int i_node);
    Eigen::Matrix3d func_el_x(int i_node);
    Eigen::Matrix3d func_el_xdot(int i_node);

    Eigen::Vector3d func_g(int i_node);
    Eigen::Matrix3d func_g_1x(int i_node);
    Eigen::Matrix3d func_g_x(int i_node);
    Eigen::Matrix3d func_g_xdot(int i_node);

    double func_ad_const();

    Eigen::Matrix3d func_fd_1x(int i_node);
    Eigen::Matrix3d func_fd_x(int i_node);
    Eigen::Matrix3d func_fd_xdot(int i_node);

    Eigen::Matrix3d func_fl_1x(int i_node);
    Eigen::Matrix3d func_fl_x(int i_node);
    Eigen::Matrix3d func_fl_xdot(int i_node);

    Eigen::Matrix3d get_skewsym(Eigen::Vector3d v3);
};

#endif
