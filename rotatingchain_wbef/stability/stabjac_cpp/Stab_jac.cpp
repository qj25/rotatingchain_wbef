#include "Stab_jac.h"
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

Stab_jac::Stab_jac(
    const double in_rho_a,
    const double in_d,
    const double in_c_f,
    const double in_c_n,
    const double in_k,
    bool in_error_verbose
)
{
    // Class to determine the stability of a certain configuration of a rotating chain.
    // For fixed end, jacobian excludes top and bottom fixed points (N and 0),
    // there is a total of n + 1 - 2 (2 fixed points), n-1 points.
    // Jacobian is 6(n-1) x 6(n-1) matrix. (3 pos + 3 vel)

    // Init chain with chain properties:
    // n - number of discrete segments
    // mu - mass per unit length
    // l - length

    // Init class with stored values for:
    // rho_a - air density
    // d - diameter of the chain
    // c_f - skin-friction drag co-eff
    // c_n - crossflow drag co-eff
    
    // Velocity:
    // x_dot - vel wrt rotating frame
    // v_rot cross x - vel of rotating frame
    // vt - vel wrt inertial frame (airspeed)
    // vt = x_dot + (v_rot cross x)

    
    error_verbose = in_error_verbose;

    rho_a = in_rho_a;
    d = in_d;
    c_f = in_c_f;
    c_n = in_c_n;
    k = in_k;

    i3 << 1., 0., 0.,
        0., 1., 0.,
        0., 0., 1.;
}

void Stab_jac::calc_stab(
    int dim_stabjac,
    double *out_stabjac,
    int dim_y,
    double *in_y,
    const double in_mu,
    const double in_l,
    const double in_v_rot
)
{
    n = dim_y/(2*3);
    n_i = n - 2;
    l_total = in_l;
    l = l_total / (n-1);
    m = in_mu * l_total / n;
    v_rot << 0.0, 0.0, in_v_rot;
    vrot_skew = get_skewsym(v_rot);

    l_n_avg = l_total / (n - 1);

    for (int i = 0; i < n; i++) {
        Eigen::Vector3d y_extract;
        y_extract << in_y[(2*i)*3], in_y[(2*i)*3 + 1], in_y[(2*i)*3 + 2];
        x.push_back(y_extract);
        y_extract << in_y[(2*i+1)*3], in_y[(2*i+1)*3 + 1], in_y[(2*i+1)*3 + 2];
        x_dot.push_back(y_extract);

        vt.push_back(
            x_dot[i]
            + v_rot.cross(x[i])
        );
        vt_norm.push_back(vt[i].norm());

        if (i > 0) {
            l_n.push_back(x[i] - x[i-1]);
        }
        else {
            y_extract << 0.0, 0.0, 0.0;
            l_n.push_back(y_extract);
        }
        ln_norm.push_back(l_n[i].norm());
    }

    for (int i = 0; i < n_i; i++) {
        std::vector <Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > jac1;
        for (int it_twice = 0; it_twice < 2; it_twice++) {
            jac_mat.push_back(jac1);
            for (int j = 0; j < n_i; j++) {
                // std::cout << 'h' << std::endl;
                // std::cout << i << std::endl;
                // std::cout << j << std::endl;
                for (int it2_twice = 0; it2_twice < 2; it2_twice++) {
                    jac_mat[2*i + it_twice].push_back(
                        (Eigen::Matrix3d() << 0., 0., 0.,
                            0., 0., 0.,
                            0., 0., 0.).finished()
                    );
                }
            }
        }
    }

    // for (int i = 0; i < n_i; i++) {
    //     for (int j = 0; j < n_i; j++) {
    //         std::cout << jac_mat[i][j] << std::endl;
    //     }
    // }

    populate_jac2();

    for (int i = 0; i < 2*n_i; i++) {
        for (int j = 0; j < 2*n_i; j++) {
            // std::cout << i << std::endl;
            // std::cout << j << std::endl;
            // std::cout << jac_mat[i][j] << std::endl;

            for (int ii = 0; ii < 3; ii++) {
                for (int jj = 0; jj < 3; jj++) {
                    out_stabjac[
                        (3*(3*i+ii))*2*n_i + (3*j+jj)
                    ] = jac_mat[i][j](ii,jj);
                    // std::cout << (3*(3*i+ii))*2*n_i + (3*j+jj) << std::endl;
                    // std::cout << jac_mat[i][j](ii,jj) << std::endl;
                    // std::cout << out_stabjac[
                        // (3*i+ii)*n_i
                        // + (3*j+jj)
                    // ] << std::endl;
                }
            }
        }
    }
    // std::cout << out_stabjac << std::endl;

}

void Stab_jac::populate_jac2()
{
    int i_node;
    for (int i = 0; i < n_i; i++) {
        // std::cout << 'a' << std::endl;
        // std::cout << i << std::endl;
        i_node = i+1;

        // velocity part
        jac_mat[i][i+n_i] = i3;

        // acceleration part
        // std::cout << 'b' << std::endl;
        if (i_node>1) {
            // diff 1x (i-1)
            jac_mat[i+n_i][i-1] = (
                func_fc_1x(i_node)
                + func_fd_1x(i_node)
                + func_fl_1x(i_node)
            ) / m;
        }

        // std::cout << 'c' << std::endl;
        // diff x (i)
        jac_mat[i+n_i][i] = (
            func_fc_x(i_node)
            + func_ff_x()
            + func_fd_x(i_node)
            + func_fl_x(i_node)
        ) / m;

        // std::cout << 'e' << std::endl;
        // diff x1 (i+1)
        if (i_node < n_i) {
            jac_mat[i+n_i][i+1] = (
                func_fc_x1(i_node)
            ) / m;
        }
        // diff xdot (ijac)
        jac_mat[i+n_i][i+n_i] = (
            func_ff_xdot()
        ) / m;
        // std::cout << 'f' << std::endl;
        jac_mat[i+n_i][i+n_i] += (
            func_fd_xdot(i_node)
            + func_fl_xdot(i_node)
        ) / m;
    }
    // return true;
}

Eigen::Matrix3d Stab_jac::func_ff_x()
{
    // std::cout << '2' << std::endl;
    return - (m*(
        v_rot*v_rot.transpose()
        - v_rot.dot(v_rot) * i3
    ));
}
Eigen::Matrix3d Stab_jac::func_ff_xdot()
{
    return -2.0*m*(vrot_skew);
}

Eigen::Matrix3d Stab_jac::func_fc_1x(const int i_node)
{
    Eigen::Matrix3d fc_1x;
    fc_1x = (
        i3
        + l_n_avg * (
            - i3 / ln_norm[i_node]
            + (l_n[i_node] * l_n[i_node].transpose()) / std::pow(ln_norm[i_node],3)
        )
    ) * k;
    // std::cout << 'h' << std::endl;
    // std::cout << i_node << std::endl;
    // std::cout << l_n_avg - 0.0074 << std::endl;
    // std::cout << fc_1x << std::endl;
    return fc_1x;
}
Eigen::Matrix3d Stab_jac::func_fc_x(const int i_node)
{
    Eigen::Matrix3d fc_x;
    fc_x = (
        - i3
        + l_n_avg * (
            i3 / ln_norm[i_node+1]
            - (l_n[i_node+1] * l_n[i_node+1].transpose()) / std::pow(ln_norm[i_node+1],3)
        )
    );
    fc_x += (
        - i3
        + l_n_avg * (
            i3 / ln_norm[i_node]
            - (l_n[i_node] * l_n[i_node].transpose()) / std::pow(ln_norm[i_node],3)
        )
    );
    fc_x *= k;
    return fc_x;
}
Eigen::Matrix3d Stab_jac::func_fc0_x(const int i_node)
{
    // std::cout << '1' << std::endl;
    Eigen::Matrix3d fc0_x;
    fc0_x = (
        - i3
        + l_n_avg * (
            i3 / ln_norm[i_node+1]
            - (l_n[i_node+1] * l_n[i_node+1].transpose()) / std::pow(ln_norm[i_node+1],3)
        )
    ) * k;
    return fc0_x;
}
Eigen::Matrix3d Stab_jac::func_fc_x1(const int i_node)
{
    Eigen::Matrix3d fc_x1;
    fc_x1 = (
        i3
        + l_n_avg * (
            - i3 / ln_norm[i_node+1]
            + (l_n[i_node+1] * l_n[i_node+1].transpose()) / std::pow(ln_norm[i_node+1],3)
        )
    ) * k;
    return fc_x1;
}

double Stab_jac::func_sinxi(const int i_node)
{
    return (l_n[i_node].cross(vt[i_node])).norm() / (ln_norm[i_node] * vt_norm[i_node]);
}
Eigen::Vector3d Stab_jac::func_sinxi_1x(const int i_node)
{
    Eigen::Vector3d c;
    double c_norm;
    Eigen::Vector3d sinxi_1x;
    
    c = l_n[i_node].cross(vt[i_node]);
    c_norm = c.norm();
    // std::cout << (get_skewsym(vt[i_node])).transpose() * c << std::endl;
    sinxi_1x = (
        (get_skewsym(vt[i_node])).transpose()*c
    ) / (
        c_norm * ln_norm[i_node] * vt_norm[i_node]
    );
    sinxi_1x += (
        c_norm * l_n[i_node]
        / (vt_norm[i_node] * std::pow(ln_norm[i_node], 3))
    );
    return sinxi_1x;
}
Eigen::Vector3d Stab_jac::func_sinxi_x(const int i_node)
{
    Eigen::Vector3d c;
    double c_norm;
    Eigen::Vector3d sinxi_x;
    
    c = l_n[i_node].cross(vt[i_node]);
    c_norm = c.norm();
    sinxi_x = (
        (c.dot(v_rot)) * (x[i_node]+l_n[i_node])
        - (get_skewsym(x_dot[i_node]).transpose())*c
        - (c.dot(x[i_node])) * (v_rot)
        - (v_rot.dot(l_n[i_node])) * (c)
    ) / (ln_norm[i_node] * vt_norm[i_node] * c_norm);
    sinxi_x += - (
        c_norm * l_n[i_node]
        / (vt_norm[i_node] * std::pow(ln_norm[i_node], 3))
    );
    sinxi_x += - (
        c_norm
        * (vrot_skew.transpose())*(vt[i_node])
    ) / (ln_norm[i_node] * std::pow(vt_norm[i_node], 3));
    return sinxi_x;
}
Eigen::Vector3d Stab_jac::func_sinxi_xdot(const int i_node)
{
    Eigen::Vector3d c;
    double c_norm;
    Eigen::Vector3d sinxi_xdot;
    
    c = l_n[i_node].cross(vt[i_node]);
    c_norm = c.norm();
    sinxi_xdot = (
        (get_skewsym(l_n[i_node]).transpose()) * c
    ) / (c_norm * ln_norm[i_node] * vt_norm[i_node]);
    sinxi_xdot += - (
        c_norm * vt[i_node]
        / (ln_norm[i_node] * std::pow(vt_norm[i_node], 3))
    );
    return sinxi_xdot;
}

double Stab_jac::func_cosxi(const int i_node)
{
    return - l_n[i_node].dot(vt[i_node]) / (ln_norm[i_node]*vt_norm[i_node]);
}
Eigen::Vector3d Stab_jac::func_cosxi_1x(const int i_node)
{
    Eigen::Vector3d cosxi_1x;
    cosxi_1x = vt[i_node] / (ln_norm[i_node]*vt_norm[i_node]);
    cosxi_1x += - (
        l_n[i_node].dot(vt[i_node])
        * l_n[i_node]
        / (vt_norm[i_node] * std::pow(ln_norm[i_node], 3))
    );
    return cosxi_1x;
}
Eigen::Vector3d Stab_jac::func_cosxi_x(const int i_node)
{
    Eigen::Vector3d cosxi_x;
    cosxi_x = (
        - vt[i_node]
        - (vrot_skew.transpose()) * (l_n[i_node])
    ) / (ln_norm[i_node] * vt_norm[i_node]);
    cosxi_x += (
        (vt[i_node].dot(l_n[i_node])) * (l_n[i_node])
    ) / (vt_norm[i_node] * std::pow(ln_norm[i_node], 3));

    cosxi_x += (
        (
            (vt[i_node].dot(l_n[i_node]))
            * vrot_skew.transpose()
        ) * (vt[i_node])
    ) / (ln_norm[i_node] * std::pow(vt_norm[i_node], 3));
    return cosxi_x;
}
Eigen::Vector3d Stab_jac::func_cosxi_xdot(const int i_node)
{
    Eigen::Vector3d cosxi_xdot;
    cosxi_xdot = - l_n[i_node] / (ln_norm[i_node]*vt_norm[i_node]);
    cosxi_xdot += (
        l_n[i_node].dot(vt[i_node])
        * vt[i_node]
        / (ln_norm[i_node] * std::pow(vt_norm[i_node], 3))
    );
    return cosxi_xdot;
}

double Stab_jac::func_cd(const int i_node)
{
    return c_f + c_n * std::pow(func_sinxi(i_node), 3);
}
Eigen::Vector3d Stab_jac::func_cd_1x(const int i_node)
{
    return 3.0*c_n*func_sinxi_1x(i_node)*std::pow(func_sinxi(i_node), 2);
}
Eigen::Vector3d Stab_jac::func_cd_x(const int i_node)
{
    return 3.0*c_n*func_sinxi_x(i_node)*std::pow(func_sinxi(i_node), 2);
}
Eigen::Vector3d Stab_jac::func_cd_xdot(const int i_node)
{
    return 3.0*c_n*func_sinxi_xdot(i_node)*std::pow(func_sinxi(i_node), 2);
}

double Stab_jac::func_cl(const int i_node)
{
    return c_n * func_cosxi(i_node) * std::pow(func_sinxi(i_node), 2);
}
Eigen::Vector3d Stab_jac::func_cl_1x(const int i_node)
{
    Eigen::Vector3d cl_1x;
    double s;
    s = func_sinxi(i_node);
    cl_1x = (
        2.0 * func_sinxi_1x(i_node)
        * s * func_cosxi(i_node)
    );
    cl_1x += func_cosxi_1x(i_node) * std::pow(s, 2);
    cl_1x *= c_n;
    return cl_1x;
}
Eigen::Vector3d Stab_jac::func_cl_x(const int i_node)
{
    Eigen::Vector3d cl_x;
    double s;
    s = func_sinxi(i_node);
    cl_x = (
        2.0 * func_sinxi_x(i_node)
        * s * func_cosxi(i_node)
    );
    cl_x += func_cosxi_x(i_node) * std::pow(s, 2);
    cl_x *= c_n;
    return cl_x;
}
Eigen::Vector3d Stab_jac::func_cl_xdot(const int i_node)
{
    Eigen::Vector3d cl_xdot;
    double s;
    s = func_sinxi(i_node);
    cl_xdot = (
        2.0 * func_sinxi_xdot(i_node)
        * s * func_cosxi(i_node)
    );
    cl_xdot += func_cosxi_xdot(i_node) * std::pow(s, 2);
    cl_xdot *= c_n;
    return cl_xdot;
}

Eigen::Vector3d Stab_jac::func_ed(const int i_node)
{
    return - vt[i_node] / vt_norm[i_node];
}
Eigen::Matrix3d Stab_jac::func_ed_x(const int i_node)
{
    Eigen::Matrix3d ed_x;
    ed_x = (
        vt[i_node] * ((vt[i_node].transpose() * vrot_skew))
    ) / std::pow(vt_norm[i_node], 3);
    ed_x += - vrot_skew / vt_norm[i_node];
    return ed_x;
}
Eigen::Matrix3d Stab_jac::func_ed_xdot(const int i_node)
{
    Eigen::Matrix3d ed_xdot;
    ed_xdot = (
        vt[i_node] * vt[i_node].transpose()
    ) / std::pow(vt_norm[i_node], 3);
    ed_xdot += - i3 / vt_norm[i_node];
    return ed_xdot;
}

Eigen::Vector3d Stab_jac::func_el(const int i_node)
{
    Eigen::Vector3d el;
    el = - func_g(i_node);
    el /= el.norm();
    return el;
}
Eigen::Matrix3d Stab_jac::func_el_1x(const int i_node)
{
    Eigen::Vector3d g;
    double g_norm;
    Eigen::Matrix3d g_1x;
    Eigen::Matrix3d el_1x;

    g = func_g(i_node);
    g_norm = g.norm();
    g_1x = func_g_1x(i_node);
    el_1x = - g_1x / g_norm;
    el_1x += (g*(g.transpose()*g_1x)) / std::pow(g_norm, 3);
    return el_1x;
}
Eigen::Matrix3d Stab_jac::func_el_x(const int i_node)
{
    Eigen::Vector3d g;
    double g_norm;
    Eigen::Matrix3d g_x;
    Eigen::Matrix3d el_x;

    g = func_g(i_node);
    g_norm = g.norm();
    g_x = func_g_x(i_node);
    el_x = - g_x / g_norm;
    el_x += (g*(g.transpose()*g_x)) / std::pow(g_norm, 3);
    return el_x;
}
Eigen::Matrix3d Stab_jac::func_el_xdot(const int i_node)
{
    Eigen::Vector3d g;
    double g_norm;
    Eigen::Matrix3d g_xdot;
    Eigen::Matrix3d el_xdot;

    g = func_g(i_node);
    g_norm = g.norm();
    g_xdot = func_g_xdot(i_node);
    el_xdot = - g_xdot / g_norm;
    el_xdot += (g*(g.transpose()*g_xdot)) / std::pow(g_norm, 3);
    return el_xdot;
}

Eigen::Vector3d Stab_jac::func_g(const int i_node)
{
    Eigen::Vector3d g;
    g = (vt[i_node].cross(l_n[i_node])).cross(vt[i_node]);
    return g;
}
Eigen::Matrix3d Stab_jac::func_g_1x(const int i_node)
{
    Eigen::Matrix3d g_1x;
    g_1x = (
        vt[i_node] * vt[i_node].transpose()
        - vt[i_node].dot(vt[i_node]) * i3
    );
    return g_1x;
}
Eigen::Matrix3d Stab_jac::func_g_x(const int i_node)
{
    Eigen::Matrix3d g_x;
    g_x = - func_g_1x(i_node);
    g_x += 2.0 * (
        l_n[i_node]
        * (vt[i_node].transpose()*(vrot_skew))
    );
    g_x += - (
        vt[i_node]
        * (l_n[i_node].transpose()*(vrot_skew))
    );
    g_x += - (
        (l_n[i_node].dot(vt[i_node])) * (vrot_skew)
    );
    return g_x;
}
Eigen::Matrix3d Stab_jac::func_g_xdot(const int i_node)
{
    Eigen::Matrix3d g_xdot;
    g_xdot = (
        - l_n[i_node].dot(vt[i_node]) * i3
        - vt[i_node] * l_n[i_node].transpose()
        + 2.0 * l_n[i_node] * vt[i_node].transpose()
    );
    return g_xdot;
}

double Stab_jac::func_ad_const()
{
    return 0.5 * rho_a * d;
}

Eigen::Matrix3d Stab_jac::func_fd_1x(const int i_node)
{
    Eigen::Matrix3d fd_1x;
    fd_1x = - (
        func_cd(i_node)
        * std::pow(vt_norm[i_node], 2)
        * func_ed(i_node)
        * (l_n[i_node]/ln_norm[i_node]).transpose()
    );
    fd_1x += (
        ln_norm[i_node]
        * std::pow(vt_norm[i_node], 2)
        * func_ed(i_node)
        * func_cd_1x(i_node).transpose()
    );
    fd_1x *= func_ad_const();
    return fd_1x;
}
Eigen::Matrix3d Stab_jac::func_fd_x(const int i_node)
{
    Eigen::Matrix3d fd_x;
    fd_x = (
        func_cd(i_node)
        * std::pow(vt_norm[i_node], 2)
        * func_ed(i_node)
        * (l_n[i_node]/ln_norm[i_node]).transpose()
    );
    fd_x += (
        ln_norm[i_node]
        * std::pow(vt_norm[i_node], 2)
        * func_ed(i_node)
        * func_cd_x(i_node).transpose()
    );
    fd_x += 2.0 * (
        func_cd(i_node)
        * ln_norm[i_node]
        * func_ed(i_node)
        * (
            (vt[i_node].transpose()) * (vrot_skew)
        )
    );
    fd_x += (
        func_cd(i_node)
        * ln_norm[i_node]
        * std::pow(vt_norm[i_node], 2)
        * func_ed_x(i_node)
    );
    fd_x *= func_ad_const();
    return fd_x;
}
Eigen::Matrix3d Stab_jac::func_fd_xdot(const int i_node)
{
    Eigen::Matrix3d fd_xdot;
    fd_xdot = 2.0 * (
        func_cd(i_node)
        * ln_norm[i_node]
        * func_ed(i_node)
        * vt[i_node].transpose()
    );
    fd_xdot += (
        ln_norm[i_node]
        * std::pow(vt_norm[i_node], 2)
        * func_ed(i_node)
        * func_cd_xdot(i_node).transpose()
    );
    fd_xdot += (
        func_cd(i_node)
        * ln_norm[i_node]
        * std::pow(vt_norm[i_node], 2)
        * func_ed_xdot(i_node)
    );
    fd_xdot *= func_ad_const();
    return fd_xdot;
}

Eigen::Matrix3d Stab_jac::func_fl_1x(const int i_node)
{
    Eigen::Matrix3d fl_1x;
    fl_1x = - (
        func_cl(i_node)
        * std::pow(vt_norm[i_node], 2)
        * func_el(i_node)
        * (l_n[i_node]/ln_norm[i_node]).transpose()
    );
    // std::cout << fl_1x << std::endl;
    fl_1x += (
        ln_norm[i_node]
        * std::pow(vt_norm[i_node], 2)
        * func_el(i_node)
        * func_cl_1x(i_node).transpose()
    );
    // std::cout << fl_1x << std::endl;
    fl_1x += (
        func_cl(i_node)
        * ln_norm[i_node]
        * std::pow(vt_norm[i_node], 2)
        * func_el_1x(i_node)
    );
    // std::cout << fl_1x << std::endl;
    fl_1x *= func_ad_const();
    // std::cout <<  'h' << std::endl;
    // std::cout << i_node << std::endl;
    // std::cout << ln_norm[i_node] << std::endl;
    // std::cout << std::pow(vt_norm[i_node], 2) << std::endl;
    // std::cout << func_el(i_node) << std::endl;
    // std::cout << func_cl_1x(i_node) << std::endl;
    // std::cout << func_el(i_node)*func_cl_1x(i_node).transpose() << std::endl;
    return fl_1x;
}
Eigen::Matrix3d Stab_jac::func_fl_x(const int i_node)
{
    Eigen::Matrix3d fl_x;
    fl_x = (
        func_cl(i_node)
        * std::pow(vt_norm[i_node], 2)
        * func_el(i_node)
        * (l_n[i_node]/ln_norm[i_node]).transpose()
    );
    fl_x += (
        ln_norm[i_node]
        * std::pow(vt_norm[i_node], 2)
        * func_el(i_node)
        * func_cl_x(i_node).transpose()
    );
    fl_x += 2.0 * (
        func_cl(i_node)
        * ln_norm[i_node]
        * func_el(i_node)
        * (
            (vt[i_node].transpose()) * (vrot_skew)
        )
    );
    fl_x += (
        func_cl(i_node)
        * ln_norm[i_node]
        * std::pow(vt_norm[i_node], 2)
        * func_el_x(i_node)
    );
    fl_x *= func_ad_const();
    return fl_x;
}
Eigen::Matrix3d Stab_jac::func_fl_xdot(const int i_node)
{
    Eigen::Matrix3d fl_xdot;
    fl_xdot = 2.0 * (
        func_cl(i_node)
        * ln_norm[i_node]
        * func_el(i_node)
        * vt[i_node].transpose()
    );
    fl_xdot += (
        ln_norm[i_node]
        * std::pow(vt_norm[i_node], 2)
        * func_el(i_node)
        * func_cl_xdot(i_node).transpose()
    );
    fl_xdot += (
        func_cl(i_node)
        * ln_norm[i_node]
        * std::pow(vt_norm[i_node], 2)
        * func_el_xdot(i_node)
    );
    fl_xdot *= func_ad_const();
    return fl_xdot;
}

Eigen::Matrix3d Stab_jac::get_skewsym(Eigen::Vector3d v3)
{
    Eigen::Matrix3d ss_v3;
    ss_v3 << 0, -v3[2], v3[1],
        v3[2], 0, -v3[0],
        -v3[1], v3[0], 0;
    return ss_v3;
}