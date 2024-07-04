import numpy as np
from rotatingchain_wbef.algos.shootingc_cpp.Shootc import Shootc
import rotatingchain_wbef.utils.check_approx as check_approx
import os
import pickle
from time import time

def spec_shootv2c(init_params, n_steps=12):
    lbar, fbar, rho10 = init_params

    k_spring = 8490790.955648089
    l = 0.5
    mu = 0.0554054054054054
    g = 9.8

    vrot = np.sqrt(lbar*g/l)

    sr_latter = 1.0 # split ratio latter

    info_list = dict(
        l=l,
        mu=mu,
        g=g,
        n_steps=n_steps,
        k=k_spring,
    )

    u_1_soln = np.zeros(n_steps)
    sbar_soln = np.zeros(n_steps)
    rho_1_soln = np.zeros(n_steps)
    s = Shootc(
        in_v_rot=vrot,
        in_l=l,
        in_mu=mu,
        in_n_steps=n_steps,
        in_g=g,
        in_k=k_spring,
        in_srl=sr_latter,
        in_error_verbose=False
    )
    s.bang(
        fbg=fbar,
        in_rho_1_0=rho10,
    )
    rbar_soln = s.get_info(
        u_1_soln,
        sbar_soln,
        rho_1_soln
    )
    # print(f"u_1 = {u_1_soln}")
    info = dict(
        u_1=u_1_soln.copy(),
        sbar=sbar_soln.copy(),
        rho_1=rho_1_soln.copy(),
        vrot=vrot,
        fbar=fbar,
        rho10=rho10,
        rbar=rbar_soln,
        k=k_spring
    )

    data_list = [
        info['vrot'],
        info['rho10'],
        info['fbar'],
        info['u_1'],
        info['sbar'],
        info['rho_1'],
        info['rbar']
    ]

    return data_list, info_list