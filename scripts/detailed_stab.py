import numpy as np
import os
from time import time
import pickle
from rotatingchain_wbef.stability.pycppstability import pycppstability
import rotatingchain_wbef.utils.plotter as plter
from rotatingchain_wbef.utils.shape_utils import get_shape2c, check_mode
from rotatingchain_wbef.utils.stab_utils import zlr_test

stabdata_from_pickle = False
cntr_dattype = 0
cntr_dt_str = ['eigval', 'rbar', 'z0bar']
# [
#   0 - eigval,
#   1 - rbar,
#   2 - z0bar,
# ]

# pickle stuff
stabres_picklepath = 'frc_soln_c_split5_result_'+cntr_dt_str[cntr_dattype]+'.pickle'
stabres_picklepath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/rc_data/" + stabres_picklepath
)

if stabdata_from_pickle:
    print('Loading previously saved stab_data.. ..')
    with open(stabres_picklepath, 'rb') as f:
        pickle_data = pickle.load(f)
    print('Pickle loaded!')
    [[lbar_list,fbar_list,rho10_list], contour_data, zlr_pts] = pickle_data

else:
    print('Creating new stab_data.. ..')
    print('Loading shape_data.. ..')
    # pickle stuff
    s2_picklepath = 'frc_soln_c_split5_'+cntr_dt_str[cntr_dattype]+'.pickle'
    s2_picklepath = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data/rc_data/" + s2_picklepath
    )

    with open(s2_picklepath, 'rb') as f:
        pickle_data = pickle.load(f)
    print('Pickle loaded!')
    info_list = pickle_data[0]
    data_list_a = pickle_data[1]
    g = info_list['g']
    l = info_list['l']
    k_stiff = info_list['k']

    vrot_list = info_list['vrot_list']
    fbar_list = info_list['fbar_list']
    rho10_list = info_list['rho10_list']
    len_x = len(vrot_list)
    len_y = len(fbar_list)
    len_z = len(rho10_list)

    # stabtest = pystability(
    #     k=k_stiff,
    #     # verb_bool=True
    # )
    # stabtest2 = pystability2(
    #     k=k,
    #     # verb_bool=True
    # )
    # get zero-locus radius
    # zlr_data = get_zlr()

    prev_attach_r = np.zeros((len_x,len_y))
    zlr_pts = []
    contour_data = np.zeros((
        len_x,
        len_y,
        len_z,
    ))
    stab_data = []
    start_t = time()
    for i in range(len_x):
        for j in range(len_y):
            for k in range(len_z):
                y = get_shape2c(data_list_a[i][j][k], info_list, n_interp=12)
                mode_val = check_mode(y)
                stabtest = pycppstability(
                    k=k_stiff,
                    # verb_bool=True
                )
                # # find zlr
                if i > 0:
                    zlr_result = zlr_test(
                        [data_list_a[i-1][j][k][6],vrot_list[i-1]],
                        [data_list_a[i][j][k][6],vrot_list[i]]
                    )
                    if zlr_result is not None:
                        zlr_pts.append([
                            zlr_result**2*l/g,
                            fbar_list[j],
                            rho10_list[k]
                        ])
                # j
                if j > 0:
                    zlr_result = zlr_test(
                        [data_list_a[i][j-1][k][6],fbar_list[j-1]],
                        [data_list_a[i][j][k][6],fbar_list[j]]
                    )
                    if zlr_result is not None:
                        zlr_pts.append([
                            vrot_list[i]**2*l/g,
                            zlr_result,
                            rho10_list[k]
                        ])
                # k
                if k > 0:
                    zlr_result = zlr_test(
                        [data_list_a[i][j][k-1][6],rho10_list[k-1]],
                        [data_list_a[i][j][k][6],rho10_list[k]]
                    )
                    if zlr_result is not None:
                        zlr_pts.append([
                            vrot_list[i]**2*l/g,
                            fbar_list[j],
                            zlr_result
                        ])


                if cntr_dattype == 0:    # not eigen_val
                    max_eigval = stabtest.calc_stab(
                        mu=info_list['mu'],
                        l=info_list['l'],
                        v_rot=data_list_a[i][j][k][0],
                        y=y
                    )
                else:
                    max_eigval = None
                rbar = data_list_a[i][j][k][6]
                r_val = rbar*g/(vrot_list[i]**2)
                z0_val = -y[-2,-1]
                z0bar = z0_val*vrot_list[i]**2/g
                # point_dat = [max_eigval, rbar, z0bar]
                point_dat = [max_eigval, r_val, z0_val]
                
                contour_data[i,j,k] = point_dat[cntr_dattype]
                # contour_data[i,j,k] = data_list_a[i][j][k][6]
                print(f"""
                        total solved = 
                        {i*len_y*len_z + j*len_z + k+1}
                        / {len_x*len_y*len_z}
                """)
                print(f"time elapsed = {time()-start_t}")

                # print(data_list_a[i][j][4])
                # if abs(data_list_a[i][j][0] - 3.3096852748586363) < 1e-10:
                #     input('here1')
                # input()

    zlr_pts = np.array(zlr_pts)

    lbar_list = np.array(vrot_list)**2*l/g

    pickle_data = [[lbar_list,fbar_list,rho10_list], contour_data, zlr_pts]
    with open(stabres_picklepath, 'wb') as f:
        pickle.dump(pickle_data,f)
    print('Pickle saved!')
    
if cntr_dattype == 0:
    plter.plot_stab_4d_splits(
        [lbar_list,fbar_list,rho10_list],
        max_eig_3d=contour_data,
        # zlr_pts=zlr_pts
    )
else:
    if cntr_dattype == 1:
        # cmaptype = 'RdGy'
        cmaptype = 1
    elif cntr_dattype == 2:
        # cmaptype = 'Blues_r'
        cmaptype = 2

    plter.plot_contour_4d(
        [lbar_list,fbar_list,rho10_list],
        contour_data=contour_data,
        # zlr_pts=zlr_pts,
        cmaptype=cmaptype
    )
    