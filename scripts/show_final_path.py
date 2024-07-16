import sys
import numpy as np
import rotatingchain_wbef.utils.plotter as plter
import os
import pickle
from rotatingchain_wbef.utils.shoot_utils import spec_shootv2c
from rotatingchain_wbef.utils.space_utils import (
    get_ctrl_truth,
    get_eig_truth,
    get_cmode,
    nearest_id_from_raw,
    expand_freespace
)
from rotatingchain_wbef.utils.animate_utils import animate_3d


data_plot_type = sys.argv[1]

stabres_picklepath = 'frc_soln_c_stab_result.pickle'
stabres_picklepath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/rc_data/" + stabres_picklepath
)
print('Loading previously saved stab_data.. ..')
with open(stabres_picklepath, 'rb') as f:
    pickle_data = pickle.load(f)
print('Pickle loaded!')
[
    [lbar_list,fbar_list,rho10_list],
    max_eig_3d,
    zlr_pts,
    control_data,
    [mode_data, mode_data2]
] = pickle_data

# space stuff
ctrl_lim = np.zeros((3,2))
ctrl_lim[0] = np.array([-100,100])
ctrl_lim[1] = np.array([-100,100])
ctrl_lim[2] = np.array([-100,100])

# ctrl_lim[0] = np.array([0.,50])
ctrl_lim[1] = np.array([0.0,0.05])
# ctrl_lim[1] = np.array([-0.035,0.035])
# ctrl_lim[2] = np.array([-0.8,-0.5])

# v, r
# 10, 0.09
# 20, 0.015
# 25, 0.010

# make r abs
control_data2 = control_data.copy()
control_data2[:,:,:,1:] = np.abs(control_data2[:,:,:,1:])
control_truth = get_ctrl_truth(
    ctrl_data=control_data2,
    ctrl_lim=ctrl_lim
)

eig_lim = 0.
eig_truth = get_eig_truth(
    max_eig_3d=max_eig_3d,
    eig_lim=eig_lim
)

c_mode = get_cmode(
    mode_data
)

# free_truth = eig_truth.copy()
# free_space = expand_freespace(free_space,diag_expand=0)

# shape pickle stuff
s2_picklepath = 'frc_soln_c.pickle'
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
xyz_axes = [lbar_list,fbar_list,rho10_list]

final_path = [
    # mode 0
    [1.  , 0.  , 0.01],
    [8.95918367, 7.28571429, 0.41367347],
    [17.71428571, 13.71428571,  0.89808163],
    [9.75510204, 6.85714286, 0.51459184],
    [1.79591837, 0.        , 0.19165306],
    [5.7755102 , 0.85714286, 0.19165306],
    # mode 1
    [12.14285714,  0.        ,  0.19165306],
    [12.93877551,  0.        ,  0.59532653],
    [19.30612245,  2.14285714,  0.83753061],
    [40.        , 11.14285714,  0.93844898]
]

control_path = [
    [4.42718872e+00,  1.71714252e-04, 4.99997814e-01],
    [13.25141502,  0.02790669, 0.48863182],
    [1.86333035e+01, 1.34271874e-02, 3.79647627e-01],
    [1.86333035e+01, 1.34271874e-02, 3.79647627e-01],
    [13.82750881,  0.01993504, 0.48140935],
    [ 4.42718872e+00,  3.29129489e-03, 4.99124935e-01],
    [ 1.06395489e+01,  2.29748924e-03, 4.98743971e-01],
    [ 1.54272486e+01,  9.47080623e-04, 4.98393286e-01],
    [ 1.59248234e+01,  4.59353474e-03, 4.83185683e-01],
    [ 1.94525063e+01,  1.15308042e-02, 4.62678274e-01],
    [ 2.80000000e+01,  7.14170759e-03, 3.90867810e-01],
    [ 2.80000000e+01,  7.14170759e-03, 3.90867810e-01]
]

# # animating 3d plot
# animate_3d(
#     [lbar_list,fbar_list,rho10_list],
#     free_truth=free_space,
#     c_mode=c_mode,
#     vital_pts=final_path,
#     path_plot=np.array(final_path),
#     label_plot=True
# )

if data_plot_type == 'full':
    free_space = eig_truth
    plter.plot_freespace_4d(
        [lbar_list,fbar_list,rho10_list],
        # free_truth=control_truth,
        # free_truth=eig_truth,
        free_truth=free_space,
        c_mode=c_mode,
        # vital_pts=final_path,
        # path_plot=np.array(final_path),
        # zlr_pts=zlr_pts,
        label_plot=True
    )
elif data_plot_type == 'path':
    free_space = np.logical_and(control_truth,eig_truth)
    plter.plot_freespace_4d(
        [lbar_list,fbar_list,rho10_list],
        # free_truth=control_truth,
        # free_truth=eig_truth,
        free_truth=free_space,
        c_mode=c_mode,
        vital_pts=final_path,
        path_plot=np.array(final_path),
        # zlr_pts=zlr_pts,
        label_plot=True
    )
elif data_plot_type == 'controls':
    plter.plot_controls(ctrlpath=control_path, plot_label=True)

# # plot_final_shapes
# # OLD
# xyz_ids = []
# seeshapes_truth = np.zeros(control_data.shape[:-1])
# for i in range(len(final_path)):
#     xyz_ids.append(nearest_id_from_raw(xyz_axes,final_path[i]))
#     seeshapes_truth[
#         xyz_ids[i][0],
#         xyz_ids[i][1],
#         xyz_ids[i][2],
#     ] = 1.
# plter.plot_overall_b(
#     data_list_a=data_list_a,
#     info_list=info_list,
#     plot_truth=seeshapes_truth,
#     # max_plot=,
#     max_eig_3d=max_eig_3d
# )

elif data_plot_type == 'shapes':
    # Plot_final_shapes
    data_list_b = []
    spec_pts = [0,2,4,6,8,9]
    for i in range(len(final_path)):
        if i in spec_pts:
            data_list, info_list = spec_shootv2c(final_path[i], n_steps=100)
            data_list_b.append(data_list)
    plter.plot_specific_shapes(
        data_list_a=data_list_b,
        info_list=info_list,
        max_plot=1
    )