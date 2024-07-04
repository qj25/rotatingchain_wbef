import numpy as np
import rotatingchain_wbef.utils.plotter as plter
import os
import pickle
from rotatingchain_wbef.utils.space_utils import (
    get_ctrl_truth,
    get_eig_truth,
    get_cmode,
    choose_within_area
)


stabres_picklepath = 'frc_soln_c_stab_result.pickle'
stabres_picklepath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
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

eig_lim = 0.5
eig_truth = get_eig_truth(
    max_eig_3d=max_eig_3d,
    eig_lim=eig_lim
)
# plot_truth = np.logical_and(eig_truth,control_truth)
# plot_truth = control_truth

mode_truth = np.ones(mode_data.shape)
mode_truth[mode_data!=1] = 0
c_mode = get_cmode(
    mode_data
)

plot_truth = np.logical_and(mode_truth,control_truth)
plot_truth = np.logical_and(plot_truth, eig_truth)
free_truth = eig_truth.copy()
free_space = np.logical_and(control_truth,eig_truth)
# free_space = expand_freespace(free_space,diag_expand=0)

# shape pickle stuff
s2_picklepath = 'frc_soln_c.pickle'
s2_picklepath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
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

# p1
p0_xyz_lim = [
    [0.0,1.1],
    [0,0.03],
    [0,0.03]
]

p1_xyz_lim = [
    [6,12],
    [3.,8.],
    [0.325,0.425]
]

p2_xyz_lim = [
    [15.0,20.],
    [9.,14.],
    [0.8,0.9]
]

p3_xyz_lim = [
    [6,12],
    [3.,8.],
    [0.425,0.525]
]

p4_xyz_lim = [
    [0.0,3.],
    [0,0.03],
    [0.19,0.2]
]

p5_xyz_lim = [
    [5.0,10.],
    [0,2.],
    [0.19,0.2]
]

P1_xyz_lim = [
    [10.5,15.],
    [0,2.],
    [0,0.2]
]

P2_xyz_lim = [
    [10.5,15.],
    [0,2.],
    [0.4,0.6]
]

P3_xyz_lim = [
    [10.5,20.],
    [0,3.],
    [0.6,0.9]
]

P4_xyz_lim = [
    [30,40.],
    [5.,20.],
    [0.6,0.96]
]

# P5_xyz_lim = [
#     [30,40.],
#     [5.,20.],
#     [0.6,1.0]
# ]

n_top = 10
top_ids = choose_within_area(
    xyz_axes=xyz_axes,
    xyz_lim=p5_xyz_lim, # turn on mode_req when going into mode 1
    max_eig_3d=max_eig_3d.copy(),
    free_space=free_space,
    # mode_stuff=[mode_data2,[2,3]],
    n_top=n_top
)

top_control = []
top_xyz = []
seeshapes_truth = np.zeros(control_data.shape[:-1])
for i in range(len(top_ids)):
    top_control.append(control_data2[
        top_ids[i,0],
        top_ids[i,1],
        top_ids[i,2],
    ])
    top_xyz.append([
        xyz_axes[0][top_ids[i,0]],
        xyz_axes[1][top_ids[i,1]],
        xyz_axes[2][top_ids[i,2]],
    ])
    seeshapes_truth[
        top_ids[i,0],
        top_ids[i,1],
        top_ids[i,2],
    ] = 1.
top_control = np.array(top_control)
top_xyz = np.array(top_xyz)
print(f"Top control = {repr(top_control)}")
print(f"Top xyz = {repr(top_xyz)}")
# top_xyz_list = None

# plter.plot_freespace_4d(
#     [lbar_list,fbar_list,rho10_list],
#     # free_truth=control_truth,
#     # free_truth=eig_truth,
#     free_truth=free_space,
#     c_mode=c_mode,
#     vital_pts=top_xyz
#     # zlr_pts=zlr_pts
# )

plter.plot_overall_b(
    data_list_a=data_list_a,
    info_list=info_list,
    plot_truth=seeshapes_truth,
    max_plot=n_top,
    max_eig_3d=max_eig_3d
)

p0_list = np.array([[ 4.42718872e+00,  1.71714252e-04, 4.99997814e-01]])   #

p1_list = np.array([[13.25141502,  0.0313522 , 0.48849701],
                    [12.64911064,  0.03673755, 0.48877396],
                    [13.25141502,  0.02790669, 0.48863182], #
                    [12.64911064,  0.03283443, 0.48895132],
                    [13.25141502,  0.0293514 , 0.48969479],
                    [12.01665511,  0.0377671 , 0.48925932],
                    [12.64911064,  0.03445443, 0.48994562],
                    [12.01665511,  0.03966563, 0.49017082],
                    [12.64911064,  0.02881959, 0.48913332],
                    [12.64911064,  0.03563078, 0.49089832]])

p2_list = np.array([[1.86333035e+01, 1.34271874e-02, 3.79647627e-01],   #
                    [1.94525063e+01, 1.61976771e-02, 3.77924787e-01],
                    [1.86333035e+01, 8.35855362e-03, 3.80062988e-01],
                    [1.90473095e+01, 6.85259507e-03, 3.79167688e-01],
                    [1.82098874e+01, 1.42318394e-02, 3.82013934e-01],
                    [1.90473095e+01, 1.15875723e-02, 3.79747302e-01],
                    [1.90473095e+01, 1.63181750e-02, 3.80406363e-01],
                    [1.82098874e+01, 8.72589679e-03, 3.82522536e-01],
                    [1.86333035e+01, 6.95706312e-03, 3.81690224e-01],
                    [1.82098874e+01, 1.82457032e-02, 3.92403503e-01]])

p3_list = np.array([[13.82750881,  0.02824808, 0.48098806],
                    [13.25141502,  0.03387199, 0.4815563 ],
                    [13.82750881,  0.02411989, 0.48118962],
                    [13.25141502,  0.02918827, 0.48180817],
                    [13.82750881,  0.02637319, 0.48268169],
                    [13.25141502,  0.03174954, 0.48320389],
                    [12.64911064,  0.03413368, 0.48244921],
                    [13.82750881,  0.01993504, 0.48140935], #
                    [13.25141502,  0.03389297, 0.48454753],
                    [12.64911064,  0.03713393, 0.48372759]])

p4_list = np.array([[ 5.93295879e+00,  1.28554477e-04, 4.99075438e-01],
                    [ 4.42718872e+00,  3.29129489e-03, 4.99124935e-01], #
                    [ 7.12741187e+00,  2.17334832e-03, 4.99022692e-01]])


p5_list = np.array([[ 1.26491106e+01,  3.69529984e-03, 4.98706832e-01],
                    [ 1.20166551e+01,  4.41041865e-03, 4.98757062e-01],
                    [ 1.13490088e+01,  4.93906367e-03, 4.98802849e-01],
                    [ 1.06395489e+01,  5.21612513e-03, 4.98845314e-01],
                    [ 1.06395489e+01,  2.29748924e-03, 4.98743971e-01], #
                    [ 1.32514150e+01,  2.85046054e-03, 4.98651763e-01],
                    [ 1.06395489e+01,  5.21612513e-03, 4.98845314e-01]])


P1_list = np.array([[ 1.54272486e+01,  9.47080623e-04, 4.98393286e-01], #
                    [ 1.49130815e+01,  6.44130041e-06, 4.98461840e-01],
                    [ 1.64073154e+01,  2.65571465e-03, 4.98254229e-01],
                    [ 1.59248234e+01,  1.83671138e-03, 4.98323799e-01],
                    [ 1.49130815e+01,  2.85446709e-03, 4.98680208e-01],
                    [ 1.68760185e+01,  3.38876452e-03, 4.98185200e-01],
                    [ 1.49130815e+01,  2.54515889e-03, 4.98944511e-01],
                    [ 1.49130815e+01,  1.22351283e-05, 4.98769827e-01],
                    [ 1.54272486e+01,  8.54377177e-04, 4.98714911e-01],
                    [ 1.64073154e+01,  2.38339786e-03, 4.98603522e-01]])

P2_list = np.array([[ 1.68760185e+01,  9.42941904e-03, 4.81887461e-01],
                    [ 1.64073154e+01,  7.12120188e-03, 4.82534380e-01],
                    [ 1.68760185e+01,  9.19080662e-03, 4.83147657e-01],
                    [ 1.59248234e+01,  4.59353474e-03, 4.83185683e-01], #
                    [ 1.64073154e+01,  6.96187379e-03, 4.83752781e-01],
                    [ 1.68760185e+01,  8.94351364e-03, 4.84355271e-01],
                    [ 1.54272486e+01,  1.88738838e-03, 4.83835034e-01],
                    [ 1.59248234e+01,  4.51786176e-03, 4.84362071e-01],
                    [ 1.64073154e+01,  6.79392323e-03, 4.84919860e-01],
                    [ 1.68760185e+01,  8.68788712e-03, 4.85511766e-01]])

P3_list = np.array([[ 1.94525063e+01,  7.16568153e-03, 4.47824193e-01],
                    [ 1.94525063e+01,  1.04082276e-02, 4.54392521e-01],
                    [ 1.90473095e+01,  8.38385352e-03, 4.53275366e-01],
                    [ 1.94525063e+01,  1.39297818e-02, 4.59514074e-01],
                    [ 1.94525063e+01,  7.38423352e-03, 4.58249749e-01],
                    [ 1.90473095e+01,  1.30474905e-02, 4.58390426e-01],
                    [ 1.94525063e+01,  1.15308042e-02, 4.62678274e-01], #
                    [ 1.90473095e+01,  5.09994780e-03, 4.57237921e-01],
                    [ 1.94525063e+01,  5.11647343e-03, 4.61409143e-01],
                    [ 1.86333035e+01,  1.06985914e-02, 4.57521464e-01]])

P4_list = np.array([[ 2.80000000e+01,  6.46065816e-03, 3.77298443e-01],
                    [ 2.77200289e+01,  7.02569522e-03, 3.79001877e-01],
                    [ 2.74372010e+01,  7.46341103e-03, 3.80782417e-01],
                    [ 2.71514272e+01,  7.76094034e-03, 3.82646541e-01],
                    [ 2.80000000e+01,  7.14170759e-03, 3.90867810e-01], #
                    [ 2.68626134e+01,  7.90427284e-03, 3.84601639e-01],
                    [ 2.65706605e+01,  7.87814308e-03, 3.86656253e-01],
                    [ 2.62754638e+01,  7.66590462e-03, 3.88820459e-01],
                    [ 2.77200289e+01,  7.39455750e-03, 4.06809765e-01],
                    [ 2.74372010e+01,  6.15446472e-03, 4.08069652e-01]])

# P5_list = np.array([[2.80000000e+01, 6.67888973e-03, 3.62247942e-01],
#                     [2.74372010e+01, 6.79599765e-03, 3.61993689e-01],
#                     [2.80000000e+01, 6.46065816e-03, 3.77298443e-01],
#                     [2.68626134e+01, 6.52094497e-03, 3.71322540e-01],
#                     [2.77200289e+01, 7.02569522e-03, 3.79001877e-01],
#                     [2.74372010e+01, 7.46341103e-03, 3.80782417e-01],
#                     [2.62754638e+01, 6.85466310e-03, 3.71435702e-01],
#                     [2.71514272e+01, 7.76094034e-03, 3.82646541e-01],
#                     [2.80000000e+01, 7.14170759e-03, 3.90867810e-01],
#                     [2.68626134e+01, 7.90427284e-03, 3.84601639e-01]])

