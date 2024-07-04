import sys
import numpy as np
from rotatingchain_wbef.algos.shootingc_cpp.Shootc import Shootc
import rotatingchain_wbef.utils.check_approx as check_approx
import os
import pickle
from time import time

# input data_plot_type = ['3d', 'stab', 'c']
# for 3d data plot, stability plot, and custom plots, respectively.
data_plot_type = sys.argv[1]

# pickle stuff
if data_plot_type == '3d':
    s2_picklepath = 'frc_soln_3d_c.pickle'
elif data_plot_type == 'stab':
    s2_picklepath = 'frc_soln_c.pickle'
elif data_plot_type == 'c':
    cntr_dattype = 0
    cntr_dt_str = ['eigval', 'rbar', 'zbar'] # low amplitude regime
    s2_picklepath = 'frc_soln_c_split5_'+cntr_dt_str[cntr_dattype]+'.pickle'
elif data_plot_type == 'path':
    s2_picklepath = 'frc_soln_c_path.pickle'

s2_picklepath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/rc_data/" + s2_picklepath
)


k_spring = 8490790.955648089
l = 0.5
mu = 0.0554054054054054
# mu = 0.02801961015363869   # does not affect fbar - affects F
g = 9.8
if data_plot_type == '3d':
    n_steps = 30
    # vrot_list = np.sqrt(np.linspace(32.*g/l, 32.*g/l, 1))
    vrot_list = [8.7*np.pi]
    fbar_list = np.linspace(0.,3., 100)
    fbar_list = np.append(fbar_list,np.linspace(3.,13., 20))
    fbar_list = np.append(fbar_list,np.linspace(13.,22., 200))
    fbar_list = np.append(fbar_list,np.linspace(22.,80., 50))
    fbar_list = np.append(fbar_list,np.linspace(80.,130., 200))
    fbar_list = np.append(fbar_list,np.linspace(130.,200., 50))
    # fbar_list = np.linspace(100.,100., 1)
    # rho10_list = np.sin(np.linspace(0.05,0.95, 50)*np.pi/2)
    # rho10_list = np.linspace(0.01,0.999, 100)
    rho10_list = np.linspace(0.2,0.8, 100)
elif data_plot_type == 'stab':
    n_steps = 12
    # for detailed:
    vrot_list = np.sqrt(np.linspace(1.*g/l, 40.*g/l, 40))  # x
    fbar_list = np.linspace(0.,21., 31)                  # z
    # rho10_list = np.sin(np.linspace(0.01,0.999, 20)*np.pi/2) # y
    rho10_list = np.linspace(0.01,0.999, 30)

    vrot_list = np.sqrt(np.linspace(1.*g/l, 40.*g/l, 50))  # x
    fbar_list = np.linspace(0.,21., 50)                  # z
    # rho10_list = np.sin(np.linspace(0.01,0.999, 20)*np.pi/2) # y
    rho10_list = np.linspace(0.01,0.999, 50)

elif data_plot_type == 'c':
    # [
    #   0 - eigval,
    #   1 - rbar,
    #   2 - z0bar,
    # ]
    n_steps = 12
    xyz_len = [50,50,50]
    if cntr_dattype == 0:
        xyz_len[1] = 3
        fbar_lim = [0.,7.]
    else:
        xyz_len[0] = 5
        fbar_lim = [-200,200.]

    vrot_list = np.sqrt(np.linspace(0.1*g/l, 40.*g/l, xyz_len[0]))  # x
    # fbar_list = np.linspace(-255,255., 11)                  # z
    fbar_list = np.linspace(fbar_lim[0],fbar_lim[1], xyz_len[1])                    # z
    # rho10_list = np.sin(np.linspace(0.05,0.95, xyz_len[2])*np.pi/2) # y
    rho10_list = np.linspace(0.01,0.999,xyz_len[2])
elif data_plot_type == 'path':
    top_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_picklepath = 'plan_path.pickle'
    path_picklepath = os.path.join(
        top_path,
        "data/rc_data/" + path_picklepath
    )
    with open(path_picklepath, 'rb') as f:
            pickle_data = pickle.load(f)
    [fp_list, ctrlpath_list, idog_list] = pickle_data
    print("Pickle Loaded!")
    fp_pts = []
    for i, i_plan in enumerate(fp_list):
        print(idog_list[i])
        print(len(i_plan))
        for idog in idog_list[i]:
            fp_pts.append(i_plan[idog])
    fp_pts = np.array(fp_pts)
    vrot_list = fp_pts[:,0]
    fbar_list = fp_pts[:,1]
    rho10_list = fp_pts[:,2]
else:
    input('Please input appropriate data_plot_type in python script!')

# vrot_list = [32.98672286269283]
# rho10_list = [0.21814324139654256]
# fbar_list = [9.763702392678109]

sr_latter = 1.0 # split ratio latter

## print all lists
print(f"vrot = {vrot_list}")
print(f"fbarg = {fbar_list}")
print(f"rho10 = {rho10_list}")
input('Press "Enter" to continue...')

info_list = dict(
    l=l,
    mu=mu,
    g=g,
    n_steps=n_steps,
    k=k_spring,
    vrot_list=vrot_list,    # i
    fbar_list=fbar_list,    # j
    rho10_list=rho10_list   # k
)

# init data storage
data_list_a = []
# get info
u_1_soln = np.zeros(n_steps)
sbar_soln = np.zeros(n_steps)
rho_1_soln = np.zeros(n_steps)

start_t = time()
for i in range(len(vrot_list)):
    data_list_a.append([])
    
    for j in range(len(fbar_list)):
        data_list_a[i].append([])
        for k in range(len(rho10_list)):
            # input(fbar_list[k])
            u_1_soln = np.zeros(n_steps)
            sbar_soln = np.zeros(n_steps)
            rho_1_soln = np.zeros(n_steps)
            s = Shootc(
                in_v_rot=vrot_list[i],
                in_l=l,
                in_mu=mu,
                in_n_steps=n_steps,
                in_g=g,
                in_k=k_spring,
                in_srl=sr_latter,
                in_error_verbose=False
            )
            s.bang(
                fbg=fbar_list[j],
                in_rho_1_0=rho10_list[k],
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
                vrot=vrot_list[i],
                fbar=fbar_list[j],
                rho10=rho10_list[k],
                rbar=rbar_soln,
                k=k_spring
            )

            data_list_a[i][j].append([
                info['vrot'],
                info['rho10'],
                info['fbar'],
                info['u_1'],
                info['sbar'],
                info['rho_1'],
                info['rbar']
            ])
        # print(f"u_1 = {info['u_1'][3]}")
        
        print(f"rho10 = {info['rho10']}")
        print(f"fbar = {info['fbar']}")
        print(f"rbar = {rbar_soln}")
        # print(f"u_1 = {u_1_soln}")
        # print(f"u_1(2) = {data_list_a[i][j][k][3]}")
        print(f"""
                total solved = 
                {i*len(fbar_list)*len(rho10_list)+j*len(rho10_list)+k+1}
                / {len(vrot_list)*len(rho10_list)*len(fbar_list)}
        """)
        print(f"time elapsed = {time()-start_t}")
        # if np.isnan(rbar_soln):
        #     input()

pickle_data = [info_list, data_list_a]
with open(s2_picklepath, 'wb') as f:
    pickle.dump(pickle_data,f)
print('Pickle saved!')