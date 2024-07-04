import numpy as np
import rotatingchain_wbef.utils.plotter as plter
import os
import pickle


# pickle stuff
s2_picklepath = 'frc_soln_3d_c.pickle'
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
k = info_list['k']

vrot_list = info_list['vrot_list']
fbar_list = info_list['fbar_list']
rho10_list = info_list['rho10_list']
lbar_list = np.array(vrot_list)**2*l/g
len_x = len(vrot_list)
len_y = len(fbar_list)
len_z = len(rho10_list)
# for z in range(len(data_list)):
#     print(f"rho_val = {data_list[z][0]} =============================")
#     for i in range(len(data_list[z][1])):
#         print(f"r_val = {data_list[z][1][i][0]} =============================")
#         for j in range(len(data_list[z][1][i])-1):
#             # print(f"j_sol = {j}")
#             print(f"data_fval = {data_list[z][1][i][j+1][0]}")
            # print(f"r = {r_list[i]}")
            # print(f"f_guess init = {fbarg_init_list[i]}")
            # print(f"f_guess final = {info['fbar']}")
            # plter.plot_u_1_sbar(u_1=info['u_1'], sbar=info['sbar'])
            # plter.plot_chainshape(
            #     rho_1=info['rho_1'],
            #     sbar=info['sbar'],
            #     l=info['l']
            # )
            # plter.plot_chainshape2(
            #     u_1=info['u_1'],
            #     v_rot=info['v_rot'],
            #     g=info['g'],
            #     sbar=info['sbar'],
            #     l=info['l']
            # )
            # print('Troubleshoot:')
            # # print(f's = {s_step}')
            # print(f'u_1 = {info["u_1"]}')
            # print(f'rho = {info["rho"]}')
            # print(f'rho_1 = {info["rho_1"]}')
# plter.plot_overall_b(
#     data_list_a=data_list_a,
#     info_list=info_list,
#     max_plot=10
# )

plter.plot_3d_fbarrbarrho10d(
    [fbar_list,rho10_list],
    data_list=data_list_a,
    mode_eval=info_list
)
# # plter.plot_3d_z0rbarrho10(data_list=data_list, l=l)
# plter.check_rz_uniqueness(
    # [fbar_list,rho10_list],
    # data_list=data_list_a,
    # info_list=info_list
# )
