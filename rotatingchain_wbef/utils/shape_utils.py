import numpy as np

def get_shape2(data_list_z1ij1, info_list, n_interp=10):
    # y = np.zeros(((info_list['n_steps'])*2,3))
    y = np.zeros((n_interp*2,3))

    rho_1=data_list_z1ij1[3]
    sbar=data_list_z1ij1[2]
    l_step = info_list['l']/(len(sbar)-1)
    rho = np.zeros(len(sbar))
    z_1 = np.zeros(len(sbar))
    z = np.zeros(len(sbar))
    z_1 = np.sqrt(1 - rho_1**2)
    for i in range(1,len(sbar)):
        rho[i] = rho[i-1] + rho_1[i-1]*l_step
        z[i] = z[i-1] + z_1[i-1]*l_step
    # rho -= rho[-1]
    z -= z[-1]
    # remove first and last node
    pos_arr = np.transpose(np.vstack((rho[:],np.zeros(len(z[:])),z[:])))
    y[::2] = pos_arr[
        (np.linspace(0,info_list['n_steps']-1,n_interp)+0.5).astype(int)
    ]
    return y

def get_shape2c(data_list_a, info_list, n_interp=None):
    # y = np.zeros(((info_list['n_steps'])*2,3))
    # print(f"len_data = {len(data_list_a_ij[2])}")
    # input()
    # print(f"vrot = {data_list_a[0]}")
    # print(f"rho10 = {data_list_a[1]}")
    # print(f"fbar = {data_list_a[2]}")
    # print(f"u_1 = {data_list_a[3]}")
    # print(f"sbar = {data_list_a[4]}")
    # print(f"rho_1 = {data_list_a[5]}")
    # print(f"rbar = {data_list_a[6]}")
    g = info_list['g']
    l = info_list['l']
    n_steps = info_list['n_steps']
    v_rot = data_list_a[0]

    if n_interp is None:
        n_interp = info_list['n_steps']
    elif (
        (n_steps-1)
        / (n_interp-1)
    ) % 1 > 0:
        print('Error split not even! --')
        print(
            f"Recommended n_interp = (factors of {(info_list['n_steps']-1)})+1"
        )
        print(f"OR recommended n_steps = (multiples of {n_interp-1})+1")

    y = np.zeros((n_interp*2,3))

    rho_1 = data_list_a[5]
    sbar = data_list_a[4]
    s = sbar * g / (v_rot*v_rot)

    rho = np.zeros(len(sbar))
    z_1 = np.zeros(len(sbar))
    z = np.zeros(len(sbar))
    z_1 = np.sqrt(1 - rho_1**2)
    for k in range(1,len(sbar)):
        actual_step = s[k] - s[k-1]
        # print(actual_step - 0.74/11)
        rho[k] = rho[k-1] + rho_1[k-1]*actual_step
        z[k] = z[k-1] + z_1[k-1]*actual_step
    #     if k < 4:
    #         print(f"k = {k} ============")
    #         print(f"rho_1 = {rho_1[k-1]}")
    #         print(f"rho = {rho[k-1]}")
    #         print(f"z = {z[k-1]}")
    # input()

    pos_arr = np.transpose(np.vstack((rho[:],np.zeros(len(z[:])),z[:])))
    # pos_arr = pos_arr[::-1]
    
    y[::2] = pos_arr[
        (np.linspace(0,n_steps-1,n_interp)+0.5).astype(int)
    ]
    r_val = -g*data_list_a[6]/ v_rot**2
    # print(r_val-y[-2,0])
    if abs(r_val-y[-2,0]) > 1e-4:
        if (r_val == 0) or (abs(r_val-y[-2,0])/r_val > 1e-4):
            print(f"fbar = {data_list_a[2]}")
            print(f"rbar = {data_list_a[6]}")
            print(v_rot)
            # print(f"rho0 = {}")
            intg_errcheck(y, r_val)
    return y

def get_shape3(data_list_ij1, info_list, n_interp=10):
    # y = np.zeros(((info_list['n_steps'])*2,3))
    y = np.zeros((n_interp*2,3))
    rho_1 = data_list_ij1[3]
    sbar = data_list_ij1[2]
    rho0 = -info_list['g']*data_list_ij1[0] / info_list['v_rot']**2
    
    l_step = info_list['l']/(len(sbar)-1)
    rho = np.zeros(len(sbar))
    rho[0] = rho0
    z_1 = np.zeros(len(sbar))
    z = np.zeros(len(sbar))
    z_1 = np.sqrt(1 - rho_1**2)
    for k in range(1,len(sbar)):
        rho[k] = rho[k-1] + rho_1[k-1]*l_step
        z[k] = z[k-1] + z_1[k-1]*l_step
    # rho -= rho[-1]
    z -= z[-1]
    # remove first and last node
    pos_arr = np.transpose(np.vstack((rho[:],np.zeros(len(z[:])),z[:])))
    # pos_arr = pos_arr[::-1]
    
    # print(f"pos_arr = {pos_arr}")
    # print(f"info_len = {info_list['n_steps']}")
    # print(f"sbar_len = {len(sbar)}")
    
    y[::2] = pos_arr[
        (np.linspace(0,info_list['n_steps']-1,n_interp)+0.5).astype(int)
    ]
    return y

def get_shape3b(data_list_a_ij, info_list, n_interp=None):
    # y = np.zeros(((info_list['n_steps'])*2,3))
    # print(f"len_data = {len(data_list_a_ij[2])}")
    # input()
    g = info_list['g']
    l = info_list['l']
    n_steps = info_list['n_steps']
    v_rot = data_list_a_ij[4]

    if n_interp is None:
        n_interp = len(data_list_a_ij[2])
    elif (
        (n_steps-1)
        / (n_interp-1)
    ) % 1 > 0:
        print('Error split not even! --')
        print(
            f"Recommended n_interp = (factors of {(info_list['n_steps']-1)})+1"
        )
        print(f"OR recommended n_steps = (multiples of {n_interp-1})+1")

    y = np.zeros((n_interp*2,3))

    rho_1 = data_list_a_ij[3]
    sbar = data_list_a_ij[2]
    s = sbar * g / (v_rot*v_rot)
    rho0 = - (
        g*data_list_a_ij[0] 
        / v_rot**2
    )
    # spring_const = info_list['k']

    # l_step = l/(len(sbar)-1)
    rho = np.zeros(len(sbar))
    rho[0] = rho0
    z_1 = np.zeros(len(sbar))
    z = np.zeros(len(sbar))
    z_1 = np.sqrt(1 - rho_1**2)
    # m = info_list['mu'] * l / n_steps
    # u_1_test = np.zeros(len(sbar))
    # u_1_test[0] = data_list_a_ij[0]
    # u_1 = data_list_a_ij[1]
    # r = np.zeros(len(sbar))
    # r[0] = rho0
    # fnet_x = 0.
    # fnet = 0.
    for k in range(1,len(sbar)):
        # f_grav_cent = np.array([
        #     (v_rot*v_rot*m*r[k-1]),
        #     0.,
        #     -(m*g),
        # ])
        # f_grav_cent_x = (v_rot*v_rot*m*r[k-1])
        # fnet += f_grav_cent
        # fnet_x += f_grav_cent_x
        # fnet_norm = np.linalg.norm(fnet)
        # actual_step = l_step + fnet_norm/spring_const
        # r[k] = r[k-1] - fnet_x/fnet_norm*actual_step
        
        actual_step = s[k] - s[k-1]
        # print(f_grav_cent)
        # print(f_grav_cent2)
        rho[k] = rho[k-1] + rho_1[k-1]*actual_step
        z[k] = z[k-1] + z_1[k-1]*actual_step
        # print(k)
        # print(rho_1[k-1]**2 + z_1[k-1]**2)
        # print(rho[k])
        # print(z[k])
        # print(actual_step-l_step)
        # # u part
        # u_1_test[k] = u_1_test[k-1] - rho_1[k-1]*actual_step*v_rot*v_rot/g
        # print(k)
        # print(v_rot)
        # print(m)
        # print(r[k-1])
        # print(fnet)
        # print(f"{v_rot}^2 * {m} * {r[k-1]} = {f_grav_cent3_x}")
        # print(f"{r[k-1]} - {fnet_x} / {np.linalg.norm(fnet)} * {l_step} = {r[k]}")
    # print(r)
    # print(rho)
    # input()
        # # print(f_grav_cent2)
        # # print(v_rot)
        # # # print(actual_step)
        # # print('ja')
    # print(u_1)
    # # print(u_1_test)
    # print(-rho*v_rot*v_rot/g)
    # print(u_1)
    # input('yoyo')

    # rho -= rho[-1]
    # z -= z[-1]
    # remove first and last node
    pos_arr = np.transpose(np.vstack((rho[:],np.zeros(len(z[:])),z[:])))
    # pos_arr = pos_arr[::-1]
    
    # print(f"r = {r}")
    # if pos_arr[-1,0] < 0:
    #     pos_arr[:,0] = - pos_arr[:,0]
    # print(f"pos_arr = {pos_arr}")
    # norm_sec = pos_arr[1:] - pos_arr[:-1]
    # norm_sec = np.linalg.norm(norm_sec, axis=1) - 0.074
    # print(f"norm_sec = {norm_sec}")
    # print(f"vrot = {v_rot}")
    # print(f"lbar = {info_list['l']*v_rot**2/g}")
    # print(f"a = {data_list_a_ij[0]}")
    # print(f"rho_0 = {rho0}")
    # print(f"rho_0_from_a = {-data_list_a_ij[0]/(v_rot*v_rot/g)}")
    # print(f"info_len = {info_list['n_steps']}")
    # print(f"sbar_len = {len(sbar)}")
    # input()

    y[::2] = pos_arr[
        (np.linspace(0,n_steps-1,n_interp)+0.5).astype(int)
    ]
    # input((np.linspace(0,info_list['n_steps']-1,n_interp)+0.5).astype(int))
    r_val = -g*data_list_a_ij[5]/ v_rot**2
    # print(r_val-y[-2,0])
    if abs(r_val-y[-2,0]) > 1e-4:
        if (r_val == 0) or (abs(r_val-y[-2,0])/r_val > 1e-4):
            print(data_list_a_ij[0])
            print(data_list_a_ij[5])
            print(f"rho0 = {rho0}")
            print(v_rot)
            print(rho0)
            intg_errcheck(y, r_val)
    return y

def intg_errcheck(y, r_val):
    # print(data_list_a_ij[0])
    # print(data_list_a_ij[5])
    # print(f"rho0 = {rho0}")
    # print(data_list_a_ij[4])
    # print(rho0)
    print(y)
    print(r_val)
    print(abs(r_val-y[-2,0]))
    print(abs(r_val-y[-2,0])/r_val)
    print('BIG difference in r_val: calculation inaccurate!')
    input()

def check_mode(y):
    x_arr = y[::2][:,0]
    i_mode = 0
    for i in range(1,len(x_arr)):
        if x_arr[i]*x_arr[i-1] < 0:
            i_mode += 1
    return i_mode

def check_mode2(y):
    x_arr = y[::2][:,0]
    i_mode = 0
    for i in range(len(x_arr)-2):
        diff1 = x_arr[i+1] - x_arr[i]
        diff2 = x_arr[i+2] - x_arr[i+1]
        if diff1*diff2 < 0:
            i_mode += 1
    return i_mode