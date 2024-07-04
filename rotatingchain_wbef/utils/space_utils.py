import numpy as np
from scipy.optimize import curve_fit

def get_ctrl_truth(ctrl_data, ctrl_lim):
    # ctrl_truth from limits
    ctrl_truth = np.ones(ctrl_data.shape[:-1])
    for i in range(ctrl_data.shape[-1]):
        ctrl_truth[ctrl_data[:,:,:,i]<ctrl_lim[i,0]] = 0
        ctrl_truth[ctrl_data[:,:,:,i]>ctrl_lim[i,1]] = 0

    # extra_condition(s)
    ctrl_truth = np.logical_and(extra_cond(ctrl_data),ctrl_truth)
    
    # vr_pts = np.array([
    #     [10.,0.025],
    #     [20.,0.015],
    #     [25.,0.010],
    # ])
    # def fun(x, a, b, c):
    #     return a * np.log(np.abs(b * x))+ c
    # coef,_ = curve_fit(fun, vr_pts[:,0], vr_pts[:,1])
    # input(coef)

    # for i in range(ctrl_data.shape[0]):
    #     for j in range(ctrl_data.shape[1]):
    #         for k in range(ctrl_data.shape[2]):
    #             if ctrl_data[i,j,k,1]<ctrl_lim[1,1]:
    #                 if ctrl_data[i,j,k,1]>0.08:
    #                     print('new_data+++++++++++++++++++++++++++++++++')
    #                     print(f"vrot = {ctrl_data[i,j,k,0]}")
    #                     print(f"r = {ctrl_data[i,j,k,1]}")
    #                     print()
    #                     # print(f"z0 = {ctrl_data[i,j,k,2]}")
    # input()

    # function with r and z0
    # # line formula
    # p1 = np.array([25.,0.010])
    # p2 = np.array([20.,0.015])
    # line_grad = (p1[1]-p2[1])/(p1[0]-p2[0])
    # line_intrcpt = p1[1] - line_grad*p1[0]
    # ctrl_truth[
    #     np.abs(ctrl_data[:,:,:,1]) > ctrl_data[:,:,:,0]*line_grad+line_intrcpt
    # ] = 1

    # # ln formula
    # ctrl_truth[
    #     np.abs(ctrl_data[:,:,:,1])
    #     > -coef[0]*np.log(coef[1]*ctrl_data[:,:,:,0]) + coef[2]
    # ] = 0
    
    # inverse x formula
    ctrl_truth[
        (ctrl_data[:,:,:,0] > 5)
        &
        (np.abs(ctrl_data[:,:,:,1])
        > 0.35/(ctrl_data[:,:,:,0]-5.) - 0.008)
        # > 0.35/(ctrl_data[:,:,:,0]-5.) + 0.008)
    ] = 0
    return ctrl_truth

def get_eig_truth(max_eig_3d, eig_lim):
    eig_truth = np.zeros_like(max_eig_3d)
    eig_truth[max_eig_3d>eig_lim] = 0
    eig_truth[max_eig_3d<=eig_lim] = 1
    return eig_truth

def get_cmode(mode_data):
    data_shape = mode_data.shape
    color_vals = np.zeros((
        data_shape[0],
        data_shape[1],
        data_shape[2],
        4
    ))
    alp_val = 0.2
    darken_val = 0.8
    color_vals[mode_data==0] = np.array([1.,0.,0.,alp_val])
    color_vals[mode_data==1] = np.array([1.,0.5,0.,alp_val])
    color_vals[mode_data==2] = np.array([1.,1.,0.,alp_val])
    color_vals[mode_data==3] = np.array([0.,1.,0.,alp_val])
    color_vals[mode_data==4] = np.array([0.,1.,1.,alp_val])
    color_vals[mode_data==5] = np.array([0.,0.,1.,alp_val])
    color_vals[:,:,:,:3] = color_vals[:,:,:,:3] * darken_val
    return color_vals

# def get_ctrl_truth2(ctrl_data,z0_lim):
#     # ctrl_truth from function
#     ctrl_truth = np.zeros(ctrl_data.shape[:-1])
#     vrot = ctrl_data[:,:,:,0]
#     r = ctrl_data[:,:,:,1]
#     z0 = ctrl_data[:,:,:,2]
#     ctrl_truth[z0<z0_lim[0]] = 1
#     ctrl_truth[z0>z0_lim[1]] = 1

#     # function with r and z0
#     # line formula
#     line_grad = (1.5-1.0)/(25.-20.)
#     line_intrcpt = -1.
#     ctrl_truth[
#         r > vrot*line_grad+line_intrcpt
#     ] = 1
#     return ctrl_truth

def expand_freespace(narrow_freespace, diag_expand=0):
    n_configs = 3
    pc = np.array((3**n_configs,3))
    pc = np.array([ # perm_cube
        [0,0,0],

        # singles
        [1,0,0],
        [0,1,0],
        [0,0,1],

        [-1,0,0],
        [0,-1,0],
        [0,0,-1],

        # doubles
        [0,1,1],
        [1,0,1],
        [1,1,0],
        
        [0,-1,-1],
        [-1,0,-1],
        [-1,-1,0],

        [0,1,-1],
        [1,0,-1],
        [1,-1,0],

        [0,-1,1],
        [-1,0,1],
        [-1,1,0],

        # triples
        [1,1,1],
        [-1,-1,-1],
        [-1,1,1],
        [1,-1,1],
        [1,1,-1],
        [1,-1,-1],
        [-1,1,-1],
        [-1,-1,1],
    ])
    edge_cond = [
        [i for i, x in enumerate(pc[:,0]) if x == -1],   # x == 0
        [i for i, x in enumerate(pc[:,0]) if x == 1],  # x == lenx
        [i for i, x in enumerate(pc[:,1]) if x == -1],   # y == 0
        [i for i, x in enumerate(pc[:,1]) if x == 1],  # y == leny
        [i for i, x in enumerate(pc[:,2]) if x == -1],   # z == 0
        [i for i, x in enumerate(pc[:,2]) if x == 1],  # z == lenz
    ]

    wide_freespace = np.zeros_like(narrow_freespace)
    len_xyz = narrow_freespace.shape
    for i in range(0,len_xyz[0]):
        edge_x = []
        if i == 0:
            edge_x = edge_cond[0]
        if i == len_xyz[0]-1:
            edge_x = edge_cond[1]

        for j in range(0,len_xyz[1]):
            edge_y = []
            if j == 0:
                edge_y = edge_cond[2]
            if j == len_xyz[1]-1:
                edge_y = edge_cond[3]
            for k in range(0,len_xyz[2]):
                if narrow_freespace[i,j,k]:
                    edge_z = []
                    if k == 0:
                        edge_z = edge_cond[4]
                    if k == len_xyz[2]-1:
                        edge_z = edge_cond[5]
                    edge_restrict = []
                    edge_restrict = np.unique(np.concatenate((edge_x,edge_y,edge_z)))

                    for pcid in range(7):
                        if pcid in edge_restrict:
                            continue
                        # expand direct adjacent
                        wide_freespace[i+pc[pcid,0],j+pc[pcid,1],k+pc[pcid,2]] = 1
                    if diag_expand > 0:
                        for pcid in range(7,19):
                            if pcid in edge_restrict:
                                continue
                            # expand further adjacent
                            wide_freespace[i+pc[pcid,0],j+pc[pcid,1],k+pc[pcid,2]] = 1
                    if diag_expand > 1:
                        for pcid in range(19,27):
                            if pcid in edge_restrict:
                                continue
                            # expand furthest diagonal
                            wide_freespace[i+pc[pcid,0],j+pc[pcid,1],k+pc[pcid,2]] = 1
    return wide_freespace

def interp_path(path_pt_few, n_pts=20):
    # interpolate path to make longer
    path_pt_many = []
    n_sec = len(path_pt_few)-1
    len_sec = np.zeros(n_sec)
    for i in range(n_sec):
        len_sec[i] = np.linalg.norm(
            path_pt_few[i] - path_pt_few[i+1]
        )
    len_total = np.sum(len_sec)
    len_smallsec = len_total/n_pts
    id_og = []
    for i in range(n_sec):
        n_smallsec = int(len_sec[i] / len_smallsec) + 1
        id_og.append(len(path_pt_many))
        for j in range(n_smallsec):
            u = j/n_smallsec
            path_pt_many.append(
                path_pt_few[i]*(1-u) + path_pt_few[i+1]*u
            )
    id_og.append(len(path_pt_many))
    path_pt_many.append(path_pt_few[-1])
    path_pt_many = np.array(path_pt_many)
    return path_pt_many, id_og

def nearest_id_from_raw(xyz_ax, xyz_pt):
    xyz_id = np.zeros(3, dtype="int")
    for id3 in range(3):
        xyz_id[id3] = np.searchsorted(xyz_ax[id3], xyz_pt[id3])
        if xyz_id[id3] >= len(xyz_ax[id3]):
            xyz_id[id3] = len(xyz_ax[id3])-1
        elif xyz_id[id3] > 0:
            d1 = np.abs(xyz_pt[id3] - xyz_ax[id3][xyz_id[id3]-1])
            d2 = np.abs(xyz_pt[id3] - xyz_ax[id3][xyz_id[id3]])
            if d2 > d1:	# nearer to the point before
                xyz_id[id3] -= 1
    return xyz_id

def remove_badctrl(ctrlpath, fp1, id_og, ctrl_lim):
    remove_ids = []
    # do not remove start and end ctrl points
    for i in range(1,len(ctrlpath)-1):
        for j in range(len(ctrl_lim)):
            if (
                (ctrlpath[i][j]<ctrl_lim[j,0])
                or (ctrlpath[i][j]>ctrl_lim[j,1])
            ):
                remove_ids.append(i)
                for i in range(len(id_og)):
                    if id_og[i] > i:
                        id_og[i] -= 1
                if i in id_og:
                    del id_og[id_og.index(i)]
        if extra_cond(ctrlpath[i]):
            remove_ids.append(i)
            for i in range(len(id_og)):
                if id_og[i] > i:
                    id_og[i] -= 1
            if i in id_og:
                del id_og[id_og.index(i)]
    remove_ids = np.unique(remove_ids)
        
    for i in range(1,len(remove_ids)+1):
        del ctrlpath[remove_ids[-i]]
        fp1 = np.delete(fp1,remove_ids[-i],0)
    return ctrlpath, fp1, id_og

def extra_cond(ctrl_data):
    # A: if radius is less than 5mm and z0 is <0.48m
    # print(ctrl_pt)
    ctrl_data_np = np.array(ctrl_data)
    if len(ctrl_data_np.shape)<2:
        if (
            np.abs(ctrl_data_np[1])<0.005
            and np.abs(ctrl_data_np[2])<0.48
            and np.abs(ctrl_data_np[0])<17.
        ):
            return True
        return False
    else:
        ectrl_truth1 = np.ones(ctrl_data_np.shape[:-1])
        ectrl_truth2 = np.ones(ctrl_data_np.shape[:-1])
        ectrl_truth1[ctrl_data_np[:,:,:,1]<0.005] = 0
        ectrl_truth2[ctrl_data_np[:,:,:,2]<0.48] = 0
        return np.logical_or(ectrl_truth1,ectrl_truth2)

def find_xyz_from_ctrl(ctrl_xyz, control_data, xyz_axes, free_space, free_truth=None, min_cnt=2):
    (
        raw_data,
        norm_data,
    ) = normalize_ctrlaxes(control_data[free_space])
    control_data_norm = normalize_data(control_data,raw_data,norm_data)
    ctrl_xyz_norm = normalize_points(np.array(ctrl_xyz),raw_data,norm_data)
    idmin_list = []
    ctrl_discr = []
    xyz_discr = []
    for i in range(len(ctrl_xyz)):
        id_min2 = []
        diff_ctrl = np.linalg.norm(control_data_norm - ctrl_xyz_norm[i],axis=3)
        while len(id_min2) < min_cnt:
            id_arrmin = np.unravel_index(
                np.argmin(diff_ctrl),
                diff_ctrl.shape
            )
            if free_truth is not None:
                if free_truth[id_arrmin]:
                    id_min2.append(id_arrmin)
                diff_ctrl[id_arrmin] = 100.
        idmin_list.append(id_min2)
        ctrl_discr.append([control_data[id_min2[ididm]] for ididm in range(min_cnt)])
        xyz_discr.append([
            [xyz_axes[0][id_min2[ididm][0]],xyz_axes[1][id_min2[ididm][1]],xyz_axes[2][id_min2[ididm][2]]]
            for ididm in range(min_cnt)
        ])
    return idmin_list,ctrl_discr,xyz_discr

def normalize_ctrlaxes(ctrl_data):
    # ctrl_axes in the shape [3][n_pts]
    # output new_axis in the shape [3][n_pts]
    # prep raw values
    ctrl_axes = [
        ctrl_data[:,0],
        ctrl_data[:,1],
        ctrl_data[:,2],
    ]
    min_max_raw = np.array([
        [np.min(a) for a in ctrl_axes],
        [np.max(b) for b in ctrl_axes]
    ]).T
    input(min_max_raw)
    mean_raw = np.array(
        [np.mean(c) for c in ctrl_axes]
    )[np.newaxis].T
    maxdev_raw = np.max(np.abs(
        min_max_raw-mean_raw), axis=1
    )[np.newaxis].T

    # prep norm values
    min_max_norm = np.array([[0],[100]])
    mean_norm = np.mean(min_max_norm)
    maxdev_norm = min_max_norm[1][0]-mean_norm
    # # actual norm
    # new_axis = ([
    #     (ctrl_axes[d_id] - mean_raw[d_id])
    #     / maxdev_raw[d_id] * maxdev_norm + mean_norm
    #     for d_id in range(3)
    # ])
    return (
        # new_axis,
        [min_max_raw, mean_raw, maxdev_raw],
        [min_max_norm, mean_norm, maxdev_norm]
    )

def normalize_data(ctrl_data_raw, raw_data, norm_data):
    og_shape = ctrl_data_raw.shape
    total_pts = 1.
    for i in range(len(og_shape)-1):
        total_pts *= og_shape[i]
    total_pts = int(total_pts)
    ctrl_data_raw = ctrl_data_raw.reshape(total_pts,og_shape[-1])
    ctrl_data_norm = normalize_points(ctrl_data_raw, raw_data, norm_data)
    ctrl_data_norm = ctrl_data_norm.reshape(og_shape)
    return ctrl_data_norm

def normalize_points(xyz_pts, raw_data, norm_data):
    [min_max_raw, mean_raw, maxdev_raw] = raw_data
    [min_max_norm, mean_norm, maxdev_norm] = norm_data
    # xyz_pts in the shape (n_pts,3)
    # output new_pts in the shape (n_pts,3)
    if len(xyz_pts.shape) > 1:
        xyz_raw = xyz_pts.T
    else:	# when only one point given, have to add axis to transpose
        xyz_raw = xyz_pts[np.newaxis].T

    new_pts = ((
        (xyz_raw - mean_raw) / maxdev_raw
    ) * maxdev_norm + mean_norm).T

    if len(xyz_pts.shape) == 1:
        new_pts = new_pts[0]
    return new_pts

def choose_within_area(xyz_axes, xyz_lim, max_eig_3d, free_space, n_top=10, mode_stuff=None):
    # returns top 10 (most stable, lowest eigval) points within region
    if mode_stuff is not None:
        mode_data, mode_req = mode_stuff

    xyz_id_lim = np.zeros((3,2),dtype=np.uint8)
    for i in range(3):
        if (xyz_axes[i][0]>xyz_lim[i][1] or xyz_axes[i][-1]<xyz_lim[i][0]):
            print("Error: limits outside xyz_axes range.")
            input()
        for j in range(len(xyz_axes[i])):
            if xyz_axes[i][j] >= xyz_lim[i][0]:
                xyz_id_lim[i,0] = j
                break
        for j in range(len(xyz_axes[i])):
            if xyz_axes[i][-j-1] <= xyz_lim[i][1]:
                xyz_id_lim[i,1] = len(xyz_axes[i]) - j
                break
    good_ids = []
    interest_area = max_eig_3d[
        xyz_id_lim[0][0]:xyz_id_lim[0][1],
        xyz_id_lim[1][0]:xyz_id_lim[1][1],
        xyz_id_lim[2][0]:xyz_id_lim[2][1],
    ]
    shape_xyz = interest_area.shape
    n_data = 1.
    for i in range(len(shape_xyz)):
        n_data *= shape_xyz[i]
    i_while = 0.
    while len(good_ids) < n_top:
        test_goodid = np.unravel_index(
            np.argmin(interest_area),
            interest_area.shape
        )
        # input(free_space[test_goodid])
        if free_space[
            test_goodid[0]+xyz_id_lim[0,0],
            test_goodid[1]+xyz_id_lim[1,0],
            test_goodid[2]+xyz_id_lim[2,0],
        ]:
            if (mode_stuff is not None):
                # print(mode_data[
                #     test_goodid[0]+xyz_id_lim[0,0],
                #     test_goodid[1]+xyz_id_lim[1,0],
                #     test_goodid[2]+xyz_id_lim[2,0],
                # ])
                if (mode_data[
                    test_goodid[0]+xyz_id_lim[0,0],
                    test_goodid[1]+xyz_id_lim[1,0],
                    test_goodid[2]+xyz_id_lim[2,0],
                ] in mode_req):
                    good_ids.append(test_goodid)
            else:
                good_ids.append(test_goodid)
        interest_area[test_goodid] = 100.
        i_while += 1
        if i_while >= n_data: break
    good_ids = np.array(good_ids) + xyz_id_lim[:,0]
    return good_ids