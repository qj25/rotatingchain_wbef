import os
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from matplotlib import cm
import numpy as np
from rotatingchain_wbef.utils.shape_utils import get_shape2c

plt.rcParams.update({'pdf.fonttype': 42})   # to prevent type 3 fonts in pdflatex
figsave_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data/figs/"
)

def plot_para(plot_leg=True,ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    # plt.xlabel(r"$\rho$")  # add X-axis label
    # plt.ylabel("$z$")  # add Y-axis label
    plt.grid(linestyle='dotted')
    # plt.title("$z$" + 'against' + r"$\rho$")
    if plot_leg:
        plt.legend(bbox_to_anchor=(3.1, 1.05))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.scatter(
        0.,0.,
        s=30.,
        c='b',
        zorder=2
    )
    ax.set_axisbelow(True)
    # plot rotation axis
    ax.plot([0,0],[0.5,0.0],'--',linewidth=1.5,color='k', alpha=0.5, zorder=0)
    ax.scatter(0,0.5,marker='o',facecolors='none', color='k',zorder=3)
    ax.arrow(0,0.49,0.0,0.02,zorder=3)
    return ax

def plot_u_1_sbar(u_1, sbar):
    plt.plot(sbar, u_1)
    plt.xlabel(r"$\bar s$")  # add X-axis label
    plt.ylabel("$u'$")  # add Y-axis label
    plt.grid()
    plt.title(r"$u'" + 'against' + r"$\bar s$")
    plt.show()

# def plot_chainshape(rho_1, sbar, l):
#     # sbar_step = (sbar[-1] - sbar[0]) / (len(sbar)-1)
#     # l_step = sbar_step * 9.81 / (11.*11.)
#     l_step = l/(len(sbar)-1)
#     rho = np.zeros(len(sbar))
#     z_1 = np.zeros(len(sbar))
#     z = np.zeros(len(sbar))
#     z_1 = np.sqrt(1 - rho_1**2)
#     for i in range(1,len(sbar)):
#         rho[i] = rho[i-1] + rho_1[i-1]*l_step
#         z[i] = z[i-1] + z_1[i-1]*l_step
#     plt.plot(z, rho)
#     plt.xlabel("$z$")  # add Y-axis label
#     plt.xlabel(r"$\rho$")  # add X-axis label
#     plt.grid()
#     plt.title(r"$\rho$" + 'against' + "$z$")

#     plt.show()

# def plot_chainshape2(rho_1, sbar, l):
#     # sbar_step = (sbar[-1] - sbar[0]) / (len(sbar)-1)
#     # l_step = sbar_step * 9.81 / (11.*11.)
#     l_step = l/(len(sbar)-1)
#     rho = np.zeros(len(sbar))
#     z_1 = np.zeros(len(sbar))
#     z = np.zeros(len(sbar))
#     z_1 = np.sqrt(1 - rho_1**2)
#     for i in range(1,len(sbar)):
#         rho[i] = rho[i-1] + rho_1[i-1]*l_step
#         z[i] = z[i-1] + z_1[i-1]*l_step
#     # rho -= rho[-1]
#     z -= z[-1]
#     plt.plot(rho, z)

def plot_chainshape3(data_list_a_ij, info_list):
    # sbar_step = (sbar[-1] - sbar[0]) / (len(sbar)-1)
    # l_step = sbar_step * 9.81 / (11.*11.)
    rho_1 = data_list_a_ij[3]
    sbar = data_list_a_ij[2]
    rho0 = - (
        info_list['g']*data_list_a_ij[0] 
        / data_list_a_ij[4]**2
    )

    l_step = info_list['l']/(len(sbar)-1)
    rho = np.zeros(len(sbar))
    rho[0] = rho0
    z_1 = np.zeros(len(sbar))
    z = np.zeros(len(sbar))
    z_1 = np.sqrt(1 - rho_1**2)
    for k in range(1,len(sbar)):
        rho[k] = rho[k-1] + rho_1[k-1]*l_step
        z[k] = z[k-1] + z_1[k-1]*l_step
        # print(k)
        # print(rho_1[k-1]**2 + z_1[k-1]**2)
        # print(rho[k])
        # print(z[k])
    # rho -= rho[-1]
    z -= z[-1]
    plt.plot(rho, z)

def calc_z0(rho_1, sbar, l):
    l_step = l/(len(sbar)-1)
    z_1 = np.zeros(len(sbar))
    z = np.zeros(len(sbar))
    z_1 = np.sqrt(1 - rho_1**2)
    for i in range(1,len(sbar)):
        z[i] = z[i-1] + z_1[i-1]*l_step
    z -= z[-1]
    return z[0]


# def plot_overall(data_list, l=1.):
#     for z in range(len(data_list)):
#         for i in range(len(data_list[z][1])):
#             for j in range(len(data_list[z][1][i])-1):
#                 plot_chainshape2(
#                     rho_1=data_list[z][1][i][j+1][3],
#                     sbar=data_list[z][1][i][j+1][2],
#                     l=l
#                 )
#                 # if z == 12:
#                     # if i == 0:
#                         # plot_chainshape2(
#                             # rho_1=data_list[z][1][i][j+1][3],
#                             # sbar=data_list[z][1][i][j+1][2],
#                             # l=l
#                         # )
#         # plt.show()
#     plt.xlabel(r"$\rho$")  # add X-axis label
#     plt.ylabel("$z$")  # add Y-axis label
#     plt.grid()
#     plt.title("$z$" + 'against' + r"$\rho$")
#     plt.show()

def plot_overall_2b(data_list_a, info_list):
    for i in range(len(data_list_a)):
        # print(abs(data_list[i][0]))
        # print(len(data_list[i]))
        for j in range(len(data_list_a[i])):
            plot_chainshape3(
                    data_list_a_ij=data_list_a[i][j],
                    info_list=info_list
                )
        plt.xlabel(r"$\rho$")  # add X-axis label
        plt.ylabel("$z$")  # add Y-axis label
        plt.grid()
        plt.title("$z$" + 'against' + r"$\rho$")
        # plt.title("z against rho")
        plt.show()

def plot_specific_shapes(
    data_list_a,
    info_list,
    max_plot=0.2,
    x_lim = 0.17
):
    plot_no = 0
    n_data = len(data_list_a)
    if n_data > 7:
        for i in range(len(data_list_a)):
            plot_shape2c(
                data_list_a[i],
                info_list,
                info_list['n_steps'],
                plot_head=True
            )
            plot_no += 1
            if plot_no % max_plot == 0:
                # print("Graph break ===================")
                ax = plot_para(plot_leg=False)
                ax.set(xlim=(-x_lim,x_lim), ylim=(-0.02, 0.52))
                ax.annotate(
                    chr(65+i),
                    xy=(-0.15,0.45),
                    bbox=dict(facecolor='0.9', edgecolor='none', pad=3.0),
                    size=20
                )
                plt.show()
                plot_no = 0
        if plot_no > 0:
            ax = plot_para(plot_leg=False)
            ax.set(xlim=(-x_lim,x_lim), ylim=(-0.02, 0.52))
            plt.show()
    else:
        fig = plt.figure()
        for i in range(n_data):
            ax = fig.add_subplot(1,n_data,i+1)
            plot_shape2c(
                data_list_a[i],
                info_list,
                info_list['n_steps'],
                plot_head=True,
                ax=ax
            )
            ax = plot_para(plot_leg=False,ax=ax)
            ax.annotate(
                chr(65+i),
                xy=(-0.15,0.45),
                bbox=dict(facecolor='0.9', edgecolor='none', pad=3.0),
                size=20
            )
            ax.set(xlim=(-x_lim,x_lim), ylim=(-0.02, 0.52))
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(16, 9) # set figure's size manually to your full screen (32x18)
        plt.savefig(figsave_path+"path_shapes.pdf",bbox_inches='tight')
        # plt.show()

def plot_overall_b(
    data_list_a,
    info_list,
    plot_truth=None,
    max_plot=0.2,
    max_eig_3d=None
):
    
    # plt.gca().set_aspect('equal', adjustable='box')
    plot_no = 0
    for i in range(len(data_list_a)):
        # print(abs(data_list[i][0]))
        # print(len(data_list[i]))
        for j in range(len(data_list_a[i])):
            for k in range(len(data_list_a[i][j])):
                if plot_truth is not None:
                    if not plot_truth[i,j,k]:
                        continue
                if max_eig_3d is None:
                    leg = [i,j,k]
                else:
                    leg = [i,j,k,max_eig_3d[i,j,k]]
                plot_shape2c(
                    data_list_a[i][j][k],
                    info_list,
                    info_list['n_steps'],
                    leg,
                    plot_head=True
                )
                # print(y)
                # print(rho)
                # input(z)
                plot_no += 1
                if plot_no % max_plot == 0:
                    # print("Graph break ===================")
                    plot_para()
                    plt.show()
                    plot_no = 0
    if plot_no > 0:
        plot_para()
        plt.show()

def plot_select_b(data_list_a, info_list, plot_id):
    for id_3d in plot_id:
        i,j,k = id_3d
        plot_shape2c(
            data_list_a[i][j][k],
            info_list,
            info_list['n_steps'],
            [i,j,k]
        )
    plt.xlabel(r"$\rho$")  # add X-axis label
    plt.ylabel("$z$")  # add Y-axis label
    plt.grid()
    plt.title("$z$" + 'against' + r"$\rho$")
    # plt.title("z against rho")
    plt.show()

def plot_shape2c(data_a,info_list,n_interp,id_3d=None,ln=2.5,plot_head=False,ax=None):
    y = get_shape2c(
        data_a,
        info_list,
        n_interp=n_interp
    )
    rho = y[::2,0]
    z = y[::2,2]
    # print(y)
    # print(rho)
    # input(z)
    if id_3d is not None:
        if len(id_3d) == 3:
            label_id = f"id = {[id_3d[0],id_3d[1],id_3d[2]]}"
        else:
            label_id = f"id = {[id_3d[0],id_3d[1],id_3d[2],id_3d[3]]}"
        if ax is None:
            plt.plot(rho, z, label=label_id,linewidth=ln)
        else:
            ax.plot(rho, z, label=label_id,linewidth=ln)

    else:
        if ax is None:
            plt.plot(rho, z,linewidth=ln,color='teal')
        else:
            ax.plot(rho, z,linewidth=ln,color='teal')

    if plot_head:
        if ax is None:
            plt.plot([rho[0],rho[-1]],[z[0],z[-1]],'o',color='0.8', alpha=0.6)
        else:
            ax.plot([rho[0],rho[-1]],[z[0],z[-1]],'o',color='0.8', alpha=0.6)

def plot_2d_arbar(data_list):
    for i in range(len(data_list)):
        for j in range(len(data_list[i])-1):
            plt.scatter(
                data_list[i][j+1][0],
                abs(data_list[i][0])
            )
            # print(data_list[i][j+1][0])
            # print(data_list[i][0])
    plt.xlabel("$a$")  # add X-axis label
    plt.ylabel(r"$\bar r$")  # add Y-axis label
    plt.grid()
    plt.title(r"$\bar r$ against $a$")
    plt.show()


def plot_3d_fbarrbarrho10(data_list):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt_data = []
    for z in range(len(data_list)):
        plt_data.append([data_list[z][0]])
        fbarrbar = []
        for i in range(len(data_list[z][1])):
            # print(data_list[z][0])
            for j in range(len(data_list[z][1][i])-1):
                # z0 = calc_z0(
                #     rho_1=data_list[z][1][i][j+1][3],
                #     sbar=data_list[z][1][i][j+1][2],
                #     l=1.
                # )
                # if abs(z0+0.99) > 1e-4:
                    # continue
                # if (data_list[z][1][i][j+1][0]) > 5:
                #     continue

                fbarrbar.append(np.array([
                    data_list[z][1][i][j+1][0],
                    abs(data_list[z][1][i][0])
                ]))
        fbarrbar = np.array(fbarrbar)
        if len(fbarrbar) < 1:
            continue
        # print(fbarrbar)
        plt_data[z].append(fbarrbar)
        rho10_arr = np.ones(len(fbarrbar)) * plt_data[z][0]
        ax.scatter(fbarrbar[:,0], fbarrbar[:,1], rho10_arr, marker='o')

    ax.set_xlabel(r"$\bar F$")
    ax.set_ylabel(r"$\bar r$")
    ax.set_zlabel(r"$\rho'_{0}$")
    plt.show()

def plot_3d_fbarrbarrho10b(axes_data_xz, data_list):
    fbar_list = axes_data_xz[0]
    rho10_list = axes_data_xz[1]
    data_list = data_list[0]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt_data = []
    for i in range(len(fbar_list)):
        for j in range(len(rho10_list)):
            # print(data_list[i][j])
            plt_data.append([
                fbar_list[i],
                data_list[i][j][6],
                rho10_list[j]
            ])
    plt_data = np.array(plt_data)
    ax.scatter(plt_data[:,0], np.abs(plt_data[:,1]), plt_data[:,2], marker='o')

    ax.set_xlabel(r"$\bar T$")
    ax.set_ylabel(r"$\bar r$")
    ax.set_zlabel(r"$c$")
    # plt.title(r"$\bar F$" + ' v ' + r"$\bar r$" + ' v ' + r"$\rho'_{0}$")
    plt.show()

def plot_3d_fbarrbarrho10c(axes_data_xz, data_list):
    fbar_list = axes_data_xz[0]
    rho10_list = axes_data_xz[1]
    data_list = data_list[0]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt_data = []
    for i in range(len(fbar_list)):
        fbar_set = []
        for j in range(len(rho10_list)):
            # print(data_list[i][j])
            fbar_set.append([
                fbar_list[i],
                data_list[i][j][6],
                rho10_list[j]
            ])
        plt_data.append(np.array(fbar_set))
    for i in range(len(plt_data)):
        cr_val = i / len(plt_data)
        ax.plot(
            plt_data[i][:,0], np.abs(plt_data[i][:,1]), plt_data[i][:,2],
            color=(cr_val, 0., 0., 1.0)
        )

    # ax.scatter(plt_data[:,0], np.abs(plt_data[:,1]), plt_data[:,2], marker='o')

    ax.set_xlabel(r"$\bar T$")
    ax.set_ylabel(r"$\bar r$")
    ax.set_zlabel(r"$c$")
    # plt.title(r"$\bar F$" + ' v ' + r"$\bar r$" + ' v ' + r"$\rho'_{0}$")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(20, 12) # set figure's size manually to your full screen (32x18)
    plt.savefig(figsave_path+"shoot_3d.pdf",bbox_inches='tight')
    plt.show()

def plot_3d_fbarrbarrho10d(axes_data_xz, data_list, mode_eval=None):
    fbar_list = axes_data_xz[0]
    rho10_list = axes_data_xz[1]
    data_list = data_list[0]
    
    plt_data = []
    for i in range(len(rho10_list)):
        plt_data.append([])
    if mode_eval is not None:
        from rotatingchain.utils.shape_utils import get_shape2c, check_mode, check_mode2
        mode_data = []
        for i in range(len(rho10_list)):
            mode_data.append([])
    for i in range(len(fbar_list)):
        for j in range(len(rho10_list)):
            # print(data_list[i][j])
            if mode_eval is not None:
                y = get_shape2c(data_list[i][j], mode_eval)
                # print(fbar_list[i])
                # plot_shape2c(
                #     data_list[i][j],
                #     mode_eval,
                #     mode_eval['n_steps'],
                # )
                # plt.show()
                mode_data[j].append(check_mode(y))
            plt_data[j].append([
                fbar_list[i],
                data_list[i][j][6],
                rho10_list[j]
            ])
    for i in range(len(rho10_list)):
        plt_data[i] = np.array(plt_data[i])
    alp_val = 1.0
    mode2colors = [
        np.array([1.,0.,0.,alp_val]),
        np.array([1.,0.5,0.,alp_val]),
        np.array([1.,1.,0.,alp_val]),
        np.array([0.,1.,0.,alp_val]),
        np.array([0.,1.,1.,alp_val]),
        np.array([0.,0.,1.,alp_val])
    ]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(len(plt_data)):
        latest_id = 0
        for j in range(1,len(plt_data[i])):
            
            if mode_data[i][j] != mode_data[i][j-1]:
                ax.plot(
                    plt_data[i][latest_id:j+1,0],
                    np.abs(plt_data[i][latest_id:j+1,1]),
                    plt_data[i][latest_id:j+1,2],
                    color=mode2colors[mode_data[i][j-1]]
                )
                latest_id = j
        ax.plot(
            plt_data[i][latest_id:,0],
            np.abs(plt_data[i][latest_id:,1]),
            plt_data[i][latest_id:,2],
            color=mode2colors[mode_data[i][-1]]
        )

    # # plot constant rbar plane
    # x_plane = np.linspace(2,2,2)
    # y_plane = np.linspace(0,200,2)
    # z_plane = np.linspace(0.2,0.8,2)
    # [xx_p, yy_p] = np.meshgrid(y_plane,x_plane, indexing='xy')
    # [_, zz_p] = np.meshgrid(x_plane,z_plane, indexing='xy')
    # surf = ax.plot_surface(xx_p, yy_p, zz_p, internal_faces=True,alpha=0.3)

    # plot 2d constant r
    inter_list = [7,19,73.5,100.8]
    for i_pt in inter_list:
        ax.scatter(
            i_pt, 2, 0.8,
            s=35.,
            c='b'
        )
    ax.plot(
        [7,100.8],
        [2,2],
        [0.8,0.8],
        color='g',
        linewidth=3
    )
    # ax.scatter(plt_data[:,0], np.abs(plt_data[:,1]), plt_data[:,2], marker='o')
    ax_fs = 15
    ax.set_xlabel(r"$\bar T$",fontsize=ax_fs)
    ax.set_ylabel(r"$\bar r$",fontsize=ax_fs)
    ax.set_zlabel(r"$c$",fontsize=ax_fs)
    # plt.title(r"$\bar F$" + ' v ' + r"$\bar r$" + ' v ' + r"$\rho'_{0}$")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 9) # set figure's size manually to your full screen (32x18)
    plt.tight_layout(rect=[0,0,1,1.3])
    plt.savefig(figsave_path+"shoot_3d_b.pdf",bbox_inches='tight')
    # plt.show()

def check_rz_uniqueness(axes_data_xz, data_list, info_list):
    fbar_list = axes_data_xz[0]
    rho10_list = axes_data_xz[1]
    data_list = data_list[0]
    plt_data = []
    n_data = len(fbar_list)*len(rho10_list)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(len(fbar_list)):
        plt_data = []
        for j in range(len(rho10_list)):
            y = get_shape2c(data_list[i][j], info_list, n_interp=12)
            plt_data.append([
                y[-2,0],
                -y[-2,-1],
                fbar_list[i]
            ])
            # print(data_list[i][j+1][0])
            # print(data_list[i][0])
        plt_data = np.array(plt_data)
        ax.scatter(
            plt_data[:,0],
            plt_data[:,1],
            plt_data[:,2],
            color=((i*j+j)/n_data,0,0,1),
            marker='o'
        )
    ax.set_xlabel("$r$")  # add X-axis label
    ax.set_ylabel(r"$z_{0}$")
    ax.set_zlabel(r"$\bar F$")
    plt.grid()
    plt.title(r"$z_{0}$ against $r$")
    plt.show()

def plot_3d_z0rbarrho10(data_list, l=1.):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt_data = []
    for z in range(len(data_list)):
        plt_data.append([data_list[z][0]])
        z0rbar = []
        for i in range(len(data_list[z][1])):
            # print(data_list[z][0])
            for j in range(len(data_list[z][1][i])-1):
                z0 = calc_z0(
                    rho_1=data_list[z][1][i][j+1][3],
                    sbar=data_list[z][1][i][j+1][2],
                    l=l
                )
                # if abs(z0+0.95) > 1e-4:
                #     continue
                z0rbar.append(np.array([
                    z0,
                    abs(data_list[z][1][i][0])
                ]))
        z0rbar = np.array(z0rbar)
        if len(z0rbar) < 1:
            continue
        # print(fbarrbar)
        plt_data[0].append(z0rbar)
        rho10_arr = np.ones(len(z0rbar)) * plt_data[z][0]
        ax.scatter(z0rbar[:,0], z0rbar[:,1], rho10_arr, marker='o')
        # if z == 7:
            # plt.show()

    ax.set_xlabel(r"$\z_{0}$")
    ax.set_ylabel(r"$\bar r$")
    ax.set_zlabel(r"$\rho'_{0}$")
    plt.show()

def plot_2d_stability(stab_data):
    import matplotlib.tri as tri

    fig, ax1 = plt.subplots(nrows=1)

    # Create grid values first.
    ngridx = 100
    ngridy = 200
    xi = np.linspace(0., 40., ngridx)
    yi = np.linspace(0., 5., ngridy)

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(stab_data[:,0], stab_data[:,1])
    interpolator = tri.LinearTriInterpolator(triang, stab_data[:,2])
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    
    ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
    cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")
    ax1.plot(stab_data[:,0], stab_data[:,1], 'ko', ms=3)
    for m in range(len(stab_data)):
        # print(stab_data[m,4])
        if stab_data[m,4]:
            ax1.plot(stab_data[m,0], stab_data[m,1], 'ro', ms=3)
    fig.colorbar(cntr1, ax=ax1)
    ax1.set_xlabel(r"$\bar F$")
    ax1.set_ylabel('$a$')
    ax1.set(xlim=(0, 40), ylim=(0, 5))

    # ax2 = fig.add_subplot(projection='3d')
    # ax2.scatter(stab_data[:,0], stab_data[:,1], np.abs(stab_data[:,5]), marker='o')
    plt.show()

def plot_stability_wzlr(eig_data, zlr_data):
    # Normalize
    min_eig = np.min(eig_data)
    max_eig = np.max(eig_data)
    eig_data[np.where(eig_data >= 0)] = eig_data[np.where(eig_data >= 0)] / max_eig
    eig_data[np.where(eig_data < 0)] = eig_data[np.where(eig_data < 0)] / 0.005
    
    plt.rcParams['axes.labelsize'] = 'small'
    plt.rcParams['legend.fontsize'] = 6
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
    fig, ax = plt.subplots(figsize=(3.5, 2))
    cb = ax.imshow(eig_data.T, origin='lower',
                cmap='seismic', extent=(0, 40, 0, 5), aspect='auto',
                # interpolation='quadric',
                vmin=-1, vmax=1
                )

    bar = plt.colorbar(cb)
    bar.set_ticks([-1, -0.5, 0, 0.5, 1])
    # bar.set_ticklabels(['<=-5e-3', '-2.5e-3', '0', '3.5e0','>=7.0e0'])
    bar.set_ticklabels(['<=-5e-3', '-2.5e-3', '0', '%.1fe0' % (max_eig/2),'>=%.1fe0' % max_eig])
    for i in range(len(zlr_data)):
        ax.plot(zlr_data[i][0], zlr_data[i][1], c='black', lw=2)
    ax.set_xlim(0, 40)
    ax.set_ylim(0.1, 5)
    ax.set_xlabel(r"$\bar L$")
    ax.set_ylabel('$a$')
    plt.tight_layout()
    # plt.savefig('/home/hung/git/hung/Papers/2015-RotatingChainPaper/fig/stability_map.pdf')
    # plt.savefig(stab_picklepath)
    plt.show()

def plot_stab_4d_splits(data_xyz, max_eig_3d, zlr_pts=None):
    import matplotlib as mpl
    # prep data
    [x, y, z] = data_xyz
    # [x, y, z] = np.meshgrid(x,y,z,indexing='ij')
    [x1, z1] = np.meshgrid(x,z,indexing='ij')
    max_eig_3d = np.array(max_eig_3d)
    
    # Normalize
    min_eig = np.min(max_eig_3d)
    max_eig = np.max(max_eig_3d)
    max_eig_3d[np.where(max_eig_3d >= 0)] = max_eig_3d[np.where(max_eig_3d >= 0)] / max_eig
    max_eig_3d[np.where(max_eig_3d < 0)] = max_eig_3d[np.where(max_eig_3d < 0)] / -(min_eig)

    # plot
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X,Y, csd_matrix,cmap =cm.seismic, alpha = 0.5)
    levels = np.linspace(max_eig_3d.min(), max_eig_3d.max(), 100)

    for j in range(len(y)):
        # if j != 2:
        #     continue
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        zz = max_eig_3d[:,j,:]
        cset = ax.contourf(x1, zz, z1,
            levels=levels,
            zdir ='y',
            offset = y[j],
            cmap='seismic',
            vmin=-1, vmax=1,
            zorder=-j)
        
    ax.set_box_aspect((1.3,2,1))
    
    # bar = plt.colorbar(cset)
    norm = mplc.Normalize(vmin=-1, vmax=1)
    bar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='seismic'), ax=ax,shrink=0.65)
    bar.set_ticks([-1, -0.5, 0, 0.5, 1])
    # bar.set_ticklabels(['<=%.1fe0' % min_eig, '%.1fe0' % (min_eig/2), '0', '%.1fe0' % (max_eig/2),'>=%.1fe0' % max_eig])
    bar.set_ticklabels(['<=-5e-3', '-2.5e-3', '0', '%.1fe0' % (max_eig/2),'>=%.1fe0' % max_eig])

    ax.set_xlim3d(np.min(x), np.max(x))
    ax.set_ylim3d(np.min(y), np.max(y))
    ax.set_zlim3d(np.min(z), np.max(z))

    axis_fontsize = 20
    if zlr_pts is not None:
        ax.scatter(zlr_pts[:,0], zlr_pts[:,1], zlr_pts[:,2], alpha=0.5)
    ax.set_xlabel(r"$\bar L$", fontsize=axis_fontsize)  # add X-axis label
    ax.set_ylabel(r"$\bar T$", fontsize=axis_fontsize)  # add Y-axis label
    ax.set_zlabel(r"$c$", fontsize=axis_fontsize)  # add Z-axis label

    ax.view_init(elev=17., azim=-145)
    plt.grid()
    # plt.title(r"$\bar L$" + ' v ' + r"$\bar F$" + ' v ' + r"$\rho'_{0}$")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(20, 12) # set figure's size manually to your full screen (32x18)
    plt.tight_layout(rect=[0,0,1,1.3])
    plt.savefig(figsave_path+"slice_stable.pdf",bbox_inches='tight')
    # plt.show()

def plot_stab_4d(
        data_xyz,
        max_eig_3d,
        zlr_pts,
        eig_lim=0.,
        plot_internal=False ## change computation for alpha_vals
    ):
    from rotatingchain.utils.plotter_utils import voxels
    import types
    [x, y, z] = data_xyz
    x1 = midpt_extend(x)
    y1 = midpt_extend(y)
    z1 = midpt_extend(z)

    max_eig_3d = np.array(max_eig_3d)
    eig_truth = np.zeros_like(max_eig_3d)
    eig_truth[max_eig_3d>eig_lim] = 0
    eig_truth[max_eig_3d<=eig_lim] = 1
    max_eig_all = np.max(max_eig_3d)
    data_shape = max_eig_3d.shape
    ## change computation for alpha_vals
    alpha_vals = np.array([
        mev/max_eig_all if mev>eig_lim else 0 for mev in max_eig_3d.flatten()
    ])
    alpha_vals.resize(data_shape)
    color_vals = np.zeros((
        data_shape[0],
        data_shape[1],
        data_shape[2],
        4
    ))
    color_vals[:,:,:,0] = 1.0
    color_vals[:,:,:,3] = alpha_vals

    [x, y, z] = np.meshgrid(x,y,z, indexing='ij')
    [x1, y1, z1] = np.meshgrid(x1,y1,z1, indexing='ij')
    # x1 = midpoints_extend(x)
    # y1 = midpoints_extend(y)
    # z1 = midpoints_extend(z)
    # print(x, y, z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel(r"$\bar L$")  # add X-axis label
    ax.set_ylabel(r"$\bar F$")  # add Y-axis label
    ax.set_zlabel(r"$\rho'_{0}$")  # add Z-axis label
    plt.grid()
    plt.title(r"$\bar L$" + ' v ' + r"$\bar F$" + ' v ' + r"$\rho'_{0}$")

    new_cmap()

    if plot_internal:
        ax.voxels = types.MethodType(voxels, ax)
        ax.voxels(
            x1,y1,z1,
            eig_truth,
            shade=True,
            facecolors=color_vals,
            internal_faces=True    
        )
    else: 
        ax.voxels(
            x1,y1,z1,
            # np.logical_not(eig_truth),
            eig_truth,
            shade=True,
            # facecolors=color_vals,
        )
    ax.scatter(zlr_pts[:,0], zlr_pts[:,1], zlr_pts[:,2], alpha=0.05)
    # ax.voxels(eig_truth, shade=True, alpha=0.75)
    # img = ax.scatter(x, y, z, c=max_eig_3d, cmap="Reds2")
    # fig.colorbar(img)
    plt.show()

def midpt_extend(x):
    if len(x) < 2:
        return np.array([x[0],x[0]])
    x1 = []
    for i in range(len(x)-1):
        diff_step = (x[i+1]-x[i])/2
        if i < 1:
            x1.append(x[i] - diff_step)
        x1.append(x[i] + diff_step)
    x1.append(x[-1] + diff_step)
    return np.array(x1)

def plot_contour_4d(data_xyz, contour_data, zlr_pts=None, cmaptype='PiYG'):
    # custom colormap
    if not isinstance(cmaptype, str):
        cmaptype = custom_cmap(cmaptype)

    # prep data
    [x, y, z] = data_xyz
    # [x, y, z] = np.meshgrid(x,y,z,indexing='ij')
    [y1, z1] = np.meshgrid(y,z,indexing='ij')
    contour_data = np.array(contour_data)
    
    # Normalize
    min_val = np.min(contour_data)
    max_val = np.max(contour_data)
    # contour_data[np.where(contour_data >= 0)] = contour_data[np.where(contour_data >= 0)] / max_val
    # contour_data[np.where(contour_data < 0)] = contour_data[np.where(contour_data < 0)] / -(min_val)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X,Y, csd_matrix,cmap =cm.seismic, alpha = 0.5)
    levels = np.linspace(contour_data.min(), contour_data.max(), 100)

    for k in range(len(x)):
        # if j != 2:
        #     continue
        # cset = ax.contourf(x, z, contour_data[:,j,:],
        #     zdir ='z',
        #     # offset = np.max(y),
        #     cmap='seismic')
        # cset = ax.contourf(x, z, contour_data[:,j,:],
        #     zdir ='x',
        #     # offset = np.min(x),
        #     cmap='seismic')
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        zz = contour_data[k,:,:]
        cset = ax.contourf(y1, z1, zz,
            levels=levels,
            zdir ='z',
            offset = x[k],
            cmap=cmaptype,
            vmin=-1, vmax=1)
    
    # bar = plt.colorbar(cset)
    norm = mplc.Normalize(vmin=min_val, vmax=max_val)
    bar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmaptype), ax=ax)
    # bar.set_ticks([-1, -0.5, 0, 0.5, 1])
    # bar.set_ticklabels(['<=%.1fe0' % min_val, '%.1fe0' % (min_val/2), '0', '%.1fe0' % (max_val/2),'>=%.1fe0' % max_val])
    # bar.set_ticklabels(['<=%.1fe0' % min_val, '%.1fe0' % (min_val/2), '0', '%.1fe0' % (max_val/2),'>=%.1fe0' % max_val])

    ax.set_xlim3d(np.min(y), np.max(y))
    ax.set_ylim3d(np.min(z), np.max(z))
    ax.set_zlim3d(np.min(x), np.max(x))

    # ax.scatter(zlr_pts[:,0], zlr_pts[:,1], zlr_pts[:,2], alpha=0.5)
    ax.set_xlabel(r"$\bar F$")  # add Y-axis label
    ax.set_ylabel(r"$\rho'_{0}$")  # add Z-axis label
    ax.set_zlabel(r"$\bar L$")  # add X-axis label
    plt.grid()
    plt.title(r"$\bar L$" + ' v ' + r"$\bar F$" + ' v ' + r"$\rho'_{0}$")
    plt.show()

def plot_stab_ctrl_4d(data_xyz, ctrl_truth, zlr_pts):
    from rotatingchain.utils.plotter_utils import voxels
    [x, y, z] = data_xyz
    x1 = midpt_extend(x)
    y1 = midpt_extend(y)
    z1 = midpt_extend(z)

    [x, y, z] = np.meshgrid(x,y,z, indexing='ij')
    [x1, y1, z1] = np.meshgrid(x1,y1,z1, indexing='ij')
    # x1 = midpoints_extend(x)
    # y1 = midpoints_extend(y)
    # z1 = midpoints_extend(z)
    # print(x, y, z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel(r"$\bar L$")  # add X-axis label
    ax.set_ylabel(r"$\bar F$")  # add Y-axis label
    ax.set_zlabel(r"$\rho'_{0}$")  # add Z-axis label
    plt.grid()
    plt.title("Control restrictions plot")

    ax.voxels(
        x1,y1,z1,
        ctrl_truth,
        shade=True,
        # alpha=0.3,
        # facecolors=color_vals,
    )
    ax.scatter(zlr_pts[:,0], zlr_pts[:,1], zlr_pts[:,2], alpha=0.05)
    # ax.voxels(eig_truth, shade=True, alpha=0.75)
    # img = ax.scatter(x, y, z, c=max_eig_3d, cmap="Reds2")
    # fig.colorbar(img)
    plt.show()

def plot_freespace_4d(
    data_xyz,
    free_truth,
    zlr_pts=None,
    c_mode=None,
    vital_pts=None,
    path_plot=None,
    show_plot=True,
    label_plot=False
):
    # takes path points in shape (n_pts,3)
    from rotatingchain.utils.plotter_utils import voxels
    import types
    [x, y, z] = data_xyz
    x1 = midpt_extend(x)
    y1 = midpt_extend(y)
    z1 = midpt_extend(z)

    # [x, y, z] = np.meshgrid(x,y,z, indexing='ij')
    [x1, y1, z1] = np.meshgrid(x1,y1,z1, indexing='ij')
    # x1 = midpoints_extend(x)
    # y1 = midpoints_extend(y)
    # z1 = midpoints_extend(z)
    # print(x, y, z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    axis_fontsize = 15
    ax.set_xlabel(r"$\bar L$", fontsize=axis_fontsize)  # add X-axis label
    ax.set_ylabel(r"$\bar T$", fontsize=axis_fontsize)  # add Y-axis label
    ax.set_zlabel(r"$c$", fontsize=axis_fontsize)  # add Z-axis label
    ax.view_init(elev=17., azim=-125)

    min_max_xyz = np.array([np.min(data_xyz,axis=1), np.max(data_xyz,axis=1)])
    xyz_limoffset = (min_max_xyz[1] - min_max_xyz[0]) * 0.01

    ax.set_xlim3d(np.min(x)-xyz_limoffset[0], np.max(x)+xyz_limoffset[0])
    ax.set_ylim3d(np.min(y)-xyz_limoffset[1], np.max(y)+xyz_limoffset[1])
    ax.set_zlim3d(np.min(z)-xyz_limoffset[2], np.max(z)+xyz_limoffset[2])
    plt.grid()
    # plt.title("Free restrictions plot")

    if c_mode is not None:
        ax.voxels = types.MethodType(voxels, ax)
        ax.voxels(
            x1,y1,z1,
            free_truth,
            shade=True,
            facecolors=c_mode,
            internal_faces=False   
        )
    else:
        ax.voxels(
            x1,y1,z1,
            free_truth,
            shade=True,
            # facecolors=color_vals,
            alpha=0.1
        )
    
    if zlr_pts is not None:
        ax.scatter(zlr_pts[:,0], zlr_pts[:,1], zlr_pts[:,2], alpha=0.05)
    
    if label_plot:
        vital_labels = [
            'A','','B','','C','',
            'D','','E','F'
        ]
        label_offset = [0.5,0,-0.07]

    if vital_pts is not None:
        for i, i_pt in enumerate(vital_pts):
            ax.scatter(
                i_pt[0], i_pt[1], i_pt[2],
                s=30.,
                c='b'
            )
            if label_plot:
                ax.text(
                    i_pt[0]+label_offset[0],
                    i_pt[1]+label_offset[1],
                    i_pt[2]+label_offset[2],
                    vital_labels[i],
                    None,
                    # backgroundcolor=(1,1,1,1.0),
                    fontweight='semibold'
                )

    if path_plot is not None:
        ax.plot(
            path_plot[:,0],
            path_plot[:,1],
            path_plot[:,2],
            'g*-',
            lw=2,
            markersize=2
        )

    # ax.voxels(eig_truth, shade=True, alpha=0.75)
    # img = ax.scatter(x, y, z, c=max_eig_3d, cmap="Reds2")
    # fig.colorbar(img)
    if show_plot:
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(20, 12) # set figure's size manually to your full screen (32x18)
        if path_plot is None:
            plt.savefig(figsave_path+"full_stab_3d_rmode.pdf",bbox_inches='tight')
        else:
            plt.savefig(figsave_path+"path_stab_3d_rmode.pdf",bbox_inches='tight')
        # plt.show()
    else:
        return fig, ax

def plot_mode_4d(
        data_xyz,
        mode_data,
        m_val,
        zlr_pts,
        plot_internal=True
    ):
    from rotatingchain.utils.plotter_utils import voxels
    import types
    [x, y, z] = data_xyz
    x1 = midpt_extend(x)
    y1 = midpt_extend(y)
    z1 = midpt_extend(z)

    mode_diff = mode_data - m_val
    mode_data = np.array(mode_data)
    mode_truth = np.zeros_like(mode_data)
    mode_truth[mode_diff>0.] = 1
    mode_truth[mode_diff<=0.] = 0

    [x, y, z] = np.meshgrid(x,y,z, indexing='ij')
    [x1, y1, z1] = np.meshgrid(x1,y1,z1, indexing='ij')
    # x1 = midpoints_extend(x)
    # y1 = midpoints_extend(y)
    # z1 = midpoints_extend(z)
    # print(x, y, z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel(r"$\bar L$")  # add X-axis label
    ax.set_ylabel(r"$\bar F$")  # add Y-axis label
    ax.set_zlabel(r"$\rho'_{0}$")  # add Z-axis label
    plt.grid()
    plt.title(r"$\bar L$" + ' v ' + r"$\bar F$" + ' v ' + r"$\rho'_{0}$")

    data_shape = mode_data.shape
    color_vals = np.zeros((
        data_shape[0],
        data_shape[1],
        data_shape[2],
        4
    ))
    color_vals[mode_data==0] = np.array([1.,0.,0.,0.2])
    color_vals[mode_data==1] = np.array([1.,0.5,0.,0.2])
    color_vals[mode_data==2] = np.array([1.,1.,0.,0.2])
    color_vals[mode_data==3] = np.array([0.,1.,0.,0.2])
    color_vals[mode_data==4] = np.array([0.,1.,1.,0.2])
    color_vals[mode_data==5] = np.array([0.,0.,1.,0.2])

    if plot_internal:
        ax.voxels = types.MethodType(voxels, ax)
        ax.voxels(
            x1,y1,z1,
            mode_data,
            shade=True,
            facecolors=color_vals,
            internal_faces=True    
        )
    else: 
        ax.voxels(
            x1,y1,z1,
            mode_truth,
            shade=True,
            # facecolors=color_vals,
        )
    ax.scatter(zlr_pts[:,0], zlr_pts[:,1], zlr_pts[:,2], alpha=0.05)
    # ax.voxels(eig_truth, shade=True, alpha=0.75)
    # img = ax.scatter(x, y, z, c=max_eig_3d, cmap="Reds2")
    # fig.colorbar(img)
    plt.show()

def plot_controls(ctrlpath, plot_label=False):
    ctrlpath_np = np.array(ctrlpath)
    sec_step = 5.0 # seconds per step
    x_axis = np.linspace(0,len(ctrlpath_np)-1,len(ctrlpath_np)) * sec_step
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)

    twin1 = ax.twinx()
    twin2 = ax.twinx()

    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    twin2.spines.right.set_position(("axes", 1.2))

    p1, = ax.plot(x_axis,ctrlpath_np[:,0], "-",color='maroon', label=r"$\omega$")
    p2, = twin1.plot(x_axis,ctrlpath_np[:,1], "-",color='steelblue', label=r"$r$")
    p3, = twin2.plot(x_axis,ctrlpath_np[:,2], "-",color='darkolivegreen', label=r"$h$")

    ax.set_xlim(0, np.max(x_axis))
    ax.set_ylim(0, 30)
    twin1.set_ylim(0, 0.03)
    twin2.set_ylim(0, 0.50)
    # twin1.set_ylim(np.min(ctrlpath_np[:,1]), np.max(ctrlpath_np[:,1]))
    # twin2.set_ylim(np.min(ctrlpath_np[:,2]), np.max(ctrlpath_np[:,2]))
    # ax.set_yticks(np.linspace(ax.get_ybound()[0], ax.get_ybound()[1], 5))
    # twin1.set_yticks(np.linspace(twin1.get_ybound()[0], twin1.get_ybound()[1], 5))
    # twin2.set_yticks(np.linspace(twin2.get_ybound()[0], twin2.get_ybound()[1], 5))

    ax.set_yticks(np.linspace(0., 30., 5))
    twin1.set_yticks(np.linspace(0., 0.03, 5))
    twin2.set_yticks(np.linspace(0., 0.50, 5))


    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$\omega$")
    twin1.set_ylabel(r"$r$")
    twin2.set_ylabel(r"$h$")

    ax.spines['top'].set_visible(False)
    twin1.spines['top'].set_visible(False)
    twin2.spines['top'].set_visible(False)

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)

    # ax.grid(linestyle='dashed')
    # twin1.grid(linestyle='dashed')
    # ax.legend(handles=[p1, p2, p3])

    # labels
    label_offset = [
        [0.5,0.],
        [0.2,0.5],
        [0.5,0.],
        [0.2,0.5],
        [0.5,0.],
        [0.2,0.5],
    ]
    if plot_label:
        for i, id in enumerate([0,2,5,7,9,10]):
            ax.plot(
                [x_axis[id],x_axis[id]],
                [0.,30.],
                '--',linewidth=1.0,color='k', alpha=0.5, zorder=-1
            )
            ax.text(
                x_axis[id] + 0.5,
                0+0.7,
                chr(i+65), fontsize=11,
                color='w',
                backgroundcolor=(0,0,0,0.5),
            )

    # ax.grid()

    plt.show()

def new_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    # get colormap
    ncolors = 256
    color_array = plt.get_cmap('Reds')(range(ncolors))

    # change alpha values
    base_val = np.linspace(0.0,1.0,ncolors)
    new_alph = (-1/(5*base_val+0.2)+5)/5
    color_array[:,-1] = new_alph

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='Reds2',colors=color_array)

    # register this new colormap with matplotlib
    plt.colormaps.register(cmap=map_object)

def custom_cmap(custom_id):
    if custom_id == 1:
        start_color = np.array([0.40392157, 0., 0.12156863, 1.0])
        mid_color = np.array([0.9, 0.9, 0.9, 1.0])
        end_color = np.array([0.05, 0.05, 0.05, 1.0])
        N = int(256/2)
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(start_color[0], mid_color[0], N)
        vals[:, 1] = np.linspace(start_color[1], mid_color[1], N)
        vals[:, 2] = np.linspace(start_color[2], mid_color[2], N)
        vals2 = np.ones((N, 4))
        vals2[:, 0] = np.linspace(mid_color[0], end_color[0], N)
        vals2[:, 1] = np.linspace(mid_color[1], end_color[1], N)
        vals2[:, 2] = np.linspace(mid_color[2], end_color[2], N)
        val_final = np.concatenate((vals,vals2))
    elif custom_id == 2:
        blue_r = cm.get_cmap('Blues_r', 256)
        id_vals = np.linspace(0,1,256)**0.3
        val_final = blue_r(id_vals)
        # val_final = np.sqrt(blue_r(np.linspace(0,1,256)))
        # print(val_final)
    return mplc.ListedColormap(val_final)