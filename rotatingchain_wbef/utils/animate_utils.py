import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import rotatingchain_wbef.utils.plotter as plter
from rotatingchain_wbef.utils.plotter_utils import voxels
import types

def animate_3d(data_xyz, free_truth, c_mode, vital_pts=None, path_plot=None, label_plot=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    def init():
        [x, y, z] = data_xyz
        x1 = plter.midpt_extend(x)
        y1 = plter.midpt_extend(y)
        z1 = plter.midpt_extend(z)

        [x1, y1, z1] = np.meshgrid(x1,y1,z1, indexing='ij')

        ax.set_xlabel(r"$\bar L$")  # add X-axis label
        ax.set_ylabel(r"$\bar T$")  # add Y-axis label
        ax.set_zlabel(r"$c$")  # add Z-axis label
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
        return fig,

    def animate(i):
        print(f"view loaded: elev = {10}; azim {i}")
        ax.view_init(elev=17., azim=i-125)
        return fig,

    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=180, interval=20, blit=True)
    # Save
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    anim.save('basic_animation.mp4', writer = writer)
    # anim.save('basic_animation.gif', fps=30)#, extra_args=['-vcodec', 'libx264'])
    # anim.save("html_example.html", writer="html")
    # anim.save("pillow_example.apng", writer="pillow")
    print('animation_saved')