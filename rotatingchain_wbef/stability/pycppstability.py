import numpy as np
from rotatingchain_wbef.stability.stabjac_cpp.Stab_jac import Stab_jac
from time import time

class pycppstability:
    def __init__(
        self,
        rho_a=1.225,
        d=2e-3,
        c_f=0.038,
        c_n=1.17,
        # k=8e7,
        k=8490790.955648089,
        verb_bool=False,
    ):
        """
        Class to determine the stability of a certain configuration of a rotating chain.
        For fixed end, jacobian excludes top and bottom fixed points (N and 0),
        there is a total of n - 2 (2 fixed points).
        Jacobian is 6(n-2) x 6(n-2) matrix. (3 pos + 3 vel)

        Init chain with chain properties:
        n - number of discrete segments
        mu - mass per unit length
        l - length

        Init class with stored values for:
        rho_a - air density
        d - diameter of the chain
        c_f - skin-friction drag co-eff
        c_n - crossflow drag co-eff
        
        Velocity:
        x_dot - vel wrt rotating frame
        v_rot cross x - vel of rotating frame
        vt - vel wrt inertial frame (airspeed)
        vt = x_dot + (v_rot cross x)

        """
        self.verb_bool = verb_bool

        self.sj2 = Stab_jac(
            in_rho_a=rho_a,
            in_d=d,
            in_c_f=c_f,
            in_c_n=c_n,
            in_k=k,
            in_error_verbose=verb_bool
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Main Stuff ||~~~~~~~~~~~~~~~~~~~~~~~~~~
    def calc_stab(
        self,
        mu,
        l,
        v_rot,
        y,
    ):
        self.n = int(len(y)/2)
        self.jac_mat = np.zeros((6*(self.n-2),6*(self.n-2))).flatten()
        np.set_printoptions(threshold=np.inf)
        self.sj2.calc_stab(
            self.jac_mat,
            y.flatten(),
            in_mu=mu,
            in_l=l,
            in_v_rot=v_rot
        )
        np.set_printoptions(threshold=np.inf)
        # for i in range(len(self.jac_mat)):
        #     print(f'id = {i}')
        #     print(self.jac_mat[i])
        # print(len(self.jac_mat))
        # input()

        self.jac_mat = self.jac_mat.reshape((6*(self.n-2),6*(self.n-2)))

        if self.verb_bool:
            np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})
        # self.populate_jac()
        # print(self.jac_mat)
        # print(self.jac_mat.shape)
        # input('hi')
        # print(self.i3)
        
        if self.verb_bool:
            self.print_jac_data()

        eigval, _ = np.linalg.eig(self.jac_mat)

        maxreallambda = np.max(np.real(eigval))
        # print(self.jac_mat)
        # print(f"eigenvalues = {eigval}")
        # print(f"maxreallambda = {maxreallambda}")
        # input()
        if self.verb_bool:
            print(f"eigenvalues = {eigval}")
            print(f"maxreallambda = {maxreallambda}")
        if maxreallambda > 0:
            # print(f"maxreallambda = {maxreallambda}")
            # print("Unstable!")
            return maxreallambda
        else:
            # print(f"maxreallambda = {maxreallambda}")
            # print("Stable!")
            return maxreallambda
        
    def print_jac_data(self):
        self.verb_bool = True
        n_i = (self.n-2)
        
        for i in range(n_i):
            i_node = i + 1
            np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})
            ijac = i * 3
            dij = (i+n_i) * 3
            print('identity3')
            print(f"{i} -- [{ijac}:{ijac+3}, {dij}:{dij+3}]")
            print(f"{self.jac_mat[ijac:ijac+3, dij:dij+3]}")

            # self.jac_mat[ijac:ijac+3, dij:dij+3] = self.i3.copy()

            ijac = (i+n_i) * 3
            if i_node > 1:
                dij = (i-1)*3
                if self.verb_bool:
                    print('C')
                    print(f"{i} -- [{ijac}:{ijac+3}, {dij}:{dij+3}]")
                    print(f"{self.jac_mat[ijac:ijac+3, dij:dij+3]}")

            dij = i*3
            if self.verb_bool:
                print('CF')
                print(f"{i} -- [{ijac}:{ijac+3}, {dij}:{dij+3}]")
                print(f"{self.jac_mat[ijac:ijac+3, dij:dij+3]}")

            if i_node < n_i:
                dij = (i+1)*3
                if self.verb_bool:
                    print('C_diff_a_wrt_x1')
                    print(f"{i} -- [{ijac}:{ijac+3}, {dij}:{dij+3}]")
                    print(f"{self.jac_mat[ijac:ijac+3, dij:dij+3]}")

            dij = ijac
            if self.verb_bool:
                print('F')
                print(f"{i} -- [{ijac}:{ijac+3}, {dij}:{dij+3}]")
                print(f"{self.jac_mat[ijac:ijac+3, dij:dij+3]}")

        if self.verb_bool:
            input()