import numpy as np

"""
mu - density
g - gravitational acceleration
"""

class func_define:
    def __init__(
        self,
        v_rot,
        rbar,
        lbar,
        mu,
        rho_1_0,
        grad_ratio,
        u_1_0,
        g=9.81,
        fg_lim=[0,1000],
        verbose=False
    ):
        self.v_rot = v_rot
        self.rbar = rbar
        self.lbar = lbar
        self.mu = mu
        self.rho_1_0 = rho_1_0
        self.grad_ratio = grad_ratio
        self.u_1_0 = u_1_0
        self.g = g
        self.fg_lim = fg_lim
        self.verbose = verbose

    def set_fbar_guess(self, fbar_guess):
        f_out = False
        if fbar_guess < self.fg_lim[0]:
            if self.verbose:
                print("Error: fbar_guess below lower limit!")
                input()
            f_out = True
            fbar_guess = self.fg_lim[0]
        elif fbar_guess > self.fg_lim[1]:
            if self.verbose:
                print("Error: fbar_guess above upper limit!")
                input()
            f_out = True
            fbar_guess = self.fg_lim[1]
        if abs(fbar_guess) < 1e-10:
            if self.verbose:
                print("Error: fbar_guess can not be 0!")
                input()
            fbar_guess = 1e-10

        self.fbar_guess = fbar_guess
        return fbar_guess, f_out

    def fixed2(self, u, sbar):
        # must set fbar_guess first
        return (
            -u
            /np.sqrt(
                u*u
                +(sbar+self.fbar_guess)**2
            )
        )
    
    def calc_u_0(self):
        return (
            self.grad_ratio 
            * self.fbar_guess
        )
    
    def get_f(self):
        return self.fbar_guess*self.g*self.g*self.mu/(self.v_rot*self.v_rot)