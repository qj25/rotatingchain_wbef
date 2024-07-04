import numpy as np

e_tol = 2e-2

def foundinlist(val, cur_list):
    if len(cur_list) > 0:
        for i in range(len(cur_list)):
            if abs(val-cur_list[i][0]) < e_tol:
                return True, i
    return False, None