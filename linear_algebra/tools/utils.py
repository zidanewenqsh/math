import numpy as np


def ele_row_tran(A: np.ndarray, P: list):
    E = np.eye(A.shape[0])
    for p in P:
        if p[0] == 0:
            Temp = np.copy(E[p[1]])
            E[p[1]] = E[p[2]]
            E[p[2]] = Temp
        elif p[0] == 1:
            E[p[1]] *= p[2]
        elif p[0] == 2:
            E[p[3]] += E[p[1]] * p[2]
    return np.dot(E, A), E


def ele_col_tran(A: np.ndarray, Q: list):
    E = np.eye(A.shape[1])
    for q in Q:
        if q[0] == 0:
            Temp = np.copy(E[:, q[1]])
            E[:, q[1]] = E[:, q[2]]
            E[:, q[2]] = Temp
        elif q[0] == 1:
            E[:, q[1]] *= q[2]
        elif q[0] == 2:
            E[:, q[3]] += E[:, q[1]] * q[2]
    return np.dot(A, E), E

def simplest_form(A:np.ndarray):
    '''
    化最简形
    :param A:
    :return:
    '''
    E = np.eye(A.shape[0])
    P=[]
    if np.any(A):
        for j in range(0, min(A.shape[0], A.shape[1],)):
            for i in range(j, A.shape[0]):
                if A[i,j]==0:
                    mask = A[i:,j]!=0
                    if np.any(mask):
                        idx = np.nonzero(mask)#tuple
                        p = [0, i, idx[0][0]+i]
                        P.append(p)
                        A, E1 = ele_row_tran(A, [p])
                        E = np.dot(E1,E)
                    else:
                        break

                p = [1, i, (1 / A[i, j])]
                P.append(p)
                A, E1 = ele_row_tran(A, [p])
                E = np.dot(E1, E)
                mask = A[(i + 1):, j] != 0

                if np.any(mask):
                    idx = np.nonzero(mask)  # tuple

                    for id in idx[0]:
                        p = [2, i, (-A[id+i+1, j]), id+i+1]
                        P.append(p)
                        A, E1 = ele_row_tran(A, [p])
                        E = np.dot(E1, E)

                mask = A[:i, j] != 0

                if np.any(mask):
                    idx = np.nonzero(mask)  # tuple
                    for id in idx[0]:
                        p = [2, i, (-A[id, j]), id]
                        P.append(p)
                        A, E1 = ele_row_tran(A, [p])
                        E = np.dot(E1, E)
                break
    return A, E, P

if __name__ == '__main__':
    print(0)