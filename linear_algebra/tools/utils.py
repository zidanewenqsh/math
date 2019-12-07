import numpy as np

def ele_row_tran(A: np.ndarray, P: list):
    '''
    初等行变换
    :param A:
    :param P:
    :return:
    '''
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
    '''
    初等列变换
    :param A:
    :param Q:
    :return:
    '''
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

def row_simplest_form(A:np.ndarray):
    '''
    化为行最简形
    :param A:
    :return:
    '''
    E = np.eye(A.shape[0])
    P=[]
    R = min(A.shape[0], A.shape[1])
    if np.any(A):
        for j in range(0, min(A.shape[0], A.shape[1])):
            # for i in range(j, A.shape[0]):
            i = j
            if A[i,j]==0:
                mask = A[i:,j]!=0
                if np.any(mask):
                    idx = np.nonzero(mask)#tuple
                    p = [0, i, idx[0][0]+i]
                    P.append(p)
                    A, E1 = ele_row_tran(A, [p])
                    E = np.dot(E1,E)
                else:
                    R -= 1
                    continue

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

    return A, E, R, P

def col_simplest_form(A:np.ndarray):
    '''
    化为列最简形
    :param A:
    :return:
    '''
    E = np.eye(A.shape[1])
    P=[]
    R = min(A.shape[0], A.shape[1])
    if np.any(A):
        for i in range(0, min(A.shape[0], A.shape[1])):
            # for j in range(i, A.shape[0]):
            j = i
            if A[i,j]==0:
                mask = A[i,j:]!=0
                if np.any(mask):
                    idx = np.nonzero(mask)#tuple
                    p = [0, j, idx[0][0]+j]
                    P.append(p)
                    A, E1 = ele_col_tran(A, [p])
                    E = np.dot(E,E1)
                else:
                    R -= 1
                    continue

            p = [1, i, (1 / A[i, j])]
            P.append(p)
            A, E1 = ele_col_tran(A, [p])
            E = np.dot(E, E1)

            mask = A[i, (j+1):] != 0
            if np.any(mask):
                idx = np.nonzero(mask)  # tuple
                for id in idx[0]:
                    p = [2, i, (-A[i, id+j+1]), id+i+1]
                    P.append(p)
                    A, E1 = ele_col_tran(A, [p])
                    E = np.dot(E, E1)

            mask = A[i, :j] != 0
            if np.any(mask):
                idx = np.nonzero(mask)  # tuple
                for id in idx[0]:
                    p = [2, i, (-A[i, id]), id]
                    P.append(p)
                    A, E1 = ele_col_tran(A, [p])
                    E = np.dot(E, E1)

    return A, E, R, P

def det_upptritran(A):
    '''
    化为上三角行列式
    :param A:
    :return:
    '''
    if A.shape[0]==A.shape[1]:
        E = np.eye(A.shape[1])
        # D = 0
        P = []
        if np.any(A):
            D = 1
            for j in range(0, A.shape[0]):
                # for i in range(j, A.shape[0]):
                i = j
                # print("aij",A[i,j],i,j)
                if A[i, j] == 0:
                    mask = A[i:, j] != 0
                    if np.any(mask):
                        idx = np.nonzero(mask)  # tuple
                        D *= -1
                        p = [0, i, idx[0][0] + i]
                        P.append(p)
                        A, E1 = ele_row_tran(A, [p])
                        E = np.dot(E1, E)
                    else:
                        return 0, 0, 0, 0
                D *= A[i, j]
                p = [1, i, (1 / A[i, j])]
                P.append(p)
                A, E1 = ele_row_tran(A, [p])
                E = np.dot(E1, E)

                mask = A[(i + 1):, j] != 0
                if np.any(mask):
                    idx = np.nonzero(mask)  # tuple

                    for id in idx[0]:
                        p = [2, i, (-A[id + i + 1, j]), id + i + 1]
                        P.append(p)
                        A, E1 = ele_row_tran(A, [p])
                        E = np.dot(E1, E)

            return D, A, E, P
        else:return 0, 0, 0, 0
    else:
        raise ValueError

def det_lowtritran(A):
    if A.shape[0] == A.shape[1]:
        E = np.eye(A.shape[1])
        # D = 0
        P = []
        if np.any(A):
            D = 1
            for i in range(0, min(A.shape[0], A.shape[1])):
                # for j in range(i, A.shape[0]):
                j = i
                if A[i, j] == 0:
                    mask = A[i, j:] != 0
                    if np.any(mask):
                        D *= -1
                        idx = np.nonzero(mask)  # tuple
                        p = [0, j, idx[0][0] + j]
                        P.append(p)
                        A, E1 = ele_col_tran(A, [p])
                        E = np.dot(E, E1)
                    else:
                        return 0, 0, 0, 0

                D *= A[i, j]
                p = [1, i, (1 / A[i, j])]
                P.append(p)
                A, E1 = ele_row_tran(A, [p])
                E = np.dot(E1, E)

                mask = A[i, (j + 1):] != 0
                if np.any(mask):
                    idx = np.nonzero(mask)  # tuple

                    for id in idx[0]:
                        p = [2, i, (-A[i, id + j + 1]), id + i + 1]
                        P.append(p)
                        A, E1 = ele_col_tran(A, [p])
                        E = np.dot(E, E1)
            return D, A, E, P
        else:
            return 0, 0, 0, 0
    else:
        raise ValueError

if __name__ == '__main__':
    np.set_printoptions(precision=2,suppress=True)
    print(0)
    a = np.arange(12.).reshape(4, 3)
    # print(a)
    P = [[0, 1, 2], [1, 2, 2], [2, 1, 3, 0]]  # ,[1,2,2],[2,0,1,3]
    b, e = ele_row_tran(a, P)
    # print(b)
    # print(e)
    # s_a = a[np.argsort(-a[:, 0])]
    # print(s_a)
    # print(np.any(np.array([0,0,0,1])))
    # Q = [[0, 1, 2], [1, 2, 2], [2, 1, 3, 0]]
    # c, f = ele_col_tran(a, Q)
    # print(f)
    # print(c)
    # # a[2],a[1] = a[1],a[2]
    # # print(a)
    a = np.array([3, 1, 2, 0, 0, 9, 0, 0, 0, 1, 2, 3]).reshape(3, 4)
    a = np.array([0, 1, 2, 0, 1, 0, 0, 0, 3]).reshape(3, 3)
    # a = np.array([1, 1, -3, -1, 1, 3, -1, -3, 4, 4, 1, 5, -9, -8, 0]).reshape(5, -1)
    # print(a)
    s1, e1, r1, p1 = row_simplest_form(a)
    print("s1", s1)
    print(e1)
    print(np.dot(e1, a))
    print("r1", r1)
    s2, e2, r2, p2 = col_simplest_form(a)
    print("s2", s2)
    print(e2)
    print(np.dot(a, e2))
    print("r2", r2)
    a = np.array([3, 1, 2, 9, 4, 0, 3, 2, 3]).reshape(3, 3)
    a = np.array([3, 1, 0, 6, 0, 0, 0, 2, 2]).reshape(3, 3)
    # print(a)
    print("------------------")
    d1, a1, e1, p1 = det_upptritran(a)
    print(a1)
    print("d1",d1)
    # print(p1)
    print(np.linalg.det(a))
    d2, a2, e2, p2 = det_lowtritran(a)
    print(a2)
    print("d2",d2)
    # print(np.linalg.det(a))
    # print(np.dot(e,a))
    # print(np.dot(a,e))
    #
    # print(p)
    # print("**************")
    # print(ele_row_tran(a, p))
    # print(ele_col_tran(a, p))
    # a = np.array([1,1,-3,-1,1,3,-1,-3,4,4,1,5,-9,-8,0]).reshape(3, 5)
    # print(a)
    # s1, e1 = simplest_form1(a)
    # print(s1)
    # print(s)
    # print(e1)
    # print(e)
    # print(np.dot(e1, a))
    # print(np.dot(e, a))
    # print(s)
    # print(s)
    # a = np.array([0,1,2,0,1,0,0,0,3]).reshape(3,3)
    # print(a)
    # print(a==0)

    #行列式
    #列变换