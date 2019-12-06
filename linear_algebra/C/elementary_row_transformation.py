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

# def simplest_form(A:np.ndarray):
#     E = np.eye(A.shape[0])
#     for i in range(A.shape[0]):
#         for j in range(i, A.shape[1]):
#             if np.any(A[i:,j]):
#                 A[i:] = A[np.argsort(-a[i:,j])]
#                 print("----------------")
#                 print(A.shape)
#                 print(A)
#
#                 print(i,j)
#                 print("----------------")
#                 if A[i,j]!= 0:
#                     print(A[i])
#                     print(A[i,j],"00000000000000")
#                     A[i] = A[i]/A[i,j]
#                     print("**************")
#                     print(A)
#
#                     print("**************")
#                 break
#
#
#     return A

def simplest_form1(A:np.ndarray):
    E = np.eye(A.shape[0])
    P=[]
    if np.any(A):
        for j in range(0, min(A.shape[0], A.shape[1],)):
            print("JJJJJJJJJJJJ",j, A.shape[0])
            for i in range(j, A.shape[0]):
                print("IIIIIIIIIIIII",i, A[i,j] )

                if A[i,j]==0:
                    mask = A[i:,j]!=0
                    print("mask,i,j",mask,i,j)
                    if np.any(mask):
                        idx = np.nonzero(mask)#tuple
                        print(idx)
                        p = [0, i, idx[0][0]+i]
                        print("p1", p, i,j )
                        P.append(p)
                        A, E1 = ele_row_tran(A, [p])
                        E = np.dot(E1,E)
                        p = [1, i, (1/A[i,j])]
                        print("p2", p)
                        P.append(p)
                        A, E1 = ele_row_tran(A, [p])
                        E = np.dot(E1, E)
                        print(A)
                        for id in idx[0][1:]:
                            p = [2, i, (-A[id, j]), id]
                            A, E1 = ele_row_tran(A, [p])
                            E = np.dot(E1, E)
                            print(A)
                    else:
                        break

                    mask = A[:i, j] != 0
                    print("mask,i,j22", mask, i, j)
                    if np.any(mask):
                        idx = np.nonzero(mask)  # tuple
                        print("IDX",idx)
                        # p = [0, i, idx[0][0] + i]
                        # print("p1", p, i, j)
                        # P.append(p)
                        # A, E1 = ele_row_tran(A, [p])
                        # E = np.dot(E1, E)
                        # p = [1, i, (1 / A[i, j])]
                        # print("p2", p)
                        # P.append(p)
                        # A, E1 = ele_row_tran(A, [p])
                        # E = np.dot(E1, E)
                        # print(A)
                        for id in idx[0]:
                            p = [2, i, (-A[id, j]), id]
                            A, E1 = ele_row_tran(A, [p])
                            E = np.dot(E1, E)
                            print(A)
                else:

                    p = [1, i, (1 / A[i, j])]
                    print("p3", p)
                    P.append(p)
                    A, E1 = ele_row_tran(A, [p])
                    E = np.dot(E1, E)
                    print(A)
                    mask = A[(i + 1):, j] != 0

                    print("MASK1", mask, i,  j)
                    if np.any(mask):
                        idx = np.nonzero(mask)  # tuple
                        print(idx)
                        print(idx[0][1:])
                        # p = [0, i+1, idx[0][0]+i+1]
                        # print("p3", p)
                        # P.append(p)
                        # A, E1 = ele_row_tran(A, [p])
                        # E = np.dot(E1, E)

                        # print(A)
                        for id in idx[0]:
                            print("id",id)
                            p = [2, i, (-A[id+i+1, j]), id+i+1]
                            print("p", p)
                            A, E1 = ele_row_tran(A, [p])
                            E = np.dot(E1, E)
                            print(A)

                    mask = A[:i, j] != 0

                    print("MASK2", mask, j)
                    if np.any(mask):
                        idx = np.nonzero(mask)  # tuple
                        print(idx)
                        print(idx[0][1:])
                        # p = [0, i , idx[0][0] ]
                        # print("p3", p)
                        # P.append(p)
                        # A, E1 = ele_row_tran(A, [p])
                        # E = np.dot(E1, E)
                        #
                        # print(A)
                        for id in idx[0]:
                            print("id", id)
                            p = [2, i, (-A[id, j]), id]
                            print("p", p)
                            A, E1 = ele_row_tran(A, [p])
                            E = np.dot(E1, E)
                            print(A)

                break


    return A, E

def simplest_form(A:np.ndarray):
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
    a = np.array([0, 1, 2, 0, 0, 9, 0, 0, 0, 1,2,3]).reshape(3, 4)
    # a = np.array([3, 1, 2, 0, 1, 0, 0, 0, 3]).reshape(3, 3)
    # a = np.array([1, 1, -3, -1, 1, 3, -1, -3, 4, 4, 1, 5, -9, -8, 0]).reshape(3, 5)
    print(a)
    s,e, p = simplest_form(a)
    print(s)
    print(e)
    print(np.dot(e,a))
    print(p)
    print("**************")
    print(ele_row_tran(a,p))
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