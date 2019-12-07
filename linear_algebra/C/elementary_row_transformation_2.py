import numpy as np


def mat_ele_row_tran(A: np.ndarray, P: list):
    E = np.eye(A.shape[-2])
    for p in P:
        if p[0] == 0:
            Temp = np.copy(E[p[1]])
            E[p[1]] = E[p[2]]
            E[p[2]] = Temp
        elif p[0] == 1:
            E[p[1]] *= p[2]
        elif p[0] == 2:
            E[p[3]] += E[p[1]] * p[2]
    return np.matmul(E, A), E


def mat_ele_col_tran(A: np.ndarray, Q: list):
    E = np.eye(A.shape[-1])
    for q in Q:
        if q[0] == 0:
            Temp = np.copy(E[:, q[1]])
            E[:, q[1]] = E[:, q[2]]
            E[:, q[2]] = Temp
        elif q[0] == 1:
            E[:, q[1]] *= q[2]
        elif q[0] == 2:
            E[:, q[3]] += E[:, q[1]] * q[2]
    return np.matmul(A, E), E


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

def simplest_form1(A: np.ndarray):
    E = np.eye(A.shape[0])
    P = []
    if np.any(A):
        for j in range(0, min(A.shape[0], A.shape[1], )):
            print("JJJJJJJJJJJJ", j, A.shape[0])
            for i in range(j, A.shape[0]):
                print("IIIIIIIIIIIII", i, A[i, j])

                if A[i, j] == 0:
                    mask = A[i:, j] != 0
                    print("mask,i,j", mask, i, j)
                    if np.any(mask):
                        idx = np.nonzero(mask)  # tuple
                        print(idx)
                        p = [0, i, idx[0][0] + i]
                        print("p1", p, i, j)
                        P.append(p)
                        A, E1 = mat_ele_row_tran(A, [p])
                        E = np.matmul(E1, E)
                        p = [1, i, (1 / A[i, j])]
                        print("p2", p)
                        P.append(p)
                        A, E1 = mat_ele_row_tran(A, [p])
                        E = np.matmul(E1, E)
                        print(A)
                        for id in idx[0][1:]:
                            p = [2, i, (-A[id, j]), id]
                            A, E1 = mat_ele_row_tran(A, [p])
                            E = np.matmul(E1, E)
                            print(A)
                    else:
                        break

                    mask = A[:i, j] != 0
                    print("mask,i,j22", mask, i, j)
                    if np.any(mask):
                        idx = np.nonzero(mask)  # tuple
                        print("IDX", idx)
                        # p = [0, i, idx[0][0] + i]
                        # print("p1", p, i, j)
                        # P.append(p)
                        # A, E1 = ele_row_tran(A, [p])
                        # E = np.matmul(E1, E)
                        # p = [1, i, (1 / A[i, j])]
                        # print("p2", p)
                        # P.append(p)
                        # A, E1 = ele_row_tran(A, [p])
                        # E = np.matmul(E1, E)
                        # print(A)
                        for id in idx[0]:
                            p = [2, i, (-A[id, j]), id]
                            A, E1 = mat_ele_row_tran(A, [p])
                            E = np.matmul(E1, E)
                            print(A)
                else:

                    p = [1, i, (1 / A[i, j])]
                    print("p3", p)
                    P.append(p)
                    A, E1 = mat_ele_row_tran(A, [p])
                    E = np.matmul(E1, E)
                    print(A)
                    mask = A[(i + 1):, j] != 0

                    print("MASK1", mask, i, j)
                    if np.any(mask):
                        idx = np.nonzero(mask)  # tuple
                        print(idx)
                        print(idx[0][1:])
                        # p = [0, i+1, idx[0][0]+i+1]
                        # print("p3", p)
                        # P.append(p)
                        # A, E1 = ele_row_tran(A, [p])
                        # E = np.matmul(E1, E)

                        # print(A)
                        for id in idx[0]:
                            print("id", id)
                            p = [2, i, (-A[id + i + 1, j]), id + i + 1]
                            print("p", p)
                            A, E1 = mat_ele_row_tran(A, [p])
                            E = np.matmul(E1, E)
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
                        # E = np.matmul(E1, E)
                        #
                        # print(A)
                        for id in idx[0]:
                            print("id", id)
                            p = [2, i, (-A[id, j]), id]
                            print("p", p)
                            A, E1 = mat_ele_row_tran(A, [p])
                            E = np.matmul(E1, E)
                            print(A)

                break

    return A, E


def mat_row_simplest_form(A: np.ndarray):
    print(A)
    h = A.shape[-2]
    w = A.shape[-1]
    # E = np.eye(h)
    E = np.ones(shape=(*A.shape[:-2], h, h)) * np.eye(h)
    P = []

    R = np.ones(shape=A.shape[:-2]) * min(w, h)
    print("R",R,A.shape)
    if np.any(A):

        for j in range(0, min(A.shape[-2], A.shape[-1])):
            # for i in range(j, A.shape[0]):

            i = j
            # mask1 = A[..., i, j] != 0
            # mask1 = A[..., i, j] == 0

            mask2 = A[..., i:, j] == 0

            # print(mask2*mask3)

            idx = np.nonzero(np.all(mask2,axis=-1))
            # idx = np.stack(idx,axis=0)
            # print(idx)
            # R[idx[0],idx[1]] -= 1
            print(R)
            R[idx] -= 1
            # R[]
            print(R)
            # print(A[...,id[:1],j])
            # E[mask3] = 0
            # R[mask3] = 0

            mask1 = A[..., i:, j] != 0
            mask3 = A[..., i, j] == 0
            # print(mask1)
            # print(mask3)
            # print(mask1*mask3[...,None])
            mask1 = mask1*mask3[...,None]
            # print(mask3[..., None].shape)


            idx = np.nonzero(mask1)  # tuple

            # idx = np.stack(idx, axis=0)
            # print(mask1)
            # print(idx)

            # print(A[idx[:-2]])
            # print(A[idx[:-1]])
            # print(A[idx])
            # print(idx[-1:])
            # print("idx-2",idx[-2])
            # print(idx)
            idx1 = np.stack(idx)
            # print("idx", idx)
            # print(idx1)
            # print(mask1.shape)
            # for k, A1 in enumerate(A[idx[:-1]]):
            #     print(idx1[:,k])
            for id in idx1.T:
                # print("id", id)
                p = [0, i, id[-1]]
                P.append(p)
                id1 = tuple(id)
                # print("ida", id1, A[id1])
                A[id1[:-1]], E1 = mat_ele_row_tran(A[id1[:-1]], [p])

            mask1 = A[..., i, j] != 0
            # mask2 = A[...,i,:] !=0
            print("mask1",mask1, A[..., i, j])
            # print(mask2)
            idx = np.nonzero(mask1)
            idx1 = np.stack(idx)
            print("idxidx", idx1,j)
            for id in idx1.T:
                id =id.tolist()
                id.extend([i])
                id1 = tuple(id)
                print("id0", id1, A[id1])
                p = [1, i, 1/(A[id1][j])]
                P.append(p)
                A[id1[:-1]], E1 = mat_ele_row_tran(A[id1[:-1]], [p])
            '''
            p = [2, i, (-A[id + i + 1, j]), id + i + 1]'''
            mask1 = A[..., (i+1):, j] != 0

            idx = np.nonzero(mask1)
            idx1 = np.stack(idx)
            idx1[-1,:]+= (i+1)
            print("PPPPPPPPPPPP", A[..., i + 1, j], idx1)
            for id in idx1.T:
                id1 = tuple(id)
                print("id111111111", id1, id1[-1], A[id1])
                p = [2, i, (-A[id1][j]), id1[-1]]
                P.append(p)
                A[id1[:-1]], E1 = mat_ele_row_tran(A[id1[:-1]], [p])

            mask1 = A[..., :i , j] != 0
            idx = np.nonzero(mask1)
            idx1 = np.stack(idx)
            # idx1[-1, :] += (i + 1)
            for id in idx1.T:
                id1 = tuple(id)
                print("id22222222222",id1, id1[-1])
                p = [2, i, (-A[id1][j]), id1[-1]]
                P.append(p)
                A[id1[:-1]], E1 = mat_ele_row_tran(A[id1[:-1]], [p])

            print(A)
    return A
            # mask1 = A[..., i:, j] != 0
            # idx = np.nonzero(mask1)
            # idx1 = np.stack(idx)
            # for id in idx1.T:
            #     p = [0, i, id[-1]]
            #     # print("p",p)
            #     P.append(p)
            #     # print(A1)
            # print(A)



            # P.append(p)
            # A[mask1], E1 = mat_ele_row_tran(A[mask1], [p])
            # E[mask1] = np.matmul(E1, E[mask1])
            # print(A)
            # else:
            # R -= 1
            # continue


#     p = [1, i, (1 / A[i, j])]
#     P.append(p)
#     A, E1 = mat_ele_row_tran(A, [p])
#     E = np.matmul(E1, E)
#
#     mask = A[(i + 1):, j] != 0
#     if np.any(mask):
#         idx = np.nonzero(mask)  # tuple
#
#         for id in idx[0]:
#             p = [2, i, (-A[id + i + 1, j]), id + i + 1]
#             P.append(p)
#             A, E1 = mat_ele_row_tran(A, [p])
#             E = np.matmul(E1, E)
#
#     mask = A[:i, j] != 0
#     if np.any(mask):
#         idx = np.nonzero(mask)  # tuple
#         for id in idx[0]:
#             p = [2, i, (-A[id, j]), id]
#             P.append(p)
#             A, E1 = mat_ele_row_tran(A, [p])
#             E = np.matmul(E1, E)
#
#
# return A, E, R, P


def col_simplest_form(A: np.ndarray):
    E = np.eye(A.shape[1])
    P = []
    R = min(A.shape[0], A.shape[1])
    if np.any(A):
        for i in range(0, min(A.shape[0], A.shape[1])):
            # for j in range(i, A.shape[0]):
            j = i
            if A[i, j] == 0:
                mask = A[i, j:] != 0
                if np.any(mask):
                    idx = np.nonzero(mask)  # tuple
                    p = [0, j, idx[0][0] + j]
                    P.append(p)
                    A, E1 = mat_ele_col_tran(A, [p])
                    E = np.matmul(E, E1)
                else:
                    R -= 1
                    continue

            p = [1, i, (1 / A[i, j])]
            P.append(p)
            A, E1 = mat_ele_col_tran(A, [p])
            E = np.matmul(E, E1)

            mask = A[i, (j + 1):] != 0
            if np.any(mask):
                idx = np.nonzero(mask)  # tuple
                for id in idx[0]:
                    p = [2, i, (-A[i, id + j + 1]), id + i + 1]
                    P.append(p)
                    A, E1 = mat_ele_col_tran(A, [p])
                    E = np.matmul(E, E1)

            mask = A[i, :j] != 0
            if np.any(mask):
                idx = np.nonzero(mask)  # tuple
                for id in idx[0]:
                    p = [2, i, (-A[i, id]), id]
                    P.append(p)
                    A, E1 = mat_ele_col_tran(A, [p])
                    E = np.matmul(E, E1)

    return A, E, R, P


def det_upptritran(A):
    if A.shape[0] == A.shape[1]:
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
                        A, E1 = mat_ele_row_tran(A, [p])
                        E = np.matmul(E1, E)
                    else:
                        return 0, 0, 0, 0
                D *= A[i, j]
                p = [1, i, (1 / A[i, j])]
                P.append(p)
                A, E1 = mat_ele_row_tran(A, [p])
                E = np.matmul(E1, E)

                mask = A[(i + 1):, j] != 0
                if np.any(mask):
                    idx = np.nonzero(mask)  # tuple

                    for id in idx[0]:
                        p = [2, i, (-A[id + i + 1, j]), id + i + 1]
                        P.append(p)
                        A, E1 = mat_ele_row_tran(A, [p])
                        E = np.matmul(E1, E)

                # mask = A[(i + 1):, j] != 0
                # if np.any(mask):
                #     idx = np.nonzero(mask)  # tuple
                #     for id in idx[0]:
                #         p = [2, i, (-A[id + i + 1, j]), id + i + 1]
                #         P.append(p)
                #         A, E1 = ele_row_tran(A, [p])
                #         E = np.matmul(E1, E)

            return D, A, E, P
        else:
            return 0, 0, 0, 0
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
                        A, E1 = mat_ele_col_tran(A, [p])
                        E = np.matmul(E, E1)
                    else:
                        return 0, 0, 0, 0

                D *= A[i, j]
                p = [1, i, (1 / A[i, j])]
                P.append(p)
                A, E1 = mat_ele_row_tran(A, [p])
                E = np.matmul(E1, E)

                mask = A[i, (j + 1):] != 0
                if np.any(mask):
                    idx = np.nonzero(mask)  # tuple
                    for id in idx[0]:
                        p = [2, i, (-A[i, id + j + 1]), id + i + 1]
                        P.append(p)
                        A, E1 = mat_ele_col_tran(A, [p])
                        E = np.matmul(E, E1)

            return D, A, E, P
        else:
            return 0, 0, 0, 0
    else:
        raise ValueError


# def det_lowtritran1(A):#这种不行，必须用列变换

#     if A.shape[0] == A.shape[1]:
#         E = np.eye(A.shape[1])
#         # D = 0
#         P = []
#         if np.any(A):
#             D = 1
#             for j in range(0, A.shape[0]):
#                 # for i in range(j, A.shape[0]):
#                 i = j
#                 # print("aij",A[i,j],i,j)
#                 if A[i, j] == 0:
#                     mask = A[i:, j] != 0
#                     if np.any(mask):
#                         idx = np.nonzero(mask)  # tuple
#                         D *= -1
#                         p = [0, i, idx[0][0] + i]
#                         P.append(p)
#                         A, E1 = ele_row_tran(A, [p])
#                         E = np.matmul(E1, E)
#                     else:
#                         return 0, A, 0, 0
#                 D *= A[i, j]
#                 p = [1, i, (1 / A[i, j])]
#                 P.append(p)
#                 A, E1 = ele_row_tran(A, [p])
#                 E = np.matmul(E1, E)
#
#                 mask = A[:i, j] != 0
#                 if np.any(mask):
#                     idx = np.nonzero(mask)  # tuple
#                     for id in idx[0]:
#                         p = [2, i, (-A[id, j]), id]
#                         P.append(p)
#                         A, E1 = ele_row_tran(A, [p])
#                         E = np.matmul(E1, E)
#
#                 # mask = A[(i + 1):, j] != 0
#                 # if np.any(mask):
#                 #     idx = np.nonzero(mask)  # tuple
#                 #     for id in idx[0]:
#                 #         p = [2, i, (-A[id + i + 1, j]), id + i + 1]
#                 #         P.append(p)
#                 #         A, E1 = ele_row_tran(A, [p])
#                 #         E = np.matmul(E1, E)
#
#             return D, A, E, P
#         else:
#             return 0, 0, 0, 0
#     else:
#         raise ValueError


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    print(0)
    a = np.arange(24.).reshape(2, 4, 3)
    # print(a)
    # P = [[0, 1, 2], [1, 2, 2], [2, 1, 3, 0]]  # ,[1,2,2],[2,0,1,3]
    # b, e = mat_ele_row_tran(a, P)
    # print(b)
    # print(e)
    # Q = [[0, 1, 2], [1, 2, 2], [2, 1, 3, 0]]
    # c, f = mat_ele_col_tran(a, Q)
    # print(f)
    # print(c)

    a = np.array([0, 1, 2, 2, 0, 0, 9, 0, 0, 0, 1,  0, 0, 9, 2, 3, 0, 1, 0, 3,0, 1, 0, 3]).reshape(1, 4, 3, 2)
    # a = np.array([0, 1, 2, 0, 1, 0, 0, 0, 3]).reshape(3, 3)
    # a = np.array([1, 1, -3, -1, 1, 3, -1, -3, 4, 4, 1, 5, -9, -8, 0]).reshape(5, -1)
    # print(a)
    s = mat_row_simplest_form(a)
    # print(s)
    # print("s1", s1)
    # print(e1)
    # print(np.dot(e1, a))
    # print("r1", r1)
