import numpy as np
import SVD


if __name__ == '__main__':
    
    c = np.array([[17, 24, 31, 45],
                [21, 32, 78, 10],
                [9, 15, 2, 62],
                [20, 42, 54, 73],
                [38, 25, 2, 19]])

    # d = np.array([[1, 2, 3, 13],
    #               [2, 3, 4, 22],
    #               [3, 4, 9, 10],
    #               [7, 6, 5, 24]])

    # t = np.array([[52, 30, 49, 28],
    #               [30, 50, 8, 44],
    #               [49, 8, 46, 16],
    #               [28, 44, 16, 22]])

    # test svd decomposition
    u_test, sig_test, v_test = SVD.svd_dec(c)
    u, s, v = np.linalg.svd(c)
    print("===testing===")
    print("matrix A: ", c, "\n")
    print("u = ", u_test, "\n")
    print("s = ", sig_test, "\n")
    print("v = ", v_test, "\n")

    result = np.matmul(u_test, sig_test)
    result = np.matmul(result, v_test)
    print("recover result: ", result, "\n")
    print("===using inbuilt function===")

    print("u = ", u, "\n")
    print("s = ", s, "\n")
    print("v = ", v, "\n")

    # a, b, d = SVD.svd_dec_alt(c, 200)
    # print("u = ", a, "\n")
    # print("s = ", b, "\n")
    # print("v = ", d, "\n")
    # test = np.matmul(a, b)
    # test = np.matmul(test, d)
    # print("recover result = ", test, "\n")