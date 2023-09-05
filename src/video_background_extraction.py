import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg

def power_iteration_rayleigh(A, iter):
    tmp_mat = A 
    identity_mat = sparse.identity(tmp_mat.shape[0])
    iter_mat = sparse.eye(m = A.shape[0], n = 1)
    shift = 1
    for i in range(iter):
        iter_mat = (tmp_mat + shift * identity_mat.toarray()) @ iter_mat
        iter_mat = iter_mat / np.linalg.norm(iter_mat)
        shift = iter_mat.T @ tmp_mat @ iter_mat
        
    return shift ** 0.5, iter_mat

def largest_singular_SVD(A):
    sa = sparse.csr_matrix(A.astype(np.float64))
    A = A.astype(np.float64)
    singular_val, v = power_iteration_rayleigh(A.T @ A, 1000)
    u = (1 / singular_val) * (A @ v)
    print('=====compute=====')
    print(u, '\n')
    print(singular_val, '\n')
    print(v, '\n')
    
    test_u, test_s, test_v = linalg.svds(A, k = 1)
    print('=====inbuilt=====')
    print(test_u, '\n')
    print(test_s, '\n')
    print(test_v, '\n')
    e = sparse.eye(m = v.shape[0], n = 1)
    vec_B = (singular_val * (v.T @ e)) * u
    # print(vec_B, '\n')
    return vec_B

def stack(video_path, frame_list):
    cap = cv.VideoCapture(video_path)
    iter = 0
    frame_mat = cap.read()[1]
    m = frame_mat.shape[0]
    n = frame_mat.shape[1]
    for frame_num in frame_list:
        
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
        frame_mat = cap.read()[1]
        # cv.imshow('frame', frame_mat)
        # cv.waitKey(2000)
        frame_mat = frame_mat.reshape(m * n, frame_mat.shape[2])
        # print(frame_mat, '\n')
        if iter == 0:
            R = frame_mat[:, 0].reshape(m * n, 1)
            G = frame_mat[:, 1].reshape(m * n, 1)
            B = frame_mat[:, 2].reshape(m * n, 1)
        else:
            R = np.concatenate((R, frame_mat[:, 0].reshape(m * n, 1)), axis = 1)
            G = np.concatenate((G, frame_mat[:, 1].reshape(m * n, 1)), axis = 1)
            B = np.concatenate((B, frame_mat[:, 2].reshape(m * n, 1)), axis = 1)
        iter += 1
    cap.release()
    return R, G, B, m, n

def construct_background(R_b, G_b, B_b, m, n):
    tmp = R_b
    tmp = np.concatenate((tmp, G_b), axis = 1)
    tmp = np.concatenate((tmp, B_b), axis = 1)
    tmp = tmp.reshape(m, n, 3)
    return tmp

if __name__ == '__main__':

    path = 'test_videos/1280_720/bangkok.mp4'
    frame_num = np.linspace(0, 300, 100)
    r, g, b, dim1, dim2 = stack(video_path = path, frame_list = frame_num)
    r_b = largest_singular_SVD(r)
    g_b = largest_singular_SVD(g)
    b_b = largest_singular_SVD(b)
    fig = construct_background(r_b, g_b, b_b, dim1, dim2)
    fig = fig.astype(np.float64) / 255

    plt.figure(1)
    plt.axis('off')
    plt.gray()
    plt.imshow(fig)
    plt.show()  

    # cap = cv.VideoCapture('test_videos/640_360/walking.mp4')
    # cap.set(cv.CAP_PROP_POS_FRAMES, 50)
    # b = cap.read()[1]
    # b = b.reshape(b.shape[0] * b.shape[1], b.shape[2])
    # test = largest_singular_SVD(b)
    # print(test)

    # cv.imshow('b',  b[1])
    # cv.waitKey(1000)
    # len_video = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    # print("number of frame:", len_video, '\n')

    # while (cap.isOpened()):
    #     ret, frame = cap.read()
    #     if ret == True:
    #         cv.imshow('frame', frame)

    #         if cv.waitKey(25) & 0xff == ord('q'):
    #             break

    #     else:
    #         break
    
    # cap.release()
    # cv.destroyAllWindows()