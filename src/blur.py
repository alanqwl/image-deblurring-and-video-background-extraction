import numpy as np
import SVD
import math
from scipy import sparse
from matplotlib import pyplot as plt
from PIL import Image
from time import time

def construct_kernel(img):
    sz = img.shape[0]

    # generate n numbers, the sum of the n number is 1
    element = np.random.dirichlet(np.ones(sz), size=1)
    data = list()
    for i in range(sz):
        data.append(np.ones(sz))
        data[i] = element[0, i] * data[i]
    data = np.array(data)
    diags = list(range(sz))
    dgs = np.array(diags)
    
    A = sparse.spdiags(data, dgs, sz, sz)
    AT = A.transpose()
    A = A + AT
    return A

def construct_kernel_2(img, k):
    sz = img.shape[0]
    delta = 0.1
    diag_entry = (2 + delta) / (4 + delta)
    diag_2_entry = 1 / (4 + delta)
    A = np.zeros((sz, sz))
    for i in range(sz):
        if i == 0:
            A[i, i] = diag_entry
            A[i, i + 1] = diag_2_entry
        elif i == sz - 1:
            A[i, i - 1] = diag_2_entry
            A[i, i] = diag_entry
        else:
            A[i, i - 1] = diag_2_entry
            A[i, i] = diag_entry
            A[i, i + 1] = diag_2_entry
    A = np.linalg.matrix_power(A, k)
    return A

def blur_image(image, Al, Ar):
    sz = image.shape[0]
    try:
        dims = image.shape[2]
        for dim in range(dims):
            tmp_mat = (Al.toarray() @ image[:, :, dim].reshape(sz, sz) @ Ar.toarray()).reshape(sz * sz, 1)
            blur_im_mat = tmp_mat if dim == 0 else np.concatenate((blur_im_mat, tmp_mat), axis = 1)
        blur_im_mat = blur_im_mat.reshape(sz, sz, dims)
    except:
        blur_im_mat = Al.toarray() @ image @ Ar.toarray()

    return blur_im_mat

def blur_image_2(image, A):
    sz = image.shape[0]
    try:
        dims = image.shape[2]
        for dim in range(dims):
            tmp_mat = (A @ image[:, :, dim].reshape(sz, sz) @ A).reshape(sz * sz, 1)
            blur_im_mat = tmp_mat if dim == 0 else np.concatenate((blur_im_mat, tmp_mat), axis = 1)
        blur_im_mat = blur_im_mat.reshape(sz, sz, dims)
    except:
        blur_im_mat = A @ image @ A

    return blur_im_mat

def trunc_SVD(A, _trunc):
    
    try:
        t1 = time()
        U, sigma, V, index_list = SVD.svd_dec_2(A.toarray())
        t2 = time()
    except:
        t1 = time()
        U, sigma, V, index_list = SVD.svd_dec_2(A)
        t2 = time()

    print("run time: ", t2 - t1)
    VT = V.T
    # generate truncated reconstruction
    A_inv = np.zeros((A.shape[0], A.shape[1]))
    for i in range(_trunc):
        pos = index_list[sigma.shape[1] - 1 - i]
        ui = U[:, pos].reshape(1, A.shape[1])
        vi = VT[:, pos].reshape(A.shape[0], 1)
        sigma_i = sigma[pos, pos]
        tmp = (np.matmul(vi, ui)) / sigma_i
        A_inv = A_inv + tmp
    return A_inv

def deblur_image(image, Al_inv, Ar_inv):
    sz = image.shape[0]
    try:
        dims = image.shape[2]
        for dim in range(dims):
            tmp_mat = image[:, :, dim].reshape(sz, sz)
            tmp_mat = np.matmul(Al_inv, tmp_mat)
            tmp_mat = np.matmul(tmp_mat, Ar_inv)
            tmp_mat = tmp_mat.reshape(sz * sz, 1)
            recover_mat = tmp_mat if dim == 0 else np.concatenate((recover_mat, tmp_mat), axis = 1)
        recover_mat = recover_mat.reshape(sz, sz, dims)
    except:
        recover_mat = Al_inv @ image @ Ar_inv   
    return recover_mat

def PSNR(rec_image, org_image, sz):
    if org_image.shape != (sz, sz):
        rec_image = rec_image.reshape(sz * sz, rec_image.shape[2])
        org_image = org_image.reshape(sz * sz, org_image.shape[2])
    psnr = 10 * math.log10((sz ** 2) / (np.linalg.norm(rec_image - org_image, ord = 'fro') ** 2))
    return psnr

if __name__ == '__main__':
    pass