import numpy as np

# Implment Singular Value Decomposition

# takes a column as input and perform householder transformation (Lecture 11)
def householder_col(a):
    if a.shape[0] == 1:
        return np.identity(1)
    # a is a vector, construct v
    e1 = np.zeros((a.shape[0], 1))
    e1[0, 0] = 1
    alpha = np.linalg.norm(x = a, ord = 2)
    if a[0] < 0:
        alpha = -alpha
    v = a + alpha * e1
    
    # construct Hv and calculate Hva
    Hv = np.identity(a.shape[0]) - (2 / ((np.linalg.norm(x = v)) ** 2)) * np.matmul(v, v.T)
    return Hv

# take a row as input and perform a right 
def householder_row(b):
    if b.shape[1] <= 2:
        return np.identity(b.shape[1] - 1)
    # construct v
    e1 = np.zeros((b.shape[1] - 1, 1))
    e1[0, 0] = 1
    temp = np.array(b[0, 1:]).reshape(b.shape[1] - 1, 1)
    beta = np.linalg.norm(temp)
    if temp[0, 0] < 0:
        beta = -beta
    v = temp + beta * e1

    # construct Hv
    Hv = np.identity(v.shape[0]) - (2 / ((np.linalg.norm(x = v)) ** 2)) * np.matmul(v, v.T)
    return Hv

# Phase I: implement Golub-Kahan bidiagonalization
def bi_diag(A):
    iter = 0
    P = np.identity(A.shape[0])
    Q = np.identity(A.shape[1])
    if A.shape[0] >= A.shape[1]:
        iter = A.shape[1]
    else:
        iter = A.shape[0]
    
    for r in range(iter):

        # update the column
        tmp_col = np.array(A[r:, r]).reshape(A.shape[0] - r, 1)
        Hv_col = householder_col(tmp_col)
        
        dim_diff = A.shape[0] - Hv_col.shape[1]
        if dim_diff != 0:
            tmp_mat_1 = np.concatenate((np.identity(dim_diff), np.zeros((dim_diff, Hv_col.shape[1]))), axis = 1)
            tmp_mat_2 = np.concatenate((np.zeros((Hv_col.shape[0], dim_diff)), Hv_col), axis = 1)
            Hv_col = np.concatenate((tmp_mat_1, tmp_mat_2), axis = 0)
        P = np.matmul(P, Hv_col)
        A = np.matmul(Hv_col, A)

        # update the row
        tmp_row = np.array(A[r, r:]).reshape(1, A.shape[1] - r)
        Hv_row = householder_row(tmp_row)
        
        if Hv_row.shape[1] == 0:
            Hv_row = np.identity(A.shape[1])
        else:
            dim_diff = A.shape[1] - Hv_row.shape[0]
            tmp_mat_1 = np.concatenate((np.identity(dim_diff), np.zeros((dim_diff, Hv_row.shape[1]))), axis = 1)
            tmp_mat_2 = np.concatenate((np.zeros((Hv_row.shape[0], dim_diff)), Hv_row), axis = 1)
            Hv_row = np.concatenate((tmp_mat_1, tmp_mat_2), axis = 0)
        Q = np.matmul(Q, Hv_row)
        A = np.matmul(A, Hv_row)
        # for i in range(A.shape[0]):
        #     for j in range (A.shape[1]):
        #         if abs(A[i, j]) < 1e-14:
        #             A[i, j] = 0
    return A, P, Q

# Phase II-A: implement the QR iteration with Wilkinson shift and deflation on matrix $B^TB$
def qr_wilkinson(B):
    eigen_val = list()
    B = np.matmul(B.T, B)
    Q = np.identity(B.shape[0])
    Q_tmp = np.identity(B.shape[0])
    X_old = np.copy(B)
    shift = 0

    # qr iteration with deflation
    while X_old.shape != (1, 1):
        shift_mat = shift * np.identity(X_old.shape[0])     # compute wilkinson shift
        q, r = np.linalg.qr(X_old - shift_mat)
        Q_tmp = np.matmul(q.T, Q_tmp)                       # compute Q_k = q_{k-1}^T ... q_1^T 
        X_old = np.matmul(r, q) + shift_mat
        shift = wilkinson_shift(np.array(X_old[X_old.shape[0] - 2 : X_old.shape[0], X_old.shape[1] - 2 : X_old.shape[1]]))
        val = np.linalg.norm(x = np.array(X_old[0 : X_old.shape[0] - 1, X_old.shape[1] - 1]))

        if val <= 1e-11:
            eigen_val.append(X_old[X_old.shape[0] - 1, X_old.shape[1] - 1])
            X_old = X_old[0 : X_old.shape[0] - 1, 0 : X_old.shape[1] - 1]
            if Q_tmp.shape[0] < Q.shape[0]:
                dim_diff = Q.shape[0] - Q_tmp.shape[0]
                tmp_mat_1 = np.concatenate((Q_tmp, np.zeros((Q_tmp.shape[0], dim_diff))), axis = 1)
                tmp_mat_2 = np.concatenate((np.zeros((dim_diff, Q_tmp.shape[1])), np.identity(dim_diff)), axis = 1)
                Q_tmp = np.concatenate((tmp_mat_1, tmp_mat_2), axis = 0)
            Q = np.matmul(Q_tmp, Q)                         # Q = Q_kQ_{k-1}...Q_1
            Q_tmp = np.identity(X_old.shape[0])
            shift = 0
    eigen_val.append(X_old[0, 0])
    eigen_val.reverse()                                # such that the eigenvalues may coincide with the eigenvectors
    return Q.T, eigen_val

# wilkinson_shift: take a 2 by 2 matrix as input
def wilkinson_shift(M):
    sigma = 0
    x_n = M[1, 1]
    val, vec = np.linalg.eig(M)
    if val[1] - x_n >= val[0] - x_n:
        sigma = val[0]
    else:
        sigma = val[1] 
    return sigma

# construct sigma, u and v matrix, A: m by n, assume that m > n
def svd_dec(A):

    if A.shape[0] >= A.shape[1]:
        # construct V
        A_bidiag, L1, R1 = bi_diag(A)
        V, ATA_val = qr_wilkinson(A_bidiag)
        # V = np.matmul(R1, V)
        # V = V.T
        
        # construct sigma
        singular_val = [(i ** 0.5) for i in ATA_val]
        Sigma_eco = np.diag(singular_val)
        Sigma = np.concatenate((Sigma_eco, np.zeros((A.shape[0] - A.shape[1], Sigma_eco.shape[1]))), axis = 0)

        # construct U
        AT_bidiag, L2, R2 = bi_diag(A_bidiag.T)
        U_tmp, AAT_val = qr_wilkinson(np.array(AT_bidiag))  
        tmp = np.matmul(A_bidiag, V)     # solving U sigma = A V^T
        U = np.matmul(tmp, np.linalg.inv(Sigma_eco))
        U = np.concatenate((U, np.array(U_tmp[:, A.shape[1] : A.shape[0]])), axis = 1)
        U = np.matmul(L1, U)
        V = np.matmul(V.T, R1.T)
    else:
        B = A.T
        V, Sigma, U = svd_dec(B)
        Sigma = Sigma.T
        U = U.T
        V = V.T

    return U, Sigma, V

# This is for part 2 where full decomposition of the matrix U or V is not necessary
def svd_dec_2(A):
    if A.shape[0] >= A.shape[1]:
        # construct V
        A_bidiag, L1, R1 = bi_diag(A)
        V, ATA_val = qr_wilkinson(A_bidiag)

        # construct sigma
        singular_val = np.array([(i ** 0.5) for i in ATA_val])
        singular_val_idx = np.argsort(singular_val, kind = 'mergesort')
        Sigma = np.diag(singular_val)

        # construct U 
        tmp = np.matmul(A_bidiag, V)     # solving U sigma = A V^T
        U = np.matmul(tmp, np.linalg.inv(Sigma))
        U = np.matmul(L1, U)
        V = np.matmul(V.T, R1.T)
    else:
        B = A.T
        V, Sigma, U = svd_dec_2(B)
        Sigma = Sigma.T
        U = U.T
        V = V.T

    return U, Sigma, V, singular_val_idx

# phase II-B: an alternative iteration of the qr iteration with Wilkinson shift
def qr_cholesky(B, iteration):
    X = B
    org_dim = X.shape[0]
    sig_list = list()               # The return values are the singular values of matrix B
    Q = np.identity(B.shape[0])     # the eigenvectors of $B^TB$
    Q_tmp = np.identity(B.shape[0])
    for i in range(iteration):
        if X.shape == (1, 1):
            sig_list.append(X[0, 0])
            break
        q, r = np.linalg.qr(X.T)
        Q_tmp = np.matmul(q.T, Q_tmp)
        tmp = np.matmul(r, r.T)
        l = np.linalg.cholesky(tmp)
        X = l.T
        # conduct the deflation
        if abs(X[X.shape[0] - 2, X.shape[1] - 1]) < 1e-12:
            sig_list.append(X[X.shape[0] - 1, X.shape[1] - 1])
            dim_diff = org_dim - Q_tmp.shape[0]
            if dim_diff != 0:
                mat_tmp1 = np.concatenate((Q_tmp, np.zeros((Q_tmp.shape[0], dim_diff))), axis = 1)
                mat_tmp2 = np.concatenate((np.zeros((dim_diff, Q_tmp.shape[1])), np.identity(dim_diff)), axis = 1)
                Q_tmp = np.concatenate((mat_tmp1, mat_tmp2), axis = 0)
            
            Q = np.matmul(Q_tmp, Q)
            X = np.array(X[0 : X.shape[0] - 1, 0 : X.shape[1] - 1])
            Q_tmp = np.identity(X.shape[0])
    
    sig_list.reverse()
    return Q.T, sig_list

# phase II-B: SVD decompostion using the alternative iteration
def svd_dec_alt(A, iter):
    bd_mat, l, r = bi_diag(A)
    flag = 0
    if bd_mat.shape[0] >= bd_mat.shape[1]:
        bd_mat = bd_mat[0 : bd_mat.shape[1], 0 : bd_mat.shape[1]]
    else:
        bd_mat = bd_mat[0 : bd_mat.shape[0], 0 : bd_mat.shape[0]]
        flag = 1
    
    if flag == 0:
        V, sig_val = qr_cholesky(bd_mat, iter)

        Sig_eco = np.diag(sig_val)
        if flag == 0:
            Sigma = np.concatenate((Sig_eco, np.zeros((A.shape[0] - A.shape[1], Sig_eco.shape[1]))), axis = 0)

        # construct U
        AT_bdmat, l2, r2 = bi_diag(A.T)
        if flag == 0:
            AT_bdmat = AT_bdmat[0 : AT_bdmat.shape[0], 0 : AT_bdmat.shape[0]]
        else:
            AT_bdmat = AT_bdmat[0 : AT_bdmat.shape[1], 0 : AT_bdmat.shape[1]]

        tmp = np.matmul(bd_mat, V)
        
        U = np.matmul(tmp, np.linalg.inv(Sig_eco))
        dim_diff = A.shape[0] - A.shape[1]
        tmp_mat_1 = np.concatenate((U, np.zeros((U.shape[0], dim_diff))), axis = 1)
        tmp_mat_2 = np.concatenate((np.zeros((dim_diff, U.shape[1])), np.identity(dim_diff)), axis = 1)
        U = np.concatenate((tmp_mat_1, tmp_mat_2), axis = 0)
        U = np.matmul(l, U)
        V = np.matmul(V.T, r.T)
    else:
        B = A.T
        V, Sigma, U = svd_dec_alt(B, iter)
        Sigma = Sigma.T
        U = U.T
        V = V.T
    return U, Sigma, V

def svd_dec_alt_2(A, iter):
    bd_mat, l, r = bi_diag(A)
    flag = 0
    if bd_mat.shape[0] >= bd_mat.shape[1]:
        bd_mat = bd_mat[0 : bd_mat.shape[1], 0 : bd_mat.shape[1]]
    else:
        bd_mat = bd_mat[0 : bd_mat.shape[0], 0 : bd_mat.shape[0]]
        flag = 1
    
    if flag == 0:
        V, sig_val = qr_cholesky(bd_mat, iter)

        singular_val_idx = np.argsort(sig_val, kind = 'mergesort')
        Sigma = np.diag(sig_val)

        # construct U
        AT_bdmat, l2, r2 = bi_diag(A.T)
        if flag == 0:
            AT_bdmat = AT_bdmat[0 : AT_bdmat.shape[0], 0 : AT_bdmat.shape[0]]
        else:
            AT_bdmat = AT_bdmat[0 : AT_bdmat.shape[1], 0 : AT_bdmat.shape[1]]

        tmp = np.matmul(bd_mat, V)
        
        U = np.matmul(tmp, np.linalg.inv(Sigma))
        dim_diff = A.shape[0] - A.shape[1]
        tmp_mat_1 = np.concatenate((U, np.zeros((U.shape[0], dim_diff))), axis = 1)
        tmp_mat_2 = np.concatenate((np.zeros((dim_diff, U.shape[1])), np.identity(dim_diff)), axis = 1)
        U = np.concatenate((tmp_mat_1, tmp_mat_2), axis = 0)
        U = np.matmul(l, U)
        V = np.matmul(V.T, r.T)
    else:
        B = A.T
        V, Sigma, U, singular_val_idx = svd_dec_alt_2(B, iter)
        Sigma = Sigma.T
        U = U.T
        V = V.T
    return U, Sigma, V, singular_val_idx    