import numpy as np

def MEF(KGlobal, s,r, Us, Fr):
    N = KGlobal.shape[1]
    F = np.zeros([N, 1])
    U = np.zeros([N, 1])
    U[s] = np.transpose([Us])
    F[r] = np.transpose([Fr])

    KRed = KGlobal[np.ix_(r,r)]
    KVin = KGlobal[np.ix_(r,s)]
    U[r] = np.linalg.solve(KRed, F[r]-KVin.dot(U[s]))
    F[s] = KGlobal[s,:].dot(U)

    return U,F