import numpy as np


def solve(K, r, Fr, s, Us):
    """
    INPUTS:
      K  = Matriz K global (relaciona los desplazamientos con las fuerzas)
      r  = Vector con los nodos con condiciones de vínculo de fuerza
      Fr = Vector con las fuerzas en cada nodo del vector 'r'
      s  = Vector con los nodos con condiciones de vínculo de desplazamiento
      Us = Vector con los desplazamientos en cada nodo del vector 's'
    OUTPUTS:
      F = Vector de fuerzas en cada nodo
      U = Vector de desplazamientos de cada nodo
    """
    N = np.shape(K)[1]
    F = np.zeros([N, 1])
    U = np.zeros([N, 1])
    U[s] = Us
    F[r] = Fr
    Kr = K[np.ix_(r, r)]
    Kv = K[np.ix_(r, s)]
    U[r] = np.linalg.solve(Kr, F[r]-Kv.dot(U[s]))
    F[s] = K[s, :].dot(U)
    return F, U