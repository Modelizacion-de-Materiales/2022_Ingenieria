import numpy as np

def solve(K, r, Fr, s, Us):
    """
    Entradas:
      K  = Matriz K global (relaciona los desplazamientos con las fuerzas)
      r  = Vector con los nodos con condiciones de vínculo de fuerza
      Fr = Vector con las fuerzas en cada nodo del vector 'r'
      s  = Vector con los nodos con condiciones de vínculo de desplazamiento
      Us = Vector con los desplazamientos en cada nodo del vector 's'
    Salidas:
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


def Kelemental(MN, MC, E, A, n_element):
    """
    Entradas:
      MN = Matriz de nodos
      MC = Matriz de conectividad
      E = Módulo elástico del elemento
      A = Sección del elemento
      n_element  = Número de elemento
    Salidas:
      Ke = Matriz K elemental
    """
    Lx = MN[MC[n_element, 1], 0]-MN[MC[n_element, 0], 0]
    Ly = MN[MC[n_element, 1], 1]-MN[MC[n_element, 0], 1]
    L = np.sqrt(Lx**2+Ly**2)
    phi = np.arctan2(Ly, Lx)
    cos = np.cos(phi)
    sin = np.sin(phi)
    Ke = (E*A/L)*np.array([[cos**2, cos*sin, -cos**2, -cos*sin],
                           [cos*sin, sin**2, -cos*sin, -sin**2],
                           [-cos**2, -cos*sin, cos**2, cos*sin],
                           [-cos*sin, -sin**2, cos*sin, sin**2]])
    Ke[np.abs(Ke/Ke.max()) < 1E-15] = 0
    return Ke


def Kglobal(MN, MC, E, A, glxn):
    """
    Entradas:
      MN   = Matriz de nodos
      MC   = Matriz de conectividad
      E    = Vector de módulos elásticos de cada elemento
      A    = Vector de secciones de cada elemento
      glxn = Grados de libertad por nodo
    Salidas:
      Kg = Matriz K global
    """
    Nn = MN.shape[0]
    Ne, Nnxe = MC.shape
    Kg = np.zeros([glxn*Nn, glxn*Nn])
    for e in range(Ne):
        Ee = E[e]
        Ae = A[e]
        Ke = Kelemental(MN, MC, Ee, Ae, e)
        for i in range(Nnxe):
            rangoi = np.linspace(i*glxn, (i+1)*glxn-1, Nnxe).astype(int)
            rangoni = np.linspace(MC[e, i]*glxn, (MC[e, i]+1)*glxn-1, Nnxe).astype(int)
            for j in range(Nnxe):
                rangoj = np.linspace(j*glxn, (j+1)*glxn-1, Nnxe).astype(int)
                rangonj = np.linspace(MC[e, j]*glxn, (MC[e, j]+1)*glxn-1, Nnxe).astype(int)
                Kg[np.ix_(rangoni, rangonj)] += Ke[np.ix_(rangoi, rangoj)]
    return Kg


def MG(MN, MC, glxn, ME):
    """
    Entradas:
      MN   = Matriz de nodos
      MC   = Matriz de conectividad
      glxn = Grados de libertad por nodo
      ME   = Matriz elemental

    Salidas:
      MG = Matriz global
    """
    n_nodos = MN.shape[0]
    n_element, n_nxe = MC.shape
    MG = np.zeros([glxn*Nn, glxn*Nn])
    MG = np.zeros([glxn*n_nodos, glxn*n_nodos])
    for e in range (n_element):
        for i in range(n_nxe):
            rangoi = np.linspace(i*glxn, (i+1)*glxn-1, glxn).astype(int)
            rangoni = np.linspace(MC[e, i]*glxn, (MC[e, i]+1)*glxn-1, glxn).astype(int)
            for j in range(n_nxe):
                rangoj = np.linspace(j*glxn, (j+1)*glxn-1, glxn).astype(int)
                rangonj = np.linspace(MC[e, j]*glxn, (MC[e, j]+1)*glxn-1, glxn).astype(int)
                MG[np.ix_(rangoni, rangonj)] += ME[np.ix_(rangoi, rangoj)]
    return MG