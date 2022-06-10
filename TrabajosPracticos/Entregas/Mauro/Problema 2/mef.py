import numpy as np

def solve(K, R, FR, S, US):
    """
    Entradas:
      K  = Matriz K global (relaciona los desplazamientos con las fuerzas)
      R  = Vector con los nodos con condiciones de vínculo de fuerza
      FR = Vector con las fuerzas en cada nodo del vector 'r'
      S  = Vector con los nodos con condiciones de vínculo de desplazamiento
      US = Vector con los desplazamientos en cada nodo del vector 's'
      
    Salidas:
      F = Vector de fuerzas en cada nodo
      U = Vector de desplazamientos de cada nodo
    """
    N = np.shape(K)[1]
    F = np.zeros([N, 1])
    U = np.zeros([N, 1])
    U[S] = US
    F[R] = FR
    Kr = K[np.ix_(R, R)]
    Kv = K[np.ix_(R, S)]
    U[R] = np.linalg.solve(Kr, F[R]-Kv.dot(U[S]))
    F[S] = K[S, :].dot(U)
    
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