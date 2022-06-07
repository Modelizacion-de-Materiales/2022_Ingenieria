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


def Kelemental1(MN, MC, Ee, Ae, e):
    """
    INPUTS:
      MN = Matriz de nodos
      MC = Matriz de conectividad
      Ee = Módulo elástico del elemento
      Ae = Sección del elemento
      e  = Número de elemento
    OUTPUTS:
      Ke = Matriz K elemental
    """
    L = MN[-1, 0]/MC.shape[0]
    Ke = (Ee*Ae/L)*np.array([[ 1, -1],
                             [-1,  1]])
    Ke[np.abs(Ke/Ke.max()) < 1E-15] = 0
    return Ke


def Kelemental2(MN, MC, Ee, Ae, e):
    """
    INPUTS:
      MN = Matriz de nodos
      MC = Matriz de conectividad
      Ee = Módulo elástico del elemento
      Ae = Sección del elemento
      e  = Número de elemento
    OUTPUTS:
      Ke = Matriz K elemental
    """
    Lx = MN[MC[e, 1], 0]-MN[MC[e, 0], 0]
    Ly = MN[MC[e, 1], 1]-MN[MC[e, 0], 1]
    L = np.sqrt(Lx**2+Ly**2)
    phi = np.arctan2(Ly, Lx)
    cos = np.cos(phi)
    sin = np.sin(phi)
    Ke = (Ee*Ae/L)*np.array([[cos**2, cos*sin, -cos**2, -cos*sin],
                             [cos*sin, sin**2, -cos*sin, -sin**2],
                             [-cos**2, -cos*sin, cos**2, cos*sin],
                             [-cos*sin, -sin**2, cos*sin, sin**2]])
    Ke[np.abs(Ke/Ke.max()) < 1E-15] = 0
    return Ke


def Kglobal(MN, MC, E, A, glxn):
    """
    INPUTS:
      MN   = Matriz de nodos
      MC   = Matriz de conectividad
      E    = Vector de módulos elásticos de cada elemento
      A    = Vector de secciones de cada elemento
      glxn = Grados de libertad por nodo
    OUTPUTS:
      Kg = Matriz K global
    """

    Ke = {}
    file1 = 'Ke.txt'
    with open(file1, 'w') as f:
        f.write('Matrices elementales\n=================================================')
    file2 = 'Kg.txt'
    with open(file2, 'w') as f:
        f.write('Matriz global\n=================================================')

    Nn = MN.shape[0]
    Ne, Nnxe = MC.shape
    Kg = np.zeros([glxn*Nn, glxn*Nn])
    for e in range(Ne):

        if glxn == 1:
            Ke = Kelemental1(MN, MC, E[e], A[e], e)
        elif glxn == 2:
            Ke = Kelemental2(MN, MC, E[e], A[e], e)

        fe = np.abs(Ke.max())
        with open(file1, 'a') as f:
            f.write(f'\nMatriz elemental {e}, fe ={fe:4e}\n')
            f.write(f'{Ke/fe}\n')

        for i in range(Nnxe):
            rangoi = np.linspace(i*glxn, (i+1)*glxn-1, Nnxe).astype(int)
            rangoni = np.linspace(MC[e, i]*glxn, (MC[e, i]+1)*glxn-1, Nnxe).astype(int)
            for j in range(Nnxe):
                rangoj = np.linspace(j*glxn, (j+1)*glxn-1, Nnxe).astype(int)
                rangonj = np.linspace(MC[e, j]*glxn, (MC[e, j]+1)*glxn-1, Nnxe).astype(int)
                Kg[np.ix_(rangoni, rangonj)] += Ke[np.ix_(rangoi, rangoj)]

    fe = np.abs(Kg.max())
    with open(file2, 'a') as f:
        f.write(f'\nMatriz global, fe ={fe:4e}\n')
        f.write(f'{Kg/fe}\n')
    return Kg


def Ej03_solve(c, L, Ne, E, A):

    glxn = 1

    MC = np.array([[i, i+1] for i in range(Ne)])

    MN = np.linspace(0, L, Ne+1).reshape([-1, 1])
    MN = np.append(MN, np.zeros([Ne+1, 2]).reshape([-1, 2]), axis=1)

    K = Kglobal(MN, MC, E, A, glxn)

    FT = 0.5*c*(L/Ne)**2
    f0 = FT/3
    f1 = 2*FT/3
    f = np.zeros([Ne+1]).reshape([-1, 1])
    for i in range(Ne):
        Fu = FT*i
        f[i:(i+2)] += np.array([[Fu+f0], [Fu+f1]])

    U = np.linalg.solve(K[:Ne, :Ne], f[:Ne])
    U = np.append(U, np.array([[0]]), 0)
    F = K.dot(U)

    eps = np.zeros([Ne, 1])
    sig = np.zeros([Ne, 1])
    for i in range(Ne):
        eps[i] = (U[i+1]-U[i])/(L/Ne)
        sig[i] = eps[i]*E[i]

    R = F[-1]-f[-1]

    return U, F, sig, R