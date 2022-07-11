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
    Ke = (Ee*Ae/L)*np.array([[1, -1],
                             [-1, 1]])
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


def Kglobal(MN, MC, E, glxn, DEl, nu='None', t='None', A='None'):
    """
    INPUTS:
      MN   = Matriz de nodos
      MC   = Matriz de conectividad
      E    = Vector de módulos elásticos de cada elemento
      glxn = Grados de libertad por nodo
      DEl  = Dimensión de los elementos
      nu   = Coeficiente de Poisson
      t    = Espesor de los elementos
      A    = Vector de secciones de cada elemento
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

    if DEl == 2:
        # Elementos triangulares
        alfa = np.zeros([3, Ne])
        beta = np.zeros([3, Ne])
        gama = np.zeros([3, Ne])
        A = np.zeros(Ne)
        B = {}
        D = {}

    for e in range(Ne):
        if DEl == 1:
            if glxn == 1:
                Ke = Kelemental1(MN, MC, E[e], A[e], e)
            elif glxn == 2:
                Ke = Kelemental2(MN, MC, E[e], A[e], e)
        elif DEl == 2:
            nodos = MC[e, :].astype(int)
            alfa[0, e] = MN[nodos[1], 0]*MN[nodos[2], 1]-MN[nodos[2], 0]*MN[nodos[1], 1]
            alfa[1, e] = MN[nodos[0], 0]*MN[nodos[2], 1]-MN[nodos[2], 0]*MN[nodos[0], 1]
            alfa[2, e] = MN[nodos[0], 0]*MN[nodos[1], 1]-MN[nodos[1], 0]*MN[nodos[0], 1]
            beta[0, e] = MN[nodos[1], 1]-MN[nodos[2], 1]
            beta[1, e] = MN[nodos[2], 1]-MN[nodos[0], 1]
            beta[2, e] = MN[nodos[0], 1]-MN[nodos[1], 1]
            gama[0, e] = MN[nodos[2], 0]-MN[nodos[1], 0]
            gama[1, e] = MN[nodos[0], 0]-MN[nodos[2], 0]
            gama[2, e] = MN[nodos[1], 0]-MN[nodos[0], 0]
            A[e] = (alfa[0, e]-alfa[1, e]+alfa[2, e])/2
            B[e] = 1/(2*A[e])*np.array([[beta[0, e], 0, beta[1, e], 0, beta[2, e], 0],
                                        [0, gama[0, e], 0, gama[1, e], 0, gama[2, e]],
                                        [gama[0, e], beta[0, e], gama[1, e], beta[1, e], gama[2, e], beta[2, e]]])
            D[e] = E[e]/(1-nu**2)*np.array([[1, nu, 0],
                                            [nu, 1, 0],
                                            [0, 0, (1-nu)/2]])
            Ke = t*np.abs(A[e])*np.transpose(B[e]).dot(D[e].dot(B[e]))

        fe = np.abs(Ke.max())
        with open(file1, 'a') as f:
            f.write(f'\nMatriz elemental {e}, fe ={fe:4e}\n')
            f.write(f'{Ke/fe}\n')

        for i in range(Nnxe):
            rangoi = np.linspace(i*glxn, (i+1)*glxn-1, glxn).astype(int)
            rangoni = np.linspace(MC[e, i]*glxn, (MC[e, i]+1)*glxn-1, glxn).astype(int)
            for j in range(Nnxe):
                rangoj = np.linspace(j*glxn, (j+1)*glxn-1, glxn).astype(int)
                rangonj = np.linspace(MC[e, j]*glxn, (MC[e, j]+1)*glxn-1, glxn).astype(int)
                Kg[np.ix_(rangoni, rangonj)] += Ke[np.ix_(rangoi, rangoj)]

    fe = np.abs(Kg.max())
    with open(file2, 'a') as f:
        f.write(f'\nMatriz global, fe ={fe:4e}\n')
        f.write(f'{Kg/fe}\n')

    if DEl == 2:
        return Kg, D, B
    elif DEl == 1:
        return Kg


def Kglobal_desdeKe(MN, MC, Ke, glxn):
    """
    INPUTS:
      MN   = Matriz de nodos
      MC   = Matriz de conectividad
      Ke   = Matriz K elemental
      glxn = Grados de libertad por nodo
    OUTPUTS:
      Kg = Matriz K global
    """
    Nn = MN.shape[0]
    Ne, Nnxe = MC.shape
    Kg = np.zeros([glxn*Nn, glxn*Nn])
    for e in range(Ne):
        for i in range(Nnxe):
            rangoi = np.linspace(i*glxn, (i+1)*glxn-1, glxn).astype(int)
            rangoni = np.linspace(MC[e, i]*glxn, (MC[e, i]+1)*glxn-1, glxn).astype(int)
            for j in range(Nnxe):
                rangoj = np.linspace(j*glxn, (j+1)*glxn-1, glxn).astype(int)
                rangonj = np.linspace(MC[e, j]*glxn, (MC[e, j]+1)*glxn-1, glxn).astype(int)
                Kg[np.ix_(rangoni, rangonj)] += Ke[np.ix_(rangoi, rangoj)]
    return Kg