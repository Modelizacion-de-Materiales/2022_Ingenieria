import numpy as np
from scipy import linalg

def k_elemental_2D(MN, MC, Ee, nu_e, t_e, element):
    '''
    Construcción del K elemental
    MN: Matriz de nodos del sistema
    MC: Matriz de conectividad
    Ee: Módulo de Young para el elemento
    nu_e: Coeficiente de Poisson del elemento
    t_e: espesor del elemento de la chapa. element thickness
    element: número de elemento
    '''

    alpha = np.zeros(3)
    beta = np.zeros(3)
    gamma = np.zeros(3)
    nodos_del_elemento = MC[element, :].astype(int)

    alpha[0] = MN[nodos_del_elemento[1], 0] * MN[nodos_del_elemento[2], 1] - MN[nodos_del_elemento[2], 0] * MN[nodos_del_elemento[1], 1]
    alpha[1] = MN[nodos_del_elemento[0], 0] * MN[nodos_del_elemento[2], 1] - MN[nodos_del_elemento[2], 0] * MN[nodos_del_elemento[0], 1]
    alpha[2] = MN[nodos_del_elemento[0], 0] * MN[nodos_del_elemento[1], 1] - MN[nodos_del_elemento[1], 0] * MN[nodos_del_elemento[0], 1]
    beta[0] = MN[nodos_del_elemento[1], 1] - MN[nodos_del_elemento[2], 1]
    beta[1] = MN[nodos_del_elemento[2], 1] - MN[nodos_del_elemento[0], 1]
    beta[2] = MN[nodos_del_elemento[0], 1] - MN[nodos_del_elemento[1], 1]
    gamma[0] = MN[nodos_del_elemento[2], 0] - MN[nodos_del_elemento[1], 0]
    gamma[1] = MN[nodos_del_elemento[0], 0] - MN[nodos_del_elemento[2], 0]
    gamma[2] = MN[nodos_del_elemento[1], 0] - MN[nodos_del_elemento[0], 0]
    Ae = (alpha[0] - alpha[1] + alpha[2]) / 2 #
    B = np.array([[beta[0], 0, beta[1], 0, beta[2], 0],
				  [0, gamma[0], 0, gamma[1], 0, gamma[2]],
                  [gamma[0], beta[0], gamma[1], beta[1], gamma[2], beta[2]]]) / (2 * Ae)

    D = Ee / (1 - nu_e ** 2) * np.array([[1, nu_e, 0], [nu_e, 1, 0], [0, 0, (1 - nu_e) / 2]])
    Ke = t_e * np.abs(Ae) * np.transpose(B).dot(D.dot(B))

    return Ke, B, D

def k_ensemble2D(MN, MC, E, glxn, v=None, t=None):
	'''
	Ensamblado de la matriz K.
	MN: Matriz de nodos del sistema
	MC: Matriz de conectividad
	E: Módulo de Young
	glxn: Grados de libertado por nodo
	dimension: dimensión del problema (acotado a 1 o 2)
	A: Vector de área de los elementos
	v: Vector de coeficientes de Poisson
	t: Vector de espesores de los elementos
	'''

	Nn = MN.shape[0]  # Número de nodos
	Ne, Nnxe = MC.shape  # Ne: Número de elementos, # Número de nodos x elemento
	B = []
	D = []

	Kg = np.zeros([glxn * Nn, glxn * Nn])
	for element in range(Ne):
		Ee = E[element]
		nu_e = v[element]
		t_e = t[element]
		Ke, Be, De = k_elemental_2D(MN, MC, Ee, nu_e, t_e, element)
		B.append(Be)
		D.append(De)

		for i in range(Nnxe):
			indices_i = np.linspace(i * glxn, (i + 1) * glxn - 1, Nnxe).astype(int)
			rangoni = np.linspace(MC[element, i] * glxn, (MC[element, i] + 1) * glxn - 1, Nnxe).astype(int)
			for j in range(Nnxe):
				indices_j = np.linspace(j * glxn, (j + 1) * glxn - 1, Nnxe).astype(int)
				rangonj = np.linspace(MC[element, j] * glxn, (MC[element, j] + 1) * glxn - 1, Nnxe).astype(int)
				Kg[np.ix_(rangoni, rangonj)] += Ke[np.ix_(indices_i, indices_j)]

	return Kg, B, D

def solve(KGlobal, s,r, Us, Fr):
    """
    KGlobal: Matriz de rigidez del sistema.
    s : Contiene los desplazamientos de las cond de contorno. numpy array que contiene los indices de estas filas
    KGlobal: Matriz de rigidez del sistema.
    r : Contiene los desplazamientos incognitas. numpy array que contiene los indices de estas filas
    Us: Vector que contiene el valor de los desplazamientos conocidos
    Fr: Vector que contiene el valor de las fuerzas conocidas
    """
    N = KGlobal.shape[1]
    U = np.zeros(N)
    F = np.zeros(N)
    U[s] = Us
    F[r] = Fr
    Kred = KGlobal[np.ix_(r,r)]
    Kvin = KGlobal[np.ix_(r,s)]
    U[r] = np.linalg.solve(Kred, F[r]-Kvin.dot(U[s]))
    F[s] = KGlobal[s,:].dot(U)
    return U,F

def KGlobal_Barra(MN, MC, GLXN, Ke):
	"""
	MN:Matriz de Nodos 
	MC:Matriz de Conectividad
	E: Modulo elástico
	A:Seccion de la barra
	GLXN: Grados de libertad por nodo
	NN:Numero de nodos
	NE: Numero de elementos
	"""
	NN=len(MN)
	NE,NNXE=MC.shape
	K_Global = np.zeros([GLXN*NN, GLXN*NN])  
	for e in range (NE):
		for i in range(NNXE):
			rangoi = np.linspace(i*GLXN, (i+1)*GLXN-1, NNXE).astype(int)
			rangoni = np.linspace(MC[e, i]*GLXN, (MC[e, i]+1)*GLXN-1, NNXE).astype(int)
			for j in range(NNXE):
				rangoj = np.linspace(j*GLXN, (j+1)*GLXN-1, NNXE).astype(int)
				rangonj = np.linspace(MC[e, j]*GLXN, (MC[e, j]+1)*GLXN-1, NNXE).astype(int)
				K_Global[np.ix_(rangoni, rangonj)] += Ke[np.ix_(rangoi, rangoj)]
	return K_Global

