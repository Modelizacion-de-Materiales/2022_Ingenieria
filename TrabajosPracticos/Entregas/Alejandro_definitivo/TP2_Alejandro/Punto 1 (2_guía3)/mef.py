import numpy as np
from scipy import linalg

# La función "solve" calcula los vectores de fuerzas "F" y de desplazamientos "U", empleando MEF.
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
    N = np.shape(K)[1]   # Número de nodos
    F = np.zeros([N, 1]) # Vector de fuerzas "F", una fila por cada nodo.
    U = np.zeros([N, 1]) # Vector de desplazamientos "U", una fila por cada nodo.
    U[s] = Us            # Completo "U" con los desplazamientos "Us" que ya conozco.
    F[r] = Fr            # Completo "F" con las fuerzas "Fr" que ya conozco.
    # Teniendo en cuenta que los desplazamientos "Us" y las fuerzas "Fr" son conocidas, buscaré identificar el subsistema de
    # ecuaciones tal que "Ur" sean las incógnitas que podría calcular con "Fr". De aquí surge la partición de la matriz 
    # global "K" en las submatrices "Kr" (MATRIZ REDUCIDA, asociada a desplazamientos incógnitas "Ur") y "Kv" (MATRIZ DE
    # VÍNCULOS, asociada a desplazamientos conocidos "Us").
    Kr = K[np.ix_(r, r)]
    Kv = K[np.ix_(r, s)]    
    U[r] = np.linalg.solve(Kr, F[r]-Kv.dot(U[s])) # Obtengo los desplazamientos incógnita "Ur", a partir de "Fr", "Us" y "Kv".
    F[s] = K[s, :].dot(U) # Tengo todos los desplazamientos "U" conocidos. Calculo las fuerzas "Fs" incógnitas que me faltan.
    return F, U

# La función "Kelemental" calcula la matriz elemental "Ke" del elemento "e".
def Kelemental(MN, MC, Ee, Ae, e):
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
    nodo1 = MC[e, 0]   # Primer nodo que conforma al elemento "e".
    nodo2 = MC[e, 1]   # Segundo nodo que conforma al elemento "e".
    Lx = MN[nodo2, 0]-MN[nodo1, 0]   # Longitud en eje "x".
    Ly = MN[nodo2, 1]-MN[nodo1, 1]   # Longitud en eje "y".
    L = np.sqrt(Lx**2+Ly**2)         # Longitud del elemento "e" (calculé la norma).
    phi = np.arctan2(Ly, Lx)         # Ángulo de inclinación del elemento "e".
    cos = np.cos(phi)
    sin = np.sin(phi)
    Ke = (Ee*Ae/L)*np.array([[cos**2, cos*sin, -cos**2, -cos*sin],
                             [cos*sin, sin**2, -cos*sin, -sin**2],
                             [-cos**2, -cos*sin, cos**2, cos*sin],
                             [-cos*sin, -sin**2, cos*sin, sin**2]])
    Ke[np.abs(Ke/Ke.max()) < 1e-15] = 0   # Para que me aparezca "0" en lugar de exponentes "1e-34", por ejemplo.
    return Ke

# La función "Kglobal" calcula la matriz global "K".
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
    Nn = MN.shape[0]      # "Nn" es número de nodos.
    Ne, Nnxe = MC.shape   # "Ne" es número de elementos. "Nnxe" es número de nodos por elemento.
    Kg = np.zeros([glxn*Nn, glxn*Nn])   # Defino matriz global "Kg".
    
    archivo= 'Matrices_elementales.txt'
    with open(archivo,'w') as f:   # Creo archivo desde cero, por eso uso "w".
        f.write('Matrices Elementales\n ===============')
    archivo1= 'Matriz_global.txt'
    with open(archivo1,'w') as f:   # Creo archivo desde cero, por eso uso "w".
        f.write('Matriz Global\n ===============')
    
    for e in range(Ne): 
        if glxn == 1:
            Ke = np.array([[1,-1],[-1,1]])*A*E/(MN[-1]/Ne)   # MN[-1] es la longitud "L" de la barra entera.
        elif glxn == 2:
            Ke = Kelemental(MN, MC, E[e], A[e], e)
        fe = np.abs(Ke.max()) # Factor de escala, para que los números en "Ke" se lean mejor.
        with open(archivo,'a') as f:   # Voy reescribiendo el archivo con nuevas "Ke", por eso uso "a".
            f.write(f'\nelemento {e}, fe = {fe:4e}\n')
            f.write(f'{Ke/fe}\n')
        for i in range(Nnxe):
            rangoi = np.linspace(i*glxn, (i+1)*glxn-1, Nnxe).astype(int)
            rangoni = np.linspace(MC[e, i]*glxn, (MC[e, i]+1)*glxn-1, Nnxe).astype(int)
            for j in range(Nnxe):
                rangoj = np.linspace(j*glxn, (j+1)*glxn-1, Nnxe).astype(int)
                rangonj = np.linspace(MC[e, j]*glxn, (MC[e, j]+1)*glxn-1, Nnxe).astype(int)
                Kg[np.ix_(rangoni, rangonj)] += Ke[np.ix_(rangoi, rangoj)]
    fe = np.abs(Kg.max())
    with open(archivo1,'a') as f:   # Reescribo el archivo con la matriz global "Kg" obtenida, por eso uso "a".
        f.write(f'\nMatriz Global, fe = {fe:4e}\n')
        f.write(f'{Kg/fe}\n')
    return Kg

# La función "subdiv" particiona un elemento de longitud "L" en "Ne" elementos de igual longitud.
def subdiv(E, A, L, C, Ne, glxn, Nnxe=2):
    """
    INPUTS:
      E    = Módulo de elasticidad del elemento
      A    = Sección del elemento
      L    = Longitud del elemento
      C    = Carga axial aplicada (DISTRIBUIDA)
      Ne   = Cantidad de elementos en que dividiré mi elemento original
      glxn = Grados de libertad por nodo
    OUTPUTS:
      F = Vector de fuerzas en cada nodo
      U = Vector de desplazamientos de cada nodo
      R_emp = Reacción del empotramiento sobre el último nodo
      sigma = Vector de tensiones en cada barra
      f = Vector que contiene fuerza aplicada sobre cada nodo DEBIDA a CARGA DISTRIBUIDA
      Kg = Matriz global
    """
    
    MN = np.linspace(0,L,Ne+1).reshape([-1,1])         # Matriz de nodos.
    Nn = MN.shape[0]                                   # "Nn" es número de nodos.
    
    MC = np.array([[i, i+1] for i in range(Ne)])       # Matriz de conectividad
    Ne, Nnxe = MC.shape                                # "Ne" es número de elementos. "Nnxe" es número de nodos por elemento.
    Le = L/Ne                                          # Longitud de los elementos resultantes.
   
    K = Kglobal(MN, MC, E, A, glxn)                    # Matriz global
    
    # Elemento "0"
    FT = 0.5*C*(Le**2)   # Fuerza total "FT" actuante sobre el elemento "0".
    # Distrubuyo la carga distribuida "T(x)" en los nodos "0" y "1".
    f0 = FT/3     # - Al nodo "0" le corresponde "f0 = FT/3".
    f1 = 2*FT/3   # - Al nodo "1" le corresponde "f1 = 2*FT/3".
    
    f = np.zeros([Ne+1])
    f[0] = f0
    f[1] = f1
    
    # RESTO de elementos
    for i in range(1,Ne): 
        # Contribución debida a RECTÁNGULOS.
        # - "Ne=2" hace que a lo sumo "i=1", entonces "1" rectángulo de área "2*FT" entrega "FT" a cada nodo.
        # - "Ne=3" hace que a lo sumo "i=2", entonces "2" rectángulos de área "2*FT" entregan "2*FT" a cada nodo.
        # - "Ne=4" hace que a lo sumo "i=3", entonces "3" rectángulos de área "2*FT" entregan "3*FT" a cada nodo.
        Ft = FT*i   
        # Fuerza en nodos de la barra i = suma de todos los rectangulos (Ft*i) + carga de la primer barra(triangulito) (Ft)
        f[i] += f0+Ft
        f[i+1] += f1+Ft
    
    s = np.array([Ne])   # Vector "s" que contiene los nodos con condiciones de vínculo en desplazamiento.
    Us = [[0]]           # Vector "Us" con los valores de las condiciones de vínculo. EMPOTRAMIENTO.
    r = np.array([i for i in range(Nn*glxn) if i not in s])   # Vector "r" que contiene los nodos con condiciones de vínculo en fuerza.
    Fr = np.array([[f[i]] for i in range(Ne)])   # Vector "Fr" con los valores de las condiciones de vínculo. CARGAS DISTRIBUIDAS.
    
    F, U = solve(K, r, Fr, s, Us)
    
    R_emp = F[-1] - f[-1]   # Calculo la reacción "R_emp" del empotramiento sobre el último nodo "i = -1".
    
    eps = np.zeros([Ne,1])
    sigma = np.zeros([Ne,1])
    for i in range(Ne):
        eps[i] = (U[i+1]-U[i])/(Le)
        sigma[i] = eps[i]*E
    
    return F, U, R_emp, sigma, f, K


# La función "s_y_r" devuelve los vectores "s" (coordenadas X e Y de nodos con condiciones de vínculo en DESPLAZAMIENTO) y "r" (coordenadas X e Y de nodos con condiciones de vínculo en FUERZA).
def s_y_r(MN,glxn):
    Nn = MN.shape[0]   # Número de NODOS.
    s = []   # Guarda coordenadas X,Y de los nodos empotrados. No Z.  
    for n in range (Nn):
        if MN[n,0] == 0:
            s.append(glxn*n) # Guardo coordenada X.
            s.append(glxn*n + 1) # Guardo coordenada Y.
    r = np.array([i for i in range(Nn*glxn) if i not in s])
    return s, r


# La función "subdiv_barra" particiona una BARRA de longitud "long" en "Ne" elementos de igual longitud.
def subdiv_barra_empotrada(E, A, long, I, Ne, glxn):
    """
    INPUTS:
      E    = Módulo de elasticidad de la barra
      A    = Sección de la barra
      long = Longitud de la barra
      I    = Momento de inercia de la barra
      Ne   = Cantidad de elementos en que dividiré mi elemento original
      glxn = Grados de libertad por nodo
    OUTPUTS:
      Kg = Matriz global
    """
    
    MN_x = np.linspace(0,long,Ne+1).reshape([-1,1])         # Matriz de nodos sólo con coordenada x.
    Nn = MN_x.shape[0]                                      # "Nn" es número de nodos.
    MN = np.hstack([MN_x,np.zeros([Nn,2])])                 # Matriz de nodos.
    
    MC = np.array([[i, i+1] for i in range(Ne)])       # Matriz de conectividad.
    Nnxe = MC.shape[1]                                 # "Nnxe" es número de nodos por elemento.
    L = long/Ne                                        # Longitud de los elementos resultantes.
    
    # Defino la matriz elemental "Ke" de los elementos que subviden la barra.
    Ke = (E*I/L**3)*np.array([[12,6*L,-12,6*L],
                           [6*L,4*(L**2),-6*L,2*(L**2)],
                           [-12,-6*L,12,-6*L],
                           [6*L,2*(L**2),-6*L,4*(L**2)]])
    
    archivo= 'Matrices_elementales.txt'
    with open(archivo,'w') as f:   # Creo archivo desde cero, por eso uso "w".
        f.write('Matrices Elementales\n ===============')
    archivo1= 'Matriz_global.txt'
    with open(archivo1,'w') as f:   # Creo archivo desde cero, por eso uso "w".
        f.write('Matriz Global\n ===============')
    
    Kg = np.zeros([glxn*Nn, glxn*Nn])   # Defino matriz global "Kg".

    for e in range(Ne):
        fe = np.abs(Ke.max()) # Factor de escala, para que los números en "Ke" se lean mejor.
        with open(archivo,'a') as f:   # Voy reescribiendo el archivo con nuevas "Ke", por eso uso "a".
            f.write(f'\nelemento {e}, fe = {fe:4e}\n')
            f.write(f'{Ke/fe}\n')

        for i in range(Nnxe):
            rangoi = np.linspace(i*glxn, (i+1)*glxn-1, Nnxe).astype(int)
            rangoni = np.linspace(MC[e, i]*glxn, (MC[e, i]+1)*glxn-1, Nnxe).astype(int)
            for j in range(Nnxe):
                rangoj = np.linspace(j*glxn, (j+1)*glxn-1, Nnxe).astype(int)
                rangonj = np.linspace(MC[e, j]*glxn, (MC[e, j]+1)*glxn-1, Nnxe).astype(int)
                Kg[np.ix_(rangoni, rangonj)] += Ke[np.ix_(rangoi, rangoj)]
    fe = np.abs(Kg.max())
    with open(archivo1,'a') as f:   # Reescribo el archivo con la matriz global "Kg" obtenida, por eso uso "a".
        f.write(f'\nMatriz Global, fe = {fe:4e}\n')
        f.write(f'{Kg/fe}\n')
    
    return Kg


#
def frec_M_consistente_barra_empotrada(E, A, long, I, rho, Ne, glxn):
    """
    INPUTS:
      E    = Módulo de elasticidad de la barra
      A    = Sección de la barra
      long = Longitud de la barra
      I    = Momento de inercia de la barra
      rho  = Densidad del material de la barra
      Ne   = Cantidad de elementos en que dividiré mi elemento original
      glxn = Grados de libertad por nodo
    OUTPUTS:
      frec = Frecuencias
      d    = Desplazamientos en Y (M.O. transversales)
    """
    
    MN_x = np.linspace(0,long,Ne+1).reshape([-1,1])         # Matriz de nodos sólo con coordenada x.
    Nn = MN_x.shape[0]                                      # "Nn" es número de nodos.
    MN = np.hstack([MN_x,np.zeros([Nn,2])])                 # Matriz de nodos.
    
    MC = np.array([[i, i+1] for i in range(Ne)])       # Matriz de conectividad.
    Nnxe = MC.shape[1]                                 # "Nnxe" es número de nodos por elemento.
    L = long/Ne                                        # Longitud de los elementos resultantes.
    
    # Calculo vectores "s" (condiciones de vínculo en DESPLAZAMIENTO) y "r" (condiciones de vínculo en FUERZA).
    s, r = s_y_r(MN,glxn)
    
    # Calculo matriz global "Kg".
    Kg = subdiv_barra_empotrada(E, A, long, I, Ne, glxn)
    
    # Matriz consistente "Me", para cada elemento.
    Me = (rho*A*L/420)*np.array ([[156,22*L,54,-13*L],
                                   [22*L,4*L**2,13*L,-3*L**2],
                                   [54,13*L,156,-22*L],
                                   [-13*L,-3*L**2,-22*L,4*L**2]])
    
    archivo= 'Matrices_consistentes_elementales.txt'
    with open(archivo,'w') as f:   # Creo archivo desde cero, por eso uso "w".
        f.write('Matrices Consistentes Elementales\n ===============')
    archivo1= 'Matriz_consistente_global.txt'
    with open(archivo1,'w') as f:   # Creo archivo desde cero, por eso uso "w".
        f.write('Matriz Consistente Global\n ===============')
    
    Mg = np.zeros([glxn*Nn, glxn*Nn])   # Defino matriz consistente global "Mg".
    
    for e in range(Ne):
        fe = np.abs(Me.max()) # Factor de escala, para que los números en "Me" se lean mejor.
        with open(archivo,'a') as f:   # Voy reescribiendo el archivo con nuevas "Me", por eso uso "a".
            f.write(f'\nelemento {e}, fe = {fe:4e}\n')
            f.write(f'{Me/fe}\n')

        for i in range(Nnxe):
            rangoi = np.linspace(i*glxn, (i+1)*glxn-1, Nnxe).astype(int)
            rangoni = np.linspace(MC[e, i]*glxn, (MC[e, i]+1)*glxn-1, Nnxe).astype(int)
            for j in range(Nnxe):
                rangoj = np.linspace(j*glxn, (j+1)*glxn-1, Nnxe).astype(int)
                rangonj = np.linspace(MC[e, j]*glxn, (MC[e, j]+1)*glxn-1, Nnxe).astype(int)
                Mg[np.ix_(rangoni, rangonj)] += Me[np.ix_(rangoi, rangoj)]
    fe = np.abs(Mg.max())
    with open(archivo1,'a') as f:   # Reescribo el archivo con la matriz global "Mg" obtenida, por eso uso "a".
        f.write(f'\nMatriz Consistente Global, fe = {fe:4e}\n')
        f.write(f'{Mg/fe}\n')
    
    # Resuelvo problema de autovalores y autovectores, que son respectivamente los modos normales de vibración "omega_2" y los
    # desplazamientos "Vr" de los modos normales 
    omega_2, Vr = linalg.eigh(Kg[np.ix_(r,r)],Mg[np.ix_(r,r)])
    frec = np.sqrt(omega_2)/(2*np.pi)        # Calculo frecuencia de los modos normales.
    
    # "V" no está considerando los nodos con condición de vínculo, de "s", así que lo soluciono agregando "len(s)" filas nulas.
    V = np.vstack([np.zeros([len(s), len(omega_2)]),Vr])
    
    d = V[0::2, :]   # Nos quedamos con los desplazamientos en el eje Y (por eso salto cada 2). MODOS TRANSVERSALES es EJE Y
    d = d/d[-1, :]     # Normalizamos los desplazamientos con respecto al último.
    
    return frec, d


#
def frec_M_concentrada_barra_empotrada(E, A, long, I, rho, Ne, glxn):
    """
    INPUTS:
      E    = Módulo de elasticidad de la barra
      A    = Sección de la barra
      long = Longitud de la barra
      I    = Momento de inercia de la barra
      rho  = Densidad del material de la barra
      Ne   = Cantidad de elementos en que dividiré mi elemento original
      glxn = Grados de libertad por nodo
    OUTPUTS:
      frec = Frecuencias
      d    = Desplazamientos en Y (M.O. transversales)
    """
    
    MN_x = np.linspace(0,long,Ne+1).reshape([-1,1])         # Matriz de nodos sólo con coordenada x.
    Nn = MN_x.shape[0]                                      # "Nn" es número de nodos.
    MN = np.hstack([MN_x,np.zeros([Nn,2])])                 # Matriz de nodos.
    
    MC = np.array([[i, i+1] for i in range(Ne)])       # Matriz de conectividad.
    Nnxe = MC.shape[1]                                 # "Nnxe" es número de nodos por elemento.
    L = long/Ne                                        # Longitud de los elementos resultantes.
    
    # Calculo vectores "s" (condiciones de vínculo en DESPLAZAMIENTO) y "r" (condiciones de vínculo en FUERZA).
    s, r = s_y_r(MN,glxn)
    
    # Calculo matriz global "Kg".
    Kg = subdiv_barra_empotrada(E, A, long, I, Ne, glxn)
    
    # Matriz concentrada "Me", para cada elemento.
    Me = (rho*A*L/24)*np.array([[12, 0, 0, 0],
                                [0, L**2, 0, 0],
                                [0, 0, 12, 0],
                                [0, 0, 0, L**2]])
    
    archivo= 'Matrices_concentradas_elementales.txt'
    with open(archivo,'w') as f:   # Creo archivo desde cero, por eso uso "w".
        f.write('Matrices Concentradas Elementales\n ===============')
    archivo1= 'Matriz_concentrada_global.txt'
    with open(archivo1,'w') as f:   # Creo archivo desde cero, por eso uso "w".
        f.write('Matriz Concentrada Global\n ===============')
    
    Mg = np.zeros([glxn*Nn, glxn*Nn])   # Defino matriz consistente global "Mg".
    
    for e in range(Ne):
        fe = np.abs(Me.max()) # Factor de escala, para que los números en "Me" se lean mejor.
        with open(archivo,'a') as f:   # Voy reescribiendo el archivo con nuevas "Me", por eso uso "a".
            f.write(f'\nelemento {e}, fe = {fe:4e}\n')
            f.write(f'{Me/fe}\n')

        for i in range(Nnxe):
            rangoi = np.linspace(i*glxn, (i+1)*glxn-1, Nnxe).astype(int)
            rangoni = np.linspace(MC[e, i]*glxn, (MC[e, i]+1)*glxn-1, Nnxe).astype(int)
            for j in range(Nnxe):
                rangoj = np.linspace(j*glxn, (j+1)*glxn-1, Nnxe).astype(int)
                rangonj = np.linspace(MC[e, j]*glxn, (MC[e, j]+1)*glxn-1, Nnxe).astype(int)
                Mg[np.ix_(rangoni, rangonj)] += Me[np.ix_(rangoi, rangoj)]
    fe = np.abs(Mg.max())
    with open(archivo1,'a') as f:   # Reescribo el archivo con la matriz global "Mg" obtenida, por eso uso "a".
        f.write(f'\nMatriz Concentrada Global, fe = {fe:4e}\n')
        f.write(f'{Mg/fe}\n')
    
    # Resuelvo problema de autovalores y autovectores, que son respectivamente los modos normales de vibración "omega_2" y los
    # desplazamientos "Vr" de los modos normales 
    omega_2, Vr = linalg.eigh(Kg[np.ix_(r,r)],Mg[np.ix_(r,r)])
    frec = np.sqrt(omega_2)/(2*np.pi)        # Calculo frecuencia de los modos normales.
    
    # "V" no está considerando los nodos con condición de vínculo, de "s", así que lo soluciono agregando "len(s)" filas nulas.
    V = np.vstack([np.zeros([len(s), len(omega_2)]),Vr])
    
    d = V[0::2, :]   # Nos quedamos con los desplazamientos en el eje Y (por eso salto cada 2). MODOS TRANSVERSALES es EJE Y
    d = d/d[-1, :]     # Normalizamos los desplazamientos con respecto al último.
    
    return frec, d