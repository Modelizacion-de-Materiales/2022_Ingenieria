import numpy as np
def Kelemental (MN,MC,Ae,Ee,e):
    """
    MN:Matriz de nodos
    MC:Matriz de conectividad
    A:Area del elemento (np.array)
    E:Modulo de Young del elemento (np.array)
    e:Elemento
    """
    Nodo1 = MC[e,0]
    Nodo2 = MC[e,1]
    LonX = MN[Nodo2,0]-MN[Nodo1,0]
    LonY = MN[Nodo2,1]-MN[Nodo1,1]
    L=np.sqrt(LonX**2+LonY**2)
    Ang = np.arctan2(LonY,LonX)
    Ke_constante = Ee*Ae/L #K elemental.
    c=np.cos(Ang)
    s=np.sin(Ang)
    Ke=np.array([[c**2,c*s,-c**2,-c*s],
                         [c*s,s**2,-c*s,-s**2],
                         [-c**2,-c*s,c**2,c*s],
                         [-c*s,-s**2,c*s,s**2]])*Ke_constante
    
    return Ke