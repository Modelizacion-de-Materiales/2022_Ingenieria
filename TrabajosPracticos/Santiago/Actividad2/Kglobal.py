import numpy as np
from Kelemental import Kelemental

def Kglobal (MN, MC, E, A, GLXN):
    NN = MN.shape[0] #Cantidad de nodos.
    NE,NNXE = MC.shape #Número de elementos y numero de nodos por elemento.
    K_Global = np.zeros([GLXN*NN, GLXN*NN])    
    for e in range(NE): #Para cada elemento calculo la Kelemental
        Ee = E[e]
        Ae = A[e]
        Ke = Kelemental(MN, MC, Ee, Ae, e)
        #print(Ke/np.max(Ke))
        for i in range(NNXE):
            rangoi = np.linspace(i*GLXN, (i+1)*GLXN-1, NNXE).astype(int)
            rangoni = np.linspace(MC[e, i]*GLXN, (MC[e, i]+1)*GLXN-1, NNXE).astype(int)
            for j in range(NNXE):
                rangoj = np.linspace(j*GLXN, (j+1)*GLXN-1, NNXE).astype(int)
                rangonj = np.linspace(MC[e, j]*GLXN, (MC[e, j]+1)*GLXN-1, NNXE).astype(int)
                K_Global[np.ix_(rangoni, rangonj)] += Ke[np.ix_(rangoi, rangoj)]
    return K_Global