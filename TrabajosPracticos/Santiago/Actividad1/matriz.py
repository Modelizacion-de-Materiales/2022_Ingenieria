import numpy as np

def matriz(Nx,Ny,lim,limtipo):
    """
    Nx:Numero de nodos en x
    Ny:Numero de nodos en y
    lim: 'A' 'B' 'C' 'D' es decir los contornos
    limtipo: 'Flujo' o 'Temp'
    """
    N = Nx*Ny #el Nk que vimos en clase.
    beta = Nx/Ny
    M = np.eye(N)
    b = np.zeros([N,1]) #armo las salidas

    for k in range(N):
        if k==0: #Vertice AB
            if limtipo['A']=='Flujo' and limtipo['B']=='Temp':#Si me paro en un vértice con temperatura y
                b[k]=lim['A']                             #flujo, le asigno la temperatura del que la posea.
            elif limtipo['A']=='Temp' and limtipo['B']=='Flujo':
                b[k]=lim['B']
            else: #Sólo queda el caso de las 2 temperaturas, donde se promedian.
                b[k]=(lim['A']+lim['B'])/2

        elif k==Nx-1: #Vértice BC
            if limtipo['B']=='Flujo' and limtipo['C']=='Temp':
                b[k]=lim['C']
            elif limtipo['B']=='Temp' and limtipo['C']=='Flujo':
                b[k]=lim['C']
            else:
                b[k]=(lim['B']+lim['C'])/2
       
        elif k==N-1: #Vértice CD
            if limtipo['C']=='Flujo' and limtipo['D']=='Temp':
                b[k]=lim['D']
            elif limtipo['C']=='Temp' and limtipo['D']=='Flujo':
                b[k]=lim['C']
            else:
                b[k]=(lim['C']+lim['D'])/2
      
        elif k==N-Nx: #Vértice AD
            if limtipo['A']=='Flujo' and limtipo['D']=='Temp':
                b[k]=lim['D']
            elif limtipo['D']=='Temp' and limtipo['A']=='Flujo':
                b[k]=lim['D']
            else:
                b[k]=(lim['A']+lim['D'])/2
        
        elif k%Nx==0: #Para el borde A uso, k múltiplo de Nx.
            if limtipo['A']=='Temp':
                b[k]=lim['A']
            elif limtipo['A']=='Flujo': #en b especializado en k ya hay un cero que queda cuando la creamos.
                M[k,k]=-2*(1+beta**2)
                M[k,k-1]=0
                M[k,k+1]=2
                M[k,k+Nx]=beta**2
                M[k,k-Nx]=beta**2
     
        elif k<Nx: #Borde B.
            if limtipo['B']=='Temp':
                b[k]=lim['B']
            elif limtipo['B']=='Flujo':
                M[k,k]=-2*(1+beta**2)
                M[k,k-1]=1
                M[k,k+1]=1
                M[k,k-Nx]=0
                M[k,k+Nx]=2*beta**2
        
        elif (k+1)%Nx==0: #Para el borde C, si k+1 es multiplo de Nx.
            if limtipo['C']=='Temp':
                b[k]=lim['C']
            elif limtipo['C']=='Flujo':
                M[k,k]=-2*(1+beta**2)
                M[k,k-1]=2
                M[k,k+1]=0
                M[k,k+Nx]=beta**2
                M[k,k-Nx]=beta**2
    
        elif (k<N-1) and (k>N-Nx): #Para el borde D uso los últimos valores en N sin contar los vértices.
            if limtipo['D']=='Temp':
                b[k]=lim['D']
            elif limtipo['D']=='Flujo':
                M[k,k]=-2*(1+beta**2)
                M[k,k-1]=1
                M[k,k+1]=1
                M[k,k-Nx]=2*beta**2
        else: #Termino de armar la matriz.
            M[k,k]=-2*(1+beta**2)
            M[k,k-1]=1
            M[k,k+1]=1
            M[k,k+Nx]=beta**2
            M[k,k-Nx]=beta**2
    return M,b