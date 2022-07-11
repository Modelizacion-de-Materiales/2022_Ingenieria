plt.rc('figure', figsize=(18,9))
plt.rc('font', size=22)

def plot_forces_wmpl(mn, f, ax=None, fig=None):
    if ax is  None:
        fig,ax = plt.subplots()
    selection = (np.abs(f[:,0]) > 0) | (np.abs(f[:,1])>0)
    plt.quiver(mn[selection,0], mn[selection,1], f[selection,0], np.zeros_like(f[selection,0]),scale=200, scale_units='x')
    plt.quiver(mn[selection,0], mn[selection,1], np.zeros_like(f[selection,1]), f[selection,1],scale=10, scale_units='y')
    return fig, ax

def plot_stress_wmpl(mn, mc, stress, label, ax=None, fig=None):
    if ax is None:
        fig, ax = plt.subplots()
    mapp = ax.tripcolor(mn[:,0], mn[:,1], mc, facecolors=stress)
    cbar = plt.colorbar(mapp, ax=ax)
    cbar.set_label(label)
    return fig, ax

#mdfcomment gmsh.fltk.run() estoy revisando en remote
def mpl_mesh_plot(mn, mc):
    fig, ax = plt.subplots()
    ax.triplot(mn[:,0], mn[:,1], mc)
    return fig, ax



from IPython.display import Image
from IPython.core.display import HTML
path='../../../EjemplosEnClase/Guia4/FrecuanciavsN.png'
Image(path)



HTML('<div style="background:#999900"> la idea era comparar las frecuencias entre métodos</div>')