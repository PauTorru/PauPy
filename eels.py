
import matplotlib.pyplot as plt
import numpy as np
import hyperspy.signals as signals
import hyperspy.api as hs
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster,fclusterdata,cophenet,inconsistent
from scipy.spatial.distance import pdist
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans



def adapt_eaxis(s_list):
    '''Interpolate spectra from all the images in s_list so that they have a common eaxis'''
    #decide common energy axis
    emin=max([s.axes_manager[-1].axis.min() for s in s_list])
    emax=min([s.axes_manager[-1].axis.max() for s in s_list])
    scale=min([s.axes_manager[-1].scale for s in s_list])
    eaxis=np.linspace(emin,emax,int((emax-emin)/scale))

    s_adapt_list=[]

    for s in s_list:
        sc=s.deepcopy()
        sc.unfold()
    #create s_adapt
        interp_data=np.zeros((sc.data.shape[0],eaxis.shape[0]))
        xp1=s.axes_manager[-1].axis

        for i in range(sc.data.shape[0]):
            interp_data[i,:]=np.interp(eaxis,xp1,sc.data[i,:])

        shape=list(s.data.shape[:-1])+[eaxis.shape[0]]
        s_adapt=hs.signals.EELSSpectrum(interp_data.reshape(shape))
        s_adapt.axes_manager[-1].scale=scale
        s_adapt.axes_manager[-1].offset=emin
        s_adapt.axes_manager[-1].axis=eaxis
        s_adapt.axes_manager[-1].copy_traits(s.axes_manager[-1],['units','scale'])
        s_adapt_list.append(s_adapt)

    return s_adapt_list

#######################################################3

def zlmirrormodel(s,energy_preZL=-10.):

    ''' s is a single, aligned, calibrated, spectrum. Outputs mirrored  in 0 model.'''
    zl=s.deepcopy()
    prezl=zl.isig[energy_preZL[0]:energy_preZL[1]]
    zl=zl.isig[energy_preZL[0]:]

    zl.unfold()
    prezl.unfold()

    prezl_length=prezl.axes_manager[-1].size
    index0=zl.inav[0].data.argmax(-1)


    for i in range(zl.data.shape[0]):
        noise_level=prezl.data[i,:].sum(-1)/prezl_length
        zl.data[i,:]-=noise_level
        zl.data[i,index0:2*index0]=zl.data[i,0:index0][::-1]
        zl.data[i,2*index0:]=0.


    zl.fold()

    return zl


####################################################################################################################
def correct_h_drift(s,drift_list):
    sc=s.deepcopy()
    drift_list=np.array(drift_list)
    drift_list=np.floor(drift_list).astype('int')
    for i in range(len(drift_list)):
        if drift_list[i]==0:
            pass
        else:
            sc.data[i,:-drift_list[i],:]=sc.data[i,drift_list[i]:,:]
    print(max(drift_list))
    sc.crop(0,0,s.data.shape[1]-max(drift_list))
    return sc
####################################################################################################################
def copy_eaxis(s_copy,s_origin):
    '''Pots fer servir s.axes_manager.copy_traits(). returns s_copy with scale,offset,name, and units of the energy axis from s_origin.
    s_copy and s_origin must both be signals.EELSspectrum.'''

    if not (isinstance(s_copy,signals.EELSSpectrum) and isinstance(s_origin,signals.EELSSpectrum)):
        print('Both inputs must be EELS spectra')
        return
    else:

        s_copy.axes_manager[-1].scale=s_origin.axes_manager[-1].scale
        s_copy.axes_manager[-1].offset=s_origin.axes_manager[-1].offset
        s_copy.axes_manager[-1].name=s_origin.axes_manager[-1].name
        s_copy.axes_manager[-1].units=s_origin.axes_manager[-1].units

    return s_copy

##################################################################################################################3
def compare_spectra(s1,s2,same_size=True):
    '''Compares two EELS spectra, spectrum lines or spectrum images. Currently same_size=False is not suported.'''

    if same_size==False:
        raise NameError('Sorry, function under development')
        return

    if same_size==True and s1.data.shape!=s2.data.shape:
        raise NameError('Actually, what you want to compare is not the same size.')
        return


    scomp=signals.EELSSpectrum(s1.data+1j*s2.data)
    copy_eaxis(scomp,s1)
    scomp.plot()
    return scomp

###################################################################################
def norm(to_norm):
    '''plot_dm3 uses dis to norm the curves'''
    factor=(to_norm.data.max()-to_norm.data.min())
    normed=(to_norm.data-to_norm.data.min())/factor
    return normed,factor

###################
def plot_dm3(dm3,normit=False,shift=0.,scale=1.,offset=0.,**kwargs):
    ''' yo, use dis to plot several dm3 profiles. pretty handy'''

    factor=1.
    if normit:
        dm3.data,factor=norm(dm3)
        print(dm3.metadata.General.title, factor)
    plt.plot(dm3.axes_manager[-1].axis+shift,scale*(dm3.data+offset),**kwargs)
    return factor
###################################################
def play_wdeco_si(factors,loadings,mix):
    ''' pass factors and loadings from PCA or BSS and manually unmix them using the mix matrix of size n*n where n
    is the number of components of the decomposition'''

    if factors.data.shape[0]==loadings.data.shape[0]==len(mix[0])==len(mix[1]):
        nf=factors.data.shape[0]

    print(nf)

    new_factors=factors.deepcopy()
    new_factors.data=np.zeros(new_factors.data.shape)
    new_loadings=loadings.deepcopy()
    new_loadings.data=np.zeros(new_loadings.data.shape)
    inv_mix=np.linalg.inv(mix)
    for i in range(nf):
        for j in range(nf):
            new_factors.data[i,:]+=factors.data[j,:]*mix[j][i]
            new_loadings.data[i,:,:]+=loadings.data[j,:,:]*inv_mix[i][j]

    return new_factors,new_loadings
###################################
def compare(s,s1):
    '''compare two SI in the same window,s in red, s1 in blue'''
    comp=s.deepcopy()
    comp.data=s.data+1j*s1.data
    comp.plot()
#############################################
def rebuild(factors,loadings):
    '''Rebuild spectrum images from decomposition. Useful if you have played with components after the PCA/BSS'''

    e=factors.data.shape[1]
    x=loadings.data.shape[1]
    y=loadings.data.shape[2]
    n=factors.data.shape[0]
    data=np.zeros((x,y,e))

    for i in range(e):
        for j in range(x):
            for k in range(y):
                for l in range(n):
                    data[j,k,i]+=factors.data[l,i]*loadings.data[l,j,k]


    s=hs.signals.EELSSpectrum(data)
    s.axes_manager[-1].scale=factors.axes_manager[-1].scale
    s.axes_manager[-1].offset=factors.axes_manager[-1].offset
    s.axes_manager[-1].units=factors.axes_manager[-1].units

    return s
#####################################################
def dark_correct_from_image(s,d):
    '''Apply dark reference (d) to spectrum image, spectrum line or signle spectrum (s). returns spectral data dark corrected'''
    dark=d.deepcopy()
    dark.unfold()
    dr=dark.data.sum(0)/dark.data.shape[0]
    s_corrected=s.deepcopy()
    s_corrected.data-=dr
    return s_corrected

def dark_correct_from_th(s,th):
    '''Apply dark reference (d) to spectrum image, spectrum line or signle spectrum (s). returns spectral data dark corrected'''
    d=s.data[s.data.sum(-1)<th]
    dr=np.average(d,0)
    s_corrected=s.deepcopy()
    s_corrected.data-=dr
    return s_corrected
####################################
def load_emi(fname):
    s=hs.load(fname).inav[1:,:]
    s.axes_manager[1].scale*=-1
    s.data=s.data.astype("float64")
    return s
######
class clustering():
    def __init__(self,s,algorithm="hierachical",on_pca_scores=False,
                 pca_comps=2,pca_kwargs={},cluster_kwargs={},normalize=True):


        s.unfold()
        self.objects=np.copy(s.data)
        s.fold()
        self.si=s

        if normalize:
            self.objects-=self.objects.min()
            self.objects/=self.objects.sum(-1)[:,np.newaxis]

        if on_pca_scores:
            self.si.decomposition(**pca_kwargs)
            l=self.si.get_decomposition_loadings().inav[:pca_comps]
            l.unfold()
            self.objects=np.rollaxis(l.data,1)
            for i in range(pca_comps):
                self.objects[:,i]-=self.objects[:,i].min()
                self.objects[:,i]/=self.objects[:,i].max()


        switcher={"hierachical":self.hierachical,"kmeans":self.kmeans,"Agglomerative":self.agglo}
        self.algorithm=algorithm
        switcher[algorithm](cluster_kwargs)


        return

    def hierachical(self,cluster_kwargs):
        self.linktree=linkage(self.objects,method="ward",**cluster_kwargs)
        return

    def plot_hierachical_distance(self):
        fig=plt.figure()
        plt.plot(self.linktree.shape[0]+np.array(range(50))*-1,self.linktree[-50:,2][::-1],'bo')
        ax=plt.gca()
        ax.set_xlim(1+self.linktree.shape[0],self.linktree.shape[0]-50)
        ax.set_xlabel("linking step")
        ax.set_ylabel("link distance")
        ax.set_yscale("log")

        return self.linktree.shape[0]+np.array(range(50))*-1,self.linktree[-50:,2][::-1]

    def plot_dendogram(self,nclusters,depth=20):
        dn=dendrogram(self.linktree,
        truncate_mode='lastp',
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=False,  # to get a distribution impression in truncated branches
        color_threshold=self.linktree[-nclusters+1,2])
        return dn

    def hierachical_cluster(self,nclusters=2):
        self.nclusters=nclusters
        self.labels=fcluster(self.linktree,self.linktree[-nclusters,2],criterion="distance").reshape(self.si.data.shape[:-1])
        self.labels=np.array(self.labels)

    def plot_cluster_image(self):

        colors=list(sns.color_palette("bright",self.nclusters))
        fig=plt.figure()
        ax1=plt.subplot(121)
        cm = LinearSegmentedColormap.from_list('dunno', colors)
        plt.imshow(self.labels,cmap=cm)

        ax2=plt.subplot(122)
        for i in range(self.nclusters):
            plt.plot(self.si.axes_manager[-1].axis,np.average(self.si.data[self.labels==i+1],axis=0),color=colors[i])

        ax2.set_xlabel("Energy Loss (eV)")
        return

    def kmeans(self,cluster_kwargs):
        km = KMeans(**cluster_kwargs).fit(self.objects)
        self.labels=km.labels_.reshape(self.si.data.shape[:-1])
        self.labels+=1
        self.nclusters=cluster_kwargs["n_clusters"]
        return

    def agglo(self):

        return
