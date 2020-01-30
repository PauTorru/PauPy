# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:46:53 2019

@author: ptbe
"""

import hyperspy.api as hs
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import skimage as ski
import os
import time
import xlsxwriter
import copy
import cv2
import imutils
import inspect

def thresholdpau2(im):
    b=np.histogram(im.ravel(),bins=500)
    pixel_n,I=b[0],b[1]
    d=pixel_n[1:]-pixel_n[:-1]
    Ith=I[-(np.argmax(d[::-1]<np.average(d[-100:])-np.std(d[-100:])**2))]

    return Ith

def get_shifts_image_series(imlist,filter_func,number_of_iterations = 10000,termination_eps = 1e-20):
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    #Align
    warp_mode = cv2.MOTION_TRANSLATION
    wms=[]

    for i,_ in enumerate(imlist[:-1]):

        im1=imlist[i]
        im2=imlist[i+1]

        im1_to_align=filter_func(im1)
        im2_to_align=filter_func(im2)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        (cc, warp_matrix) = cv2.findTransformECC (im1_to_align,im2_to_align,warp_matrix, warp_mode, criteria)

        wms.append(warp_matrix)

    wms=np.dstack(wms)
    wms_summed=np.zeros(wms.shape)
    for i in range(wms.shape[-1]-1):
        wms_summed[:,-1,i]=wms[:,-1,:i+1].sum(-1)

    wms_summed[:,-1,0]=wms[:,-1,0]
    wms_summed[:,:-1,:]=wms[:,:-1,:]
    return wms_summed

def apply_shifts_image_series(imlist,shifts):

        buffer=int(np.ceil(shifts.max()))
        im0=np.zeros(np.array(im.data.shape)+np.array([buffer*2,buffer*2]))
        im0[buffer:-buffer,buffer:-buffer]=hr.to8bit(imlist[0])

        aligned=[im0]


        for i in range(shifts.data.shape[-1]):

            im1=hr.to8bit(imlist[i])
            im2=hr.to8bit(imlist[i+1])

            a=np.zeros(np.array(im.data.shape)+np.array([buffer*2,buffer*2]))
            a[buffer:-buffer,buffer:-buffer]=im2

            im2_aligned = cv2.warpAffine(a,shifts[:,:,i],
                                     (a.shape[1],a.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            aligned.append(im2_aligned)

        return aligned

def remove_outliers(im,out_threshold=0.01):
    data=im.data.ravel()
    high=np.percentile(data,100-out_threshold)
    low=np.percentile(data,out_threshold)
    new_im=im.deepcopy()
    new_im.data[new_im.data>high]=high
    new_im.data[new_im.data<low]=low
    return new_im

def label2coord(imlabel):
    return np.dstack(np.where(imlabel))[0]

def thresholdpau(im):
    d=np.histogram(im.ravel(),bins=100)
    x=(d[1][:-1]+d[1][1:])/2
    cder=np.convolve(d[0][1:]-d[0][:-1],np.ones(4))
    mask=np.zeros(x.shape)
    mask[max(int(0.3*x.argmax()),10):]=1
    n=np.logical_and((cder>0)[1:-1], mask==1).argmax()
    th=x[n]
    return th

def to8bit(im2):
    im=copy.deepcopy(im2).astype("float")
    im-=im.min()
    im/=im.max()
    im*=255
    return im.astype("uint8")

def ifft(fft):
    return abs(np.fft.ifft2(np.fft.fftshift(fft)))

def create_fft_mask(fft,position,radius,simetric=False,add0=False):

    mask=np.zeros(fft.shape)
    circle=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (2*radius+1,2*radius+1))

    center=np.array(fft.shape)/2

    xi=int(position[0]-radius)
    xf=int(position[0]+radius+1)

    yi=int(position[1]-radius)
    yf=int(position[1]+radius+1)

    mask[yi:yf,xi:xf]=circle

    if simetric:
        p2=2*center-np.array(position)
        xi=int(p2[0]-radius)
        xf=int(p2[0]+radius+1)

        yi=int(p2[1]-radius)
        yf=int(p2[1]+radius+1)

        mask[yi:yf,xi:xf]=circle

    if add0:
        p2=center
        xi=int(p2[0]-radius)
        xf=int(p2[0]+radius+1)
        yi=int(p2[1]-radius)
        yf=int(p2[1]+radius+1)

        mask[yi:yf,xi:xf]=circle


    return mask


class plane_analysis_v2():

    def __init__(self,fname,save_images=False,
                planes2measure={"101":(3.417,3.617),
                                "004":(2.279,2.479),
                                "200":(1.793,1.993)},
                mask_radius=30,
                filter_by_simmetry={"101":False,
                                    "004":False,
                                    "200":False},
                printduration=True,
                use_lap=False,
                peak_detection_threshold=150,
                minarea=80,
                filter_edge_particles=True,
                peak_detection="default",
                reduction=8,
                scan_detection_threshold=20,
                threshold_method=0,
                filter_convex=False,
                convex_th=0.8):

        self.minarea=minarea
        self.filter_convex=filter_convex
        self.filter_by_simmetry=filter_by_simmetry
        self.mask_radius=mask_radius
        self.image=hs.load(fname)
        self.save_images=save_images
        self.printduration=printduration
        self.use_lap=use_lap
        self.peak_detection_threshold=peak_detection_threshold
        self.filter_edge_particles=filter_edge_particles
        self.threshold=threshold_method
        self.convex_th=convex_th

        self.fft=getfft(self.image)
        self.rfft=to8bit(np.log(abs(self.fft)))
        self.nice_rfft=cv2.blur(self.rfft,(5,5))

        self.defplanes=planes2measure
        if peak_detection=="default":
            self.allfftpeaks=self.get_fftpeaks()
        elif peak_detection=="scan":
            self.allfftpeaks=self.get_fftpeaks_by_scan(reduction,scan_detection_threshold)


        self.fftpeaks=self.filter_planes(self.allfftpeaks)

    def get_fftpeaks_by_scan(self,reduction,scan_detection_threshold):
        im=self.image
        sz=int(im.data.shape[0]/reduction)
        sz_shift=int(sz/2)
        fft=np.zeros((sz,sz))
        for i in range(2*reduction-1):
            for j in range(2*reduction-1):

                section=im.isig[sz_shift*i:sz_shift*i+sz,sz_shift*j:sz_shift*j+sz]
                section=remove_outliers(section)
                fft=np.maximum(fft,cv2.blur(abs(getfft(section)),(5,5)))

        fft=np.log(fft)
        x1=cv2.blur(fft,(10,10))
        x1=to8bit(fft-x1)
        x1=x1-np.average(x1)
        x1[x1<0]=0

        params=cv2.SimpleBlobDetector_Params()
        params.filterByArea=False
        params.filterByCircularity=False
        params.filterByColor=False
        params.filterByConvexity=False
        params.filterByInertia=False

        params.minDistBetweenBlobs=0
        params.minThreshold=scan_detection_threshold
        params.maxThreshold=255
        params.thresholdStep=20

        params.minRepeatability=1


        detector=cv2.SimpleBlobDetector_create(params)
        kps=detector.detect(to8bit(x1))

        return_kps=[]
        for kp in kps:
            if (kp.pt[0]-sz_shift)>0:
                return_kps.append(np.array(kp.pt)*reduction)

        #fft[sz_shift-5:sz_shift+5,sz_shift-5:sz_shift+5]=np.average(fft.ravel())
        self.scanfft=fft
        return return_kps


    def get_fftpeaks(self):



        b=cv2.blur(self.rfft,(500,500))
        c=self.nice_rfft-b
        c[c>200]=0

        n=21
        kernel=np.ones((n,n))*-1
        kernel[int(np.floor(n/2)),int(np.floor(n/2))]=n**2-1
        kernel/=n
        d=cv2.filter2D(c,0,kernel)

        e=cv2.morphologyEx(d,cv2.MORPH_OPEN,np.ones((5,5)))

        params=cv2.SimpleBlobDetector_Params()
        params.minArea=self.minarea
        params.filterByCircularity=False
        params.filterByColor=False
        params.filterByConvexity=False
        params.filterByInertia=False
        params.minThreshold=0
        params.maxThreshold=self.peak_detection_threshold
        params.filterByArea=True
        params.minDistBetweenBlobs=10
        dt=cv2.SimpleBlobDetector_create(params)

        kp=dt.detect(255-to8bit(e))

        return [np.array(k.pt) for k in kp]

    def filter_planes(self,kps):

        center=np.array(self.image.data.shape)/2

        ds=[2*center[0]*self.image.axes_manager[0].scale/np.linalg.norm(k-center) for k in kps]
        planes={}
        for key in list(self.defplanes.keys()):
            ps=[]
            for d,kp in zip(ds,kps):
                if d>self.defplanes[key][0]*0.1 and d<self.defplanes[key][1]*0.1 and kp[0]-center[0]>0:
                    ps.append(kp)
            planes[key]=ps

        return planes


    def plot_fft_peaks(self,plot_all=False):
        plt.clf()
        plt.imshow(self.nice_rfft)
        if plot_all:
            for k in self.allfftpeaks:
                plt.plot(k[0],k[1],"ro",markersize=3)

        else:
            for key,c in zip(list(self.defplanes.keys()),plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(self.defplanes.keys())]):
                for k in self.fftpeaks[key]:
                    plt.plot(k[0],k[1],"o",color=c,markersize=3)

    def ifft_spot(self,spot):

        mask=create_fft_mask(self.fft,spot,self.mask_radius)
        filtered=mask*self.fft
        return(ifft(filtered))

    def measure(self,im,spot):

        center=np.array(self.image.data.shape)/2
        cs=spot-center
        angle= -np.arctan(cs[1]/cs[0])*180/np.pi
        self.current_angle=angle





        r=imutils.rotate_bound(im,angle)
        r=to8bit(r)
        self.current_r=r

        if self.use_lap:
            L=cv2.Laplacian(r,0,0,5,4)
            r-=L
            r=cv2.morphologyEx(r,cv2.MORPH_OPEN,np.ones((11,11)))

        if self.threshold==0:
            th=thresholdpau(r)
        elif self.threshold==1:
            th=thresholdpau2(im)-im.min()
            th/=im.max()
            th*=255

        th=to8bit(r>th)

        if self.use_lap:
            th=cv2.morphologyEx(th,cv2.MORPH_DILATE,np.ones((7,7)))

        a,b,c,d=cv2.connectedComponentsWithStats(th,8)
        self.current_b=b


        isum=np.array([r[b==i].sum() for i in range(1,b.max()+1)])
        n=np.argmax(isum)+1

        particle=np.zeros(r.shape,"uint8")
        particle[b==n]=255

        self.ok=True

        #is bad threshold? i.e. is convex?
        if self.filter_convex:
            im2, contours, hierarchy = cv2.findContours(particle, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hull=cv2.convexHull(contours[0],False)
            drawing = np.zeros((im2.shape[0], im2.shape[1]), np.uint8)
            cv2.drawContours(drawing, [hull], 0, 255, -1, 8)
            convex=particle.sum()/drawing.sum()
            if convex<self.convex_th:
                self.ok=False

        #is on edge?
        if self.filter_edge_particles:
            edge=np.ones(self.image.data.shape)
            edge[5:-5,5:-5]=0
            redge=imutils.rotate_bound(edge,angle)>0
            if (redge*particle).sum()>0:
                self.ok=False






        length=c[n,2]*self.image.axes_manager[0].scale

        return particle,[length]


    def analyze(self):
        start=time.time()
        results={}

        for key in list(self.defplanes.keys()):
            sizes=[]
            i=0
            for spot in self.fftpeaks[key]:
                i+=1
                self.current_spot=spot

                spotifft=self.ifft_spot(spot)
                self.current_spotifft=spotifft

                image_measurement,size=self.measure(spotifft,spot)
                self.current_measure=image_measurement
                self.current_size=size

                if self.filter_by_simmetry[key]:
                    ellipsoid_angle=abs(self.get_ellipsoid_angle(image_measurement)%90)
                    if ellipsoid_angle<75 and ellipsoid_angle>15:
                        self.ok=False

                if self.ok:
                    sizes+=size

                if self.save_images:
                    fig=self.save_measurement_image()
                    fig.savefig(key+"_"+"spot"+f"{i:0>4d}"+"_"+self.image.metadata.General.original_filename.split(".")[0]+".png")

            results[key]=sizes

        self.results=results
        end=time.time()
        t=end-start
        if self.printduration:
            print("Duration of the analysis: "+str(int(t/60))+" minutes "+str(int(t%60))+" seconds" )

        return results

    def get_ellipsoid_angle(self,image_measurement):
        nzs=cv2.findNonZero(image_measurement)
        self.current_ellipse=cv2.fitEllipse(nzs)
        center,axes,angle=self.current_ellipse
        return angle

    def save_measurement_image(self):
        plt.clf()
        fig=plt.gcf()
        fig.set_size_inches(10,6)
        plt.subplot(131)
        plt.imshow(self.nice_rfft)
        plt.plot(self.current_spot[0],self.current_spot[1],"ro",markersize=3)
        ax=plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])

        plt.subplot(132)
        plt.imshow(self.current_spotifft)
        ax=plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])

        plt.subplot(133)
        plt.imshow(self.current_measure)
        ax=plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        if self.ok:
            plt.gca().set_title("OK "+f"{self.current_size[0]:.2f}"+self.image.axes_manager[0].units)

        else:
            plt.gca().set_title("NOT OK; "+f"{self.current_size[0]:.2f}"+self.image.axes_manager[0].units)

        return fig












































class plane_analysis():

    def __init__(self,fname,save_images=True, h=0.07,spot_sigma=300,
                 use_watershed=False,
                 filter_by_ellipse=False,
                 filter_by_ellipse_axes=False,
                 use_blob_detection=False):

        '''
        Class used to analyse HRTEM images of TiO2 and measure their sizes along different crystalline directions.

        Parameters
        ----------

        fname : string
            filename of the image to analyse. Uses Hyperspy load().
        h: float
            pamater used by ski.morphology.extrema.h_maxima.
        spot_sigma: float
            controls the radius of the masks applied to each diffraction spot when filtering.
        save_images: bool
            if True saves an image for the analysis of each particle.
        use_watershed: bool
            if True uses watershed filter to split touching particles.
            No suitable if superimposed particles.
        filter_by_ellipse: bool
            If true filters planes identified as 004 if not aligned with one of the minor/major axes of the particle.
            use if C/A is very different than one.
        filter_by_ellipse_axes: bool
            If true filters planes identified as 004 minor/major are very similar in size.
            use if C/A is very different than one.

        '''
        self.image=hs.load(fname)
        self.cal=self.image.axes_manager[0].scale
        self.h=h
        self.spot_sigma=spot_sigma
        self.fft=getfft(self.image)
        self.filterfft()
        self.build_mask()
        self.imshape=self.image.data.shape[0]
        self.use_watershed=use_watershed
        self.save=save_images
        self.filter_by_ellipse=filter_by_ellipse
        self.filter_by_ellipse_axes=filter_by_ellipse_axes
        self.use_blob_detection=use_blob_detection

    def label2coord(self,imlabel):
        return np.dstack(np.where(imlabel))[0]

    def fft_peak_distance(self,fft_coord):
        if all(abs(fft_coord-self.image.data.shape[0]/2.)<10):
            return 0
        d=np.sqrt((abs(fft_coord-self.imshape/2.)**2.).sum())
        return self.image.data.shape[0]*self.cal/d


    def filterfft(self):
        pfft=ski.filters.gaussian(abs(self.fft),sigma=5)
        pfft_log=np.log(pfft)
        pfft_log-=pfft_log.min()
        pfft_log/=pfft_log.max()

        self.filtered_fft=pfft_log

    def find_plane_peaks(self,plane="101"):

        '''plane="101", "004", "200"'''

        lims={"101":[0.342,0.362],
             "004":[0.2275,0.2475],
             "200":[0.1792,0.1992]}

        local_maxima = ski.morphology.extrema.h_maxima(self.filtered_fft,self.h)
        allspots=self.label2coord(local_maxima)


        d=np.array([self.fft_peak_distance(np.array((x,y))) for x,y in zip(allspots[:,1],allspots[:,0])])
        mask=np.logical_and(np.logical_and(d>lims[plane][0], d<lims[plane][1]),allspots[:,0]-self.imshape/2>0)
        self.fft_peaks=allspots[mask]

        self.angles=np.arcsin((self.fft_peaks[:,0]-self.imshape/2)/np.linalg.norm(self.fft_peaks-self.imshape/2,axis=1))
        for i in range(self.angles.shape[0]):
            if self.fft_peaks[i,1]-self.imshape/2<0:
                self.angles[i]=np.pi-self.angles[i]

        return self.fft_peaks

    def plot_fft_peaks(self):
        plt.clf()
        plt.imshow(self.filtered_fft)
        for peak in self.fft_peaks:
            plt.plot(peak[1],peak[0],"ro")
        return


    def build_mask(self):
        # build mask
        x = np.array(range(0,100))
        y = np.array(range(0,100))
        x, y = np.meshgrid(x, y)
        z = np.exp(-((x-50)**2+(y-50)**2)/self.spot_sigma)
        self.mask=z

    def spot_ifft(self,spot_index):
        s=self.fft_peaks[spot_index]
        s1=self.image.data.shape[0]-s
        mask=np.zeros(self.image.data.shape)
        wmask=int(self.mask.shape[0]/2)
        mask[s[0]-wmask:s[0]+wmask,s[1]-wmask:s[1]+wmask]=self.mask
        mask[s1[0]-wmask:s1[0]+wmask,s1[1]-wmask:s1[1]+wmask]=self.mask
        self.full_mask=mask

        fft_filtered_by_spot=self.fft*mask

        image_filtered_by_spot=np.fft.ifft2(np.fft.fftshift(fft_filtered_by_spot))

        return abs(image_filtered_by_spot)


    def blob_filter(self,image):
        #sigma and power are arbitrary
        self.current_filter=ski.filters.gaussian(image**1.25,sigma=8)[20:-20,20:-20]
        return self.current_filter

    def blob_threshold(self,image):
        x=image-image.min()
        x/=x.max()
        x*=255
        x=x.astype("uint8")
        th=ski.filters.threshold_otsu(x)

        x[x<th]=0
        seed=(x==x.max())

        if self.use_watershed:
            local_max=ski.morphology.extrema.h_maxima(x,h=50)
            markers = ndi.label(local_max)
            ws=ski.morphology.watershed(-x,markers[0],mask=x)
            maxs=[]
            for i in range(markers[1]):
                maxs.append(x[ws==i].max())
            good=np.array(maxs).argmax()

            for i in range(markers[1]):
                if i!=good:
                    x[ws==i]=0

        elif self.use_blob_detection:
            params = cv2.SimpleBlobDetector_Params()

            # Filter by Area.
            params.filterByArea = True
            params.minArea = 25/(self.image.axes_manager[0].scale)**2
            params.maxArea=np.inf
            # Filter by Circularity
            params.filterByCircularity = False
            # Filter by Convexity
            params.filterByConvexity = False
            # Filter by Inertia
            params.filterByInertia = False
            d=cv2.SimpleBlobDetector_create(params)
            kp=d.detect(255-hr.to8bit(x))
            self.im_kp=cv2.drawKeypoints(hr.to8bit(iff),kp,np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            sds=[i.pt for i in kp]
            seed=[]
            for s in sds:
                a=np.zeros(self.image.data.shape)
                a[s]=255
                seed.append()






        mask=np.copy(x)
        filled = ski.morphology.reconstruction(seed, mask, method='dilation')

        self.current_th=filled
        self.current_rot=filled

        return filled

    def calculate_ellipse_angle(self,th):
        # calculate ellipse angle
        x,y=np.nonzero(th)
        xc,yc=np.mean(x),np.mean(y)
        x=x.astype("float")-xc
        y=y.astype("float")-yc
        coords = np.vstack([x, y])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]
        x_v2, y_v2 = evecs[:, sort_indices[1]]
        return np.arctan((x_v1)/(y_v1)),evals[1]/evals[0]

    def blob_measure(self,th,angle):
        if any(th[0,:]!=0) or any(th[-1,:]!=0) or any(th[:,0]!=0) or any(th[:,-1]!=0):
            return 0.
        rot=ski.transform.rotate(th,angle*180/np.pi,True)
        self.current_rot=rot
        i=rot.max(0).argmax()
        sizepxl=(rot.sum(0)[i:]==0).argmax()+(rot.sum(0)[:i][::-1]==0).argmax()
        return sizepxl*self.cal

    def save_images(self,plane,i):
        c=self.fft_peaks

        plt.clf()
        figure=plt.gcf()
        figure.set_size_inches(7,2)
        plt.subplot(131)
        plt.imshow(self.filtered_fft)
        plt.plot(c[i,1],c[i,0],"ro")

        plt.subplot(132)
        plt.imshow(self.current_filter)

        plt.subplot(133)
        plt.imshow(self.current_rot)
        plt.plot(self.current_rot.sum(0)+1000)
        m=self.sizes[plane][-1]
        if not m>0:
            plt.gca().set_title("discarded")
        else:
            plt.gca().set_title(str(m)+" nm")
        figure=plt.savefig(self.image.metadata.General.original_filename.split(".")[0]+"_"+plane+"_"+str(i)+".png")

    def run_analysis(self):
        self.sizes={"101":[],"004":[],"200":[]}

        for plane in ["101", "004", "200"]:#["101", "004", "200"]
            print("now working on "+plane)


            c=self.find_plane_peaks(plane)
            print("with "+ str(c.shape[0])+" particles")


            for i in range(c.shape[0]):
                filtered=self.blob_filter(self.spot_ifft(i))
                th=self.blob_threshold(filtered)

                if self.filter_by_ellipse and (plane=="004" or plane=="200"):
                    a_el,raxes=self.calculate_ellipse_angle(th)
                    a_sp=self.angles[i]
                    x=(a_el-a_sp)*180/np.pi
                    r=abs(x%90)
                    if not(r<10 or r>80):
                        print(self.image.metadata.General.original_filename.split(".")[0]+"_"+plane+"_"+str(i)+" not good")
                        self.sizes[plane].append(0)
                    else:
                        if self.filter_by_ellipse_axes and abs(raxes-1)<0.4:
                            print(self.image.metadata.General.original_filename.split(".")[0]+"_"+plane+"_"+str(i)+" not good")
                            self.sizes[plane].append(0)
                        else:
                            self.sizes[plane].append(self.blob_measure(th,self.angles[i]))

                else:
                    self.sizes[plane].append(self.blob_measure(th,self.angles[i]))

                if self.save:
                    self.save_images(plane,i)

        return


def getfft(hs_im):# lacks returning a proper image
        im=hs_im.data
        fft=np.fft.fft2(im)
        fft=np.fft.fftshift(fft)
        return fft

def plane_analysis_onfolder_v2(folder=None,return_params=False,**kwargs):
    start=time.time()
    if folder==None:
        folder=os.getcwd()
    sizes={}

    for file in os.listdir(folder):
        ext=file.split(".")[-1]
        if ext=="dm3" or ext=="tif":
            print("now working on file "+file)
            a=plane_analysis_v2(fname=folder+"/"+file,printduration=False,**kwargs)
            s=a.analyze()

            for key in s.keys():
                if key in sizes.keys():
                    sizes[key]+=s[key]
                else:
                    sizes[key]=s[key]

    for key in sizes.keys():
        sizes[key]=np.array(sizes[key])

    end=time.time()
    elapsed=end-start
    print("lasted "+str(elapsed)+" seconds")
    if return_params:
        params={}
        for k,v in inspect.signature(plane_analysis_v2).parameters.items():
            params[k]=v.default
        for k in kwargs.keys():
            params[k]=kwargs[k]

        return sizes,params
    else:
        return sizes



def plane_analysis_onfolder(folder=None,**kwargs):
    '''kwargs are passed to plane_analysis'''

    start=time.time()
    if folder==None:
        folder=os.getcwd()

    sizes200=[]
    sizes004=[]
    sizes101=[]

    for file in os.listdir(folder):
        ext=file.split(".")[-1]
        if ext=="dm3" or ext=="tif":
            print("now working on file "+file)
            a=plane_analysis(fname=folder+"/"+file,**kwargs)
            a.run_analysis()
            sizes200+=a.sizes["200"]
            sizes004+=a.sizes["004"]
            sizes101+=a.sizes["101"]


    print(str(sizes200.count(0))+" discarded measurements for 200")
    print(str(sizes004.count(0))+" discarded measurements for 004")
    print(str(sizes101.count(0))+" discarded measurements for 101")


    sizes={}
    sizes["101"]=np.array(list(filter(lambda a: a != 0, sizes101)))
    sizes["200"]=np.array(list(filter(lambda a: a != 0, sizes200)))
    sizes["004"]=np.array(list(filter(lambda a: a != 0, sizes004)))

    end=time.time()
    elapsed=end-start
    print("lasted "+str(elapsed)+" seconds")
    return sizes

def anatase_results_xls(sizes,name,params=None):
    '''Returns table of results from the size distributions'''
    alpha=21.69*np.pi/180.

    workbook = xlsxwriter.Workbook('Plane Analysis Results'+name+'.xlsx',
                                    options={"nan_inf_to_errors":True})
    worksheet = workbook.add_worksheet()

    results={}
    for plane in ['200','004','101']:
        s=sizes[plane]
        if len(s)==0:
            s=np.array([0])
        results[plane]=[s.shape[0],
               np.average(s),
               np.std(s),
               s.min(),
               s.max()]

    A=results['200'][1]
    C=results['004'][1]
    L=results['101'][1]/np.sin(alpha)-A/np.tan(alpha)
    a=(results['101'][1]-C*np.sin(alpha))/np.cos(alpha)

    d101=results['101'][2]/2.
    dA=results['200'][2]/2.
    dC=results['004'][2]/2.
    dL=np.linalg.norm([d101/np.sin(alpha),dA/np.tan(alpha)])
    da=np.linalg.norm([d101/np.cos(alpha),dC*np.tan(alpha)])

    dCA=(C/A)*np.linalg.norm([dC/C,dA/A])

    worksheet.write('B7', 'Parameter')
    worksheet.write('B8', 'A')
    worksheet.write('B9', 'C')
    worksheet.write('B10', 'L')
    worksheet.write('B11', 'a')
    worksheet.write('B12', 'C/A')


    worksheet.write('C7', 'Value (nm)')
    worksheet.write('C8', A)
    worksheet.write('C9', C)
    worksheet.write('C10', L)
    worksheet.write('C11', a)
    worksheet.write('C12', C/A)

    worksheet.write('D7', 'Error (nm)')
    worksheet.write('D8', dA)
    worksheet.write('D9', dC)
    worksheet.write('D10', dL)
    worksheet.write('D11', da)
    worksheet.write('D12', dCA)

    su200=4*A*L
    su004=2*a*a
    su101=2*(A*A-a*a)/np.sin(alpha)

    tsu=su200+su004+su101
    tV=L*A**2+(A**3-a**3)/(3*np.tan(alpha))
    sv=tsu*10**9/tV #in 1/meter
    density=3900000 #in g/m3
    BET=sv/density
    worksheet.write('B13', "BET (m^2/g)")
    worksheet.write("C13", BET)



    su200p=100*su200/tsu
    su004p=100*su004/tsu
    su101p=100*su101/tsu

    dsu200=2*(dA/A+dL/L)*A*L
    dsu004=np.linalg.norm(2*[2*da*a])
    dsu101=np.linalg.norm(8*[dA*A/(2*np.sin(alpha)),da*a/(2*np.sin(alpha))])

    worksheet.write('B2', 'Direction')
    worksheet.write('B3', '200')
    worksheet.write('B4', '004')
    worksheet.write('B5', '101')

    worksheet.write('C2', 'N')
    worksheet.write('D2', 'Mean size (nm)')
    worksheet.write('E2', 'std (nm)')
    worksheet.write('F2', 'min (nm)')
    worksheet.write('G2', 'max (nm)')

    worksheet.write('H2', 'Surface (nm^2)')
    worksheet.write('H3', su200)
    worksheet.write('H4', su004)
    worksheet.write('H5', su101)

    worksheet.write('I2', 'Surface error (nm^2)')
    worksheet.write('I3', dsu200)
    worksheet.write('I4', dsu004)
    worksheet.write('I5', dsu101)

    worksheet.write('J2', 'Surface %')
    worksheet.write('J3', su200p)
    worksheet.write('J4', su004p)
    worksheet.write('J5', su101p)


    column=['C','D','E','F','G']
    for plane,row in zip(['200','004','101'],range(3,6)):
        for i in range(5):
             worksheet.write(column[i]+str(row),results[plane][i])



    if params!=None:
        worksheet.write('L2', 'Parameters')
        for i,k in enumerate(params.keys()):
            worksheet.write('L'+str(i+3), str(k))
            worksheet.write('M'+str(i+3), str(params[k]))
    workbook.close()
    return

def get_histograms(sizes,bins=80,range=(0,80)):

    for plane in ["200","004","101"]:
        plt.clf()
        plt.gcf().set_size_inches(3,3)
        plt.hist(sizes[plane],edgecolor="k",bins=bins,range=range)
        ax=plt.gca()
        ax.set_title(plane+" direction")
        ax.set_xlabel("length (nm)")
        ax.set_xlim(range[0],range[1])
        ax.set_ylabel("# of particles")
        plt.tight_layout()
        plt.gcf().savefig(plane+"hist.png")
    return
