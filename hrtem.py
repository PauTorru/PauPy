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


class plane_analysis():

    def __init__(self,fname,save_images=True, h=0.07,spot_sigma=300,
                 use_watershed=False,
                 filter_by_ellipse=False,
                 filter_by_elipse_axes=False):

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
        filter_by_elipse_axes: bool
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
        self.filter_by_elipse_axes=filter_by_elipse_axes

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


        seed=(x==x.max())
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

                if self.filter_by_ellipse and plane=="004":
                    a_el,raxes=self.calculate_ellipse_angle(th)
                    a_sp=self.angles[i]
                    x=(a_el-a_sp)*180/np.pi
                    r=abs(x%90)
                    if not(r<10 or r>80):
                        print(self.image.metadata.General.original_filename.split(".")[0]+"_"+plane+"_"+str(i)+" not good")
                        self.sizes[plane].append(0)
                    else:
                        if self.filter_by_elipse_axes and abs(raxes-1)<0.4:
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
            a=plane_analysis(fname=file,**kwargs)
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

def anatase_results_xls(sizes):
    '''Returns table of results from the size distributions'''
    alpha=21.69*np.pi/180.

    workbook = xlsxwriter.Workbook('Plane Analysis Results.xlsx')
    worksheet = workbook.add_worksheet()

    results={}
    for plane in ['200','004','101']:
        s=sizes[plane]
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


    worksheet.write('B7', 'Value (nm)')
    worksheet.write('B8', 'A')
    worksheet.write('B9', 'C')
    worksheet.write('B10', 'L')
    worksheet.write('B11', 'a')

    worksheet.write('C7', 'Value (nm)')
    worksheet.write('C8', A)
    worksheet.write('C9', C)
    worksheet.write('C10', L)
    worksheet.write('C11', a)

    worksheet.write('D7', 'Error (nm)')
    worksheet.write('D8', dA)
    worksheet.write('D9', dC)
    worksheet.write('D10', dL)
    worksheet.write('D11', da)

    su200=4*A*L
    su004=2*a*a
    su101=2*(A*A-a*a)/np.sin(alpha)

    tsu=su200+su004+su101


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
