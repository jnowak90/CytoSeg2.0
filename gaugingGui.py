from tkinter import *
from tkinter import Tk, Label, Button
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import sys
import numpy as np
import scipy as sp
import skimage
from skimage import feature
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from packaging.version import Version

def im2d3d(im):
    if(len(im.shape)==2):
        im=im[:,:,np.newaxis]
    else:
        im=im
    return im

def skeletonize_graph(imO,mask,sigma,block,small,factr):
    imO-=imO[mask].min()
    imO*=255.0/imO.max()
    ly,lx,lz=imO.shape
    imR=imO.copy()*0
    imT=imO.copy()*0
    for z in range(lz):
        imR[:,:,z] = tube_filter(imO[:,:,z],sigma)
        threshold = skimage.filters.threshold_local(imR[:,:,z],block)
        imT[:,:,z] = imR[:,:,z] > threshold
    imS=skimage.morphology.skeletonize_3d(imT>0)
    ones=np.ones((3,3,3))
    imC=skimage.morphology.remove_small_objects(imS,small,connectivity=2)>0
    for z in range(lz):
        imC[:,:,z]=imC[:,:,z]*mask
    imC=imC>0
    imL,N=sp.ndimage.label(imC,structure=ones)
    mean=imO[imC].mean()
    means=[np.mean(imO[imL==n]) for n in range(1,N+1)]
    imA=1.0*imC.copy()
    for n in range(1,N+1):
        if(means[n-1]<mean*factr):
            imA[imL==n]=0
    imA=skimage.morphology.remove_small_objects(imA>0,2,connectivity=8)
    return imA

def tube_filter(imO,sigma):
    if Version(skimage.__version__) < Version('0.15'):
        imH=skimage.feature.hessian_matrix(imO,sigma=sigma,mode='reflect')
        imM=skimage.feature.hessian_matrix_eigvals(imH[0],imH[1],imH[2])
    else:
        imH=skimage.feature.hessian_matrix(imO,sigma=sigma,mode='reflect',order='xy')
        imM=skimage.feature.hessian_matrix_eigvals(imH)
    imR=-1.0*imM[1]
    imT=255.0*(imR-imR.min())/(imR.max()-imR.min())
    return imT

# set randw, randn and depth from parameter list
pathToPlugin = sys.argv[0]
pathToPlugin = '/'.join(pathToPlugin.split('/')[:-1])
roll = 50
randw = 1
randn = 1
depth = 7.75
filename = sys.argv[1]

class GaugingGui:

    def __init__(self,root, filename):
        self.root = root
        if filename == 'None':
            self.filename = ""
        else:
            self.filename = filename
        print(self.filename)
        self.past = 1

        self.root.title('CytoSeg 2.0 - Gauging')
        self.root.geometry('600x800')

        # Menu bar buttons
        self.menu = Frame(self.root)
        self.open = Button(self.menu,text="Open Image",command=self.openImage).grid(row=0, column=0)
        self.help = Button(self.menu, text="Help", command=self.helpMessage).grid(row=0, column=1)
        self.back = Button(self.menu, text="Back to Main Menu", command=self.root.quit).grid(row=0, column=2)
        self.menu.pack(side=TOP, anchor=W, padx=10,pady=10)

        # Welcome message
        self.textVar = StringVar(self.root)
        self.LabelWelcome = Label(self.root, textvariable=self.textVar, fg='dark green').pack()
        self.textVar.set("Open image to start parameter gauging.")

        # canvas for the image
        self.canvas = Canvas(self.root, width = 500, height = 500)
        self.canvas.pack()

        # frame for the scale bars
        self.frame = Frame(self.root)
        self.LabelSigma = Label(self.frame,text="v_width")
        self.sigma = Scale(self.frame, from_=0.4, to=2.2, resolution=0.2, orient=HORIZONTAL,length=450, command=self.showValueSigma, showvalue=0)
        self.sigma.set(2.0)
        self.LabelSigmaValue = Label(self.frame, text="")
        self.sigma.bind("<ButtonRelease-1>", self.displaySkeleton)

        self.LabelBlock = Label(self.frame,text="v_thres")
        self.block = Scale(self.frame, from_=20, to=112, resolution=10.0, orient=HORIZONTAL,length=450, command=self.showValueBlock, showvalue=0)
        self.block.set(101.0)
        self.LabelBlockValue = Label(self.frame, text="")
        self.block.bind("<ButtonRelease-1>", self.displaySkeleton)

        self.LabelSmall = Label(self.frame,text="v_size")
        self.small = Scale(self.frame, from_=2.0, to=47.0, resolution=5.0, orient=HORIZONTAL,length=450, command=self.showValueSmall, showvalue=0)
        self.small.set(27.0)
        self.LabelSmallValue = Label(self.frame, text="")
        self.small.bind("<ButtonRelease-1>", self.displaySkeleton)

        self.LabelFactr = Label(self.frame,text="v_int")
        self.factr = Scale(self.frame, from_=0.1, to=2.0, resolution=0.2, orient=HORIZONTAL,length=450, command=self.showValueFactr, showvalue=0)
        self.factr.set(0.5)
        self.LabelFactrValue = Label(self.frame,text="")
        self.factr.bind("<ButtonRelease-1>", self.displaySkeleton)

        self.LabelSigma.grid(row=0, column=0, sticky=S, pady=10)
        self.sigma.grid(row=0, column=4, pady=10)
        self.LabelSigmaValue.grid(row=0, column=20, sticky=E, pady=10)
        self.LabelBlock.grid(row=1, column=0, sticky=S, pady=10)
        self.block.grid(row=1, column=4, pady=10)
        self.LabelBlockValue.grid(row=1, column=20, sticky=E, pady=10)
        self.LabelSmall.grid(row=2, column=0, sticky=S, pady=10)
        self.small.grid(row=2, column=4, pady=10)
        self.LabelSmallValue.grid(row=2, column=20, sticky=E, pady=10)
        self.LabelFactr.grid(row=3, column=0, sticky=S, pady=10)
        self.factr.grid(row=3,column=4, pady=10)
        self.LabelFactrValue.grid(row=3, column=20, sticky=E, pady=10)

        self.frame.pack(side=TOP,padx=10,pady=5)

        # button to submit parameters
        self.Final = Button(self.root,text='Choose Parameters',command=self.get_parameters).pack(anchor=CENTER)

    # select image and open in canvas
    def openImage(self):
        self.lastdir = './'
        if self.filename == "":
            self.filename = filedialog.askopenfilename(initialdir = self.lastdir, title ="Select image!",filetypes = (("png images","*.png") , ("tif images","*.tif"), ("jpeg images","*.jpg")) )
        self.img = Image.open(self.filename)
        if self.img.size[0] == self.img.size[1]:
            self.resized = self.img.resize((500, 500),Image.ANTIALIAS)
        else:
            self.max, self.argmax = np.max(self.img.size), np.argmax(self.img.size)
            self.min = (np.min(self.img.size)*500)/self.max
            if self.argmax == 0:
                self.resized = self.img.resize((500,int(self.min)),Image.ANTIALIAS)
            else:
                self.resized = self.img.resize((int(self.min),500),Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(self.resized)
        self.canvas.create_image(0, 0, anchor=NW, image=self.image)
        if self.filename != "":
            self.lastdir = os.path.dirname(self.filename)
        self.textVar.set("Change segmentation by adjusting parameter controllers.")

    # message that pops up when clicking Help
    def helpMessage(self):
        messagebox.showinfo("Parameter information","v_width: width of filamentous structures to enhance with a 2D tubeness filter,\n\nv_thres: block size for adaptive median threshold,\n\nv_size: size of small objects to be removed,\n\nv_int: lowest average intensity of a component")


    # have to cheat here, scale bar is not showing the intervals for block, small and factr as it should...
    def showValueSigma(self,ev):
        self.LabelSigmaValue.configure(text=ev)
    def showValueBlock(self,ev):
        number = int(ev)+1
        self.LabelBlockValue.configure(text=number)
    def showValueSmall(self,ev):
        number = int(ev)+2
        self.LabelSmallValue.configure(text=number)
    def showValueFactr(self,ev):
        number = format(float(ev)-0.1,".1f")
        self.LabelFactrValue.configure(text=number)

    # save the selected parameters in a file
    def get_parameters(self):
        params = "" + str(roll) + ","+ str(randw) + "," + str(randn) + "," + str(depth) + "," + str(self.sigma.get()) + "," + str(self.block.get()+1) + "," + str(self.small.get()+2) + "," + str(format(float(self.factr.get())-0.1,".1f"))
        np.savetxt(pathToPlugin+"/defaultParameter.txt",[params],fmt='%s')

    def displaySkeleton(self,ev):
        self.textVar.set("If you are satisfied with the segmentation, press 'Choose Parameters' to save\n the parameters for the CytoSeg analysis and go back to the main menu.")
        if self.filename!="":
            sig = self.sigma.get()
            blo = self.block.get()+1
            sma = self.small.get()+2
            fac = self.factr.get()-0.1

            path = self.filename
            imageName = path.split('/')[-1].split('.')[0]
            imagePath = '/'.join(path.split('/')[:-1])
            imO = skimage.io.imread(self.filename, plugin='tifffile')
            if imO.shape[2] in (3,4):
                imO = np.swapaxes(imO, -1, -3)
                imO = np.swapaxes(imO, -1, -2)
            mask = skimage.io.imread(imagePath+'/'+imageName+"_mask.tif", plugin='tifffile')>0

            shape = imO.shape
            imI = imO[0]
            imI = im2d3d(imI)
            imG = skimage.filters.gaussian(imI,sig)
            imA = skeletonize_graph(imG,mask,sig,blo,sma,fac)

            fig=plt.figure()
            plt.imshow(imI.reshape(shape[1],shape[2]),cmap='gray_r')
            imA = np.ma.masked_where(imA == 0, imA)
            plt.imshow(imA.reshape(shape[1],shape[2]),cmap='autumn')
            plt.axis('off')
            fig.savefig(imagePath+'/skeletonOnImage.png',bbox_inches='tight',dpi=300)

            self.img = Image.open(imagePath+'/skeletonOnImage.png')
            if self.img.size[0] == self.img.size[1]:
                self.resized = self.img.resize((500, 500),Image.ANTIALIAS)
            else:
                self.max, self.argmax = np.max(self.img.size), np.argmax(self.img.size)
                self.min = (np.min(self.img.size)*500)/self.max
                if self.argmax == 0:
                    self.resized = self.img.resize((500,int(self.min)),Image.ANTIALIAS)
                else:
                    self.resized = self.img.resize((int(self.min),500),Image.ANTIALIAS)
            self.image = ImageTk.PhotoImage(self.resized)
            self.canvas.create_image(0, 0, anchor=NW, image=self.image)



master = Tk()
my_gui = GaugingGui(master, filename)
master.mainloop()
