#! /usr/bin/env python
import math
import numpy as np
from numpy.linalg import svd
import cv2
import sys
import time


class WMAP():

     def __init__(self,img):
          self.img=img
          self.orows, self.ocols, self.ochannels = img.shape
   
          M = self.orows%2
          N = self.ocols%2

          #con dimensions multiplos de 2 acelerase o calculo da DFT
          if M+N>0:
               self.img=cv2.copyMakeBorder(img,0,M,0,N,cv2.BORDER_REPLICATE);
               self.orows+=M;
               self.ocols+=N;
             
          #parametros da Gaussiana para obter o mapa de densidade de atencion
          self.blurr_sizex=self.orows 
          self.blurr_sizey=self.ocols
          self.blurr_maxhw=max(0,min(np.int(self.orows/2),np.int(self.ocols/2))-1)
          if self.blurr_maxhw%2==0:
               self.blurr_maxhw+=1
          self.blurr_sig=max(self.orows,self.ocols)*2*0.01; #mellor axustalo aos pixeles de 1 grao visual
          self.blurr_mindist=max(0,min(np.int(self.orows/10),np.int(self.ocols/10))-1);
    
          #valores para o banco de filtros monoxenicos
          self.bank_Bfo=1 #distribucion as frecuencias centrais dos filtros en octavas
          self.bank_Bab=2.3548 #ancho do filtro en octavas -->sigmaOn=0.5
          self.bank_nscale=3
          self.bank_minWaveLength=8
          self.bank_mult=np.exp(self.bank_Bfo*np.log(2.0))
          self.bank_sigmaOnf=(self.bank_Bab*np.sqrt(np.log(2.0)/2.0))/2.0
    
          #parametros para compensar ruido e os puntos onde a fase se alinha ao longo das escalas
          self.bank_k=1
          self.bank_cutOff=0.25
          self.bank_g=10
          
          self.monofilt()
          self.colorDecorrelation()


     def monofilt(self):
          rows=self.blurr_sizex
          cols=self.blurr_sizey
    
          x_range=np.zeros((1,cols),np.float64)
          y_range=np.zeros((rows,1),np.float64)
          radius=np.zeros((rows,cols),np.float64) 
          
          for i in range(cols):
		     x_range[0,i]=np.float64((i-(cols-1.0)/2.0)/(cols-1.0))
          for z in range(rows):
               y_range[z,0]=np.float64((z-(rows-1.0)/2.0)/(rows -1.0))
    
          u1=np.tile(x_range, (rows,1)) 
          u2=np.tile(y_range, (1,cols))
    
          #construcion de filtros monoxenicos no dominio frecuencial            
          u1=self.shiftDFT(u1)
          u2=self.shiftDFT(u2)
    
          radius=np.sqrt(np.square(u1)+np.square(u2))
          radius[0,0]=1
     
          self.F=np.empty((rows,cols,self.bank_nscale),np.float64)
          self.H=np.empty((rows,cols,2),np.float64)
          self.H[:,:,0]=-u2/radius
          self.H[:,:,1]=u1/radius
    
          radius=np.log(radius)
     
          wavelength=self.bank_minWaveLength
          sigma2=2*self.bank_sigmaOnf*self.bank_sigmaOnf
    
    #Trasformada Inversa de Fouriere (IDFT) para el procesamiento de las componentes logGabor
          for s in range(self.bank_nscale):
               radius2=radius+math.log(wavelength)
               radius2=(-1.0/sigma2)*radius2*radius2
              
               temp=np.exp(radius2)
               temp[0,0]=0
               self.F[:,:,s]=temp;
               wavelength*=self.bank_mult


     def colorDecorrelation(self):
          rows, cols, channels=self.img.shape

          if channels>1:
               img_f=self.img.astype(np.float64)
               img_f=img_f.reshape(rows*cols,3) 
               mu=img_f.mean(axis=0)
          #valor medio de las columnas
               mu=np.tile(mu, [rows*cols,1])    
               img_f = img_f - mu
          #sustraccion(subtract) de cada canal con su valor medio
                
               A = np.dot(img_f.transpose(),img_f)
               U, sigma, V =svd(A)
                
               self.imgDecorr = np.dot(img_f,U)
               self.imgDecorr = self.imgDecorr.reshape(rows,cols,3)
          else:
               self.imgDecorr=self.img.astype(np.float64)

     def shiftDFT(self,fImage):
          cx = np.int(fImage.shape[0]/2)
          cy = np.int(fImage.shape[1]/2)

	     #reordenaccion de los cuadrantes de la imagen de Fourier
	     #el origen se va al centro de la imagen

          q0=fImage[0:cx,0:cy]
          q1=fImage[0:cx,cy:]
          q2=fImage[cx:,0:cy]
          q3=fImage[cx:,cy:]

          tmp=q0.copy()
          q0[:,:]=q3
          q3[:,:]=tmp
	
          tmp=q1.copy()
          q1[:,:]=q2
          q2[:,:]=tmp

          return fImage


     def salmap(self):
          if self.imgDecorr.shape[2]==1: #channels()==1)
               achroDecorrImg=amap.imgDecorr

               self.Salmap=mgproplocal(achroDecorrImg)
               self.Salmap=cv2.GaussianBlur(self.Salmap,(self.blurr_maxhw,self.blurr_maxhw),self_blurr_sig,self_blurr_sig,BORDER_DEFAULT)

          else:         
               achroDecorrImg=self.imgDecorr[:,:,0]
               chroDecorrImg=np.empty((self.orows,self.ocols,2),np.float64)
               chroDecorrImg[:,:,0]=self.imgDecorr[:,:,1]
               chroDecorrImg[:,:,1]=self.imgDecorr[:,:,2] 

               self.Salmap=self.mgproplocal(achroDecorrImg)
               self.Salmap=self.Salmap + self.mgproplocal(chroDecorrImg[:,:,0])
               self.Salmap=self.Salmap + self.mgproplocal(chroDecorrImg[:,:,1])
               self.Salmap=cv2.GaussianBlur(self.Salmap,(self.blurr_maxhw,self.blurr_maxhw),self.blurr_sig,self.blurr_sig,0)

          suma= self.Salmap.sum()
          Salmap = self.Salmap/suma;
          return Salmap


     def mgproplocal(self,img):
          rows=self.orows
          cols=self.ocols
    
          imfft=np.zeros((rows,cols,2),np.float64)
          imfft[:,:,0]=img

          imfft = cv2.dft(imfft)#,flags = cv2.DFT_COMPLEX_OUTPUT)
    
          amplitude=np.ones((rows,cols),np.float64)
          fase=cv2.phase(imfft[:,:,0],imfft[:,:,1])
    
          x=np.empty((rows,cols))
          y=np.empty((rows,cols))

          cv2.polarToCart(amplitude,fase,x,y)
          imfft[:,:,0]=x
          imfft[:,:,1]=y
          
          f=np.empty((rows,cols,self.bank_nscale),np.float64)
          h1f=np.empty((rows,cols,self.bank_nscale),np.float64)
          h2f=np.empty((rows,cols,self.bank_nscale),np.float64)
    
          epsilon=0.0001

          sumAn=np.zeros((rows,cols),np.float64)
          sumf=np.zeros((rows,cols),np.float64)
          sumh1f=np.zeros((rows,cols),np.float64)
          sumh2f=np.zeros((rows,cols),np.float64)
  
          
          for s in range(self.bank_nscale):
        
               imfftpb=np.zeros((rows,cols,2),np.float64)
               imfftpb[:,:,0]=self.F[:,:,s]
      
               imfftpb=cv2.mulSpectrums(imfft,imfftpb, 0)
               h=cv2.mulSpectrums(imfftpb,self.H, 0)
               imfftpb=cv2.idft(imfftpb)
               f[:,:,s]=imfftpb[:,:,0]
       
               h=cv2.idft(h)
               h1f[:,:,s]=h[:,:,0]
               h2f[:,:,s]=h[:,:,1]
       
       
               An= f[:,:,s]*f[:,:,s] + h1f[:,:,s]*h1f[:,:,s] +h2f[:,:,s]*h2f[:,:,s]
               An=np.sqrt(An)
       
               sumAn=sumAn+An
               sumf=sumf+f[:,:,s]
               sumh1f=sumh1f+h1f[:,:,s]
               sumh2f=sumh2f+h2f[:,:,s]

       
               if s==0:
                    maxAn=An
                    tau= np.median(An)/np.log(4.0)
           
               else:
                    maxAn=np.maximum(An,maxAn)

          maxAn=maxAn+epsilon       
          
          width=(sumAn/maxAn -1.0)/(self.bank_nscale-1)
    
          t=(self.bank_cutOff-width)*self.bank_g
          t=np.exp(t)
          t=t+1
    
          weight= np.ones((rows,cols),np.float64)/t

          totalTau=tau*(1.0 - pow(1.0/self.bank_mult,self.bank_nscale))/(1.0-(1.0/self.bank_mult))

          EstNoiseEnergyMean=totalTau*math.sqrt(math.pi/2.0)
          EstNoiseEnergySigma=totalTau*math.sqrt((4.0-math.pi)/2.0)
          T= EstNoiseEnergyMean + self.bank_k*EstNoiseEnergySigma
    
          XEnergy =sumf*sumf + sumh1f*sumh1f+ sumh2f*sumh2f
          XEnergy=np.sqrt(XEnergy)
          XEnergy=XEnergy+epsilon
    
          Meanf=sumf/XEnergy
          Meanh1f= sumh1f/XEnergy
          Meanh2f = sumh2f/XEnergy
    
          maxPhaseDev=np.empty((rows,cols),np.float64)
    
          for s in range(self.bank_nscale):
               PhaseDev=f[:,:,s]*Meanf + h1f[:,:,s]*Meanh1f + h2f[:,:,s]*Meanh2f
        
               if s==0:
                    maxPhaseDev=PhaseDev.copy()
               else:
                    maxPhaseDev=np.maximum(maxPhaseDev,PhaseDev)

          maxPhaseDev=cv2.max(maxPhaseDev-T,0)
        
          maxPhaseDev=weight*maxPhaseDev
    
          bordersize=10;

          temp1=maxPhaseDev[bordersize:rows-bordersize,bordersize:cols-bordersize];
          maxPhaseDev=np.zeros((rows,cols),np.float64)
          maxPhaseDev[bordersize:rows-bordersize,bordersize:cols-bordersize]=temp1.copy()
    
          return maxPhaseDev

####
####

if __name__ == "__main__":

     #lectura de imagen a procesar
     if len(sys.argv)>1:
          img = cv2.imread(sys.argv[1],1);
          img_name = str(sys.argv[1]).split('/')[-1].split('.')[0]
     else:
          img = cv2.imread("./cat.jpg",1)
          img_name = 'cat'
        
     if img is None:
	     print "Can not open image file"
	     exit(1)

     start_t = time.time()  

     #Creacion de objeto y calculo del mapa de saliencia
     wmap=WMAP(img)
     Salmap=wmap.salmap()
     
     end_t = time.time()#cv2.getTickCount()
     time_t = (end_t - start_t)#/ cv2.getTickFrequency()
     print "Execution time: {:.3f} seconds".format(time_t)
 
     #paso a formato imagen para visualizacion y almacenamiento
     cv2.normalize(Salmap,Salmap, 0, 255, cv2.NORM_MINMAX)
     A_scaled=Salmap.astype(np.uint8)

     cv2.imwrite('%s' % img_name+'_wmap.jpg', A_scaled.astype(int))

     A_scaled = cv2.applyColorMap(A_scaled, cv2.COLORMAP_JET)
     cv2.imwrite('%s' % img_name+'_sal_colormap_wmap.jpg', A_scaled.astype(int))
     
     #cv2.imshow("Saliency",A_scaled)
     #cv2.waitKey(0);
     

