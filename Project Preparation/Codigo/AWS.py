#! /usr/bin/env python
import math
import numpy as np
from numpy.linalg import svd
import cv2
import sys
import time
import pickle

class AWS():
     def __init__(self,img,filters_file_name=None):
          self.img=img
          self.orows, self.ocols, self.ochannels = img.shape
   
          M = self.orows%2
          N = self.ocols%2

          #con dimensiones multiplos de 2 se acelera el calculo de la DFT
          if M+N>0:
               self.img=cv2.copyMakeBorder(img,0,M,0,N,cv2.BORDER_REPLICATE);
               self.orows+=M;
               self.ocols+=N;

          imgDecorr=self.colorDecorrelation()
          if self.ochannels==1: #imgDecorr.shape[2]==1: 
               self.achroDecorrImg=imgDecorr
          else:         
               self.achroDecorrImg=imgDecorr[:,:,0]
               self.chroDecorrImg=np.zeros((self.orows,self.ocols,2),np.float64)
               self.chroDecorrImg[:,:,0]=imgDecorr[:,:,1]
               self.chroDecorrImg[:,:,1]=imgDecorr[:,:,2]   
          
          #parametros para componente acromatica
          self.achrScales=7
          self.achrOrient=4
          self.achrLambda_min=3.0
          self.achrScFactor=2.0
          self.achrDeltaAng = 1.0
          self.achrBwF=1.1
          
          #parametros para componentes cromaticas
          self.chrScales=5;
          self.chrOrient=4;
          self.chrLambda_min=6.0;
          self.chrScFactor=2.0;
          self.chrDeltaAng = 1.7;
          self.chrBwF=1.1;

          self.achrLambda_max= (self.achrLambda_min) * (pow(self.achrScFactor,self.achrScales-1));
          self.chrLambda_max= (self.chrLambda_min) * (pow(self.chrScFactor,self.chrScales-1));
          self.achrpadSize =int(round(self.achrLambda_max/2.0))
          self.chrpadSize =int(round(self.chrLambda_max/2.0))
                    
          if filters_file_name==None:          
               self.createFilters()
          else:#lectura de filtos precalculados

               with open(filters_file_name, 'rb') as f:
                    data=np.load(f)

                    rows=int(self.orows +2*self.achrpadSize)
                    cols=int(self.ocols +2*self.achrpadSize) 
                    self.achr_filters=data['acf'].reshape((self.achrOrient,self.achrScales,rows*cols))
                    
                    rows=int(self.orows +2*self.chrpadSize)
                    cols=int(self.ocols +2*self.chrpadSize)
                    self.chr_filters=data['cf'].reshape((self.chrOrient,self.chrScales,rows*cols))
                    self.GaussKernel=data['gk'].reshape((self.orows,self.ocols,2))



     def createFilters(self):
          self.achrsigma_r = np.exp((-(pow(2.0,self.achrBwF) -1.0))/(pow(2.0,self.achrBwF) + 1.0))
          self.achrsigma_theta = (math.pi/self.achrOrient)/self.achrDeltaAng
          self.chrsigma_r = np.exp(-(pow(2.0,self.chrBwF) -1.0)/(pow(2.0,self.chrBwF) + 1.0))
          self.chrsigma_theta = (math.pi/self.chrOrient)/self.chrDeltaAng
          
          self.achr_filters=self.achr_pre_filter_Computations()
          self.chr_filters=self.chr_pre_filter_Computations()
          self.GaussKernel=self.LowPassFilter()

          #Guardaddo de filtros para posteriores usos
          filters_file_name='AWS_filters_image_size_'+ str(self.orows)+'_' +str(self.ocols) + '.dat'

          with open(filters_file_name, 'wb') as f:
               acf=self.achr_filters.reshape(self.achr_filters.size)
               cf=self.chr_filters.reshape(self.chr_filters.size)
               gk=self.GaussKernel.reshape(self.GaussKernel.size)
               np.savez(f,acf=acf,cf=cf,gk=gk)              

     def colorDecorrelation(self):
         # rows, cols, channels=self.img.shape

          if self.ochannels>1:
               img_f=self.img.astype(np.float64)
               img_f=img_f.reshape(self.orows*self.ocols,3) 
               
               mu=img_f.mean(axis=0)
             
               #valor medio de las columnas
               mu=np.tile(mu, [self.orows*self.ocols,1])    
               img_f = img_f - mu
               #sustraccion(subtract) de cada canal con su valor medio
                     
               A = np.dot(img_f.transpose(),img_f)
               U, sigma, V =svd(A)
                     
               imgDecorr = np.dot(img_f,U)
               imgDecorr = imgDecorr.reshape(self.orows,self.ocols,3)
          else:
               imgDecorr=self.img.astype(np.float64)

          return imgDecorr


     def salmap(self):
 
          imfft=self.padded_FourierTrasform2D(self.achroDecorrImg,self.achrpadSize)
                
          # calculo del nucleo (kernel) del log Gabor
          #radius,sintheta,costheta,theta_o,r_o
          #_filter=self.achr_pre_filter_Computations()
          #sigma_r = np.exp((-(pow(2.0,self.achrBwF) -1.0))/(pow(2.0,self.achrBwF) + 1.0))
          #sigma_theta = (math.pi/self.achrOrient)/self.achrDeltaAng 
                   
          resp1=self.logGabor(imfft,self.achr_filters) #r_o,theta_o,self.achrsigma_theta,radius,sintheta,costheta,self.achrsigma_r)
          imfft=None 

          #calculo de la mapa de saliencia para el primer canal
          conspic1 = np.zeros((self.orows,self.ocols),np.float64)

          inCompImg = np.empty((self.orows*self.ocols,np.int(self.achrScales)),np.float64)

          for i_orient in range(self.achrOrient):
          # copia cada componente log Gabor en una columna de inCompImg
               for  i_scale in range(self.achrScales):
                    inCompImg[:,i_scale]=resp1[i_orient,i_scale,:]
               
               Tsquared=self.whitening_sqr_meanDistance(inCompImg)
               Tsquared = Tsquared.reshape(self.orows,self.ocols)

               #busqueda del valor maximo y division
               minVal, maxVal, minLoc, maxLoc= cv2.minMaxLoc(Tsquared)
               if maxVal==.0:
                    maxVal=1.0
               Tsquared = Tsquared/maxVal
           
               Tsquared=self.Blurring(Tsquared,self.GaussKernel)    
               conspic1 = conspic1 + Tsquared
         
          inCompImg=None
          resp1=None

          minVal, maxVal, minLoc, maxLoc= cv2.minMaxLoc(conspic1)
          if (maxVal==.0):
               maxVal=1.0
          conspic1 = conspic1/maxVal
             
          #Suavizacion (difusion) del mapa de saliencia acromatico
          conspic1=self.Blurring(conspic1,self.GaussKernel)
             
          conspic2 = np.zeros((self.orows,self.ocols),np.float64)
          if self.ochannels>1:        
               #radius,sintheta, costheta,theta_o,r_o
               #_filter=self.chr_pre_filter_Computations()
               #sigma_r = np.exp(-(pow(2.0,self.chrBwF) -1.0)/(pow(2.0,self.chrBwF) + 1.0))
               #sigma_theta = (math.pi/self.chrOrient)/self.chrDeltaAng
                 

               for i_c_comp in range(self.ochannels-1):

                    imfft=self.padded_FourierTrasform2D(self.chroDecorrImg[:,:,i_c_comp],self.chrpadSize) 
                    resp2 = self.logGabor(imfft,self.chr_filters)#r_o,theta_o,self.chrsigma_theta,radius,sintheta,costheta,self.chrsigma_r)

                   
                    conspic = np.zeros((self.orows,self.ocols),np.float64)
                    inCompImg = np.empty((self.orows*self.ocols,int(self.chrScales)),np.float64)
                     
                    for i_orient in range(self.chrOrient):
                             
                         for i_scale in range(self.chrScales):
                              inCompImg[:,i_scale]=resp2[i_orient,i_scale,:]
                         
                         Tsquared=self.whitening_sqr_meanDistance(inCompImg)
                         Tsquared = Tsquared.reshape(self.orows, self.ocols)

                     
                         minVal, maxVal, minLoc, maxLoc= cv2.minMaxLoc(Tsquared)
                         if maxVal==.0:
                              maxVal=1.0
                         Tsquared = Tsquared/maxVal
                             
                         Tsquared=self.Blurring(Tsquared,self.GaussKernel)
                         conspic=conspic + Tsquared
               

                    minVal, maxVal, minLoc, maxLoc= cv2.minMaxLoc(conspic)
                    if maxVal==.0:
                         maxVal=1.0;             
                    conspic = conspic/maxVal;
                    conspic=self.Blurring(conspic,self.GaussKernel)
                    conspic2 = conspic2 + conspic
           
          # calculo del mapa de saliencia total y normalizacion
          Salmap=2*conspic1 + conspic2
          suma= Salmap.sum()
          Salmap = Salmap/suma

          return Salmap


     def padded_FourierTrasform2D(self,img,padSize):
     #procesamiento de la Trasformada de Fourier, evitando efectos de borde
     #con el padding

          padded=np.zeros((img.shape[0] + 2*padSize,img.shape[1] + 2*padSize),np.float64)
          padded=cv2.copyMakeBorder(img,padSize,padSize,padSize,padSize,cv2.BORDER_REPLICATE)

          imfft=np.zeros((padded.shape[0],padded.shape[1],2))
          imfft[:,:,0]=padded
          imfft = cv2.dft(np.float64(imfft))#,flags = cv2.DFT_COMPLEX_OUTPUT)

          return imfft


     def chr_pre_filter_Computations(self):

          rows=int(self.orows +2*self.chrpadSize)
          cols=int(self.ocols +2*self.chrpadSize) 
          radius=np.zeros((rows,cols),np.float64)
          sintheta=np.zeros((rows,cols),np.float64)
          costheta=np.zeros((rows,cols),np.float64)
             
          x=np.zeros((1,cols),np.float64)
          y=np.zeros((rows,1),np.float64)
          theta=np.zeros((rows,cols),np.float64)
          
          epsilon = 0.000001
          col2 = np.int(cols/2.0)
          row2 = np.int(rows/2.0)
             
          for i in range(cols-1,-1,-1):
               x[0,i]=np.float64(i-col2)/col2
          
          for z in range(rows-1,-1,-1):
               y[z,0]=np.float64(row2-z)/row2

          x=np.tile(x, (rows,1))
          x=x.astype(np.float64)  
          y=np.tile(y, (1,cols))
          y=y.astype(np.float64) 
             
          radius=np.sqrt(np.square(x)+np.square(y))   
          radius[row2,col2]=1.0;
          radius=radius+epsilon; 
             
          radius=np.log(radius)

          theta=cv2.phase(x,y)

          sintheta=np.sin(theta)
          costheta=np.cos(theta)

          
          #wavelengths = self.chrLambda_min     
          r_o=(2.0/self.chrLambda_min)*np.ones((self.chrScales), np.float64)     
          for i_scale in range(self.chrScales-1):
               r_o[i_scale+1] = r_o[i_scale]/self.chrScFactor


          _filter=np.zeros((self.chrOrient,self.chrScales,rows*cols),np.float64)
          for i_orient in range (self.chrOrient):
               theta_o = (i_orient)*(math.pi/self.chrOrient)
               sintheta_o=math.sin(theta_o)
               costheta_o=math.cos(theta_o)
               #rows, cols, channels =imfft.shape
                  
               spread=np.zeros((rows,cols),np.float64)
               
               divis2=np.zeros((rows,cols),np.float64)
                  
               t2=math.log(self.chrsigma_r)
               t2_s=2*t2*t2
               sigma_theta_s= 2*self.chrsigma_theta*self.chrsigma_theta
                  
          #calculo del filtro 
               ds = (sintheta * costheta_o) - (costheta * sintheta_o)
               dc = (costheta * costheta_o) + (sintheta * sintheta_o)
	
               dalpha=cv2.phase(dc,ds)#np.angle(dc+1j*ds)
               dalpha=dalpha-np.pi;
                 
               divis = (-1.0/sigma_theta_s)*dalpha*dalpha
               cv2.exp(divis,spread)#spread=np.exp(divis)
               
   
               for i_scale in range(self.chrScales):
                    dc=radius-math.log(r_o[i_scale])
                    ds= (-1.0/t2_s)*dc*dc
                    cv2.exp(ds,divis2)#divis2=np.exp(ds)

                    filtro=spread*divis2
                    filtro=self.shiftDFT(filtro)
                    _filter[i_orient,i_scale,:]=filtro.reshape(rows*cols)
          

          return _filter 



     def achr_pre_filter_Computations(self):

          rows=self.orows+2*self.achrpadSize
          cols=self.ocols+2*self.achrpadSize #=imfft.shape
          radius=np.zeros((rows,cols),np.float64)
          sintheta=np.zeros((rows,cols),np.float64)
          costheta=np.zeros((rows,cols),np.float64)
             
          x=np.zeros((1,cols),np.float64)
          y=np.zeros((rows,1),np.float64)
          theta=np.zeros((rows,cols),np.float64)
          
          epsilon = 0.000001
          col2 = np.int(cols/2.0)
          row2 = np.int(rows/2.0)
             
          for i in range(cols-1,-1,-1):
		     x[0,i]=np.float64(i-col2)/col2


          for z in range(rows-1,-1,-1):
               y[z,0]=np.float64(row2-z)/row2

          x=np.tile(x, (rows,1))
          x=x.astype(np.float64)  
          y=np.tile(y, (1,cols))
          y=y.astype(np.float64) 
             
          radius=np.sqrt(np.square(x)+np.square(y))   
          radius[row2,col2]=1.0;
          radius=radius+epsilon; 
             
          radius=np.log(radius)

          theta=cv2.phase(x,y)

          sintheta=np.sin(theta)
          costheta=np.cos(theta)
          
          #wavelengths = self.achrLambda_min     
          r_o=(2.0/self.achrLambda_min)*np.ones((self.achrScales), np.float64)     
          for i_scale in range(self.achrScales-1):
               r_o[i_scale+1] = r_o[i_scale]/self.achrScFactor

         
          _filter=np.zeros((self.achrOrient,self.achrScales,rows*cols),np.float64)
          for i_orient in range (self.achrOrient):
               theta_o = (i_orient)*(math.pi/self.achrOrient)
               sintheta_o=math.sin(theta_o)
               costheta_o=math.cos(theta_o)
               #rows, cols, channels =imfft.shape
                  
               spread=np.zeros((rows,cols),np.float64)
               
               divis2=np.zeros((rows,cols),np.float64)
                  
               t2=math.log(self.achrsigma_r)
               t2_s=2*t2*t2
               sigma_theta_s= 2*self.achrsigma_theta*self.achrsigma_theta
                  
          #calculo del filtro 
               ds = (sintheta * costheta_o) - (costheta * sintheta_o)
               dc = (costheta * costheta_o) + (sintheta * sintheta_o)
	
               dalpha=cv2.phase(dc,ds)#np.angle(dc+1j*ds)
               dalpha=dalpha-np.pi;
                 
               divis = (-1.0/sigma_theta_s)*dalpha*dalpha
               cv2.exp(divis,spread)#spread=np.exp(divis)
               
   
               for i_scale in range(self.achrScales):
                    dc=radius-math.log(r_o[i_scale])
                    ds= (-1.0/t2_s)*dc*dc
                    cv2.exp(ds,divis2)#divis2=np.exp(ds)

                    filtro=spread*divis2
                    filtro=self.shiftDFT(filtro)
                    _filter[i_orient,i_scale,:]=filtro.reshape(rows*cols)

          return _filter #radius, sintheta, costheta,theta_o,r_o


     def logGabor(self,imfft,_filter):#r_o_v,theta_o_v,sigma_theta,radius,sintheta,costheta,sigma_r):
        
          rows, cols, channels =imfft.shape
          Orients,Scales,fs=_filter.shape 

          resp=np.array([])    

          resp=np.zeros((Orients,Scales,self.orows*self.ocols),np.float64) 

          for i_orient in range (Orients):
            
               
               for i_scale in range(Scales):
                    
                    rpad=np.zeros((rows,cols,2), np.float64)
                    rpad[:,:,0]=_filter[i_orient,i_scale,:].reshape(rows,cols)
               
                    rpad=cv2.mulSpectrums(imfft, rpad, 0);

                    cv2.idft(rpad,rpad)
                    rpad= cv2.magnitude(rpad[:,:,0],rpad[:,:,1])
                    padSize=(rows-self.orows)/2
                    rpad=rpad[padSize:rows-padSize,padSize:cols-padSize]
                    resp[i_orient,i_scale,:]=rpad.reshape(rpad.size)
                    
          return resp

      
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
  
     def whitening_sqr_meanDistance(self,X):

          Z=np.zeros((X.shape[0],1),np.float64)
          mean=np.zeros(X.shape,np.float64)
          epsilon = 0.00001

	  
	  mean[0, :] = cv2.reduce(X, 0, cv2.REDUCE_AVG, cv2.CV_64F)
          #mean[0,:]=cv2.reduce(X, 0,cv2.cv.CV_REDUCE_AVG,cv2.CV_64F)
          
          mean[:,:]=np.tile(mean[0,:], (X.shape[0],1))

          X = X-mean # sustraccion(subtract) cada columna con su valor medio
          A = np.dot(X.T,X)
          U, D, V =svd(A)
             
	     # Raiz cuadrada de la matriz inversa de Dq
          for i in range(D.shape[0]):
                 D[i] = np.sqrt(1.0/(D[i]+epsilon))
             
          whMat = np.dot(U,np.diag(D))
          Xwh = np.dot(X,whMat)
          
          Xwh=Xwh*Xwh
          
	  Z=cv2.reduce(Xwh, 1, cv2.REDUCE_SUM,cv2.CV_64F)
          #Z=cv2.reduce(Xwh, 1,cv2.cv.CV_REDUCE_SUM,cv2.CV_64F)

          return Z #distancias cuadradas al valor medio de deblanqueado
    
          
     def LowPassFilter(self):
         #Construcion de filtro pasa baja en el dominio de la frecuecnia
          aperture = np.int(np.floor(np.min([self.orows/2.0,self.ocols/2.0]))-1)
          sigma = np.max([self.orows,self.ocols])*(0.021)

          iterCoEx=0.5
          DTar=np.int((self.orows-aperture)/2)
          DTe=np.int((self.ocols-aperture)/2)

          gExc =np.zeros((self.orows,self.ocols),np.float64)
          K1D = cv2.getGaussianKernel(aperture,sigma,cv2.CV_64F)
          
          gExcPeq = gExc[DTar:DTar+aperture, DTe:DTe+aperture] 
          gExcPeq[:,:]=iterCoEx*np.dot(K1D,K1D.T)

          fgExc=np.zeros((self.orows,self.ocols,2))
          fgExc[:,:,0]=gExc
          fgExc=cv2.dft(np.float64(fgExc))#,flags = cv2.DFT_COMPLEX_OUTPUT)

          return fgExc


     def Blurring(self,_in,GaussKernel):

          infft=np.zeros((_in.shape[0],_in.shape[1],2))
          infft[:,:,0]=_in
          infft = cv2.dft(np.float64(infft))#,flags = cv2.DFT_COMPLEX_OUTPUT)
          
          infft=cv2.mulSpectrums(infft, GaussKernel, 0)
          infft=cv2.idft(infft)

          _out = cv2.magnitude(infft[:,:,0],infft[:,:,1])
          _out=self.shiftDFT(_out)

          return _out


if __name__ == "__main__":

     #lectura de imagen a procesar
     if len(sys.argv)>1:
          img = cv2.imread(sys.argv[1],1);
          img_name = str(sys.argv[1]).split('/')[-1].split('.')[0]
     else:
          img = cv2.imread("./cat.jpg",1)
          img_name = 'cat'

     #archivo con filtros precalculados
     #filters='AWS_filters_image_size_226_226.dat'
        
     if img is None:
	     print "Can not open image file"
	     exit(1)

     start_t = time.time()

     #Creacion de objeto AWS con inicializacion de parametros y filtros
     #El calculo de filtros se puede evitar si fueron calculados previamente para una
     #imagen de identico tamanho. En caso contrario pasar el parametro con valor "None"
     aws=AWS(img,None)#filters)
     middle_t = time.time()#cv2.getTickCount()
     time_m = (middle_t - start_t)#/ cv2.getTickFrequency()
     print "Initialization time: {:.3f} seconds".format(time_m)

     #Calculo del mapa de saliencia
     Salmap=aws.salmap()   
 
     end_t = time.time()
     time_e = (end_t - middle_t)
     time_t = (end_t - start_t)
 
     print "Execution time: {:.3f} seconds".format(time_e)
     print "Total time: {:.3f} seconds".format(time_t)
     
     #pasar a formato imagen para visualizacion y almacenamiento
     cv2.normalize(Salmap,Salmap, 0, 255, cv2.NORM_MINMAX)
     A_scaled=Salmap.astype(np.uint8)
     cv2.imwrite('%s' % img_name+'_sal_aws.jpg', A_scaled.astype(int))


     A_scaled = cv2.applyColorMap(A_scaled, cv2.COLORMAP_JET)
     cv2.imwrite('%s' % img_name+'_sal_colormap_aws.jpg', A_scaled.astype(int))

     #cv2.imshow("Saliency",A_scaled)
     #cv2.waitKey(0);
