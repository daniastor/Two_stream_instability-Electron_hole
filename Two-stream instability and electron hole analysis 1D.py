# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:34:20 2019

@author: Dani Astor & Jaime Hoyos
"""

#----------------------------Librerías-----------------------------------------
import numpy as np                
import matplotlib                  
import matplotlib.pyplot as plt               
import seaborn as sns              
from mpl_toolkits import mplot3d   

matplotlib.rcParams.update({'font.size': 15})
#------------------------------------------------------------------------------
#---------------------------Parámetros-----------------------------------------

N  = 15000     # Número de partículas

L  = 10        # Longitud del espacio a analizar

vt = 1.0       # Velocidad de propagación térmica (desviación)
vb = 3.0       # Velocidad de los haces "drift"   

ng = 10000     # Número de puntos en la grilla
dx = L/ng      # Paso espacial

dt = 0.05      # Paso temporal
nt = 250       # Tiempo (Número de iteraciones)

xp = 0.0       # Amplitud de perturbación
mode = 1       # Modo de perturbación

Q = -L/N       #Carga

k = np.fft.fftfreq(ng, dx)    # Número de onda
k[0] = 1.0                    # "fix frequencies" con k=0 
#------------------------------------------------------------------------------
#---------------------------Condiciones iniciales------------------------------

x = np.arange(N)*(L/N)       # Posición inicial de las partículas

  # Distribución "normal estándar" para velocidades iniciales +vb y -vb
v = vt*np.random.randn(N)     
pm = 2*np.random.randint(0,2,N) - 1
v = v+pm*vb  
#------------------------------------------------------------------------------
#-----------Oscilación del plasma o ondas de Langmuir--------------------------
 #Ajustar parámetros 
 #vt = 0.0 (se le pude agregar un poco), vb = 0.0 , xp diferente de cero
x = x + xp*np.sin(2*np.pi*mode*x/L)    #perturbación sinusoidal
#------------------------------------------------------------------------------
#-------------Evolución temporal según el modelo matemático--------------------
for i in range(nt):             #Iteraciones
    # Resuelvo por RK4: dx/dt = v
    k1 = v*dt                   
    k2 = (v+(1/2)*k1)*dt   
    k3 = (v+(1/2)*k2)*dt
    k4 = (v+k3)*dt                    
    x = x + (1/6)*(k1+2*k2+2*k3+k4) # Obtengo x

    x = np.mod(x, L)                # Condición periódica x(0) = x(L) = 0
    
    #Método de aproximación PIC
      #Calcular contribución en las celdas (Peso)
                  
    f = x/dx                
    j = np.floor(f) 
    h = j + 1.0 - f
    j = np.int32(np.mod(np.concatenate([j,j+1]), ng)) 
    h = np.concatenate([h, 1.0-h])
    rho = np.zeros(ng) 
    for jj,hh in zip(j,h):  #Función
        rho[jj] = rho[jj] + hh
    rho = Q/dx*(rho)        #Densidad    }
    #Transformada de Fourier para la densidad rho(x)->rho(k)
    rhok = np.fft.fft(rho)
    
    #Potencial eléctrico
    phi= np.fft.ifft((rhok/k**2)).real
    phi1 = phi[j]*h               
    phi2= (phi1[:N]+phi1[N:])         #Obtengo potencial
    
    #Campo eléctrico 
    ee  = np.fft.ifft(-1j*rhok/k).real 
    ep = ee[j]*h
    E = (ep[:N] + ep[N:])             #Obtengo campo eléctrico    
    # Resuelvo por RK4: dv/dt = -E
    l1 = (-E)*dt
    l2 = (-E+(1/2)*l1)*dt   
    l3 = (-E+(1/2)*l2)*dt
    l4 = (-E+l3)*dt                    
    v = v + (1/6)*(l1+2*l2+2*l3+l4)   #Obtengo v
##-----------------------------------------------------------------------------
##----------------------#Gráfica de cada iteración#----------------------------
###-------------#Oscilación del plasma o ondas de Langmuir
###    plt.figure(figsize=(10, 6), dpi=80)
###    plt.plot(x,"o",alpha=0.3)
###    plt.xlabel("t [1/\u03C9p]")
###    plt.ylabel("v [\u03BBD \u03C9p]")
###    plt.show()

##----------------#Inestabilidad de dos flujos           
#    fig=plt.figure(figsize=(20, 12), dpi=80)
#    plt.subplot(2, 2, 1)
#    plt.plot(x,v,".",alpha=0.4)
#    plt.xlabel("x [\u03BBD]")
#    plt.ylabel("v [\u03BBD \u03C9p]")
#
####----------------#Potencial eléctrico asociado
#    plt.subplot(2, 2, 2)
#    plt.plot(x,phi2,"k.")
#    plt.xlabel("x [\u03BBD]")
#    plt.ylabel("Electric Potencial \u03A6 (x) [kTe/e]")
##
####----------------#Campo eléctrico asociado
#    plt.subplot(2, 2, 3)
#    plt.plot(x,E,"k.")
#    plt.xlabel("x [\u03BBD]")
#    plt.ylabel("E(x) [kTe/e\u03BBD]")                
##
#####----------------#Función de distribución para x=5
#    m=[i for i,x1 in enumerate(x) if (x1>=4.5) and (x1<=5.5)]
#    a0=v[m]        
#    plt.hist(a0,100,orientation="horizontal", histtype='barstacked', 
#             normed=True,visible=False)
#    plt.subplot(2, 2, 4)
#    valores0 = sns.kdeplot(a0,vertical=True,color="k",visible=True)
#    c0,b0 = valores0.get_lines()[0].get_data()                                                 
#    plt.xlabel("Distribution f(x=5,v)")
#    plt.ylabel("Velocity v") 
#    fig.savefig("{0}.png".format(i)) #Guarda las imágenes en la carpeta donde 
#                                      #se encuentra el programa
#    plt.close()
    
#------------------------------------------------------------------------------
                        #---------------#
#------------------------Iteración final---------------------------------------
                        #---------------#
#----------------#Oscilación del plasma o ondas de Langmuir
#print (" \n  Resultado: iteración final") 
#plt.figure(figsize=(10, 6), dpi=80)
#plt.plot(x,"o",alpha=0.3)
#plt.xlabel("t [1/\u03C9p]")
#plt.ylabel("v [\u03BBD \u03C9p]")
#plt.title("Ondas de Langmuir")
#plt.show() 

#----------------#Inestabilidad de dos flujos           
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(x,v,".",alpha=0.4)
plt.xlabel("x [\u03BBD]")
plt.ylabel("v [\u03BBD \u03C9p]")
plt.title("Two-stream instability")
plt.show()

#----------------#Potencial eléctrico asociado
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(x,phi2,"k.")
plt.xlabel("x [\u03BBD]")
plt.ylabel("Electric Potential \u03A6 (x) [kTe/e]")
plt.title("Electric Potential")
plt.show()

#----------------#Campo eléctrico asociado
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(x,E,"k.")
plt.xlabel("x [\u03BBD]")
plt.ylabel("E(x) [kTe/e\u03BBD]")
plt.title("Electric Field")
plt.show()  
##----------------#Función de distribución de todo el vórtice
    #x=0
m=[i for i,x1 in enumerate(x) if (x1>=-0.5) and (x1<=0.5)]
a0=v[m]
plt.figure(figsize=(10, 6), dpi=80)
plt.hist(a0,100,orientation="horizontal", histtype='barstacked',
         normed=True,visible=True)    #Histograma
valores0 = sns.kdeplot(a0,vertical=True,color="k",visible=True)
plt.xlabel("Función de distribución f(v,x=0)")
plt.ylabel("v [\u03BBD \u03C9p]")
c0,b0 = valores0.get_lines()[0].get_data() #datos de la linea de ajuste
plt.plot(c0,b0)                                                  
plt.title("Función de distribución x=0")
plt.xlabel("Distribution f(x=0,v)")
plt.ylabel("Velocity  v") 
plt.show()
   # para x = 1, 2, 3, 4, 5, 6, 7,...,n             
for j1 in range(0,L):
    plt.figure(figsize=(10, 6), dpi=80)
    m=[i for i,x1 in enumerate(x) if (x1>j1+0.5) and (x1<=j1+1.5)]
    a=v[m]   
    plt.hist(a,100,orientation="horizontal", histtype='barstacked',
             normed=True,visible=False)                       
    valores = sns.kdeplot(a,vertical=True,color="k",visible=True)
    plt.title("Distribution Function x="+str(j1+1))
    plt.xlabel("Distribution f(x="+str(j1+1)+",v)")
    plt.ylabel("Velocity v")
    plt.show()
    c,b = valores.get_lines()[0].get_data()  #Data de los gráficos
    #Plot 3d  de todas das Distribuciones, solamente analiza desde x=0 a x=10
    if (j1==0):
        c1=c
        b1=b
    else:
        if(j1==1):
            c2=c
            b2=b
        else:
            if(j1==2):
                c3=c
                b3=b
            else:
                if(j1==3):
                    c4=c
                    b4=b
                else:
                    if(j1==4):
                        c5=c
                        b5=b
                    else:
                        if(j1==5):
                            c6=c
                            b6=b
                        else:
                            if(j1==6):
                                c7=c
                                b7=b
                            else:
                                if(j1==7):
                                    c8=c
                                    b8=b
                                else:
                                    if(j1==8):
                                        c9=c
                                        b9=b
                                    else:
                                        if(j1==9):
                                            c10=c
                                            b10=b                                    
fig = plt.figure()
ax  = plt.axes(projection="3d")
z1= [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10]
z2= [b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10]
z3= [0,1,2,3,4,5,6,7,8,9,10]
ax.plot3D(z1,z2,z3,"k",alpha=0.5)
ax.set_xlabel("f(v)")
ax.set_ylabel("v")
ax.set_zlabel("x")
plt.show() 
                        #---------------------#
                             #----Fin----#    




