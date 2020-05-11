# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:18:50 2020

@author: Dani Astor
"""
import numpy as np        
import matplotlib          
import matplotlib.pyplot as plt    
import seaborn as sns     
matplotlib.rcParams.update({'font.size': 15})        

N  = 15000 #Particles number
L  = 10    #Debye lengths
vt = 1.0   #Thermal velocity
vb = 3.0   #Average velocity   
ng = 3750  #Grid points umber     
dx = L/ng  #Space step 
dt = 0.05  #Time step
nt = 250   #Iterations number
Q = -L/N   #Super particle charge
k = np.fft.fftfreq(ng, dx) #Frequencies
k[0] = 1.0                 #Fix frequencies

x = np.arange(N)*(L/N)         
v = vt*np.random.randn(N)
pm = 2*np.random.randint(0,2,N) - 1
v = v+pm*vb 

for t in range(nt):
    
    k1 = v*dt                   
    k2 = (v+(1/2)*k1)*dt   
    k3 = (v+(1/2)*k2)*dt
    k4 = (v+k3)*dt                    
    x = x + (1/6)*(k1+2*k2+2*k3+k4)
    x = np.mod(x, L)

    P = x/dx
    PG = np.floor(P)
    w = 1.0 + PG - P
    G = np.int32(np.mod(np.concatenate([PG,PG+2.5]),ng))
    WW = np.concatenate([w,2.5-w])

    rho = np.zeros(ng)
    for G1,W1 in zip(G,WW):
        rho[G1] = rho[G1]+W1 
    rho = Q/dx*(rho)
    
    rhok = np.fft.fft(rho)     #Density, frequency domain
    phi= np.fft.ifft((rhok/k**2)).real
    phi1 = phi[G]
    phi2= (phi1[:N]+phi1[N:])  #Electric Potential
    ee  = np.fft.ifft(-1j*rhok/k).real                                  
    ep = ee[G]                      
    E = (ep[:N] + ep[N:])      #Electric field
    
    l1 = (-E)*dt
    l2 = (-E+(1/2)*l1)*dt   
    l3 = (-E+(1/2)*l2)*dt
    l4 = (-E+l3)*dt                    
    v = v + (1/6)*(l1+2*l2+2*l3+l4)

    fig=plt.figure(figsize=(20, 12), dpi=80)
    plt.subplot(2, 2, 1)
    plt.plot(x,v,".",alpha=0.4)
    plt.ylim(-8,8)
    plt.xlabel("x [\u03BBD]")
    plt.ylabel("v [\u03BBD \u03C9p]")

    plt.subplot(2, 2, 3)
    plt.plot(x,phi2,"k.")
    plt.ylim(-70,80)
    plt.xlabel("x [\u03BBD]")
    plt.ylabel("Electric Potencial \u03A6 (x) [kTe/e]")

    plt.subplot(2, 2, 2)
    plt.plot(x,E,"k.")
    plt.ylim(-20,20)
    plt.xlabel("x [\u03BBD]")
    plt.ylabel("E(x) [kTe/e\u03BBD]")                
    
    m=[i for i,x1 in enumerate(x) if (x1>=4.5) and (x1<=5.5)]
    a0=v[m]        
    plt.hist(a0,100,orientation="horizontal", histtype='barstacked',normed=True,visible=False)
    plt.subplot(2, 2, 4)
    valores0 = sns.kdeplot(a0,vertical=True,color="k",visible=True)
    c0,b0 = valores0.get_lines()[0].get_data()
    plt.xlim(0,0.2)
    plt.ylim(-10,10)                                                 
    plt.xlabel("Distribution f($v_{x}$), around x=5")
    plt.ylabel("Velocity v") 
    fig.savefig("{}.png".format(t+1))
    plt.close()

plt.figure(figsize=(10, 6), dpi=80)
et = (1/2)*v**2-phi2
D  = 700   
res = et[1::D]
print("\u03A6(x) minimum"+str(min(phi2)))
print("\u03A6(x) maximum"+str(max(phi2)))
print("energias: "+str(res))
v1 = np.zeros(len(phi2))
v2 = np.zeros(len(phi2))
for l in range(len(res)):
    B = res[l]
    for o in range(len(phi2)):
        v1[o]=np.sqrt(B+phi2[o])*2
        v2[o]=-np.sqrt(B+phi2[o])*2 
    plt.plot(x,v1,"k.")
    plt.plot(x,v2,"k.")
    plt.title("Energy contours")
    plt.xlabel("x [\u03BBD]")
    plt.ylabel("v [\u03BBD \u03C9p]")
plt.show()
