#%% G. A. Basso
import numpy as np
import matplotlib.pyplot as plt
#%% 
data_21 = np.loadtxt( r".\Experimento IV, Ferromagnetico\Codigo\Datos\histeresis_21C_100ks.txt",skiprows=4)
data_141 = np.loadtxt( r".\Experimento IV, Ferromagnetico\Codigo\Datos\histeresis_141C_100ks.txt",skiprows=4)

# esto se agrega a mano, con info del nombre de archivo o de su metadata
sampling=100e3 # Ss: Samples/second == 1/s
dt=1/sampling # s
T_21,T_141=21,141 #C
frec_21 = 700.040 # Hz
periodo_21 = 1/frec_21 #s 
frec_141 = 699.998 #Hz
periodo_141=1/frec_141

#defino arrays y doy escala temporal
time  = data_21[:,0]*dt #s
v_ch1 = data_21[:,1]    #mV
v_ch2 = data_21[:,2]    #mV 

w_ch1 = data_141[:,1]    #mV
w_ch2 = data_141[:,2]    #mV 

#%% Ploteo senales 
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(8,4),sharex=True,constrained_layout=True)
ax1.plot(time,v_ch1,'.-')
ax2.plot(time,v_ch2,'.-')

ax1.set_title('CH1',loc='left')
ax2.set_title('CH2',loc='left')
ax2.set_xlabel('t (s)')
for a in [ax1,ax2]:
    a.grid()
    a.set_ylabel('mV')   
    a.axvspan(0,periodo_21,0,1,color='C2',alpha=0.5,label=f'periodo = {periodo_21:.2e} s')
    a.legend()

#%%
N_21 = round(sampling / frec_21)
N_141 = round(sampling / frec_141)

print(N_21)
print(N_141)
fig2,ax=plt.subplots(figsize=(6.66,5),constrained_layout=True)
ax.plot(v_ch1[:N_21],v_ch2[:N_21],'.-',label=f'{T_21} °C')
ax.plot(w_ch1[:N_141],w_ch2[:N_141],'.-',label=f'{T_141} °C')

ax.legend(title='Temperatura')

ax.set_xlabel('H (u.a.)')
ax.set_ylabel('M (u.a.)')
ax.grid()
plt.show()
# plt.savefig('ciclos_HM_21_141.png',dpi=300)
# %%
