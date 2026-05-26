import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Datos completos
#v_p = np.array([0.187, 0.0914, 0.0645, 0.0492, 0.0394, 0.0319, 0.0271, 0.0174, 0.0147, 0.012, 0.00931, 0.189, 0.129, 0.108, 0.0211])   
#v_s = np.array([5.17, 5.12, 3.89, 3, 2.45, 2.05, 1.76, 1.13, 0.987, 0.798, 0.639, 5.06, 5.92, 5.65, 1.98])
#t = np.array([0.0158, 0.00364, 0.0021, 0.0015, 0.00116, 0.00094, 0.000788, 0.000492, 0.000406, 0.000346, 0.000276, 0.0165, 0.00784, 0.00536, 0.000604])
#f = np.array([21.1, 121.4, 218.8, 319.8, 420.2, 521.3, 628.3, 1019, 1224, 1506, 1913, 20, 49.95, 80.65, 819.7])
#mu = 4 * np.pi * 1e-7
#es =  0.00156
#des= 0.00001
#dv_p = np.array([0.02661, 0.013742, 0.007935, 0.02076, 0.007182, 0.003957, 0.003813, 0.002522, 0.002441, 0.002236, 0.0022793, 0.02667, 0.01487, 0.01424, 0.003633]) #Vienen del manual, calculados en el excel
#dv_s = np.array([0.6561, 0.6546, 0.6177, 0.291, 0.2745, 0.2625, 0.2538, 0.0849, 0.08061, 0.07494, 0.07017, 0.6528, 0.6786, 0.6705, 0.1604])
#df = np.array([1.055, 6.07, 10.94, 15.99, 21.01, 26.065, 31.415, 50.95, 61.2, 75.3, 95.65, 1, 2.4975, 4.0325, 40.985]) #5% de la medida 
#dt= np.array([0.005, 0.001, 0.0005, 0.0005, 0.00025, 0.00025, 0.0001, 0.0001, 0.00005, 0.00005, 0.00005, 0.0025, 0.001, 0.001, 0.0001]) #Escala sobre 5


#Datos sin los puntos de 20 
#v_p = np.array([0.0914, 0.0645, 0.0492, 0.0394, 0.0319, 0.0271, 0.0174, 0.0147, 0.012, 0.00931, 0.108, 0.0211, 0.129])
#v_s = np.array([5.12, 3.89, 3, 2.45, 2.05, 1.76, 1.13, 0.987, 0.798, 0.639, 5.65, 1.98, 5.90, 5.92])
#t = np.array([0.00364, 0.0021, 0.0015, 0.00116, 0.00094, 0.000788, 0.000492, 0.000406, 0.000346, 0.000276, 0.00536, 0.000604, 0.00784])
#f = np.array([121.4, 218.8, 319.8, 420.2, 521.3, 628.3, 1019, 1224, 1506, 1913, 80.65, 819.7, 49.95])
#mu = 4 * np.pi * 1e-7
#es =  0.00156
#dv_p = np.array([0.013742, 0.007935, 0.02076, 0.007182, 0.003957, 0.003813, 0.002522, 0.002441, 0.002236, 0.0022793, 0.01424, 0.003633, 0.01487])
#dv_s = np.array([0.6546, 0.6177, 0.291, 0.2745, 0.2625, 0.2538, 0.0849, 0.08061, 0.07494, 0.07017, 0.6705, 0.1604, 0.6786])
#df = np.array([6.07, 10.94, 15.99, 21.01, 26.065, 31.415, 50.95, 61.2, 75.3, 95.65, 4.0325, 40.985, 2.4975])
#dt= np.array([0.001, 0.0005, 0.0005, 0.00025, 0.00025, 0.0001, 0.0001, 0.00005, 0.00005, 0.00005, 0.001, 0.0001, 0.001])
#dt/=5


#Ajuste exp del módulo, todo libre, todos los puntos

v_p = np.array([0.187, 0.0914, 0.0645, 0.0492, 0.0394, 0.0319, 0.0271, 0.0174, 0.0147, 0.012, 0.00931, 0.189, 0.129, 0.108, 0.0211])
v_s = np.array([5.17, 5.12, 3.89, 3, 2.45, 2.05, 1.76, 1.13, 0.987, 0.798, 0.639, 5.06, 5.92, 5.65, 1.98])
t = np.array([0.0158, 0.00364, 0.0021, 0.0015, 0.00116, 0.00094, 0.000788, 0.000492, 0.000406, 0.000346, 0.000276, 0.0165, 0.00784, 0.00536, 0.000604])
f = np.array([21.1, 121.4, 218.8, 319.8, 420.2, 521.3, 628.3, 1019, 1224, 1506, 1913, 20, 49.95, 80.65, 819.7])
dv_p = np.array([0.02661, 0.013742, 0.007935, 0.02076, 0.007182, 0.003957, 0.003813, 0.002522, 0.002441, 0.002236, 0.0022793, 0.02667, 0.01487, 0.01424, 0.003633])
dv_s = np.array([0.6561, 0.6546, 0.6177, 0.291, 0.2745, 0.2625, 0.2538, 0.0849, 0.08061, 0.07494, 0.07017, 0.6528, 0.6786, 0.6705, 0.1604])
df = np.array([1.055, 6.07, 10.94, 15.99, 21.01, 26.065, 31.415, 50.95, 61.2, 75.3, 95.65, 1, 2.4975, 4.0325, 40.985])
dt= np.array([0.005, 0.001, 0.0005, 0.0005, 0.00025, 0.00025, 0.0001, 0.0001, 0.00005, 0.00005, 0.00005, 0.0025, 0.001, 0.001, 0.0001])
dt/=5
x = np.sqrt(f*2*np.pi)
y = v_s/(v_p*2*np.pi*f)
mu = 4 * np.pi * 1e-7
es =  0.00156
des= 0.00001

dx = np.sqrt((np.pi/2)*((df**2)/f))
dy = (1/(2*np.pi*v_p*f))*np.sqrt((dv_s)**2+((v_s*dv_p)/v_p)**2+((df*v_s)/f)**2)

#Ajuste

def expo(x, A, C):
    return A*np.exp(-C*x)
popt, pcov = curve_fit(expo, x, y, sigma=dy, absolute_sigma=True, bounds=([0,0],[np.inf,np.inf]))

A_fit, C_fit = popt
dA, dC = np.sqrt(np.diag(pcov))

x_fit = np.linspace(min(x), max(x), 500)
y_fit = expo(x_fit, *popt)

# Chi² reducido

y_ajustado = expo(x, A_fit, C_fit)

chi2 = np.sum(((y - y_ajustado)/dy)**2)

gl = len(x) - 2   # 2 parámetros: A y C

chi2_red = chi2 / gl

print(f"Chi-cuadrado reducido = {chi2_red:.4f}")

# grafico
plt.figure(figsize=(8,6))

plt.errorbar(x, y, xerr=dx, yerr=dy,
             fmt='o', color='blue',
             ecolor='gray', capsize=3,
             label='Datos experimentales')

plt.plot(x_fit, y_fit, 'r-',
         label=fr'Ajuste: ${A_fit:.2e}e^{{-{C_fit:.2e}x}}$')
print(f"A = {A_fit:.4e} ± {dA:.4e}")
print(f"C = {C_fit:.4e} ± {dC:.4e}")
plt.xlabel(r'$\sqrt{\omega}$', fontsize=14)
plt.ylabel(r'$\frac{|H|}{\omega}$', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
print(f"Chi-cuadrado reducido = {chi2_red:.4f}")
plt.tight_layout()
plt.show()

sigma = ((C_fit/es)**2)*(2/mu)
ds= 4*np.sqrt((((C_fit/es**2)*dC)**2)/mu**2 +((((C_fit**2)/es**3)*des)**2)/mu**2)

print(f"sigma = {sigma:.4e} ± {ds:.4e}")

#Borro un los puntos de 20 
v_p = np.array([0.0914, 0.0645, 0.0492, 0.0394, 0.0319, 0.0271, 0.0174, 0.0147, 0.012, 0.00931, 0.108, 0.0211, 0.129])
v_s = np.array([5.12, 3.89, 3, 2.45, 2.05, 1.76, 1.13, 0.987, 0.798, 0.639, 5.65, 1.98, 5.90])
t = np.array([0.00364, 0.0021, 0.0015, 0.00116, 0.00094, 0.000788, 0.000492, 0.000406, 0.000346, 0.000276, 0.00536, 0.000604, 0.00784])
f = np.array([121.4, 218.8, 319.8, 420.2, 521.3, 628.3, 1019, 1224, 1506, 1913, 80.65, 819.7, 49.95])
mu = 4 * np.pi * 1e-7
es =  0.00156
dv_p = np.array([0.013742, 0.007935, 0.02076, 0.007182, 0.003957, 0.003813, 0.002522, 0.002441, 0.002236, 0.0022793, 0.01424, 0.003633, 0.01487])
dv_s = np.array([0.6546, 0.6177, 0.291, 0.2745, 0.2625, 0.2538, 0.0849, 0.08061, 0.07494, 0.07017, 0.6705, 0.1604, 0.6786])
df = np.array([6.07, 10.94, 15.99, 21.01, 26.065, 31.415, 50.95, 61.2, 75.3, 95.65, 4.0325, 40.985, 2.4975])
dt= np.array([0.001, 0.0005, 0.0005, 0.00025, 0.00025, 0.0001, 0.0001, 0.00005, 0.00005, 0.00005, 0.001, 0.0001, 0.001])
dt/=5
x = np.sqrt(f*2*np.pi)
y = v_s/(v_p*2*np.pi*f)

dx = np.sqrt((np.pi/2)*((df**2)/f))
dy = (1/(2*np.pi*v_p*f))*np.sqrt((dv_s)**2+((v_s*dv_p)/v_p)**2+((df*v_s)/f)**2)

#Ajuste

def expo(x, A, B):
    return A*np.exp(-B*x)
popt, pcov = curve_fit(expo, x, y, sigma=dy, absolute_sigma=True, bounds=([0,0],[np.inf,np.inf]))

A_fit, B_fit = popt
dA, dB = np.sqrt(np.diag(pcov))

x_fit = np.linspace(min(x), max(x), 500)
y_fit = expo(x_fit, *popt)

# Chi² reducido

y_ajustado = expo(x, A_fit, B_fit)

chi2 = np.sum(((y - y_ajustado)/dy)**2)

gl = len(x) - 2   # 2 parámetros: A y B

chi2_red = chi2 / gl

print(f"Chi-cuadrado reducido = {chi2_red:.4f}")

# grafico
plt.figure(figsize=(8,6))

plt.errorbar(x, y, xerr=dx, yerr=dy,
             fmt='o', color='blue',
             ecolor='gray', capsize=3,
             label='Datos experimentales')

plt.plot(x_fit, y_fit, 'r-',
         label=fr'Ajuste: ${A_fit:.2e}e^{{-{B_fit:.2e}x}}$')
print(f"A = {A_fit:.4e} ± {dA:.4e}")
print(f"B = {B_fit:.4e} ± {dB:.4e}")
plt.xlabel(r'$\sqrt{\omega}$', fontsize=14)
plt.ylabel(r'$\frac{|H|}{\omega}$', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
print(f"Chi-cuadrado reducido = {chi2_red:.4f}")
plt.tight_layout()
plt.show()

sigma = ((B_fit/es)**2)*(2/mu)
ds= 4*np.sqrt((((B_fit/es**2)*dB)**2)/mu**2 +((((B_fit**2)/es**3)*des)**2)/mu**2)

print(f"sigma = {sigma:.4e} ± {ds:.4e}")

#Puntos sin los de 20. Muchos ajustes

n = (64*np.pi)/415035
B_fijo = 0.00759251
A_fijo = (64*np.pi)/415035
u = 0.00000123656

#Ajustes

# 1) A,B libres
def expo1(x, A, B):
    return A*np.exp(-B*x)

popt1, pcov1 = curve_fit(
    expo1, x, y,
    sigma=dy,
    absolute_sigma=True,
    bounds=([0,0],[np.inf,np.inf])
)
A1, B1 = popt1
dA1, dB1 = np.sqrt(np.diag(pcov1))

y1 = expo1(x, A1, B1)

chi2_1 = np.sum(((y-y1)/dy)**2)
gl1 = len(x)-2
chi2red1 = chi2_1/gl1

# 2) A=C*n (C libre), N fijo, B=D*u, u fijo D seria raiz de sigma

def expo2(x, C, D):
    A = C*n
    B = D*u

    return A*np.exp(-B*x)

popt2, pcov2 = curve_fit(
    expo2,
    x, y,
    sigma=dy,
    absolute_sigma=True,
     bounds=([0,0],[np.inf,np.inf]))

C2, D2 = popt2
dC2, dD2 = np.sqrt(np.diag(pcov2))

A2 = C2*n
dA2 = n*dC2

B2 = D2*u
dB2 = u*dD2

y2 = expo2(x, C2, D2)

chi2_2 = np.sum(((y-y2)/dy)**2)
gl2 = len(x)-2
chi2red2 = chi2_2/gl2

W= (D2)**2

# 3) A fijo, B libre

def expo3(x, B):
    return A_fijo*np.exp(-B*x)

popt3, pcov3 = curve_fit(
    expo3,
    x, y,
    sigma=dy,
    absolute_sigma=True,
    bounds=([0],[np.inf]))

B3 = popt3[0]
dB3 = np.sqrt(np.diag(pcov3))[0]

y3 = expo3(x, B3)

chi2_3 = np.sum(((y-y3)/dy)**2)
gl3 = len(x)-1
chi2red3 = chi2_3/gl3

# 4) A fijo, B fijo
def expo4(x):
    return A_fijo*np.exp(-B_fijo*x)

y4 = expo4(x)

chi2_4 = np.sum(((y-y4)/dy)**2)
gl4 = len(x)
chi2red4 = chi2_4/gl4

# Grafico
x_fit = np.linspace(min(x), max(x), 500)

y_fit1 = expo1(x_fit, A1, B1)
y_fit2 = expo2(x_fit, C2, D2)
y_fit3 = expo3(x_fit, B3)
y_fit4 = expo4(x_fit)

print("AJUSTE 1: A,B libres")
print(f"A = {A1:.4e} ± {dA1:.4e}")
print(f"B = {B1:.4e} ± {dB1:.4e}")
print(f"χ² reducido = {chi2red1:.4f}")

print("AJUSTE 2: A=C*n, B=D*u")
print(f"C = {C2:.4e} ± {dC2:.4e}")
print(f"D = {D2:.4e} ± {dD2:.4e}")
print(f"A = {A2:.4e} ± {dA2:.4e}")
print(f"B = {B2:.4e} ± {dB2:.4e}")
print(f"sigma= {W:e}" )
print(f"χ² reducido = {chi2red2:.4f}")

print("AJUSTE 3: A fijo, B libre")
print(f"A fijo = {A_fijo:.4e}")
print(f"B = {B3:.4e} ± {dB3:.4e}")
print(f"χ² reducido = {chi2red3:.4f}")

print("AJUSTE 4: A fijo, B fijo")
print(f"A fijo = {A_fijo:.4e}")
print(f"B fijo = {B_fijo:.4e}")
print(f"χ² reducido = {chi2red4:.4f}")

plt.figure(figsize=(8,6))

plt.errorbar(x, y, xerr=dx, yerr=dy, fmt='o', color='blue', ecolor='gray', capsize=3, label='Datos experimentales')

plt.plot(x_fit, y_fit1, color='red', linestyle='-', linewidth=2, label='1) A,B libres')
plt.plot(x_fit, y_fit2, color='yellow', linestyle='--', linewidth=3, label='2) A=C·n, B=D*u')
plt.plot(x_fit, y_fit3, color='black', linestyle=':', linewidth=4, label='3) A fijo, B libre')
plt.plot(x_fit, y_fit4, color='green', linestyle='-.', linewidth=2, label='4) A fijo, B fijo')

plt.xlabel(r'$\sqrt{\omega}$', fontsize=14)
plt.ylabel(r'$\frac{|H|}{\omega}$', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

#Sigmas
def calc_sigma(B, dB, es, des, mu):

    sigma = ((B/es)**2)*(2/mu)

    ds = 4*np.sqrt((((B/es**2)*dB)**2)/(mu**2)+((((B**2)/(es**3))*des)**2)/(mu**2))

    return sigma, ds


#1)
sigma1, ds1 = calc_sigma(B1, dB1, es, des, mu)

#2)
sigma2, ds2 = calc_sigma(B2, dB2, es, des, mu)

#3)
sigma3, ds3 = calc_sigma(B3, dB3, es, des, mu)

#4)
dB_fijo = 0
sigma4, ds4 = calc_sigma(B_fijo, dB_fijo, es, des, mu)

print("AJUSTE 1: A,B libres")
print(f"sigma = {sigma1:.4e} ± {ds1:.4e}")

print("AJUSTE 2: A=C*N, B=D*u")
print(f"sigma = {sigma2:.4e} ± {ds2:.4e}")

print("AJUSTE 3: A fijo, B libre")
print(f"sigma = {sigma3:.4e} ± {ds3:.4e}")

print("AJUSTE 4: A fijo, B fijo")
print(f"sigma = {sigma4:.4e} ± {ds4:.4e}")