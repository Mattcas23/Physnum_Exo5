import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pdb
import os

# Parameters
# TODO adapt to what you need (folder path executable input filename)
executable = r"/Users/a-x-3/Desktop/Exercice5_2025_student/exe"  # Name of the executable (NB: .exe extension is required on Windows)
repertoire = r"/Users/a-x-3/Desktop/Exercice5_2025_student"
os.chdir(repertoire)

input_filename = 'input_example_student'  # Name of the input file

# ------------------------------------- Variables fichier -------------------------------------- # 


#VarFichier = np.genfromtxt("input_example_student" , comments = '!')


# ------------------------------------- Simulations ----------------------------------- #

#nsteps = np.array([1,2,3,4,5,6,7,8,9])*200
#nx = np.array([1,2,3,4,5,6,7,8,9])*20

#nsteps = np.array([50e3])
#nx = np.array([10000])

nsteps = np.array([2000])
nx = np.array([64])

paramstr = 'nsteps'  # Parameter name to scan
param = nsteps  # Parameter values to scan

paramstr2 = 'nx'  # Parameter name to scan
param2 = nx  # Parameter values to scan

nsimul = len(nsteps)

# Simulations
outputs = []  # List to store output file names
convergence_list = []
for i in range(nsimul):
    output_file = f"{paramstr}={param[i]}_{paramstr2}={param2[i]}.out"
    outputs.append(output_file)
    cmd = f"{repertoire}{executable} {input_filename} {paramstr}={param[i]:.15g} output={output_file}"
    cmd = f"{executable} {input_filename} {paramstr}={param[i]:.15g} {paramstr2}={param2[i]:.15g} output={output_file}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    print('Done.')


for i in range(nsimul):  # Iterate through the results of all simulations
    data = np.loadtxt(outputs[i]+'_x')  # Load the output file of the i-th simulation
    #xx = data[-1] # position finale 
    #convergence_list.append(xx)

lw = 1.5
fs = 16

x = np.loadtxt(outputs[-1]+'_x') # position en x 
f = np.loadtxt(outputs[-1]+'_f') # 
v = np.loadtxt(outputs[-1]+'_v') 
E = np.loadtxt(outputs[-1]+'_en') # temps et énergie

#print(E)

print(x.shape)
print(f.shape)

def f_analytique (T ,  x , L = 15.0 ) : # changer les valeurs , on a que T = tfin 

    n = 2
    pi = 3.1415926535897932384626433832795028841971
    kn = ( n + 0.5 ) * pi / L 
    om = 2 * pi / T # calcul de oméga à partir de la période
    
    return np.sin( om * T ) * np.cos( kn*x ) 

def Erreur (x , f , tfin) :

    dx = abs(x[1] - x[0]) # intégrale de Riemann
    err = ( f[-1,1:] - f_analytique(tfin,x) ).sum() * dx # somme de Riemann
    # f[-1,1:] : comme la simulation d'arrête a tfin = T, on prend la dernière valeur 
    return err

def Convergence ( T = 1.91565 , norder = 1 ) : # simulation réalisée avec n = 2 , T période pour n = 2 # T = 2.39457

    err = np.array([]) 
    for output in outputs :

        xi = np.loadtxt(output+'_x')
        fi = np.loadtxt(output+'_f')
        err = np.append(err,Erreur( tfin = T , x = xi, f = fi ))
    
    plt.plot(pow(T/nsteps,norder),err,'k+-') # pow(T/nsteps,norder)
    #plt.title("$\\Beta_{CFL} = $") 
    plt.xlabel(f"$(\\Delta t)^{norder}$", fontsize = fs)
    plt.ylabel("$\\delta_{err}$", fontsize = fs)
    



def ftPlot() : 

    t = f[:,0]
    #print(t)

    plt.ion() # pour faire l'animation visuelle

    #plt.plot(x,f[5,1:])

    for i in range(f.shape[0]) :
    
        f_a_t = f[i,1:] # f au temps t 
        plt.scatter(x,f_a_t)
        plt.title(f"t = {t[i]}")
        plt.draw()
        plt.pause(0.02)
        plt.close()
        
    plt.ioff() # pour arrêter

def PlotCouleur() :

    t = f[:,0]
    fval = f[:,1:]

    plt.figure()
    plt.pcolor(x,t,fval)
    plt.xlabel("x [m]", fontsize = fs)
    plt.ylabel("t [s]", fontsize = fs)
    plt.colorbar()
    
def Eplot () :

    En = E[:,1]
    t  = E[:,0]
    print(max(En))
    
    plt.figure()
    plt.plot(t,En, color = 'black')
    plt.xlabel('Temps [s]'   , fontsize = fs)
    plt.ylabel('Energie [J]' , fontsize = fs)

def Ewplot () : # plot max(E) en fonction de omega , simulations réalisées avec n = 2 

    # Simulations
    outputs2 = []  # List to store output file names

    wn = (2.0+0.5) * np.pi * np.sqrt(4.0*9.81) / 15.0 
    print(f"wn = {wn} (pour n = 2, c-à-d deuxième mode propre)")
    Omeg = np.array([20,22,24,25,25.5,26,26.24,26.4,26.5,27,28,29,30])*0.1
    #Omeg = np.array([2.624,3.2,wn,3.3,4])

    paramstr3 = 'om'  # Parameter name to scan
    param3 = Omeg  # Parameter values to scan
    nsimul = len(Omeg)

    for i in range(nsimul):
        output_file = f"{paramstr3}={param3[i]}.out"
        outputs2.append(output_file)
        cmd = f"{executable} {input_filename} {paramstr3}={param3[i]:.15g} output={output_file}"
        print(cmd)
        subprocess.run(cmd, shell=True)
        print('Done.')

    Emax = np.array([])
    
    for i in range(nsimul) :

        Ei = np.loadtxt(outputs2[i]+'_en')
        #plt.figure()
        #plt.plot(Ei[:,0],Ei[:,1])
        #plt.axhline(y=max(Ei[:,1]), color='r', linestyle='-')
        #plt.title(f"$\\Omega =$ {Omeg[i]}")
        Emax = np.append(Emax,max(Ei[:,1]))

    plt.figure()
    plt.scatter(Omeg,Emax, color = "black")
    plt.axvline(x=wn, color='r', linestyle='-')
    plt.xlabel("$\\omega$", fontsize = fs )
    plt.ylabel("$E_{max}$", fontsize = fs )
    
    
def vPlot () :

    plt.figure() 
    plt.plot(x,np.sqrt(v),color = "black")
    plt.xlabel("x [m]", fontsize = fs)
    plt.ylabel("v [m/s]", fontsize = fs)

#ftPlot()
#Ewplot ()
#Convergence()
Eplot()
PlotCouleur()
vPlot()
plt.show()
    
