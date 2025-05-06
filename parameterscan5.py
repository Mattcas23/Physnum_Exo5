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


#VarFichier = np.genfromtxt("input_example_student" , comments = '//')


# ------------------------------------- Simulations ----------------------------------- #

nsteps = np.array([3000])
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
    xx = data[-1] # position finale 
    convergence_list.append(xx)

lw = 1.5
fs = 16

x = np.loadtxt(outputs[-1]+'_x') # position en x 
f = np.loadtxt(outputs[-1]+'_f') # 
v = np.loadtxt(outputs[-1]+'_v') 
E = np.loadtxt(outputs[-1]+'_en') # temps et énergie

print(E)

print(x.shape)
print(f.shape)

Tn = 0 # période d'oscillation calculée analytiquement

def f_analytique ( t,x,w = 0 ) : # changer les valeurs 

    n = 1
    L = 12
    return np.cos( w * t ) * np.sin( n * x * np.pi() / L ) 

def Erreur ( x , f , t = Tn ) :

    dx = abs(x[1] - x[0]) # intégrale de Riemann 
    err = ( f[t,1:] - f_analytique(t,x) ).sum() * dx # somme de Riemann
    return err

def Convergence ( t = Tn ) :

    err = []
    for output in outputs :

        xi = np.loadtxt(output+'_x')
        fi = np.loadtxt(output+'_f')
        err.append(Erreur( t = Tn , x = xi, f = fi))
    
    plt.plot(nx,err)
    plt.title(f"$\\Beta_{CFL} = $ {1}") 
    plt.xlabel("$n_x$", fontsize = fs)
    plt.ylabel("n_{steps}", fontsize = fs)
    



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
    


ftPlot()
#Convergence()
Eplot()
PlotCouleur()
plt.show()
    
