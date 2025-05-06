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

nsteps = np.array([1,2,3,4,5,6,7,8,9])*200
nx = np.array([1,2,3,4,5,6,7,8,9])*20


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

#print(E)

print(x.shape)
print(f.shape)

def f_analytique ( t, x, w = 0 ) : # changer les valeurs 

    n = 1
    L = 12
    pi = 3.1415926535897932384626433832795028841971
    return np.cos( w * t ) * np.sin( n * x * pi / L ) 

def Erreur ( x , f , tfin ) :

    dx = abs(x[1] - x[0]) # intégrale de Riemann
    err = ( f[-1,1:] - f_analytique(tfin,x) ).sum() * dx # somme de Riemann
    # f[-1,1:] : comme la simulation d'arrête a tfin = T, on prend la dernière valeur 
    return err

def Convergence ( t = 2.39457 , norder = 1 ) : # simulation réalisée avec n = 2 

    err = np.array([])
    for output in outputs :

        xi = np.loadtxt(output+'_x')
        fi = np.loadtxt(output+'_f')
        err = np.append(err,Erreur( tfin = t , x = xi, f = fi))
    
    plt.plot(pow(t/nsteps,norder),err, 'k+-')
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
        plt.pause(0.0002)
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
    


#ftPlot()
Convergence()
Eplot()
PlotCouleur()
plt.show()
    
