import numpy as np
import scipy as sci 
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

#nsteps = np.array([2,3,4,5,6,7,8,9])*200 # valeurs utilisées pour la convergence
#nx = np.array([2,3,4,5,6,7,8,9])*20 # valeurs utilisées pour la convergence

nsteps = np.array([50e3]) # valeurs utilisées pour le tsunami
nx = np.array([10000]) # valeurs utilisées pour le tsunami 

#nsteps = np.array([2000]) 
#nx = np.array([64])

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

def f_analytique (T ,  x , L = 15.0 ) : # changer les valeurs , on a que T = tfin , n_init = 2 

    n = 2
    kn = ( n + 0.5 ) * np.pi / L 
    om = (2.0 + 0.5) * np.pi * np.sqrt(4.0*9.81) / 15.0 # calcul de oméga à partir de la période

    #print(om)
    #print(np.sin( om * T ) * np.cos( kn*x ))
    
    return ( np.cos( om * T ) + np.sin( om * T ) ) * np.cos( kn*x ) 

def Erreur (x , f , tfin) :

    dx = abs(x[1] - x[0]) # intégrale de Riemann
    err = ( f[-1,1:] - f_analytique(tfin,x) ).sum() * dx # somme de Riemann
    # f[-1,1:] : comme la simulation d'arrête a tfin = T, on prend la dernière valeur 
    return err

def Convergence ( T = 1.91565 , norder = 2 ) : # simulation réalisée avec n = 2 , T période pour n = 2 # T = 2.39457

    err = np.array([]) 
    for output in outputs :

        xi = np.loadtxt(output+'_x')
        fi = np.loadtxt(output+'_f')
        err = np.append(err,Erreur( tfin = T , x = xi, f = fi ))

    plt.figure()
    plt.plot(pow(T/nsteps,norder),err,'k+-') # pow(T/nsteps,norder)
    #plt.title("$\\Beta_{CFL} = $") 
    plt.xlabel(f"$(\\Delta t)^{norder}$", fontsize = fs)
    plt.ylabel("$\\delta_{err}$", fontsize = fs)
    



def ftPlot( scatter = False ) : 

    t = f[:,0]

    plt.ion() # pour faire l'animation visuelle

    for i in range(f.shape[0]) :
    
        f_a_t = f[i,1:] # f au temps t
        if (scatter) : 
            plt.scatter(x,f_a_t)
        else :
            plt.plot(x,f_a_t)
        plt.title(f"t = {t[i]}")
        plt.draw()
        plt.pause(0.02)
        plt.close()
        
    plt.ioff() # pour arrêter

def PlotCouleur() :

    t = f[:,0]
    fval = f[:,1:]

    plt.figure()
    #plt.title("$\\beta_{CFL} = 1.01$")
    plt.pcolor(x,t,fval)
    plt.xlabel("x [m]", fontsize = fs)
    plt.ylabel("t [s]", fontsize = fs)
    plt.colorbar()

def Ana_vs_Num ( pos = x , tfin = 1.91565 ) : # plot la solution analytique et celle numérique au temps t = tfin = T ( réalisée pour n_init = 2 )

    fnum = f[-1,1:] # solution numérique ( prendre impose_nsteps = false et CFL = 1 )
    fana = f_analytique( T = tfin , x = pos , L = 15.0 )

    plt.figure()
    plt.title("n = 2", fontsize = fs - 2)
    plt.plot(pos,fnum, color = "black" , label = "$f_{numérique}$")
    plt.plot(pos,fana, color = "red" , linestyle = "dashed", label = "$f_{analytique}$")
    plt.xlabel("x [m]", fontsize = fs )
    plt.ylabel("f [m]", fontsize = fs )
    
    
def Eplot () : # plot l'énergie en fonction du temps 

    En = E[:,1]
    t  = E[:,0]
    
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
    plt.axvline(x=wn, color='r', linestyle='-', label = f"$\\omega_n$ = {wn}")
    plt.xlabel("$\\omega$", fontsize = fs )
    plt.ylabel("$E_{max}$", fontsize = fs )
    
    
def vPlot () : # plot de la vitesse calculée en fonction de la profondeur

    tcrete = f[:,0] # temps pour chaque crête
    icrete = [] #np.argmax(f[:,1:] , axis =0) # indice de position de la crête pour un certain temps t

    a = int(0) # marque l'indice de temps t[a] à partir duquel l'onde sort ( c-à-d np.argmax = 0 )
    b = int(-1)
    tf = int(0)

    for j in range(len(tcrete)) :

        idx = np.argmax(f[j,1:])

        if ( idx <= b ) : # si l'indice retourné par argmax est inférieure ou égale ou précédent , l'onde est sortie
            tf = j # indice du temps final ( on ne considère pas les temps après que l'onde soit sortie )
            break # on sort de la boucle for 
        else :
            icrete.append( idx )  # pour un certain temps t[j] on trouve l'indice de la position de la crête
            b = idx 
    
    xcrete = x[icrete] # on sélectionne les positions correspondant à ces indices
    
    dxcrete = xcrete[1:] - xcrete[:-1] # x[i+1] - x[i]
    dtcrete = tcrete[1:j] - tcrete[:j-1] # t[i+1] - t[i]
    
    vnum = dxcrete / dtcrete # on calcul la vitesse

    xvnum = ( xcrete[1:] + xcrete[:-1] ) / 2 # middle points où ont été évaluées les dérivées 
    
    plt.figure()
    plt.plot(xvnum,vnum, color = "black")
    plt.plot(x[icrete],np.sqrt(v[icrete]), color = "red" , linestyle = "dashed" , label = "WKB") # solution WKB ( u = sqrt(gh) pris du fichier Cpp )
    plt.xlabel("x [m]", fontsize = fs)
    plt.ylabel("$u$(x) [m/s]", fontsize = fs)
    

    xa = 900e3
    xb = 950e3
    plt.axvline(x=xa, color='blue', linestyle='dotted', label = f"$x_a$ = {xa:.1e}")
    plt.axvline(x=xb, color='green', linestyle='dotted', label = f"$x_b$ = {xb:.1e}")

    plt.legend(fontsize = fs - 3 )

def CretePlot () : # plot la position de la crête en fonction de la position

    fmax = np.max(f[:,1:], axis = 0)
    idx  = np.argmax(f[0,1:]) # on commence le plot à partir de la position de la crête à t = 0 
    #_ = sci.interpolate.interp1d(x , fmax , kind = "quadratic") # interpolation quadratique (ne fonctionne pas)
    #fmax_new = _(fmax)   
    #print(fmax_new)
    
    plt.figure()
    plt.plot(x[idx:], fmax[idx:] , color = "black") # crête numérique

    
    # WKB eq A
    #plt.plot(x[idx:], pow(v[idx:],0.25)/pow(v[0],0.25), color = "red" , linestyle = "dashed" , label = "WKB")
    # WKB eq B
    #plt.plot(x[idx:], pow(v[0],0.25)/pow(v[idx:],0.25), color = "red" , linestyle = "dashed" , label = "WKB") # solution WKB, au début on doit avoir une amplitude de 1 donc A0 = v[0]^1/4
    # WKB eq C 
    plt.plot(x[idx:], pow(v[0],0.75 )/pow(v[idx:],0.75 ), color = "red" , linestyle = "dashed" , label = "WKB")


    xa = 900e3
    xb = 950e3
    plt.axvline(x=xa, color='blue', linestyle='dotted', label = f"$x_a$ = {xa:.1e}")
    plt.axvline(x=xb, color='green', linestyle='dotted', label = f"$x_b$ = {xb:.1e}")

    plt.xlabel("x [m]", fontsize = fs)
    plt.ylabel("f$_{max}$ [m]", fontsize = fs)
    plt.legend(fontsize = fs - 3)
    
    

#ftPlot()
#Ewplot ()
#Ana_vs_Num ( )
#Convergence()
#Eplot()
CretePlot()
PlotCouleur()
vPlot()
plt.show()
    
