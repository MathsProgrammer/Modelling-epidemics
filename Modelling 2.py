import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Question 1 models
def SIR(t,x):
    ds = -beta*x[0]*x[1]
    di = beta*x[0]*x[1] - gamma*x[1]
    dr =  gamma*x[1] 
    return [ds,di,dr]  

#plot r = 2.5
R = 2.5
N = 1000
gamma = 1/14
beta = R*gamma/N
alpha = 0.2
omega = 0.1

ts = np.linspace(0,200, 1000)
x0 =  [N -2,2,0]            

sol = solve_ivp(SIR,[ts[0],ts[-1]],x0,t_eval=ts)

plt.plot(ts,sol.y[0],"g",label= "susceptible")
plt.plot(ts,sol.y[1],"r", label= "infected")
#plt.plot(ts,sol.y[2],"b", label = "recovery")
plt.xlabel("Time")
plt.ylabel("Units of population")
plt.title("SIR model for $R_0$=2.5")
plt.legend()
plt.show()
#####################################
#plot r = 5
R = 5
N = 1000
gamma = 1/14
beta = R*gamma/N
alpha = 0.2
omega = 0.1

ts = np.linspace(0,150, 1000)
x0 =  [N -2,2,0]            

sol = solve_ivp(SIR,[ts[0],ts[-1]],x0,t_eval=ts)

plt.plot(ts,sol.y[0],"g",label= "susceptible")
plt.plot(ts,sol.y[1],"r", label= "infected")
#plt.plot(ts,sol.y[2],"b", label = "recovery")
plt.xlabel("Time")
plt.ylabel("Units of population")
plt.title("SIR model for $R_0$=5")
plt.legend()
plt.show()
####################################### Q2
def SIR1(t,x):
    ds = -beta*x[0]*x[1] + omega*x[2]
    di = beta*x[0]*x[1] - gamma*x[1]
    dr= gamma*x[1] - omega*x[2]
    return [ds,di,dr]  

R = 1.5 
N = 1000
gamma = 1/14
beta = R*gamma/N
omega = 1/120

ts = np.linspace(0,730, 1000)   
x0 =  [N -2,2,0]             

# Now plug everything into solve_ivp
sol = solve_ivp(SIR1,[ts[0],ts[-1]],x0,t_eval=ts)

# Plot the output
plt.plot(ts,sol.y[0],"g",label= "susceptible")
plt.plot(ts,sol.y[1],"r", label= "infected")
#plt.plot(ts,sol.y[2],"b", label = "recovery")
plt.xlabel("Time")
plt.ylabel("Units of population")
plt.title("SIR model for $R_0$=1.5")
plt.legend()
plt.show()
############################################## Q3
def SEIR(t,x):
    ds = -beta*x[0]*x[2] + omega*x[3]
    de = beta*x[0]*x[2] - epsilon*x[1]
    di = epsilon*x[1] - gamma*x[2]
    dr = gamma*x[2] - omega*x[3]
    return [ds,de,di,dr]  

x0 =[N-2, 0.0, 2.0, 0.0]
R = 1.5 
N = 1000
gamma = 1/14
beta = R*gamma/N
omega = 1/120
epsilon = 1/7

ts = np.linspace(0,1000, 1000) 

sol = solve_ivp(SEIR,[ts[0],ts[-1]],x0,t_eval=ts)

plt.plot(ts,sol.y[0],"g",label= "Susceptible")
plt.plot(ts,sol.y[1],"r", label= "Exposed")
plt.plot(ts,sol.y[2],"b", label= "Infected")
#plt.plot(ts,sol.y[3],"b", label = "recovery")
plt.xlabel("Time")
plt.ylabel("Units of population")
plt.title("SIR model for $R_0$=1.5")
plt.legend()
plt.show()
############################################## Q4
def SIR2(t,x):
    ds = -beta*x[0]*x[1]
    di = beta*x[0]*x[1] - gamma*x[1] - alpha*x[1]
    dr =  gamma*x[1] 
    return [ds,di,dr]  

x0 =[N-2, 2.0, 0.0]
R = 5
N = 1000
gamma = 1/14
beta = R*gamma/N
omega = 1/120
alpha = 1/100
ts = np.linspace(0,1000, 1000) 

sol = solve_ivp(SIR2,[ts[0],ts[-1]],x0,t_eval=ts)

plt.plot(ts,sol.y[0],"g",label= "Susceptible")
plt.plot(ts,sol.y[1],"r", label= "infected")
#plt.plot(ts,sol.y[2],"b", label= "Recovered")
plt.xlabel("Time")
plt.ylabel("Units of population")
plt.title("SIR model for $R_0$=1.5")
plt.legend()
plt.show()



