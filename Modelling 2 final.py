import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.style.use('seaborn')
import scipy.stats as st
import seaborn as sns

def SIR(t,x):
    ds = -beta*x[0]*x[1]
    di = beta*x[0]*x[1] - gamma*x[1]
    dr =  gamma*x[1] 
    return [ds,di,dr]  
num=[]
for i in range(1000):
    num.append(60)
R = 2.5
N = 1000
gamma = 1/14
beta = R*gamma/N
alpha = 0.2
omega = 0.1
ts = np.linspace(0,300, 1000)
x0 =  [N -2,2,0]   
peak=[]
time=[]
dayofpeak=[]
totalinfection=[]
sol = solve_ivp(SIR,[ts[0],ts[-1]],x0,t_eval=ts)
for i in range(200):
  R = 2.5
  N=1000
  s=[N-2]
  t=[0]                # A list to store time-points, with first entry 0
  inf=[2]             # A list to store the infected densities, with some chosen starting value
  gamma=1/14           # The recovery rate parameter value
  max_time=300         # The maximum time for your simulation to run
  beta = (R*gamma)/N

  current_t=t[-1]      # Set the current time
  current_inf=inf[-1]  # Set the current infected density
  current_s = s[-1]
  while (current_t<max_time):  
      
      rho = beta*current_s*current_inf + gamma*current_inf        # Calculate the *current* total rate
      rand_num = np.random.rand()           # Draw a random number between 0 and 1
      t_step = -np.log(rand_num)/rho # Draw the time step from exponential distribution with mean 1/total_rate
      
      current_t+=t_step        # Update current_t    
      
      # Check still in time frame
      if current_t>max_time:
          t.append(max_time)
          inf.append(current_inf)
          break
      
                # Update current_inf (take one away since it is recovery)
      
                                # Update the time list
    
    
      
      rand_num2 = np.random.rand()
      
        
      if rand_num2 > (gamma*current_inf)/rho:
        current_inf += 1
        current_s -= 1
        t.append(current_t)
        inf.append(current_inf)
        s.append(current_s)
      else:
        current_inf -= 1
        t.append(current_t)
        inf.append(current_inf)
        s.append(current_s)
        
      if current_inf ==0:
        time.append(t[-1])
        break
  count=0
  for i in inf:
      count+=1
      if i == max(inf):
          dayofpeak.append(t[count])
          break
  totalinfection.append(N-s[-1]-inf[-1])    
  peak.append(max(inf))  
  plt.step(t,inf,where='post',color='r',alpha=0.2)
  plt.step(t,s,where='post',color='b', alpha=0.2)
print(dayofpeak)
print(np.mean(time))
print(np.mean(peak))
plt.plot(ts,num,"fuchsia",label="Maximum hospital capacity")
plt.plot(ts,sol.y[1],"k", label= "Deterministic")
plt.plot(ts,sol.y[0],"k")
plt.rcParams.update({'font.size': 16})
plt.step(0,0,label='Susceptible (Stochastic)',color='b')
plt.step(0,0,label='Infected (Stochastic)',color='r')
plt.xlabel('Time',fontsize=16)
plt.ylabel('Population',fontsize=16)
plt.title("SIR model for $R_0$=2.5",fontsize=16)
plt.legend(loc="upper right",fontsize=14)
plt.show()

st.t.interval(alpha=0.95, df=len(peak)-1, loc=np.mean(peak), scale=st.sem(peak))
st.t.interval(alpha=0.95, df=len(peaks)-1, loc=np.mean(peaks), scale=st.sem(peaks))
st.t.interval(alpha=0.95, df=len(times)-1, loc=np.mean(times), scale=st.sem(times))
maxval = np.max(sol.y[1])
print(maxval) ##peak infection
for i in range(len(sol.y[1])):
    if sol.y[1][i] == maxval:
        print(ts[i])    ##time of peak infection
for i in range(len(sol.y[1])):
    if sol.y[1][i] < 1:
        print(ts[i])    #day of peak infection
        break
print(1000-sol.y[0][989])   #total number of people infected     
    
R = 5
N = 1000
gamma = 1/14
beta = R*gamma/N
alpha = 0.2
omega = 0.1
ts = np.linspace(0,300, 1000)
x0 =  [N -2,2,0]   
peaks = []
times = []
dayofpeak1=[]
totalinfection1=[]
sol = solve_ivp(SIR,[ts[0],ts[-1]],x0,t_eval=ts)
for i in range(200):
  R = 5
  N=1000
  s=[N-2]
  t=[0]                # A list to store time-points, with first entry 0
  inf=[2]             # A list to store the infected densities, with some chosen starting value
  gamma=1/14           # The recovery rate parameter value
  max_time=300         # The maximum time for your simulation to run
  beta = (R*gamma)/N

  current_t=t[-1]      # Set the current time
  current_inf=inf[-1]  # Set the current infected density
  current_s = s[-1]
  while (current_t<max_time):  
      
      rho = beta*current_s*current_inf + gamma*current_inf        # Calculate the *current* total rate
      rand_num = np.random.rand()           # Draw a random number between 0 and 1
      t_step = -np.log(rand_num)/rho # Draw the time step from exponential distribution with mean 1/total_rate
      
      current_t+=t_step        # Update current_t    
      
      # Check still in time frame
      if current_t>max_time:
          t.append(max_time)
          inf.append(current_inf)
          break
      
                # Update current_inf (take one away since it is recovery)
      
                                # Update the time list
    
    
      
      rand_num2 = np.random.rand()
      
        
      if rand_num2 > (gamma*current_inf)/rho:
        current_inf += 1
        current_s -= 1
        t.append(current_t)
        inf.append(current_inf)
        s.append(current_s)
      else:
        current_inf -= 1
        t.append(current_t)
        inf.append(current_inf)
        s.append(current_s)
      if current_inf ==0:
        times.append(t[-1])  
        break
    
  count=0
  for i in inf:
      count+=1
      if i == max(inf):
          dayofpeak1.append(t[count])
          break
  totalinfection1.append(N-s[-1]-inf[-1])    
  plt.step(t,inf,where='post',color='r',alpha=0.2)
  plt.step(t,s,where='post',color='b', alpha=0.2)
  peaks.append(max(inf))
print(dayofpeak)
print(dayofpeak1)
print(np.mean(times))
maxv = np.max(sol.y[1])
print(maxv) ##peak infection
for i in range(len(sol.y[1])):
    if sol.y[1][i] == maxv:
        print(ts[i])  
print(np.mean(peaks)) 
for i in range(len(sol.y[1])):
    if sol.y[1][i] < 1:
        print(ts[i])    #day  infection dies out
        break
print(1000-sol.y[0][989])   #total number of people infected 
plt.plot(ts,sol.y[1],"k", label= "Deterministic")
plt.plot(ts,sol.y[0],"k")
plt.plot(ts,num,"fuchsia",label="Maximum hospital capacity")
plt.rcParams.update({'font.size': 16})
plt.step(0,0,label='Infected (Stochastic)',color='r')
plt.step(0,0,label='Susceptible (Stochastic)', color='b')
plt.xlabel('Time',fontsize=16)
plt.ylabel('Population',fontsize=16)
plt.title("SIR model for $R_0$=5",fontsize=16)
plt.legend(loc="upper right",fontsize=14)
plt.show()

plt.hist([peak,peaks],bins=25,label=["$R_0=2.5$","$R_0=5$"])
plt.ylabel("Frequency",fontsize=16)
plt.xlabel("Peak number of people infected",fontsize=16)
plt.title("Histogram for peak infection",fontsize=16)
plt.legend(loc="upper left",fontsize=14)
plt.show()


plt.hist([time,times],bins=25,label=["$R_0=2.5$","$R_0=5$"])
plt.ylabel("Frequency",fontsize=16)
plt.xlabel("Days taken for the infection to die out",fontsize=16)
plt.title("Histogram of time taken for infection to cease",fontsize=16)
plt.legend(loc="upper left",fontsize=14)
plt.show()

plt.hist([dayofpeak,dayofpeak1],bins=25,label=["$R_0=2.5$","$R_0=5$"])
plt.ylabel("Frequency",fontsize=16)
plt.xlabel("The day of peak infection",fontsize=16)
plt.title("Histogram for the day of peak infection",fontsize=16)
plt.legend(loc="upper left",fontsize=14)
plt.show()

plt.hist([totalinfection,totalinfection1],bins=25,label=["$R_0=2.5$","$R_0=5$"])
plt.ylabel("Frequency",fontsize=16)
plt.xlabel("Total number of people infected",fontsize=16)
plt.title("Histogram of total infection across epidemic",fontsize=16)
plt.legend(loc="upper left",fontsize=14)
plt.show()

box=plt.boxplot([peak,peaks],patch_artist=True)
plt.ylabel("Number of people infected",fontsize=16)
plt.xticks([1, 2], ['$R_0=2.5$', '$R_0=5$'],fontsize=16)
plt.title("Boxplot for peak infection",fontsize=16)
plt.legend(loc="upper left",fontsize=14)
colors = ['cyan', 'fuchsia']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.show()

box=plt.boxplot([time,times],patch_artist=True)
plt.ylabel("Time in days",fontsize=16)
plt.xticks([1, 2], ['$R_0=2.5$', '$R_0=5$'],fontsize=16)
plt.title("Boxplot for day of infection ceasing",fontsize=16)
plt.legend(loc="upper left",fontsize=14)
colors = ['cyan', 'fuchsia']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.show()

box=plt.boxplot([dayofpeak,dayofpeak1],patch_artist=True)
plt.ylabel("Time in days",fontsize=16)
plt.xticks([1, 2], ['$R_0=2.5$', '$R_0=5$'],fontsize=16)
plt.title("Boxplot for the day of peak infection",fontsize=16)
plt.legend(loc="upper left",fontsize=14)
colors = ['cyan', 'fuchsia']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.show()

box=plt.boxplot([totalinfection,totalinfection1],patch_artist=True)
plt.ylabel("Number of people infected",fontsize=16)
plt.xticks([1, 2], ['$R_0=2.5$', '$R_0=5$'],fontsize=16)
plt.title("Boxplot for total infection",fontsize=16)
plt.legend(loc="upper left",fontsize=14)
colors = ['cyan', 'fuchsia']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.show()

##extension plot

N = 1000
gamma = 1/14

alpha = 0.2
omega = 0.1

ts = np.linspace(0,150, 1000)
x0 =  [N -2,2,0]            
R = [1,1.5,2,5]
col=("g","c","m","y")

for i in range(len(R)):
    beta = R[i]*gamma/N
    sol = solve_ivp(SIR,[ts[0],ts[-1]],x0,t_eval=ts)
    plt.plot(ts,sol.y[1],str(col[i]),label= "$R_0=$" +str(R[i]),)
plt.plot(ts,num,"fuchsia",label="Maximum hospital capacity")
plt.xlabel('Time',fontsize=16)
plt.ylabel('Population',fontsize=16)
plt.title("SIR model for various $R_0$",fontsize=16)
plt.legend(loc="upper right",fontsize=14)
plt.show()


###########
###total infected, peak infection time, when infection ends, standard deviation of peak infection time
###########

#q2 The total recoveries can dip below 0 using this method, as you do susceptible +1 when waning immunity occurs but everyony can already be sucesciptible
longterminf=[]
longtermsus=[]
dayofpeak2=[]
peaks2=[]
peak=[]
time=[]
dayofpeak=[]
susser=[]
infer=[]
inner=[]
outer=[]
howmany=[]
for i in range(200):
  N =1000
  t=[0]                # A list to store time-points, with first entry 0
  inf=[2]              # A list to store the infected densities, with some chosen starting value
  sus=[N-2]            # A list to store the susceptible densities, with some chosen starting value
  gamma=1/14           # The recovery rate parameter value
  beta=(1.5*gamma)/N    # The transmission parameter value (chosen such that R0=5)
  max_time=1500   # The maximum time for your simulation to run

  current_t=t[-1]      # Set the current time
  current_inf=inf[-1]  # Set the current infected density
  current_sus=sus[-1]  # Set the current susceptible density
  omega =1/120
  R1 = 0
  
  
  # Here we use a while loop
  # You should be very careful using these as you can easily get trapped in an infinite loop
  while (current_t<max_time):  
      
      rho = gamma*current_inf + beta*current_inf*current_sus + omega*R1        # Calculate the *current* total rate
      rand_num = np.random.rand()           # Draw a random number between 0 and 1
      t_step = -np.log(rand_num)/rho # Draw the time step from exponential distribution with mean 1/total_rate
      
      
      current_t+=t_step        # Update current_t
      
      # Check still in time frame
      if current_t>max_time:
          t.append(max_time)
          inf.append(current_inf)
          sus.append(current_sus)
          break
      
      rand_num2 = np.random.rand()
      if rand_num2 < (gamma*current_inf)/rho:
          current_inf-=1           
                                 
      elif (gamma*current_inf)/rho < rand_num2  and rand_num2 < (beta*current_sus*current_inf + gamma*current_inf)/rho:
          current_inf+=1           
          current_sus-=1  
          
      else:
          current_sus +=1

      t.append(current_t)      
      inf.append(current_inf)  
      sus.append(current_sus)  
      R1 = N- current_sus - current_inf
      
      if current_inf ==0:
          howmany.append(1)
          break
      if current_t>1499:
          infer.append(current_inf)
          susser.append(current_sus)
          
  count=0
  for i in inf:
      count+=1
      if i == max(inf):
          dayofpeak2.append(t[count])
          break 
  count=0
  for i in inf:
      count+=1
      if i == max(inf):
          dayofpeak.append(t[count])
          break
      
  peak.append(max(inf))  
  longterminf.append(inf[-1])
  longtermsus.append(sus[-1])
  peaks2.append(max(inf))
  #plt.step(t,inf,where='post',color='r',alpha=0.1)
  plt.step(t,sus,where='post',color='b',alpha=0.1)
print(np.mean(dayofpeak))
print(np.mean(peak))

print(np.mean(infer))
print(np.mean(susser))
print(len(howmany))

plt.step(0,1000,label='Susceptible (Stochastic)',color='b')

print(longterminf)        
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

ts = np.linspace(0,700, 1000)   
x0 =  [N -2,2,0]             


sol = solve_ivp(SIR1,[ts[0],ts[-1]],x0,t_eval=ts)


plt.plot(ts,num,"fuchsia",label="Maximum hospital capacity")
plt.plot(ts,sol.y[1],"k",label= "Infected (Deterministic)")
plt.plot(ts,sol.y[0],"k",label= "Susceptible (Deterministic)")

plt.xlabel("Time (Days)",fontsize=16)
plt.ylabel("Units of population",fontsize=16)
plt.title("SIR model with waning immunity for $R_0$=1.5",fontsize=16)
plt.legend(loc="upper right",fontsize=14)
plt.show()  

maxval = np.max(sol.y[1])
print(maxval) ##peak infection
for i in range(len(sol.y[0])):
    if sol.y[1][i] == maxval:
        print(ts[i])    ##day of peak infection

print(sol.y[1][989])   #longterm infection 
for i in range(len(sol.y[1])): 
    if 59 < sol.y[1][i] <61:
        print(ts[i])    #times above
    




#2 plots, 1 for infection, another for susc #done
#stat plots for longterminf,longtermsus, time of peak, peak inf, #done
#relate
#inf to plgging in for I as before, 34.92..., find the time infection is under the capacity
#extension: how many stochastics havent hit 0 infection, the random motion means the longer
#time goes on the more die out and thats all, maybe quick boxes for appendix eith same code lel
#add up total of longtermsus and longterm infected for average amount of people in exposure period
#####Q3
longterminf1=[]
longtermsus1=[]
longtermexp=[]
dayofpeak3=[]
peaks3=[]
for i in range(200):
  N=1000
  t=[0]                # A list to store time-points, with first entry 0
  inf=[2]              # A list to store the infected densities, with some chosen starting value
  sus=[N-2]            # A list to store the susceptible densities, with some chosen starting value
  gamma=1/14           # The recovery rate parameter value
  beta=1.5*gamma/(N)    # The transmission parameter value (chosen such that R0=5)
  max_time=1500       # The maximum time for your simulation to run
  exp=[0]
  epsilon = 1/7
  omega = 1/120
  R = 0
  current_t=t[-1]      # Set the current time
  current_inf=inf[-1]  # Set the current infected density
  current_sus=sus[-1]  # Set the current susceptible density
  current_exp=exp[-1]

  # Here we use a while loop
  # You should be very careful using these as you can easily get trapped in an infinite loop
  while (current_t<max_time):  
      
      total_rate = gamma*current_inf + beta*current_inf*current_sus + epsilon*current_exp + omega*R      # Calculate the *current* total rate
      rand_num = np.random.rand()           # Draw a random number between 0 and 1
      t_step = -np.log(rand_num)/total_rate # Draw the time step from exponential distribution with mean 1/total_rate
      
      current_t+=t_step        # Update current_t
      
      # Check still in time frame
      if current_t>max_time:
          t.append(max_time)
          inf.append(current_inf)
          sus.append(current_sus)
          exp.append(current_exp)
          break
      
      # Draw a new random number and use it to choose which event
      rand_num2 = np.random.rand()
      if rand_num2 < gamma*current_inf/total_rate:
          current_inf-=1           # Update current_inf (take one away since it is recovery)
      elif (gamma*current_inf)/total_rate < rand_num2 and rand_num2 < (gamma*current_inf + beta*current_sus*current_inf)/total_rate:
          current_exp +=1
          current_sus -=1
      elif (gamma*current_inf + beta*current_sus*current_inf)/total_rate < rand_num2 and rand_num2 < (gamma*current_inf + epsilon*current_exp + beta*current_sus*current_inf)/total_rate:
          current_inf+=1           # Update current_inf (add one since it is transmssion)
          current_exp-=1           # Update current_sus (take one away since it is transmission)
      else:
          current_sus+=1

      t.append(current_t)      # Update the time list
      inf.append(current_inf)  # Update the infected density list
      sus.append(current_sus)
      exp.append(current_exp)
      R = N - current_sus - current_inf - current_exp
      # Check that you haven't reached extniction
      # Important to add with the while loop. Think what might happen otherwise
      if current_inf ==0:
          break

  # Plot the stochastic output
  count=0
  for i in inf:
      count+=1
      if i == max(inf):
          dayofpeak3.append(t[count])
          break
  longterminf1.append(inf[-1])
  longtermsus1.append(sus[-1])
  longtermexp.append(exp[-1])
  peaks3.append(max(inf)) 
  plt.step(t,inf,where='post',color='r',alpha=0.1)
  #plt.step(t,sus,where='post',color='b',alpha=0.1)
  plt.step(t,exp,where='post',color="lime",alpha=0.1)


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

ts = np.linspace(0,730, 1000) 

sol = solve_ivp(SEIR,[ts[0],ts[-1]],x0,t_eval=ts)

plt.step(0,0,where='post',color='r',label ="Infectious")
#plt.step(0,1000,where='post',color='b',label="Susceptible")
plt.step(0,0,where='post',color='lime',label="Exposed")
#plt.plot(ts,sol.y[0],"k", label="Deterministic curves")
plt.plot(ts,sol.y[1],"K",label="Deterministic curves")
plt.plot(ts,sol.y[2],"k")
plt.plot(ts,num,"fuchsia",label="Maximum hospital capacity")
plt.xlabel("Time",fontsize=16)
plt.ylabel("Units of population",fontsize=16)
plt.title("SEIR model for $R_0$=1.5",fontsize=16)
plt.legend(loc="upper right",fontsize=16)
plt.show()



maxval = np.max(sol.y[2])
print(maxval) ##peak infection
for i in range(len(sol.y[2])):
    if sol.y[2][i] == maxval:
        print(ts[i])    ##day of peak infection

print(sol.y[2][989])   #longterm infection 
for i in range(len(sol.y[1])): 
    if 59 < sol.y[1][i] <61:
        print(ts[i]) 

sns.distplot(longterminf, color= "lime",label="SIR",hist=False)
sns.distplot(longterminf1,color="r",label="SEIR", hist=False)
plt.xlabel("Number of people infected",fontsize=16)
plt.ylabel("Density",fontsize=16)
plt.xlim(0)
plt.title("Kernel density plot of infection rate after 2 years",fontsize=16)
plt.legend(fontsize=14)
plt.show()

sns.distplot(longtermsus, color="lime",label="SIR",hist=False)
sns.distplot(longtermsus1,color="r",label="SEIR", hist=False)
plt.xlabel("Number of people susceptible",fontsize=16)
plt.ylabel("Density",fontsize=16)
plt.xlim(0)
plt.title("Kernel density plot of the amount of people susceptible after 2 years",fontsize=16)
plt.legend(fontsize=14)
plt.show()

sns.distplot(peaks2, color="lime",label="SIR",hist=False)
sns.distplot(peaks3, color="r",label="SEIR",hist=False)
plt.xlabel("Number of people infected",fontsize=16)
plt.ylabel("Density",fontsize=16)
plt.xlim(0)
plt.title("Kernel density plot of peak number of infections",fontsize=16)
plt.legend(fontsize=14)
plt.show()

sns.distplot(dayofpeak2, hist=True)
sns.distplot(dayofpeak3, hist=True)
plt.xlabel("Time (Days)",fontsize=16)
plt.ylabel("Density",fontsize=16)
plt.xlim(0)
plt.title("Kernel density plot of when peak infection occurs",fontsize=16)
plt.legend(fontsize=14)
plt.show()

plt.hist([peaks2,peaks3],bins=8,label=["SIR","SEIR"])
plt.ylabel("Frequency",fontsize=16)
plt.xlabel("Peak number of people infected",fontsize=16)
plt.title("Histogram for peak infection",fontsize=16)
plt.legend(fontsize=14)
plt.show()

plt.hist([dayofpeak2,dayofpeak3],bins=8,label=["SIR","SEIR"])
plt.ylabel("Frequency",fontsize=16)
plt.xlabel("The day of peak infection",fontsize=16)
plt.title("Histogram for the day in peak infection",fontsize=16)
plt.legend(fontsize=14)
plt.show()

plt.hist([longterminf,longterminf1],bins=8,label=["SIR","SEIR"])
plt.ylabel("Frequency",fontsize=16)
plt.xlabel("The average number of infections on day 730 (2 years)",fontsize=16)
plt.title("Histogram for long term infection",fontsize=16)
plt.legend(fontsize=14)
plt.show()

plt.hist([longtermsus,longtermsus1],bins=8,label=["SIR","SEIR"])
plt.ylabel("Frequency",fontsize=16)
plt.xlabel("The average number of susceptibles on day 730 (2 years)",fontsize=16)
plt.title("Histogram for long term susceptibles",fontsize=16)
plt.legend(fontsize=14)
plt.show()

box=plt.boxplot([peaks2,peaks3])
plt.ylabel("Peak number of people infected",fontsize=16)
plt.xlabel("Model",fontsize=16)
plt.xticks([1, 2], ["SIR", 'SEIR'],fontsize=16)
plt.title("A boxplot for peak infection",fontsize=16)
plt.legend(loc="upper left",fontsize=14)
plt.show()

box=plt.boxplot([dayofpeak2,dayofpeak3])
plt.ylabel("Day of peak infection",fontsize=16)
plt.xlabel("Model",fontsize=16)
plt.xticks([1, 2], ["SIR", 'SEIR'],fontsize=16)
plt.title("A boxplot for the day of peak infection",fontsize=16)
plt.legend(loc="upper left",fontsize=14)
plt.show()

box=plt.boxplot([longterminf,longterminf1])
plt.ylabel("Infectected on day 730 (2 years)",fontsize=16)
plt.xlabel("Model",fontsize=16)
plt.xticks([1, 2], ["SIR", 'SEIR'],fontsize=16)
plt.title("A boxplot for longterm infection",fontsize=16)
plt.legend(loc="upper left",fontsize=14)
plt.show()

plt.boxplot([longterminf,longterminf1])
plt.ylabel("Suscepitbles on day 730 (2 years)",fontsize=16)
plt.xlabel("Model",fontsize=16)
plt.xticks([1, 2], ["SIR", 'SEIR'],fontsize=16)
plt.title("A boxplot for longterm susceptibles",fontsize=16)
plt.legend(loc="upper left",fontsize=14)
plt.show()

print(np.mean(peaks3))
#st.t.interval(alpha=0.95, df=len(peaks2)-1, loc=np.mean(peaks2), scale=st.sem(peaks2))
count=0
for i in longterminf1:
    if i == 0:
        count+=1
print(count)        

print(np.mean(longterminf1))
print(np.mean(longtermsus1))
print(np.mean(longtermexp))
print(np.mean(dayofpeak3))
print(np.mean(peaks3))

###Q4###

lendeath=[]
deathtimes=[]
for i in range(100):
  N=1000
  t=[0]                # A list to store time-points, with first entry 0
  inf=[2]              # A list to store the infected densities, with some chosen starting value
  sus=[N-2]            # A list to store the susceptible densities, with some chosen starting value
  gamma=1/14           # The recovery rate parameter value
  beta=2.5*gamma/N    # The transmission parameter value (chosen such that R0=5)
  max_time=300         # The maximum time for your simulation to run
  alpha=1/100
  current_t=t[-1]      # Set the current time
  current_inf=inf[-1]  # Set the current infected density
  current_sus=sus[-1]  # Set the current susceptible density
  
  # Here we use a while loop
  # You should be very careful using these as you can easily get trapped in an infinite loop
  while (current_t<max_time):  
      
      total_rate = gamma*current_inf + beta*current_inf*current_sus + alpha*current_inf        # Calculate the *current* total rate
      rand_num = np.random.rand()           # Draw a random number between 0 and 1
      t_step = -np.log(rand_num)/total_rate # Draw the time step from exponential distribution with mean 1/total_rate
      
      current_t+=t_step        # Update current_t
      
      # Check still in time frame
      if current_t>max_time:
          t.append(max_time)
          inf.append(current_inf)
          sus.append(current_sus)
          break
      
      # Draw a new random number and use it to choose which event
      rand_num2 = np.random.rand()
      if rand_num2 < gamma*current_inf/total_rate:
            current_inf-=1           # Update current_inf (take one away since it is recovery)
                                     # current_sus is unchanged
      elif gamma*current_inf/total_rate<rand_num2 and rand_num2<(gamma*current_inf +beta*current_inf*current_sus)/total_rate:
            current_inf+=1           # Update current_inf (add one since it is transmssion)
            current_sus-=1           # Update current_sus (take one away since it is transmission)
      else:
            current_inf-=1
            N-=1
            beta=2.5*gamma/N
            lendeath.append(1)
            deathtimes.append(current_t)
                     # Update current_sus (take one away since it is transmission)
      
      
      
      t.append(current_t)      # Update the time list
      inf.append(current_inf)  # Update the infected density list
      sus.append(current_sus)  # Update the susceptible density list
                      # Update the time list
      
      # Check that you haven't reached extniction
      # Important to add with the while loop. Think what might happen otherwise
      if current_inf ==0:
          break

  # Plot the stochastic output
  plt.step(t,inf,where='post',color='r',alpha=0.1)
  #plt.step(t,sus,where='post',color='b',alpha=0.1)




def SIR2(t,x):
    beta=R*gamma/(x[0]+x[1]+x[2])
    ds = -beta*x[0]*x[1]
    di = beta*x[0]*x[1] - gamma*x[1] - alpha*x[1]
    dr =  gamma*x[1] 
    return [ds,di,dr]  

x0 =[N-2, 2.0, 0.0]
R = 2.5
N = 1000
gamma = 1/14
omega = 1/120
alpha = 1/100
ts = np.linspace(0,300, 1000) 

sol = solve_ivp(SIR2,[ts[0],ts[-1]],x0,t_eval=ts)
plt.plot(ts,N-sol.y[0]-sol.y[1]-sol.y[2],"blue", label= "Death for $R_0=2.5$")
x0 =[N-2, 2.0, 0.0]
R = 5
N = 1000
gamma = 1/14
omega = 1/120
alpha = 1/100
sol1 = solve_ivp(SIR2,[ts[0],ts[-1]],x0,t_eval=ts)
plt.plot(ts,N-sol1.y[0]-sol1.y[1]-sol1.y[2],"darkorange", label= "Death for $R_0=5$")


#plt.plot(ts,sol.y[0],"k",label= "Deterministic curves")
#plt.plot(ts,sol.y[1],"k")
#plt.plot(ts,sol.y[2],"k", label= "Recovered")
#plt.plot(ts,N-sol.y[0]-sol.y[1]-sol.y[2],"k", label= "death")
plt.xlabel("Time (days)",fontsize=16)
plt.ylabel("Units of population",fontsize=16)
#plt.step(0,0,where='post',color='r',label ="Infectious")
#plt.step(0,1000,where='post',color='b',label="Susceptible")
#plt.step(0,0,where='post',color='lime',label="Exposed")
plt.ylim(0)
plt.title("Deterministic cumulative death plot with time",fontsize=16)
plt.legend(fontsize=16)
print(N-sol.y[0][999]-sol.y[1][999]-sol.y[2][999])
print(max(sol.y[1]))
plt.show()
print(len(lendeath))

for i in range(1000):
    if (N-sol1.y[0][i]-sol1.y[1][i]-sol1.y[2][i]) > 121:
        print(ts[i])
        break
    
counter=[]
for i in range(100):
    counter.append(10*i)
    
maxdaydeath=[]
for i in range(99):
    maxdaydeath.append((N-sol.y[0][989-10*i]-sol.y[1][989-10*i]-sol.y[2][989-10*i])-(N-sol.y[0][989-10*(i+1)]-sol.y[1][989-10*(i+1)]-sol.y[2][989-10*(i+1)]))

print(max(maxdaydeath))    
print(maxdaydeath)
deathavg=[]
death1avg=[]
time=[]
time1=[]
tt=[]
for i in range(300):
    tt.append[i]
maxdeath=[]  
  
for i in range(100):
  N=1000
  t=[0]                # A list to store time-points, with first entry 0
  inf=[2]              # A list to store the infected densities, with some chosen starting value
  sus=[N-2]  
  Ncount=[0]          # A list to store the susceptible densities, with some chosen starting value
  gamma=1/14           # The recovery rate parameter value
  beta=2.5*gamma/N    # The transmission parameter value (chosen such that R0=5)
  max_time=300         # The maximum time for your simulation to run
  alpha=1/100
  current_t=t[-1]      # Set the current time
  current_inf=inf[-1]  # Set the current infected density
  current_sus=sus[-1]  # Set the current susceptible density
 
  # Here we use a while loop
  # You should be very careful using these as you can easily get trapped in an infinite loop
  while (current_t<max_time):  
      
      total_rate = gamma*current_inf + beta*current_inf*current_sus + alpha*current_inf        # Calculate the *current* total rate
      rand_num = np.random.rand()           # Draw a random number between 0 and 1
      t_step = -np.log(rand_num)/total_rate # Draw the time step from exponential distribution with mean 1/total_rate
      
      current_t+=t_step        # Update current_t
      
      # Check still in time frame
      if current_t>max_time:
          t.append(max_time)
          inf.append(current_inf)
          sus.append(current_sus)
          Ncount.append(1000-N)
          break
      
      # Draw a new random number and use it to choose which event
      rand_num2 = np.random.rand()
      if rand_num2 < gamma*current_inf/total_rate:
            current_inf-=1           # Update current_inf (take one away since it is recovery)
                                     # current_sus is unchanged
      elif gamma*current_inf/total_rate<rand_num2 and rand_num2<(gamma*current_inf +beta*current_inf*current_sus)/total_rate:
            current_inf+=1           # Update current_inf (add one since it is transmssion)
            current_sus-=1           # Update current_sus (take one away since it is transmission)
      else:
            current_inf-=1
            N-=1
            beta=2.5*gamma/N
            #lendeath.append(1)
            #deathtimes.append(current_t)
                     # Update current_sus (take one away since it is transmission)
      
      
      Ncount.append(1000-N)
      t.append(current_t)      # Update the time list
      inf.append(current_inf)  # Update the infected density list
      sus.append(current_sus)  # Update the susceptible density list
                      # Update the time list
      
      # Check that you haven't reached extniction
      # Important to add with the while loop. Think what might happen otherwise
      if current_inf ==0:
          break
  
  
  for i in range(len(t)):
      if Ncount[i] ==Ncount[-1]:
          time.append(t[i])
          break      
  deathavg.append(Ncount[-1])
  # Plot the stochastic output
  #plt.step(t,inf,where='post',color='blue',alpha=0.1)
  plt.step(t,Ncount,where='post',color='blue',alpha=0.15)
print(np.mean(deathavg))
st.t.interval(alpha=0.95, df=len(deathavg)-1, loc=np.mean(deathavg), scale=st.sem(deathavg))


for i in range(100):
  N=1000
  t=[0]                # A list to store time-points, with first entry 0
  inf=[2]              # A list to store the infected densities, with some chosen starting value
  sus=[N-2] 
  Ncount1=[0]           # A list to store the susceptible densities, with some chosen starting value
  gamma=1/14           # The recovery rate parameter value
  beta=5*gamma/N    # The transmission parameter value (chosen such that R0=5)
  max_time=300         # The maximum time for your simulation to run
  alpha=1/100
  current_t=t[-1]      # Set the current time
  current_inf=inf[-1]  # Set the current infected density
  current_sus=sus[-1]  # Set the current susceptible density
  
  # Here we use a while loop
  # You should be very careful using these as you can easily get trapped in an infinite loop
  while (current_t<max_time):  
      
      total_rate = gamma*current_inf + beta*current_inf*current_sus + alpha*current_inf        # Calculate the *current* total rate
      rand_num = np.random.rand()           # Draw a random number between 0 and 1
      t_step = -np.log(rand_num)/total_rate # Draw the time step from exponential distribution with mean 1/total_rate
      
      current_t+=t_step        # Update current_t
      
      # Check still in time frame
      if current_t>max_time:
          t.append(max_time)
          inf.append(current_inf)
          sus.append(current_sus)
          Ncount1.append(1000-N)
          break
      
      # Draw a new random number and use it to choose which event
      rand_num2 = np.random.rand()
      if rand_num2 < gamma*current_inf/total_rate:
            current_inf-=1           # Update current_inf (take one away since it is recovery)
                                     # current_sus is unchanged
      elif gamma*current_inf/total_rate<rand_num2 and rand_num2<(gamma*current_inf +beta*current_inf*current_sus)/total_rate:
            current_inf+=1           # Update current_inf (add one since it is transmssion)
            current_sus-=1           # Update current_sus (take one away since it is transmission)
      else:
            current_inf-=1
            N-=1
            beta=5*gamma/N
            #lendeath.append(1)
            #deathtimes.append(current_t)
                     # Update current_sus (take one away since it is transmission)
      
      
      
      t.append(current_t)      # Update the time list
      inf.append(current_inf)  # Update the infected density list
      sus.append(current_sus)  # Update the susceptible density list
      Ncount1.append(1000-N)               # Update the time list
      
      # Check that you haven't reached extniction
      # Important to add with the while loop. Think what might happen otherwise
      if current_inf ==0:
          break
  for i in range(len(t)):
      if Ncount1[i] == Ncount1[-1]:
          time1.append(t[i])
          break
  death1avg.append(Ncount1[-1])
  # Plot the stochastic output
  #plt.step(t,inf,where='post',color='darkorange',alpha=0.1)
  plt.step(t,Ncount1,where='post',color='darkorange',alpha=0.15)

print(np.mean(death1avg))
st.t.interval(alpha=0.95, df=len(death1avg)-1, loc=np.mean(death1avg), scale=st.sem(death1avg))
print(np.std(deathavg))

plt.step(0,0,where='post',color='blue',label ="$R_0=2.5$")
plt.step(0,0,where='post',color='darkorange',label="$R_0=5$")
plt.xlabel("Time (days)",fontsize=16)
plt.ylabel("Units of population",fontsize=16)
plt.ylim(0)
plt.title("Stochastic cumulative death plot with time",fontsize=16)
plt.legend(fontsize=16)  
plt.show()

print(time)

print(np.mean(time))
print(np.mean(time1))
#####extension

for i in range(100):
  N=1000
  t=[0]                # A list to store time-points, with first entry 0
  inf=[2]              # A list to store the infected densities, with some chosen starting value
  sus=[N-2] 
  Ncount1=[0]
  exp=[0]           # A list to store the susceptible densities, with some chosen starting value
  gamma=1/14 
  alpha=1/100 
  omega=1/120         # The recovery rate parameter value
  beta=5*gamma/N    # The transmission parameter value (chosen such that R0=5)
  max_time=300         # The maximum time for your simulation to run
  alpha=1/100
  epsilon=1/7
  current_t=t[-1]      # Set the current time
  current_inf=inf[-1]  # Set the current infected density
  current_sus=sus[-1]  # Set the current susceptible density
  current_exp=exp[-1]
  # Here we use a while loop
  # You should be very careful using these as you can easily get trapped in an infinite loop
  while (N>0):  
      
      total_rate = gamma*current_inf + beta*current_inf*current_sus + alpha*current_inf + epsilon*current_exp        # Calculate the *current* total rate
      rand_num = np.random.rand()           # Draw a random number between 0 and 1
      t_step = -np.log(rand_num)/total_rate # Draw the time step from exponential distribution with mean 1/total_rate
      
      current_t+=t_step        # Update current_t
      
      # Check still in time frame
      if current_t>max_time:
          t.append(max_time)
          inf.append(current_inf)
          sus.append(current_sus)
          exp.append(current_exp)
          Ncount1.append(1000-N)
          break
      
      # Draw a new random number and use it to choose which event
      rand_num2 = np.random.rand()
      if rand_num2 < gamma*current_inf/total_rate:
          current_inf-=1           # Update current_inf (take one away since it is recovery)
      elif (gamma*current_inf)/total_rate < rand_num2 and rand_num2 < (gamma*current_inf + beta*current_sus*current_inf)/total_rate:
          current_inf +=1
          
      elif (gamma*current_inf + beta*current_sus*current_inf)/total_rate < rand_num2 and rand_num2 < (gamma*current_inf + epsilon*current_exp + beta*current_sus*current_inf)/total_rate:
          current_sus-=1# Update current_inf (add one since it is transmssion)
          current_exp+=1           # Update current_sus (take one away since it is transmission)
                
      elif (gamma*current_inf + epsilon*current_exp + beta*current_sus*current_inf)/total_rate< rand_num2 and rand_num2 < (gamma*current_inf + epsilon*current_exp + beta*current_sus*current_inf + alpha*current_inf)/total_rate:
            current_inf-=1
            N-=1
            
    
            
            beta=5*gamma/N
                
      else:
            current_sus+=1
            #lendeath.append(1)
            #deathtimes.append(current_t)
                     # Update current_sus (take one away since it is transmission)
      
      
      
      t.append(current_t)      # Update the time list
      inf.append(current_inf)  # Update the infected density list
      sus.append(current_sus)
      exp.append(current_exp)# Update the susceptible density list
      Ncount1.append(1000-N)               # Update the time list
      
      # Check that you haven't reached extniction
      # Important to add with the while loop. Think what might happen otherwise
      if current_inf ==0 :
          break
  
  # Plot the stochastic output
  plt.step(t,inf,where='post',color='red',alpha=0.1)
  plt.step(t,exp,where='post',color='limegreen',alpha=0.15)
  plt.step(t,Ncount1,where='post',color='darkorange',alpha=0.15)

def SEIR2(t,x):
    beta= R*gamma/(x[0]+x[1]+x[2]+x[3])
    ds = -beta*x[0]*x[2] + omega*x[3]
    de = beta*x[0]*x[2] - epsilon*x[1]
    di = epsilon*x[1] - gamma*x[2] - alpha*x[2]
    dr = gamma*x[2] - omega*x[3]
    return [ds,de,di,dr]  

x0 =[N-2, 0, 2,0.0]
R = 5
N = 1000
gamma = 1/14
omega = 1/120
alpha = 1/100
ts = np.linspace(0,600, 1000) 


sol1 = solve_ivp(SEIR2,[ts[0],ts[-1]],x0,t_eval=ts)
plt.plot(ts,N-sol1.y[0]-sol1.y[1]-sol1.y[2]-sol1.y[3],"darkorange", label= "Death from infection")
plt.plot(ts,sol1.y[2],"red", label= "Exposed")
plt.plot(ts,sol1.y[1],"limegreen", label= "Infected")



plt.xlabel("Time (days)",fontsize=16)
plt.ylabel("Units of population",fontsize=16)
plt.ylim(0)
plt.title("Plotting a more complete model",fontsize=16)
plt.legend(loc="top right",fontsize=12) 

 
plt.show()