#Run in Python 3.5
#airfoil_gen(0.04,0.6,0.2,15)
import numpy as np
import csv
import os
import copy
import operator
import subprocess as sp
import threading
import queue

os.chdir(r'C:\Users\Aniru_000\Desktop\TD-1\Airfoil\s1223\airfoil\Python Code\tempfiles')

np.set_printoptions(precision=4)

"""Constants"""
M_CONST=(0,0.04)
P_CONST=(0.15,0.6)
T_CONST=(0.075,0.2)
A_CONST=(-10,10)

M_STEP_SIZE=0.005
P_STEP_SIZE=0.01
T_STEP_SIZE=0.005
A_STEP_SIZE=0.5

SECTION_1_ELEMENTS=40
SECTION_2_ELEMENTS=20
SECTION_3_ELEMENTS=20

REYNOLDS_NO=200000
ALPHA_OPT=0

CHROMOSOME_SIZE=4
SELECTION_k=2
NUM_THREADS=2

OPT_TYPE='MAX'                            #MAX or MIN
q=queue.Queue()
"""Class Declarations"""


class Individual(object):
    def __init__(self):
        m=np.random.choice(np.arange(M_CONST[0],M_CONST[1]+M_STEP_SIZE,M_STEP_SIZE))                #STEP_SIZE added since arange() does not 
        p=np.random.choice(np.arange(P_CONST[0],P_CONST[1]+P_STEP_SIZE,P_STEP_SIZE))                #include stop 
        t=np.random.choice(np.arange(T_CONST[0],T_CONST[1]+T_STEP_SIZE,T_STEP_SIZE))
        a=np.random.choice(np.arange(A_CONST[0],A_CONST[1]+A_STEP_SIZE,A_STEP_SIZE))
        self.chromosome=np.array([m,p,t,a])
        self.fitness= -np.inf if OPT_TYPE=='MAX' else np.inf
        
    def evaluate(self,thread_name):
        self.fitness= fitness_function(self,thread_name) #FITNESS FUNCTION CALL return fitness 
        

class GAEnvironment(object):
    def __init__(self, MAX_GEN=100, POP_SIZE=50, CXPB=0.90, MXPB=0.05):
        self.pop_size=POP_SIZE
        self.gen=0
        self.maxgen=MAX_GEN
        self.mutation_rate=MXPB
        self.crossover_rate=CXPB
        self.population=self._createpopulation()
        self.best=self.population[0]                                            #Initial value 
        self.newpopulation=[]
        
    def _createpopulation(self):
        return [Individual() for i in range(self.pop_size)]
        
    def crossover(self):
        child1=self.selection()
        child2=self.selection()
        if np.random.random()< self.crossover_rate:    
            left=np.random.randint(0,np.ceil(CHROMOSOME_SIZE/2))
            right=np.random.randint(np.floor(CHROMOSOME_SIZE/2),CHROMOSOME_SIZE)
            temp_chromosome=child1
            child1.chromosome[left:right]=child2.chromosome[left:right]
            child2.chromosome[left:right]=temp_chromosome.chromosome[left:right]
        return child1, child2
        
    def evaluatepopulation(self):
        parallel_computing(self)
        #for i in range(self.pop_size):
            #self.population[i].evaluate()
    
    def selection(self):
        best=copy.copy(self.population[np.random.randint(0,self.pop_size)])
        for i in range(SELECTION_k):
            ind=self.population[np.random.randint(0,self.pop_size)]
            if best.fitness<ind.fitness:
                best=copy.copy(ind)
        return best
        
    def mutation(self):
        temp=np.random.randint(0,self.pop_size)
        index=np.random.randint(0,CHROMOSOME_SIZE)
        if index==0:
            self.population[temp].chromosome[index]=np.random.choice(np.arange(M_CONST[0],M_CONST[1]+M_STEP_SIZE,M_STEP_SIZE))                #STEP_SIZE added since arange() does not 
        elif index==1:
            self.population[temp].chromosome[index]=np.random.choice(np.arange(P_CONST[0],P_CONST[1]+P_STEP_SIZE,P_STEP_SIZE))                #include stop 
        elif index==2:
            self.population[temp].chromosome[index]=np.random.choice(np.arange(T_CONST[0],T_CONST[1]+T_STEP_SIZE,T_STEP_SIZE))
        elif index==3:
            self.population[temp].chromosome[index]=np.random.choice(np.arange(A_CONST[0],A_CONST[1]+A_STEP_SIZE,A_STEP_SIZE))
        else:
            print("Warning: MUTATION FAILED")


def y_thickness(x,t):
    return ((t/0.2)*((0.29690*(x**0.5))-(0.126*x)-(0.3516*(x**2))+(0.28430*(x**3))-(0.10150*(x**4))))

def airfoil_gen (m,p,t,a,thread_name):
    
    x=np.arange(0,0.03,(0.03-0)/(0.5*SECTION_1_ELEMENTS))
    x=np.append(x,np.arange(0.03,0.1,(0.1-0.03)/(0.5*SECTION_2_ELEMENTS)))
    x=np.append(x,np.arange(0.1,1.02,(1.001-0.1)/(0.5*SECTION_3_ELEMENTS)))
    upper=np.array([])
    lower=np.array([])
    
    for i in range(len(x)):
        if x[i]<=p:
            yc=(m/(p**2))*((2*p*x[i])-(x[i]**2))
            dyc=((2*m)/p**2)*(p-x[i])
        else:
            a_temp=np.array([[0,1,1,1],[1,p,p**2,p**3],[0,1,2*p,3*p*p],[0,1,2,3]])
            b_temp=np.array([0,m,0,-np.tan(-a/57.29)])
            b=np.linalg.solve(a_temp,b_temp)
            yc=b[0]+(b[1]*x[i])+(b[2]*(x[i]**2))+(b[3]*(x[i]**3))
            dyc=b[1]+(2*b[2]*x[i])+(3*b[3]*(x[i]**2))
        
        theta=np.arctan(dyc)
        yt=y_thickness(x[i],t)
        xu=x[i]-(yt*np.sin(theta))            #Assign directly
        yu=yc+(yt*np.cos(theta))
        xl=x[i]+(yt*np.sin(theta))
        yl=yc-(yt*np.cos(theta))
        #print("XU" +str(xu))
        #print("Theta" +str(theta))
        #print("YT "+ str(yt))
        #print("yc" + str(yc))
        #print("YU" +str(yu))
        
        upper=np.append(upper,np.array([xu,yu]))
        lower=np.append(lower,[xl,yl])

    airfoil_dat=open("airfoil_"+str(thread_name)+".dat", 'w')
    airfoil_dat.write("TEST AIRFOIL\n\n")
    for i in range(len(upper)-1,-1,-2):
        airfoil_dat.write("%f \t %f\n" %(upper[i-1], upper[i]))
    for i in range(0,len(lower),2):
        airfoil_dat.write("%f \t %f\n" %(lower[i], lower[i+1]))
        
def run_xfoil(thread_name) :
    cmd='xfoil.exe < session_'+str(thread_name)+'.txt'
    p1=sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        out,errs=p1.communicate(timeout=10)
        errorCode=0
    except sp.TimeoutExpired:
        p1.terminate()
        print ("tried so hard and got so far....")
        errorCode=1
    return(errorCode)

def change_session_file(thread_name):
    with open('session.txt','r') as file:
        data=file.readlines()
    data[0]='load airfoil_'+str(thread_name)+'.dat\n'                            #***PROGRAM WILL CRASH IF THESE INDICES ARE WRONG
    data[8]='output_'+str(thread_name)+'.txt\n'
    data[3]=str(REYNOLDS_NO)+'\n'
    
    with open('session_'+str(thread_name)+'.txt' ,'w') as file:                    #Write the session file for that thread
        file.writelines(data)
        
def read_output(thread_name):
    with open('output_'+str(thread_name)+'.txt','r', newline='',encoding='utf-8') as f:
        csv_f=csv.reader(f)
        rowNum=0
        alpha=[]
        cl=[]
        cd=[]
        cdp=[]
        cm=[]
        top_xtr=[]
        bot_xtr=[]
        for row in csv_f:
            if rowNum>=12:
                
                rowData=str(row)
                rowData=rowData.split()
                if len(rowData)>1:
                    alpha.append(float(rowData[1]))
                    cl.append(float(rowData[2]))
                    cd.append(float(rowData[3]))
                    cdp.append(float(rowData[4]))
                    cm.append(float(rowData[5]))
                    top_xtr.append(float(rowData[6]))
                    bot_xtr.append(float(rowData[7][0:6]))
                
            rowNum+=1
    cd=[cd[i]+cdp[i] for i in range(len(cdp))]
    return(dict(zip(alpha,cl)), dict(zip(alpha,cd)))
    
    
def fitness_function(individual,thread_name):
    m=individual.chromosome[0]
    p=individual.chromosome[1]
    t=individual.chromosome[2]
    a=individual.chromosome[3]
    airfoil_gen(m,p,t,a,thread_name)
    change_session_file(thread_name)
    errorCode=run_xfoil(thread_name)
    if errorCode is 1:
        print('Hello')
        flag=0
    else:    
        cl_dict,cd_dict =read_output(thread_name)
        flag=0
        try:
            cl=cl_dict[ALPHA_OPT]
            cd=cd_dict[ALPHA_OPT]
            flag=1
        except KeyError:
            try:
                cl=cl_dict[ALPHA_OPT+1]
                cd=cd_dict[ALPHA_OPT+1]
                flag=1
            except KeyError:
                try:
                    cl=cl_dict[ALPHA_OPT-1]
                    cd=cd_dict[ALPHA_OPT-1]
                    flag=1
                except KeyError:
                    flag=0
        
    os.remove('airfoil_'+str(thread_name)+'.dat')
    os.remove('output_'+str(thread_name)+'.txt')
    os.remove('session_'+str(thread_name)+'.txt')
    return -np.inf if flag==0 else cl/cd

def get_task(thread_name):
    while True:
        individual=q.get()
        print("Getting next individual, Thread:" +thread_name)
        individual.evaluate(thread_name)
        print(individual.fitness)
        
        q.task_done()
    
def parallel_computing(ga_object):
    
    for ind in ga_object.population[0:10]:
        q.put(ind)
    
    thread_list=[]
    for i in range(NUM_THREADS):
        thread_list.append(threading.Thread(target=get_task, args=('Thread_'+str(i),), name='Thread_'+str(i)))
        thread_list[i].setDaemon(True)    
        thread_list[i].start()
        
    q.join()
    print("Completed")
        
        
if __name__=='__main__':
    g=GAEnvironment()
    # g.evaluatepopulation()
    # g.population.sort(key=operator.attrgetter('fitness'))
    # g.newpopulation=g.population[0:10]
    # while len(g.newpopulation)<g.pop_size:
        # child1, child2=g.crossover()
        # g.newpopulation.append(child1)
        # g.newpopulation.append(child2)
    # g.mutation()
    # g.population=copy.copy(g.newpopulation)
        
        
        
        
        
        