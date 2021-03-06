#Run in Python 3.5
"Parallel Genetic Algorithm for airfoil optimization"
import numpy as np
import csv
import os
import glob
import copy
import operator
import subprocess as sp
import threading
import queue
import time
import psutil
import sys

os.chdir(r'C:\Users\Aniru_000\Desktop\TD-1\Airfoil\s1223\airfoil\Python Code\tempfiles')

np.set_printoptions(precision=4)

"""Constants"""
M_CONST=(0,0.04)
P_CONST=(0.15,0.6)
T_CONST=(0.075,0.2)
A_CONST=(-15,15)

M_STEP_SIZE=0.001
P_STEP_SIZE=0.005
T_STEP_SIZE=0.001
A_STEP_SIZE=0.25

SECTION_1_ELEMENTS=40
SECTION_2_ELEMENTS=20
SECTION_3_ELEMENTS=20

REYNOLDS_NO=250000
ALPHA_OPT=5
XFOIL_STEP_SIZE=1

CHROMOSOME_SIZE=4
MAX_GEN=200
POP_SIZE=50
CXPB=0.95
MXPB=0.10
SELECTION_k=4

NUM_THREADS=2
THREAD_LIST=[]
XFOIL_TIMEOUT=20

OPT_TYPE='MAX'                            #MAX or MIN
q=queue.Queue()
"""Class Declarations"""


class Individual(object):
    def __init__(self,id):
        m=np.random.choice(np.arange(M_CONST[0],M_CONST[1]+M_STEP_SIZE,M_STEP_SIZE))                #STEP_SIZE added since arange() does not 
        p=np.random.choice(np.arange(P_CONST[0],P_CONST[1]+P_STEP_SIZE,P_STEP_SIZE))                #include stop 
        t=np.random.choice(np.arange(T_CONST[0],T_CONST[1]+T_STEP_SIZE,T_STEP_SIZE))
        a=np.random.choice(np.arange(A_CONST[0],A_CONST[1]+A_STEP_SIZE,A_STEP_SIZE))
        self.chromosome=np.array([m,p,t,a])
        self.id=id
        self.fitness= -np.inf if OPT_TYPE=='MAX' else np.inf
        
    def evaluate(self,thread_name):
        self.fitness= fitness_function(self,thread_name) #FITNESS FUNCTION CALL return fitness 
        

class GAEnvironment(object):
    def __init__(self, MAX_GEN=50, POP_SIZE=100, CXPB=0.90, MXPB=0.05):
        self.pop_size=POP_SIZE
        #self.gen=0
        self.maxgen=MAX_GEN
        self.mutation_rate=MXPB
        self.crossover_rate=CXPB
        self.population=self.createpopulation(self.pop_size)
        self.best=self.population[0]                                            #Initial value 
        self.newpopulation=[]
        self.hof=self.best
        
    def createpopulation(self,size):
        return [Individual(i) for i in range(size)]
    
    def halloffame(self,ind):
        self.best=ind
        if self.best.fitness> self.hof.fitness:
            self.hof=copy.deepcopy(self.best)
        
    def crossover(self):
        child1=self.selection()
        child2=self.selection()
        if np.random.random()< self.crossover_rate:    
            left=np.random.randint(0,np.ceil(CHROMOSOME_SIZE/2))
            right=np.random.randint(np.floor(CHROMOSOME_SIZE/2),CHROMOSOME_SIZE)
            temp_chromosome=copy.deepcopy(child1)
            child1.chromosome[left:right]=copy.deepcopy(child2.chromosome[left:right])
            child2.chromosome[left:right]=copy.deepcopy(temp_chromosome.chromosome[left:right])
        return child1, child2
        
    def evaluatepopulation(self):
        parallel_computing(self)
    
    def selection(self):
        best=self.population[np.random.randint(0,self.pop_size)]
        for i in range(SELECTION_k):
            ind=self.population[np.random.randint(0,self.pop_size)]
            if best.fitness<ind.fitness:
                best=copy.deepcopy(ind)                    #Removed Copy Copy
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

def airfoil_gen (m,p,t,a,thread_name,id):
    
    x=np.arange(0,0.03,(0.03-0)/(0.5*SECTION_1_ELEMENTS))
    x=np.append(x,np.arange(0.03,0.1,(0.1-0.03)/(0.5*SECTION_2_ELEMENTS)))
    x=np.append(x,np.arange(0.1,1.02,(1.001-0.1)/(0.5*SECTION_3_ELEMENTS)))           #1.02
    upper=np.array([])
    lower=np.array([])
    
    for i in range(len(x)):
        if x[i]<=p:
            yc=(m/(p**2))*((2*p*x[i])-(x[i]**2))
            dyc=((2*m)/p**2)*(p-x[i])
        else:
            a_temp=np.array([[1,1,1,1],[1,p,p**2,p**3],[0,1,2*p,3*p*p],[0,1,2,3]])
            b_temp=np.array([0,m,0,np.tan(-a/57.29)])
            b=np.linalg.solve(a_temp,b_temp)
            yc=b[0]+(b[1]*x[i])+(b[2]*(x[i]**2))+(b[3]*(x[i]**3))
            dyc=b[1]+(2*b[2]*x[i])+(3*b[3]*(x[i]**2))
        
        theta=np.arctan(dyc)
        yt=y_thickness(x[i],t)
        xu=x[i]-(yt*np.sin(theta))            #Assign directly
        yu=yc+(yt*np.cos(theta))
        xl=x[i]+(yt*np.sin(theta))
        yl=yc-(yt*np.cos(theta))
        upper=np.append(upper,np.array([xu,yu]))
        lower=np.append(lower,np.array([xl,yl]))

    airfoil_dat=open("airfoil_"+str(thread_name)+'_'+str(id)+".dat", 'w')
    airfoil_dat.write("TEST AIRFOIL\n\n")
    for i in range(len(upper)-1,-1,-2):
        airfoil_dat.write("%f \t %f\n" %(upper[i-1], upper[i]))
    for i in range(0,len(lower),2):
        airfoil_dat.write("%f \t %f\n" %(lower[i], lower[i+1]))
        
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            continue
    try:        
        process.kill()
    except psutil.NoSuchProcess:
        pass
        
def run_xfoil(thread_name,id) :
    cmd='xfoil.exe < session_'+str(thread_name)+'_'+str(id)+'.txt'
    p1=sp.Popen(cmd, shell=True,creationflags=sp.CREATE_NEW_CONSOLE)
    try:     
        p1.wait(timeout=XFOIL_TIMEOUT)
        errorCode=0
    except sp.TimeoutExpired:
        kill(p1.pid)
        errorCode=1
    return(errorCode)

def change_session_file(thread_name,id):
    with open('session.txt','r') as file:
        data=file.readlines()
    data[0]='load airfoil_'+str(thread_name)+'_'+str(id)+'.dat\n'                            #***PROGRAM WILL CRASH IF THESE INDICES ARE WRONG
    data[12]='output_'+str(thread_name)+'_'+str(id)+'.txt\n'
    data[7]=str(REYNOLDS_NO)+'\n'
    data[14]='aseq 0 '+str(ALPHA_OPT)+' '+str(XFOIL_STEP_SIZE)+'\n'
    
    with open('session_'+str(thread_name)+'_'+str(id)+'.txt' ,'w') as file:                    #Write the session file for that thread
        file.writelines(data)
        
def read_output(thread_name,id):
    with open('output_'+str(thread_name)+'_'+str(id)+'.txt','r', newline='',encoding='utf-8') as f:
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
    id=individual.id
    airfoil_gen(m,p,t,a,thread_name,id)
    change_session_file(thread_name,id)
    errorCode=run_xfoil(thread_name,id)
    if errorCode is 1:
        return -np.inf
    else:    
        cl_dict,cd_dict =read_output(thread_name,id)
        flag=0
        try:
            cl=[cl_dict[k] for k in range(0, ALPHA_OPT+1)]
            cd=[cd_dict[k] for k in range(0, ALPHA_OPT+1)]
            flag=1
            fitness=[cl[k]/cd[k] for k in range(len(cl))][0]
        except KeyError:
            fitness=-np.inf
            pass
            # try:
                # cl=cl_dict[ALPHA_OPT+1]
                # cd=cd_dict[ALPHA_OPT+1]
                # flag=1
            # except KeyError:
                # try:
                    # cl=cl_dict[ALPHA_OPT-1]
                    # cd=cd_dict[ALPHA_OPT-1]
                    # flag=1
                # except KeyError:
                    # flag=0
        
        return fitness

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix))    #sys.stdout.write
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def get_task(thread_name):
    while True:
        individual=q.get()
        individual.evaluate(thread_name)       
        q.task_done()
        # if(q.qsize()%10==0):
            # print(str(q.qsize())+" remaining")
            
def create_threads():
    for i in range(NUM_THREADS):
        THREAD_LIST.append(threading.Thread(target=get_task, args=('Thread_'+str(i),), name='Thread_'+str(i)))
        THREAD_LIST[i].setDaemon(True)    
        THREAD_LIST[i].start()
    
def parallel_computing(ga_object):
    
    for ind in ga_object.population:
        q.put(ind)
      
    i=0
    l=q.qsize()
    printProgress(i, l, prefix = 'Evaluating:', suffix = 'Complete', barLength = 50)
    while q.qsize()>0:
        i=l-q.qsize()
        printProgress(i, l, prefix = 'Evaluating:', suffix = 'Complete', barLength = 50)
        time.sleep(0.3)
    printProgress(l, l, prefix = 'Evaluating:', suffix = 'Complete', barLength = 50)
    q.join()
    while True:
        try:
            [os.remove(x) for x in glob.glob("session_Thread*.txt")]
            [os.remove(x) for x in glob.glob("airfoil_Thread*.dat")]
            [os.remove(x) for x in glob.glob("output_Thread*.txt")]
        except PermissionError:
            print("Unable To delete")
        else:
            break
    print("Completed")
        
        
if __name__=='__main__':
    program_start=time.time()
    g=GAEnvironment(MAX_GEN,POP_SIZE,CXPB,MXPB)
    create_threads()
    for i in range(g.maxgen):
        time_start=time.time()
        g.evaluatepopulation()
        g.population.sort(key=operator.attrgetter('fitness'),reverse=True)
        print(np.array([g.population[i].fitness for i in range(10)]))
        g.halloffame(g.population[0])
        g.newpopulation=g.population[0:int(g.pop_size*0.2)]
        while len(g.newpopulation)<int(g.pop_size*0.7):
            child1, child2=g.crossover()
            g.newpopulation.append(child1)
            g.newpopulation.append(child2)
        
        temp=g.createpopulation(g.pop_size-len(g.newpopulation))
        for j in range(len(temp)):
            g.newpopulation.append(temp[j])
        g.mutation()
        g.population=copy.deepcopy(g.newpopulation)
        print("Generation No: " +str(i))
        print("Best Fitness : %0.3f" %(g.best.fitness))
        print("Execution Time: %0.3fs" %(time.time()-time_start))
    fitness_function(g.hof,"Optimized")
    print("Program Execution Time: %0.3fs" %(time.time()-program_start))
        
        
        
        
        