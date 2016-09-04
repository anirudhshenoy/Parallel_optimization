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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing

os.chdir(r'C:\Users\Aniru_000\Desktop\TD-1\Airfoil\s1223\airfoil\Python Code\tempfiles')

np.set_printoptions(precision=4)

"""Constants"""
KINEMATIC_VISCOSITY=1.50375e-5
OPERATING_VELOCITY=12
RHO=1.225
E_1=0.8

M_CONST=(0,0.04)
P_CONST=(0.15,0.6)
T_CONST=(0.075,0.2)
A_CONST=(-15,15)
B_WING_CONST=(0.5,5)
C_WING_CONST=(0.2,0.4)
ALPHA_WING_CONST=(0,3)
TAPER_WING_CONST=(0.5,0.95)

CONSTRAINTS=(M_CONST,P_CONST,T_CONST,A_CONST,B_WING_CONST,C_WING_CONST,ALPHA_WING_CONST,TAPER_WING_CONST)

M_STEP_SIZE=0.001
P_STEP_SIZE=0.005
T_STEP_SIZE=0.001
A_STEP_SIZE=0.25
B_WING_STEP_SIZE=0.05
C_WING_STEP_SIZE=0.05
ALPHA_WING_STEP_SIZE=0.5
TAPER_WING_STEP_SIZE=0.01

SECTION_1_ELEMENTS=40
SECTION_2_ELEMENTS=20
SECTION_3_ELEMENTS=20


XFOIL_STEP_SIZE=ALPHA_WING_STEP_SIZE
XFOIL_LOWER_BOUND=-2

CHROMOSOME_SIZE=8
MAX_ITER=5
POP_SIZE=10

NUM_THREADS=1
THREAD_LIST=[]
XFOIL_TIMEOUT=20

OPT_TYPE='MIN'                            #MAX or MIN
q=queue.Queue()
"""Class Declarations"""


class Individual(object):
    def __init__(self,id):
        m=np.random.choice(np.arange(M_CONST[0],M_CONST[1]+M_STEP_SIZE,M_STEP_SIZE))                #STEP_SIZE added since arange() does not 
        p=np.random.choice(np.arange(P_CONST[0],P_CONST[1]+P_STEP_SIZE,P_STEP_SIZE))                #include stop 
        t=np.random.choice(np.arange(T_CONST[0],T_CONST[1]+T_STEP_SIZE,T_STEP_SIZE))
        a=np.random.choice(np.arange(A_CONST[0],A_CONST[1]+A_STEP_SIZE,A_STEP_SIZE))
        b_wing=np.random.choice(np.arange(B_WING_CONST[0],B_WING_CONST[1]+B_WING_STEP_SIZE,B_WING_STEP_SIZE))
        c_wing=np.random.choice(np.arange(C_WING_CONST[0],C_WING_CONST[1]+C_WING_STEP_SIZE,C_WING_STEP_SIZE))
        alpha_wing=np.random.choice(np.arange(ALPHA_WING_CONST[0],ALPHA_WING_CONST[1]+ALPHA_WING_STEP_SIZE,ALPHA_WING_STEP_SIZE))
        taper_wing=np.random.choice(np.arange(TAPER_WING_CONST[0],TAPER_WING_CONST[1]+TAPER_WING_STEP_SIZE,TAPER_WING_STEP_SIZE))
       
        self.dimensions=np.array([m,p,t,a,b_wing,c_wing,alpha_wing,taper_wing])
        self.velocity=np.array([0,0,0,0,0,0,0,0])                   #Fix this when free
        self.lbest=np.array([0,0,0,0,0,0,0,0]) 
        self.lbest_fitness=-np.inf if OPT_TYPE=='MAX' else np.inf
        self.id=id
        self.fitness= -np.inf if OPT_TYPE=='MAX' else np.inf
        
    def evaluate(self,thread_name):
        self.fitness= fitness_function(self,thread_name)                            #FITNESS FUNCTION CALL return fitness 
        
    def update_lbest(self):
        if self.lbest_fitness>self.fitness:
            self.lbest=copy.deepcopy(self.dimensions)
            self.lbest_fitness=self.fitness
            
        

    
class swarm_optimizer(object):
    def __init__(self, MAX_ITER=50, POP_SIZE=100, C_1=2.05, C_2=2.05):
        self.pop_size=POP_SIZE
        #self.gen=0
        self.maxiter=MAX_ITER
        self.c1=C_1
        self.c2=C_2
        self.phi=self.c1+self.c2
        self.chi=2/np.fabs(2-self.phi-np.sqrt((self.phi**2)-(4*self.phi)))
        self.population=self.createpopulation(self.pop_size)
        self.best=self.population[0]                                            #Initial value 
        self.newpopulation=[]
        self.gbest=self.best
        
    def createpopulation(self,size):
        return [Individual(i) for i in range(size)]
    
    def halloffame(self,ind):
        self.best=ind
        if self.best.fitness< self.gbest.fitness:
            self.gbest=copy.deepcopy(self.best)
        
        
    def evaluatepopulation(self):
        parallel_computing(self)
        
          
    def update_vector(self):
        for i in range(self.pop_size):
            ind=self.population[i]
            ind.velocity=self.chi*(ind.velocity+(self.c1*np.random.random()*(ind.lbest-ind.dimensions))+(self.c2*np.random.random()*(self.gbest.dimensions-ind.dimensions)))
            #print(ind.dimensions)
            #print(ind.velocity)
            ind.dimensions=ind.dimensions+ind.velocity
            #print(ind.dimensions)
            for j in range(len(ind.dimensions)):
                if ind.dimensions[j]<CONSTRAINTS[j][0]:
                    ind.dimensions[j]=CONSTRAINTS[j][0]                    
                elif ind.dimensions[j]>CONSTRAINTS[j][1]:
                    ind.dimensions[j]=CONSTRAINTS[j][1]
                    
    def update_individuals(self):
        for i in range(self.pop_size):
            self.population[i].update_lbest()


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

def change_session_file(thread_name,reynolds_no,alpha_opt,id):
    with open('session.txt','r') as file:
        data=file.readlines()
    data[0]='load airfoil_'+str(thread_name)+'_'+str(id)+'.dat\n'                            #***PROGRAM WILL CRASH IF THESE INDICES ARE WRONG
    data[12]='output_'+str(thread_name)+'_'+str(id)+'.txt\n'
    data[7]=str(reynolds_no)+'\n'
    data[14]='aseq '+str(XFOIL_LOWER_BOUND)+' '+str(alpha_opt)+' '+str(XFOIL_STEP_SIZE)+'\n'
    
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
    m=individual.dimensions[0]
    p=individual.dimensions[1]
    t=individual.dimensions[2]
    a=individual.dimensions[3]
    b_wing=individual.dimensions[4]
    c_wing=individual.dimensions[5]
    alpha_wing=individual.dimensions[6]
    taper_wing=individual.dimensions[7]
    id=individual.id
    
    """Initial Calculations"""
    reynolds_no=(OPERATING_VELOCITY*c_wing)/KINEMATIC_VISCOSITY
    MAC_wing=(2/3)*c_wing*((taper_wing**2 + taper_wing +1)/(taper_wing+1))
    wing_area=b_wing*MAC_wing
    AR_wing=(b_wing**2)/wing_area
    
    airfoil_gen(m,p,t,a,thread_name,id)
    change_session_file(thread_name,reynolds_no,alpha_wing,id)
    errorCode=run_xfoil(thread_name,id)
    if errorCode is 1:
        return np.inf
    else:    
        cl_dict,cd_dict =read_output(thread_name,id)
        flag=0
        cl=np.array([])
        cd=np.array([])
        alpha_airfoil=np.array([])
        for k in np.arange(XFOIL_LOWER_BOUND, alpha_wing+1,XFOIL_STEP_SIZE):               #Check for exceptions   
            try:
                cl=np.append(cl,cl_dict[k])
                cd=np.append(cd,cd_dict[k])
                alpha_airfoil=np.append(alpha_airfoil,k)
            except KeyError:
                continue
        
        if len(cl)>0:                                                              #Valid values for airfoil
            """Begin Wing Calculations"""        
            airfoil_equation=np.polyfit(alpha_airfoil,cl,1)
            a0=airfoil_equation[0]
            alpha_0=-airfoil_equation[1]/a0
            
            """Slope Correction"""
            a_wing=(a0*AR_wing)/(2+(4+AR_wing**2)**0.5)
            
            """Wing Lift Calculations"""
            CL_wing=a_wing*(alpha_wing-(a0))
            lift_wing=0.5*RHO*(OPERATING_VELOCITY**2)*wing_area*CL_wing
            
            """Wing Drag Calculations"""
            cd_p=0
            for k in np.arange(alpha_wing,XFOIL_LOWER_BOUND-1,XFOIL_STEP_SIZE):                   #Use 2 for loops to look ahead and behind (or nested)
                try:
                    cd_p=cd_dict[k]
                    break
                except KeyError:
                    continue
                    
            CD_i_wing=(CL_wing**2)/(np.pi*E_1*AR_wing)
            CD_wing=CD_i_wing+cd_p
            drag_wing=0.5*RHO*(OPERATING_VELOCITY**2)*wing_area*CD_wing
            print(CL_wing)
            print(lift_wing)
            print(CD_wing)
            print(drag_wing)
            #fitness=-CL_wing/CD_wing
            fitness=-100*np.exp(-((lift_wing-35.28)**2)/(2*35**2))
           
        else:
            fitness=np.inf
        
        
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
    print('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix))    #sys.stdout.write
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def get_task(thread_name):
    while True:
        individual=q.get()
        individual.evaluate(thread_name)       
        q.task_done()
      
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


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, g):
        self.g=g
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        plt.ion()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=10, 
                                           init_func=self.setup_plot, blit=True)
    
    def sphere_plot(self,x,y):
        return 20+((x**2-(10*np.cos(2*np.pi*x)))+(y**2-(10*np.cos(2*np.pi*y))))
        


    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x=np.array([self.g.population[i].dimensions[0] for i in range(self.g.pop_size)])
        y=np.array([self.g.population[i].dimensions[1] for i in range(self.g.pop_size)])
        x=(x-np.min(x))/(np.max(x)-np.min(x))
        y=(y-np.min(y))/(np.max(y)-np.min(y))
        self.scat = self.ax.scatter(x, y, animated=True)
        self.prev_x=copy.deepcopy(x)
        self.prev_y=copy.deepcopy(y)
        self.prev2_x=copy.deepcopy(x)
        self.prev2_y=copy.deepcopy(y)
        self.ax.axis([0,1, 0,1])
        # x=np.linspace(M_CONST[0], M_CONST[1])
        # y=np.linspace(P_CONST[0], P_CONST[1])
        # xv,yv=np.meshgrid(x,y)
        # z=self.sphere_plot(xv,yv)
        #self.ax.scatter(512,404.2319,marker='x')
        # self.ax.imshow(z, cmap='autumn_r', interpolation='none',extent=[M_CONST[0], M_CONST[1], P_CONST[0], P_CONST[1]])

        return self.scat,
    
    

    def update(self, i):
        """Update the scatter plot."""
        x=np.array([self.g.population[i].dimensions[0] for i in range(self.g.pop_size)])
        y=np.array([self.g.population[i].dimensions[1] for i in range(self.g.pop_size)])
        xv=np.array([self.g.population[i].velocity[0] for i in range(self.g.pop_size)])
        yv=np.array([self.g.population[i].velocity[1] for i in range(self.g.pop_size)])
        
        
        if (self.prev_x==x).all():      
            del_x=self.prev_x-self.prev2_x
            del_y=self.prev_y-self.prev2_y
            y=self.prev2_y+(0.001*del_y)
            x=self.prev2_x+(0.001*del_x)
            self.prev2_x=copy.deepcopy(x)
            self.prev2_y=copy.deepcopy(y)
            data=[x,y]
            
        else:
            self.prev_x=copy.deepcopy(x)
            self.prev_y=copy.deepcopy(y)
            self.time_start=time.time()
            del_x=self.prev_x-self.prev2_x
            del_y=self.prev_y-self.prev2_y
            y=self.prev2_y+(0.001*del_y)
            x=self.prev2_x+(0.001*del_x)
            self.prev2_x=copy.deepcopy(x)
            self.prev2_y=copy.deepcopy(y)
            data=[x,y]
                 
        self.scat.set_offsets(data)  
        return self.scat,
     
    def plt_update(self):
        plt.pause(5)

    def show(self):
        plt.show()
        

def update_plot(g):
    a = AnimatedScatter(g)
    a.show()
    
    while True:
        a.plt_update()
        
if __name__=='__main__':
    
    program_start=time.time()
    g=swarm_optimizer(MAX_ITER,POP_SIZE)
    # plt_update_thread=threading.Thread(target=update_plot, args=(g,))                 #Uncomment for animation
    # plt_update_thread.setDaemon(True)
    # plt_update_thread.start()
    
    
    create_threads()
    input('Ready')
    
    
    for iter in range(g.maxiter):
        time_start=time.time()
        g.evaluatepopulation()
        g.update_individuals()  
        g.population.sort(key=operator.attrgetter('fitness'))
        print(np.array([g.population[i].fitness for i in range(10)]))
        g.halloffame(g.population[0])
        g.update_vector()

        print("Generation No: " +str(iter))
        print("Best Fitness : %0.3f" %(g.best.fitness))
        print("Execution Time: %0.3fs" %(time.time()-time_start))
    fitness_function(g.gbest,"Optimized")
    print("Program Execution Time: %0.3fs" %(time.time()-program_start))
    print("Best Individual:" )
    print(g.gbest.dimensions)
    print(g.gbest.fitness)
        
        
        
        