#Run in python3.5
#g=GAEnvironment()
import random
import numpy as np
import copy
import operator

np.set_printoptions(precision=4)

"""Constants"""
M_CONST=(0,0.04)
P_CONST=(0.15,0.6)
T_CONST=(0.075,0.2)
A_CONST=(-10,15)

M_STEP_SIZE=0.005
P_STEP_SIZE=0.01
T_STEP_SIZE=0.005
A_STEP_SIZE=0.5


CHROMOSOME_SIZE=4
SELECTION_k=2

OPT_TYPE='MAX'							#MAX or MIN

"""Class Declarations"""


class Individual(object):
	def __init__(self):
		m=np.random.choice(np.arange(M_CONST[0],M_CONST[1]+M_STEP_SIZE,M_STEP_SIZE))				#STEP_SIZE added since arange() does not 
		p=np.random.choice(np.arange(P_CONST[0],P_CONST[1]+P_STEP_SIZE,P_STEP_SIZE))				#include stop 
		t=np.random.choice(np.arange(T_CONST[0],T_CONST[1]+T_STEP_SIZE,T_STEP_SIZE))
		a=np.random.choice(np.arange(A_CONST[0],A_CONST[1]+A_STEP_SIZE,A_STEP_SIZE))
		self.chromosome=np.array([m,p,t,a])
		self.fitness= -np.inf if OPT_TYPE=='MAX' else np.inf
		
	def evaluate(self):
		self.fitness= 0 #FITNESS FUNCTION CALL return fitness 

class GAEnvironment(object):
	def __init__(self, MAX_GEN=100, POP_SIZE=50, CXPB=0.90, MXPB=0.05):
		self.pop_size=POP_SIZE
		self.gen=0
		self.maxgen=MAX_GEN
		self.mutation_rate=MXPB
		self.crossover_rate=CXPB
		self.population=self._createpopulation()
		self.best=self.population[0]											#Initial value 
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
		for i in range(self.pop_size):
			self.population[i].evaluate()
	
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
			self.population[temp].chromosome[index]=np.random.choice(np.arange(M_CONST[0],M_CONST[1]+M_STEP_SIZE,M_STEP_SIZE))				#STEP_SIZE added since arange() does not 
		elif index==1:
			self.population[temp].chromosome[index]=np.random.choice(np.arange(P_CONST[0],P_CONST[1]+P_STEP_SIZE,P_STEP_SIZE))				#include stop 
		elif index==2:
			self.population[temp].chromosome[index]=np.random.choice(np.arange(T_CONST[0],T_CONST[1]+T_STEP_SIZE,T_STEP_SIZE))
		elif index==3:
			self.population[temp].chromosome[index]=np.random.choice(np.arange(A_CONST[0],A_CONST[1]+A_STEP_SIZE,A_STEP_SIZE))
		else:
			print("Warning: MUTATION FAILED")
		
		
if __name__=='__main__':
	g=GAEnvironment()
	g.evaluatepopulation()
	g.population.sort(key=operator.attrgetter('fitness'))
	g.newpopulation=g.population[0:10]
	while len(g.newpopulation)<g.pop_size:
		g.newpopulation.append(g.crossover())
	g.mutation()
	g.population=copy.copy(g.newpopulation)
		