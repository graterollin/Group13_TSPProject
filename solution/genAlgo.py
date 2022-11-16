# Group 13 
# Andres Graterol
# Christopher Hinkle
# TODO: INSERT NAME HERE
# ---------------------------
from cProfile import label
import re
import numpy as np
import matplotlib.pyplot as plt
from fcmeans import FCM
import math
import random

# DATA PREPROCESSING
def preprocess_data_from_file(filepath): 
    # Parsing the content from the file         
    file = open(filepath)
    
    content = file.readlines()
    length_of_file = len(content)
    
    entries = content[6:length_of_file-1]
    #print(entries) 
    
    # Remove leading and trailing characters
    for i in range(len(entries)):
        entries[i] = entries[i].strip()
        
    #print(entries)
    
    nodes = np.zeros(len(entries))
    x = np.zeros(len(entries))
    y = np.zeros(len(entries))
    
    # Split each string into 3 substrings: nodes, x, and y
    index = 0
    for entry in entries:
        # Remove extra whitespace just in case
        entry = re.sub(' +', ' ', entry)
        data = entry.split(' ')
        
        nodes[index] = int(data[0])
        
        # X is a 2d array with x, y coordinates 
        x[index] = int(data[1])
        y[index] = int(data[2])
        
        index += 1
      
    # Combine x and y coordinates into a 2d array of shape: (len(dataset), 2)
    X = []    
    for i in range(len(x)):
        row = [x[i], y[i]]
        X.append(row)
        
    X = np.array(X) 
    #print(X.shape)
    #print(X)
    plt.scatter(X[:,0], X[:,1])    
    
    return nodes, X

def createClusters(X, numClusters):
        my_model = FCM(n_clusters=numClusters, m=2)
        my_model.fit(X)
        centers = my_model.centers
        labels = my_model.predict(X)
        
        return labels, centers

# Base chromosome will be a list of all of the cities for the given cluster
def createBasechromosome(nodes,labels,targetCluster):
    chromosome = []
    
    for i, cluster in enumerate(labels):
        if cluster == targetCluster:
            chromosome.append(int(nodes[i]))
            
    return chromosome

# Initial population is created by creating s randomly 
# shuffled chromosomes (where s = number of cities in a cluster) 
def generateInitialPop(chromosome):
    population = []
    totalChromos = len(chromosome)
    for _ in range(totalChromos):
        random.shuffle(chromosome)
        tmp = chromosome.copy()
        population.append(tmp)
    
    return population
        
def getFitnessScore(chromosome, cityCoord): #returns total distance of the tour
    totalDistance = 0
    startGene = chromosome[0]
    prevGene = startGene
    
    for gene in chromosome[1:]:
        totalDistance += math.dist(cityCoord[prevGene], cityCoord[gene])
        prevGene = gene
    
    totalDistance += math.dist(cityCoord[prevGene],cityCoord[startGene])
    
    return totalDistance            

def pMX(parent1, parent2): #Partially-Matched Crossover
    child1 = [None] * len(parent1)
    child2 = [None] * len(parent2)
    lengthOfChromo = len(parent1)
    
    cityList1 = parent1.copy()
    cityList2 = parent2.copy()
    
    num1 = random.randint(0,lengthOfChromo - 1)
    num2 = num1
    while num2 == num1: #This while loop gurantees that the two random number choosen will not be the same
        num2 = random.randint(0,lengthOfChromo - 1)
    
    if num1 < num2:
        A = num1
        B = num2
    else:
        B = num1
        A = num2
        
    for i in range(A,B+1): #Copy cities between A and B from parent 1 into child
        
        child1[i] = parent1[i]
        child2[i] = parent2[i]
        
        cityList1.remove(child1[i])
        cityList2.remove(child2[i])
        
        
    
    for i in range(lengthOfChromo):  #For childs array outside of [A,B] copy cities from parent 2 that havent been taken yet
        
        if parent2[i] not in child1 and child1[i] is None:
            child1[i] = parent2[i]
            cityList1.remove(child1[i])
            
        if parent1[i] not in child2 and child2[i] is None:
            child2[i] = parent1[i]
            cityList2.remove(child2[i])
    
    for i in range(lengthOfChromo): #Fill in the gaps with cities that havent been taken yet
        if child1[i] is None:
            child1[i] = cityList1.pop(0)
            
        if child2[i] is None:
            child2[i] = cityList2.pop(0)
        
    if None in child1 or None in child2:  #Checking that 
        print('PMX crossover did not work correclty')
    
    return child1, child2

def swapMutation(chromosome): #swap mutation function
    lengthOfChromo = len(chromosome)
    
    num1 = random.randint(0,lengthOfChromo - 1)
    num2 = num1
    while num2 == num1: #This while loop gurantees that the two random number choosen will not be the same
        num2 = random.randint(0,lengthOfChromo - 1)
    
    if num1 < num2:
        A = num1
        B = num2
    else:
        B = num1
        A = num2
        
    chromosome[A], chromosome[B] = chromosome[B], chromosome[A]
    

def tournamentSelection():
    return None
        
def gaForCluster(nodes, labels, cityCoordinates, clusterNum):
    prob_cross = .8
    prob_mut = .02
    t_max = 100
    t = 0
    
    baseChromo = createBasechromosome(nodes,labels,clusterNum)
    pop = generateInitialPop(baseChromo)   #randomly generate population P(0)
    
    # TODO: Evaluate all the indidvuals in the population
    
    # TODO: code condition1
    
    # TODO: go through this psuedo code
    # while condition1() and t < t_max:
    #     nextPop or maybe parentpairs = tournamentSelection()
    #     children = []
        
    #     for parent1, parent 2 in parentpairs:
    #         if random.random() < prob_cross:
    #             child1, child2 = pMX(parent1,parent2)
    #         else:
    #             child1, child2 = parent1, parent2
            
    #         if random.random() < prob_mut:
    #             child1 = swapMutation(child1)
    #         if random.random() < prob_mut:
    #             child2 = swapMutation(child2)
                
    #         children.append(child1)
    #         children.append(child2)
            
    #     pop = children
    

    return pop
    


def main():
    # Using the most basic symmetric TSP file: a280.tsp
    # optimal length: 2579
    tsp_file = '../testCases/a280.tsp'
    nodes, cityCoordinates = preprocess_data_from_file(tsp_file)
    
    # TODO: Experiment with different number of clusters
    num_clusters = 5
    
    labels, centers = createClusters(cityCoordinates, num_clusters)
    nodes -= 1 #node name will now correlate to the index in cityCoordinates
    

    chromo0 = createBasechromosome(nodes,labels,0)
    pop0 = generateInitialPop(chromo0)
    
main()
