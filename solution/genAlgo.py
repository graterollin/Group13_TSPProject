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
        population.append(chromosome)
    
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


def tournamentSelection():
    return None
        


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
    print(chromo0)
    print(getFitnessScore(chromo0,cityCoordinates))
    pop0 = generateInitialPop(chromo0)
    
main()
