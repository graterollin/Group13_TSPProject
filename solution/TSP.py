# Group 13 
# Andres Graterol
# Christopher Hinkle
# Nicolas Leocadio
# ---------------------------
# Main entrypoint of TSP program/algorithm
# ---------------------------
from BasicClustering import *
from genAlgo import *
import timeit
import os
import numpy as np

def probabilisticAssn(membership):
    numClusters = len(membership)
    rand = random.random()
    x = 0
    for m in range(numClusters):
        x += membership[m]
        if rand < x:
            return m

#-------------------------------------------------------------------------------------------------------

def createClusterMembership(membership, numClusters):
    citiesPerCluster = [[] for _ in range(numClusters)]

    for i, probabilities in enumerate(membership):
        m = probabilisticAssn(probabilities)
        citiesPerCluster[m].append(i)

    return citiesPerCluster

#-------------------------------------------------------------------------------------------------------

def TSP(tsp_file):
    start = timeit.default_timer()
    # Using the most basic symmetric TSP file: a280.tsp
    # optimal length: 2579
    nodes, cityCoordinates = preprocess_data_from_file(tsp_file)
    nodes -=1
    
    finalMemDegree, finalCenters, finalM, finalNumClusters = UFL_FCM_VAL(cityCoordinates)
    
    # Visualize the data before & after our algorithms have run
    #visualize_data(cityCoordinates, finalNumClusters, finalM) 
    
    # Make the cluster membership from finalMemDegree
    citiesPerCluster = createClusterMembership(finalMemDegree, finalNumClusters)

    #print(citiesPerCluster)
    #print("Shape of cities per cluster: ", np.array(citiesPerCluster).shape)

    bestChromosomes = []
    bestDistances = []
    totalDistance = 0
    for i in range(finalNumClusters):
        bestDistance, bestChromosome = gaForCluster(cityCoordinates, citiesPerCluster[i])
        bestDistances.append(bestDistance)
        totalDistance += bestDistance
        bestChromosomes.append(bestChromosome)

    stop = timeit.default_timer()
    print("File: ", tsp_file, ", Tour length: ", totalDistance, ", sub tours: ", bestDistances, ", Time: ", stop - start)
    print(bestChromosomes)

#-------------------------------------------------------------------------------------------------------

def main():
    np.set_printoptions(threshold=np.inf) # Print everything in big matrices :)
    tsp_dir = "../testCases/"
    # files = os.listdir(tsp_dir)
    # for file in tsp_file:
    #     print("Running TSP for file: ", file)
    #     TSP(tsp_dir+file)

    # tsp_file = 'a280.tsp'
    # tsp_file = 'lin105.tsp'
    # tsp_file = "lin318.tsp"
    tsp_file = "att532.tsp"
    TSP(tsp_dir+tsp_file)

#-------------------------------------------------------------------------------------------------------

main()