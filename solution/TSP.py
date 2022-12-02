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
import random

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

def generateCenterpop(centers):

    centersPop = []
    centersindexs = [i for i in range(len(centers))]
    for i in range(len(centers)):
        clusterLeft = centersindexs.copy()
        clusterLeft.remove(i)
        chromo = [i]
        appendClosest(chromo, centers, clusterLeft)
        centersPop.append(chromo)
    
    return centersPop


#-------------------------------------------------------------------------------------------------------

def appendClosest(chromosome, clustersCoord, clustersLeft):
    rightAppend = chromosome[-1]
    leftAppend = chromosome[0]
    minDist = float('inf')
    minIndex = 0
    for i in clustersLeft:
        dist = math.dist(clustersCoord[i], clustersCoord[rightAppend])
        if dist < minDist:
            minDist = dist
            minIndex = i
    chromosome.append(minIndex)
    clustersLeft.remove(minIndex)

    if len(clustersLeft) == 0:
        return chromosome

    minDist = float('inf')
    minIndex = 0
    for i in clustersLeft:
        dist = math.dist(clustersCoord[i], clustersCoord[leftAppend])
        if dist < minDist:
            minDist = dist
            minIndex = i

    chromosome.insert(0, minIndex)
    clustersLeft.remove(minIndex)
    if len(clustersLeft) == 0:
        return chromosome
    
    return appendClosest(chromosome, clustersCoord, clustersLeft)

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

    centersPopMatrix = centersPopulation(finalCenters, citiesPerCluster, cityCoordinates)
    print("Shape of the final population matrix: ", np.array(centersPopMatrix).shape)
    print(centersPopMatrix)

#-------------------------------------------------------------------------------------------------------

# TODO: Ensure that this generalizes for multiple cluster sizes
def centersPopulation(centers, citiesPerCluster, cityCoordinates):

    #numberOfCenters = len(centers)
    centersPopulationMatrix = []
    
    centersPickList = (centers.tolist()).copy()
    iteration = 0
    while(len(centersPopulationMatrix) < len(centers)):
        centersList = centers.tolist()
        centersHold = centersList.copy()

        centersIndividual = []
        print("Size of centers pick: ", len(centersPickList), "for iteration: ", iteration)
        # Pick a random c0 from the list of centers
        #centerPick = random.choice(centersList)
        centerPick = random.choice(centersPickList)
        centersPickList.remove(centerPick)
        #centersPickList = [c for c in centersPickList if c != centerPick]
        print("We pick center:", centerPick)
        # Pick a random point from this cluster 
        c0 = random.choice(cityCoordinates[citiesPerCluster[centersList.index(centerPick)]])
        print("Point we pick: ", c0)
        # Remove the first center choice from the list of centers 
        centersList.remove(centerPick)

        # calculate the distance from c0 to all of the centers except our original pick
        distances = []
        for d in range(len(centersList)):
            distance = math.dist(c0, centersList[d])
            print("Distance between: ", c0, "and: ", centersList[d], "is: ", distance)
            distances.append(distance)

        print(distances)
        # Keep track of indices to tie back to the centers 
        sortIndices = np.argsort(distances)
        distances = distances.sort()
        print("Sort for iteration: ", iteration)
        print(sortIndices)
        print("Initial CentersHold:", centersHold)

        c0 = c0.tolist()
        centersIndividual.append(c0)
        print("current view of the individual:", centersIndividual)
        if (len(centersIndividual) == len(centers)):
            centersPopulationMatrix.append(centersIndividual)
            continue
        centersHold.remove(centerPick)
        print("current view of the centers left in centers hold: ", centersHold)

        # Insert the closest center (c1) to c0
        centersIndividual.append(centersList[sortIndices[0]])
        print("current view of the individual:", centersIndividual)
        if (len(centersIndividual) == len(centers)):
            centersPopulationMatrix.append(centersIndividual)
            continue
        print("Center to add to the individual and remove from centersHold", centersList[sortIndices[0]])
        centersHold = [c for c in centersHold if c != centersList[sortIndices[0]]]
        #centersHold.remove(centers[sortIndices[0]])
        #del centersHold[sortIndices[0]]
        print("current view of the centers left in centers hold: ", centersHold)

        # Insert the second closest center (c2) before c0
        centersIndividual.insert(0, centersList[sortIndices[1]])
        print("current view of the individual:", centersIndividual)
        if (len(centersIndividual) == len(centers)):
            centersPopulationMatrix.append(centersIndividual)
            continue
        print("Center to add to the individual and remove from centersHold", centersList[sortIndices[1]])
        centersHold = [c for c in centersHold if c != centersList[sortIndices[1]]]
        #centersHold.remove(centers[sortIndices[1]])
        #del centersHold[sortIndices[1]]
        print("current view of the centers left in centers hold: ", centersHold)

        # Save the centers left after the initial steps
        centersLeft = centersHold.copy()

        while (len(centersIndividual) < len(centers)):
            # Calculate the distances from the center in the last index 
            # to all centers that have not been used yet
            distances = []

            centerToCompare = centersIndividual[-1]

            #centersLeft.remove(centersHold[-1])
            print("Centers left:", centersLeft)
            # Now we compute the distances from the last element
            for d in range(len(centersLeft)):
                distance = math.dist(centerToCompare, centersLeft[d])
                print("Distance between ", centerToCompare, "and: ", centersLeft[d], "is: ", distance)
                distances.append(distance)

            print(distances)
            sortIndices = np.argsort(distances)
            distances = distances.sort()
            print("SUB Distance shape, distances, and sort for iteration: ", iteration)
            print(sortIndices)

            centersIndividual.append(centersLeft[sortIndices[0]])
            print("current view of the individual:", centersIndividual)
            if (len(centersIndividual) == len(centers)):
                centersPopulationMatrix.append(centersIndividual)
                break
            #centersHold.remove(centers[sortIndices[0]])
            centersHold = [c for c in centersHold if c != centersLeft[sortIndices[0]]]  
            print("current view of the centers left in centers hold: ", centersHold)

            centersIndividual.insert(0, centersLeft[sortIndices[1]])
            print("current view of the individual:", centersIndividual)
            if (len(centersIndividual) == len(centers)):
                centersPopulationMatrix.append(centersIndividual)
                break
            #centersHold.remove(centers[sortIndices[1]])
            centersHold = [c for c in centersHold if c != centersLeft[sortIndices[1]]]
            print("current view of the centers left in centers hold: ", centersHold)

        iteration += 1

    return centersPopulationMatrix

#-------------------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------------------

def main():
    np.set_printoptions(threshold=np.inf) # Print everything in big matrices :)
    tsp_dir = "../testCases/"
    # files = os.listdir(tsp_dir)
    # for file in tsp_file:
    #     print("Running TSP for file: ", file)
    #     TSP(tsp_dir+file)

    tsp_file = 'a280.tsp'
    # tsp_file = 'lin105.tsp'
    # tsp_file = "lin318.tsp"
    #tsp_file = "att532.tsp"
    # TSP(tsp_dir+tsp_file)
    centers = [(0,0), (8,2), (2,5), (0,3), (6,4)]
    chromo = [2]
    clustersleft = [0,1,3,4]
    pop = generateCenterpop(centers)
    print("Population: ", pop)
    clusterTour = gaForClusterCenters(centers, pop)
    print("Cluster Tour: ", clusterTour)
    print(getFitnessScoreCenters(clusterTour[1], centers))
    print(getFitnessScore(clusterTour[1], centers))


    

#-------------------------------------------------------------------------------------------------------

main()