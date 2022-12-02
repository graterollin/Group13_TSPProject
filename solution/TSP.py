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
    #print(bestChromosomes)

    for route in bestChromosomes:
        x = [cityCoordinates[point][0] for point in route]
        y = [cityCoordinates[point][1] for point in route]
        x = x + [cityCoordinates[route[0]][0]]
        y = y + [cityCoordinates[route[0]][1]]
        plt.plot(x, y, '-o')

    plt.show()
    centerpop = generateCenterpop(finalCenters)
    # centersPopMatrix, centersNumberMatrix = centersPopulation(finalCenters, citiesPerCluster, cityCoordinates)
    clusterRoute = gaForClusterCenters(finalCenters, centerpop)
    finalTour = connectClusters(clusterRoute,bestChromosomes, cityCoordinates)
    x = [cityCoordinates[point][0] for point in finalTour]
    y = [cityCoordinates[point][1] for point in finalTour]
    x = x + [cityCoordinates[finalTour[0]][0]]
    y = y + [cityCoordinates[finalTour[0]][1]]
    plt.plot(x, y, '-o')
    plt.show()
    print('finalTour',finalTour)

    print('Final Tour length:',getFitnessScore(finalTour, cityCoordinates))

#-------------------------------------------------------------------------------------------------------

def centersPopulation(centers, citiesPerCluster, cityCoordinates):
    
    centersPopulationMatrix = []
    centerNumbersMatrix = []

    centersPickList = (centers.tolist()).copy()
    iteration = 0
    while(len(centersPopulationMatrix) < len(centers)):
        centersList = centers.tolist()
        centersHold = centersList.copy()

        centersIndividual = []
        centersNumber = []

        # Pick a random c0 from the list of centers
        centerPick = random.choice(centersPickList)
        centersPickList.remove(centerPick)

        # Pick a random point from this cluster 
        c0 = random.choice(cityCoordinates[citiesPerCluster[centersList.index(centerPick)]])
        # Remove the first center choice from the list of centers 
        centersList.remove(centerPick)

        # calculate the distance from c0 to all of the centers except our original pick
        distances = []
        for d in range(len(centersList)):
            distance = math.dist(c0, centersList[d])
            distances.append(distance)

        # Keep track of indices to tie back to the centers 
        sortIndices = np.argsort(distances)
        distances = distances.sort()

        c0 = c0.tolist()
        centersIndividual.append(c0)

        if (len(centersIndividual) == len(centers)):
            centersPopulationMatrix.append(centersIndividual)
            # Replace the randomized cluster with the orginial pick 
            centersIndividual[centersIndividual.index(c0)] = centerPick
            for i, coord in enumerate(centersIndividual):
                centersNumber.append(np.where(centers == coord)[0][0])
            
            centerNumbersMatrix.append(centersNumber)
            continue
        centersHold.remove(centerPick)

        # Insert the closest center (c1) to c0
        centersIndividual.append(centersList[sortIndices[0]])
        if (len(centersIndividual) == len(centers)):
            centersPopulationMatrix.append(centersIndividual)
            centersIndividual[centersIndividual.index(c0)] = centerPick
            for i, coord in enumerate(centersIndividual):
                centersNumber.append(np.where(centers == coord)[0][0])

            centerNumbersMatrix.append(centersNumber)
            continue
        centersHold = [c for c in centersHold if c != centersList[sortIndices[0]]]

        # Insert the second closest center (c2) before c0
        centersIndividual.insert(0, centersList[sortIndices[1]])
        if (len(centersIndividual) == len(centers)):
            centersPopulationMatrix.append(centersIndividual)
            centersIndividual[centersIndividual.index(c0)] = centerPick
            for i, coord in enumerate(centersIndividual):
                centersNumber.append(np.where(centers == coord)[0][0])
                
            centerNumbersMatrix.append(centersNumber)
            continue
        centersHold = [c for c in centersHold if c != centersList[sortIndices[1]]]

        # Save the centers left after the initial steps
        centersLeft = centersHold.copy()

        while (len(centersIndividual) < len(centers)):
            # Calculate the distances from the center in the last index 
            # to all centers that have not been used yet
            distances = []

            centerToCompare = centersIndividual[-1]

            # Now we compute the distances from the last element
            for d in range(len(centersLeft)):
                distance = math.dist(centerToCompare, centersLeft[d])
                distances.append(distance)

            sortIndices = np.argsort(distances)
            distances = distances.sort()

            centersIndividual.append(centersLeft[sortIndices[0]])
            if (len(centersIndividual) == len(centers)):
                centersPopulationMatrix.append(centersIndividual)
                centersIndividual[centersIndividual.index(c0)] = centerPick
                for i, coord in enumerate(centersIndividual):
                    centersNumber.append(np.where(centers == coord)[0][0])
                
                centerNumbersMatrix.append(centersNumber)
                break
            centersHold = [c for c in centersHold if c != centersLeft[sortIndices[0]]]  

            centersIndividual.insert(0, centersLeft[sortIndices[1]])
            if (len(centersIndividual) == len(centers)):
                centersPopulationMatrix.append(centersIndividual)
                centersIndividual[centersIndividual.index(c0)] = centerPick
                for i, coord in enumerate(centersIndividual):
                    centersNumber.append(np.where(centers == coord)[0][0])

                centerNumbersMatrix.append(centersNumber)
                break
            centersHold = [c for c in centersHold if c != centersLeft[sortIndices[1]]]

        iteration += 1

    return centersPopulationMatrix, centerNumbersMatrix

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
    TSP(tsp_dir+tsp_file)
    

#-------------------------------------------------------------------------------------------------------

main()