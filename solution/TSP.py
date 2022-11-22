from BasicClustering import *
from genAlgo import *

def probabilisticAssn(membership):
    numClusters = len(membership)
    rand = random.random()
    x = 0
    for m in range(numClusters):
        x += membership[m]
        if rand < x:
            return m

def createClusterMembership(membership, numClusters):
    citiesPerCluster = [[] for _ in range(numClusters)]

    for i, probabilities in enumerate(membership):
        m = probabilisticAssn(probabilities)
        citiesPerCluster[m].append(i)

    return citiesPerCluster

def main():
    start = timeit.default_timer()
    # Using the most basic symmetric TSP file: a280.tsp
    # optimal length: 2579
    tsp_file = '../testCases/lin105.tsp'
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
    for i in range(finalNumClusters):
        bestChromosome = gaForCluster(cityCoordinates, citiesPerCluster[i])
        bestChromosomes.append(bestChromosome)

    print(bestChromosomes)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

main()