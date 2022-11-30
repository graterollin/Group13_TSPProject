# Group 13 
# Andres Graterol
# Christopher Hinkle
# Nicolas Leocadio
# ---------------------------
import numpy as np
# import matplotlib.pyplot as plt
# from fcmeans import FCM
import math
import random

# Base chromosome will be a list of all of the cities for the given cluster
def createBasechromosome(nodes,labels,targetCluster):
    chromosome = []
    
    for i, cluster in enumerate(labels):
        if cluster == targetCluster:
            chromosome.append(int(nodes[i]))
            
    return chromosome

#-------------------------------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------------------------------

def getFitnessScore(chromosome, cityCoord): #returns total distance of the tour
    totalDistance = 0
    startGene = chromosome[0]
    prevGene = startGene
    
    # for gene in chromosome[1:]:
    #     totalDistance += math.dist(cityCoord[prevGene], cityCoord[gene])
    #     prevGene = gene
    
    # totalDistance += math.dist(cityCoord[prevGene],cityCoord[startGene])

    ndx = np.array(chromosome[1:])
    copy_coords = np.array(cityCoord)
    chromosome_coords = copy_coords[ndx]
    totalDist = np.sum(np.sqrt(np.sum(np.square(np.diff(chromosome_coords, axis=0)), axis=1)))
    totalDist += np.sum(np.sqrt(np.sum(np.square(np.diff([chromosome_coords[-1],chromosome_coords[0]], axis=0)), axis=1)))

    # print("dists: ", totalDistance, totalDist, ", diff: ", totalDistance - totalDist)
    
    # return int(totalDistance)
    return int(totalDist)

#-------------------------------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------------------------------

def inverseMutation(chromosome):

    num1 = random.randint(0,len(chromosome) - 1)
    num2 = num1
    while num2 == num1: #This while loop gurantees that the two random number choosen will not be the same
        num2 = random.randint(0,len(chromosome) - 1)
    
    if num1 < num2:
        A = num1
        B = num2
    else:
        B = num1
        A = num2
    
    B += 1
    print(A, B)

    chromosome[A:B] = chromosome[A:B][::-1]

#-------------------------------------------------------------------------------------------------------

def sortFn(item):
    return item[0]

#-------------------------------------------------------------------------------------------------------

def sortPopulation(pop, cityCoord):
    
    sortedPop = []
    # Call the fitness function for each chromosome in the population
    for city in pop:
        distance = getFitnessScore(city, cityCoord)
        sortedPop.append([distance, city])

    #sorted(l_key, l_key[0])
    sortedPop.sort(key=sortFn)

    return sortedPop

#-------------------------------------------------------------------------------------------------------

def tournamentSelection(population, cityCoord, l_sorted):
    # Number of cities in the population/population size
    N = len(population)
    # print("Size of the population: ", N)

    # Indices that map to individual (chromosome in a population)
    pop = population.copy()

    tournamentWinners = []

    k = 0
    l = 0
    #j = 0
    while (l < N):
        C1 = random.choice(pop)
        
        m = 1
        while (m < k):
            # Condition to break out of loop if we are in an invalid index range
            #if ((j+m) >= N):
            #    break

            C2 = random.choice(pop)
            if (getFitnessScore(C1, cityCoord) > getFitnessScore(C2, cityCoord)):
                C1 = C2
            
            m += 1

        pop.remove(C1)
        parentPair = [C1,l_sorted[l][1]]
        tournamentWinners.append(parentPair)
        
        l += 2
        k += 1
        #j += 2

    return tournamentWinners

#-------------------------------------------------------------------------------------------------------

def condition1(sortedPop):
    N = len(sortedPop)
    bestDist = sortedPop[0][0]
    numberOfBest = 0
    
    #print('Current Optimal path length: ', bestDist)
    
    for dist, gene in sortedPop:
        if dist == bestDist:
            numberOfBest += 1
        else:
            break
    
    val = (numberOfBest / N) * 100
    
    if val < 95:
        return True
    else:
        return False

#-------------------------------------------------------------------------------------------------------

def gaForCluster(cityCoordinates, baseChromo):
    prob_cross = .8
    prob_mut = .02
    t_max = 100
    t = 0
    
    pop = generateInitialPop(baseChromo)   #randomly generate population P(0)
    sortedPop = sortPopulation(pop,cityCoordinates)
    
    while condition1(sortedPop) and t < t_max:
        parents = tournamentSelection(pop, cityCoordinates, sortedPop)
        children = []
        
        for p1, p2 in parents:
            if random.random() < prob_cross:
                child1, child2 = pMX(p1, p2)
            else:
                child1, child2 = p1.copy() , p2.copy()
            
            if random.random() < prob_mut:
                swapMutation(child1)
                swapMutation(child2)
                
            children.append(child1)
            children.append(child2)
        
        if len(children) > len(pop): # If we have too many children get rid of the last one
            children.pop()
            
        elif len(children) < len(pop): #If we are missing a child we choose random chromosome to live on
            children.append(random.choice(pop))
        
        pop = children
        sortedPop = sortPopulation(pop, cityCoordinates)
        t += 1
    # print("cluster tour length: ", sortedPop[0][0])
    return sortedPop[0][0], sortedPop[0][1] # Returns the shortes tour that the GA found


def needRotate(X,a,b):
    lastIndex = len(X) - 1
    if a == 0 and b == lastIndex or b == 0 and a == lastIndex:
        return False
    else:
        return True



def mergeTour(finalTour, nextTour, city_coords):
    A1, A2, B1, B2 = findClosestEdge(finalTour,nextTour,city_coords)
    cityA1 = finalTour[A1]
    cityA2 = nextTour[A2]

    if needRotate(finalTour, A1,B1):
        if A1 < B1:
            ind = A1
        else:
            ind = B1
        rollamount = len(finalTour) - 1 - ind
        print('roll Amount', rollamount)
        finalTour = list(np.roll(finalTour,rollamount))

    print ('finaltour after rotate',finalTour)
        
    if needRotate(nextTour, A2, B2):
        if A2 < B2:
           ind = A2
        else:
          ind = B2
    
        rollamount = len(nextTour) - 1 - ind
        print('roll Amount', rollamount)
        nextTour = list(np.roll(nextTour,rollamount))

    print ('nexttour after rotate',nextTour)

    if finalTour[-1] == cityA1 and nextTour[0] == cityA2 or finalTour[0] == cityA1 and nextTour[-1] == cityA2:
        finalTour = finalTour + nextTour
    else:
        nextTour.reverse()
        finalTour = finalTour + nextTour
    
    print(finalTour)




def connectClusters(clusterTour, sub_tours, cityCoordinates):
    finalTour = sub_tours[0]
    for i in range(1,len(clusterTour)):
        nextTour = sub_tours[i]
        mergeTour(finalTour, nextTour,cityCoordinates)

    return finalTour
        

def gaForClusterCenters(clusterCenters):
    prob_cross = .8
    prob_mut = .02
    t_max = 100
    t = 0
    
    pop = heuristicInitialization(clusterCenters)   #randomly generate population P(0) figure 5

    sortedPop = sortPopulation(pop, clusterCenters)
    
    while condition1(sortedPop) and t < t_max:
        parents = tournamentSelection(pop, clusterCenters, sortedPop)
        children = []
        
        for p1, p2 in parents:
            if random.random() < prob_cross:
                child1, child2 = pMX(p1, p2)
            else:
                child1, child2 = p1.copy() , p2.copy()
            
            if random.random() < prob_mut:
                inverseMutation(child1)
                inverseMutation(child2)
                
            children.append(child1)
            children.append(child2)
        
        if len(children) > len(pop): # If we have too many children get rid of the last one
            children.pop()
            
        elif len(children) < len(pop): #If we are missing a child we choose random chromosome to live on
            children.append(random.choice(pop))
        
        pop = children
        sortedPop = sortPopulation(pop, clusterCenters)
        t += 1
    # print("cluster tour length: ", sortedPop[0][0])
    return sortedPop[0][0], sortedPop[0][1] # Returns the shortes tour that the GA found
   

#-------------------------------------------------------------------------------------------------------

# return A1,A2 and B1,B2 (connect a1 to a2, b1 to b2)
# def connectClusters(cluster_tour, centers, sub_tours):
def findClosestEdge(cluster1, cluster2, city_coords):
    min_dist = math.inf
    A = [0,0]
    B = [0,0]
    flip = 0
    for i, city_a1 in enumerate(cluster1):
        if i == len(cluster1)-1:
            k = -1
        else:
            k = i+1
        city_b1 = cluster1[k]
            
        for j, city_a2 in enumerate(cluster2):
            if j == len(cluster2)-1:
                l = -1
            else:
                l = i+1
            city_b2 = cluster2[l]
            
            dist_forward = math.dist(city_coords[city_a1], city_coords[city_a2]) + math.dist(city_coords[city_b1], city_coords[city_b2])
            dist_backward = math.dist(city_coords[city_a1], city_coords[city_b2]) + math.dist(city_coords[city_b1], city_coords[city_a2])
            cur_dist = math.dist(city_coords[city_a1], city_coords[city_b1]) + math.dist(city_coords[city_a2], city_coords[city_b2])
            dist_forward -= cur_dist
            dist_backward -= cur_dist
            
            if dist_forward < dist_backward:
                if dist_forward < min_dist:
                    min_dist = dist_forward
                    # A = [i, j]
                    # B = [k, l]
                    A1 = i
                    A2 = j
                    B1 = k
                    B2 = l
            else:
                if dist_backward < min_dist:
                    min_dist = dist_backward
                    # A = [i, l]
                    # B = [k, j]
                    A1 = i
                    A2 = l
                    B1 = k
                    B2 = j
                    flip = 1

    return A1, A2, B1, B2, flip

#-------------------------------------------------------------------------------------------------------

# def main():
#     # Using the most basic symmetric TSP file: a280.tsp
#     # optimal length: 2579
#     tsp_file = '../testCases/lin105.tsp'
#     nodes, cityCoordinates = preprocess_data_from_file(tsp_file)
    
#     # TODO: Experiment with different number of clusters
#     num_clusters = 5
    
#     labels, centers = createClusters(cityCoordinates, num_clusters)
#     nodes -= 1 #node name will now correlate to the index in cityCoordinates
    
#     chromo0 = createBasechromosome(nodes,labels,0)
#     #print(chromo0)
#     #print(getFitnessScore(chromo0,cityCoordinates))
#     pop0 = generateInitialPop(chromo0)
#     #print("Shape of the initial population: ", np.array(pop0).shape)
#     sortedPop = sortPopulation(pop0, cityCoordinates)
    
#     bestChromosome = gaForCluster(nodes,labels,cityCoordinates,0)

    
    # print("Shape of the winners: ", np.array(tournamentWinners).shape)
    # initializing the list
    # random_list = ['A', 'A', 'B', 'C', 'B', 'D', 'D', 'A', 'B']
    # frequency = {}
    
    # print(tournamentWinners)
    # # printing the frequency
    # print(frequency)
    
#main()

if __name__ == "__main__":
    tour1 = [5,2,11,8, 13,15,16,18]
    tour2 = [1,12,12,12,21,3]
    A1 = 2
    A2 = 3

    B1 = 5
    B2 = 1

    mergeTour(tour1,tour2,A1,B1,A2,B2)
