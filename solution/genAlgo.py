# Group 13 
# Andres Graterol
# Christopher Hinkle
# Nicolas Leocadio
# ---------------------------
import numpy as np
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

def getFitnessScore(chromosome, cityCoord, forCenters=False): #returns total distance of the tour
    totalDist = 0

    ndx = np.array(chromosome)
    copy_coords = np.array(cityCoord)
    chromosome_coords = copy_coords[ndx]
    totalDist = np.sum(np.sqrt(np.sum(np.square(np.diff(chromosome_coords, axis=0)), axis=1)))
    if forCenters:
        totalDist += np.sum(np.sqrt(np.sum(np.square(np.diff([chromosome_coords[-1],chromosome_coords[0]], axis=0)), axis=1)))

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

    chromosome[A:B] = chromosome[A:B][::-1]

#-------------------------------------------------------------------------------------------------------

def sortFn(item):
    return item[0]

#-------------------------------------------------------------------------------------------------------

def sortPopulation(pop, cityCoord, forCenters=False):
    
    sortedPop = []
    # Call the fitness function for each chromosome in the population
    for city in pop:
        distance = getFitnessScore(city, cityCoord, forCenters)
        sortedPop.append([distance, city])

    sortedPop.sort(key=sortFn)

    return sortedPop

#-------------------------------------------------------------------------------------------------------

def tournamentSelection(population, cityCoord, l_sorted, forCenters=False):
    # Number of cities in the population/population size
    N = len(population)

    # Indices that map to individual (chromosome in a population)
    pop = population.copy()

    tournamentWinners = []

    k = 0
    l = 0
    while (l < N):
        C1 = random.choice(pop)
        
        m = 1
        while (m < k):
            C2 = random.choice(pop)
            if (getFitnessScore(C1, cityCoord, forCenters) > getFitnessScore(C2, cityCoord, forCenters)):
                C1 = C2
            
            m += 1

        pop.remove(C1)
        parentPair = [C1,l_sorted[l][1]]
        tournamentWinners.append(parentPair)
        
        l += 2
        k += 1

    return tournamentWinners

#-------------------------------------------------------------------------------------------------------

def condition1(sortedPop):
    N = len(sortedPop)
    bestDist = sortedPop[0][0]
    numberOfBest = 0
    
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

def gaForCluster(cityCoordinates, baseChromo=None, pop=None, forCenters=False):
    prob_cross = .8
    prob_mut = .02
    t_max = 100
    t = 0
    
    if not forCenters:
        pop = generateInitialPop(baseChromo)   #randomly generate population P(0)
    sortedPop = sortPopulation(pop,cityCoordinates, forCenters)
    
    while condition1(sortedPop) and t < t_max:
        parents = tournamentSelection(pop, cityCoordinates, sortedPop, forCenters)
        children = []
        # bestRoute = sortedPop[0][1]
        # x = [cityCoordinates[point][0] for point in bestRoute]
        # y = [cityCoordinates[point][1] for point in bestRoute]
        # x = x + [cityCoordinates[bestRoute[0]][0]]
        # y = y + [cityCoordinates[bestRoute[0]][1]]
        # plt.plot(x, y, '-o')
        # plt.show()
        for p1, p2 in parents:
            if random.random() < prob_cross:
                child1, child2 = pMX(p1, p2)
            else:
                child1, child2 = p1.copy() , p2.copy()
            
            if random.random() < prob_mut:
                if forCenters:
                    inverseMutation(child1)
                    inverseMutation(child2)
                else:
                    swapMutation(child1)
                    swapMutation(child2)
                
            children.append(child1)
            children.append(child2)
        
        if len(children) > len(pop): # If we have too many children get rid of the last one
            children.pop()
            
        elif len(children) < len(pop): #If we are missing a child we choose random chromosome to live on
            children.append(random.choice(pop))
        
        pop = children
        sortedPop = sortPopulation(pop, cityCoordinates, forCenters)
        t += 1

    return sortedPop[0][0], sortedPop[0][1] # Returns the shortes tour that the GA found
   
#-------------------------------------------------------------------------------------------------------

def needRotate(X,a,b):
    lastIndex = len(X) - 1
    if a == 0 and b == lastIndex or b == 0 and a == lastIndex:
        return False
    else:
        return True

#-------------------------------------------------------------------------------------------------------

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
        finalTour = list(np.roll(finalTour,rollamount))
        
    if needRotate(nextTour, A2, B2):
        if A2 < B2:
            ind = A2
        else:
            ind = B2
    
        rollamount = len(nextTour) - 1 - ind
        nextTour = list(np.roll(nextTour,rollamount))

    if finalTour[-1] == cityA1 and nextTour[0] == cityA2 or finalTour[0] == cityA1 and nextTour[-1] == cityA2:
        finalTour = finalTour + nextTour
    else:
        nextTour.reverse()
        finalTour = finalTour + nextTour
    
    return finalTour

#-------------------------------------------------------------------------------------------------------

def connectClusters(clusterTour, sub_tours, cityCoordinates):
    finalTour = sub_tours[clusterTour[0]]
    for i in clusterTour[1:]:
        nextTour = sub_tours[i]
        finalTour = mergeTour(finalTour, nextTour, cityCoordinates)

    return finalTour
        
#-------------------------------------------------------------------------------------------------------

# return A1,A2 and B1,B2 (connect a1 to a2, b1 to b2)
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
                l = j+1
            city_b2 = cluster2[l]
            
            dist_forward = math.dist(city_coords[city_a1], city_coords[city_a2]) + math.dist(city_coords[city_b1], city_coords[city_b2])
            dist_backward = math.dist(city_coords[city_a1], city_coords[city_b2]) + math.dist(city_coords[city_b1], city_coords[city_a2])
            cur_dist = math.dist(city_coords[city_a1], city_coords[city_b1]) + math.dist(city_coords[city_a2], city_coords[city_b2])
            dist_forward -= cur_dist
            dist_backward -= cur_dist
            
            if dist_forward < dist_backward:
                if dist_forward < min_dist:
                    min_dist = dist_forward
                    A1 = i
                    A2 = j
                    B1 = k
                    B2 = l
            else:
                if dist_backward < min_dist:
                    min_dist = dist_backward
                    A1 = i
                    A2 = l
                    B1 = k
                    B2 = j

    return A1, A2, B1, B2

#-------------------------------------------------------------------------------------------------------