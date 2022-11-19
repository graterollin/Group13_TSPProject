# Group 13 
# Andres Graterol
# Christopher Hinkle
# Nicolas Leocadio
# ---------------------------
# %% 
from cProfile import label
import re
import numpy as np
import matplotlib.pyplot as plt
from fcmeans import FCM
import math
import timeit

#Equation 2 figure 1
def S(i,j):
    return (1 - ((math.dist(i,j) * math.dist(i,j))/2))
    

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

    return nodes, X
    

# DATA VISUALIZATION 
def visualize_data(X, num_clusters, m):
        # Call FCM with the optimal m and number of clusters found by our algorithm 
        optimal_model = FCM(n_clusters=num_clusters, m=m)
        optimal_model.fit(X)
        centers = optimal_model.centers

        # Obtain hard and soft labels for visualization 
        labels = optimal_model.predict(X)
        print("Labels:", labels)
        soft = optimal_model.soft_predict(X)

        alphas = list(map(max,soft[:]))
        #print(alphas)

        # Visualize data with and without clusters
        f, axes = plt.subplots(1, 2, figsize=(11,5))
        axes[0].scatter(X[:,0], X[:,1], alpha=1)
        axes[1].scatter(X[:,0], X[:,1], c=labels, alpha=alphas)
        axes[1].scatter(centers[:,0], centers[:,1], marker="+", s=500, c='black')
        plt.show()

        return None
        
         
def fcm(X, num_clusters, m, centers=0):
        my_model = FCM(cluster_centers=centers, n_clusters=num_clusters, m=m)
        my_model.fit(X)
        memDegree = my_model.soft_predict(X)
        centers = my_model.centers

        return memDegree, centers
    
    
# H(U) Function from figure 2
def entropy(U):
    c = len(U[0]) # number of clusters
    n = len(U) # number of cities
    x = 0
    i = 0

    for i in range(n):
        for j in range(c):
            x += U[i][j] * math.log(U[i][j])
    
    if c == 1 or n == 0:
        return np.inf
    else:
        return -(1/math.log(c)) * 1/n * x
    
    
# Figure 2 from Paper
def UFL_FCM_VAL(X):
    # TODO: revert steps to normal, reduced because runtime is VERY LONG
    m_min = 1.1
    m_max = 3.1
    m_step = 0.1
    finalNumClusters = 2
    h_min = 1
    # S_min = 0.09
    # S_max = 0.99
    S_min = 0.01
    S_max = 0.99
    S_step = 0.01
    n = len(X) # number of cities
    
    c = 5 # This is a temp c; c will be created by UFL 
    
    m = m_min
    while m < m_max:
        print("m: ", m)
        # m_start = timeit.default_timer()
        S = S_min
        while S < S_max:
            print("S: ", S)
            c, C, U = UFL(X, S, m)
            
            # Apply FCM
            # U , centers = fcm(X, c, m, C)
            
            # Calculate Entropy
            h = entropy(U)
            
            if h_min > h:
                h_min = h
                finalNumClusters = c
                finalClusters = C
                finalMemDegree = U
                finalM = m

            S += S_step
        m += m_step
        # m_end = timeit.default_timer()
        # print('Time: ', stop - start)
    
    # Output:
    # Labels, Centers, Fuzzy Degree (M)
    print("Final m: ", finalM)
    print("Final C: ", finalClusters)
    print("Final c: ", finalNumClusters)

    return finalMemDegree, finalClusters, finalM, finalNumClusters

def e_dist(a, b):
    dist = np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    return dist

def UFL(X, S_min, m):
    np.set_printoptions(threshold=np.inf) # Print everything in big matrices :)
    n = len(X) # Number of cities
    c = 1 # Number of clusters, start with 1
    C = np.zeros((1,2)) # Location of cluster centers
    C[0] = X[0] + (1,1) # Start with center 1 = X_0 + 1 to avoid issues with diving by 0 later

    # First, get the max distance to calculate ratios later. Ratios need to be between 0 and 1
    initial_dist = np.zeros((n,1))
    for i in range(n):
        initial_dist[i] = e_dist(X[0], X[i])
    max_dist = np.amax(initial_dist)
    
    # Initialize U matrix with a single row for 1 cluster
    U = np.zeros((1,n))

    # Next, calculate S for each city to each cluster so that we can normalize the data from 0 to 1
    for i in range(n-1):
        S = np.zeros(c) # Initialize S with a slot for each cluster
        for k in range(c):
            S[k] = 1 - (e_dist(X[i], C[k])/max_dist)**2

        # If this city's similarity to all clusters < S_min
        # Create a new cluster centered on current city
        if np.amax(S) < S_min:
            c = c + 1
            C = np.vstack((C, X[i])) # Apend new center to C
            
        U, C = fcm(X, c, m, C)
    

    # print("memDegree: ", memDegree)
    # print("S_min: ", S_min)
    # print("m: ", m)

    return c, C, U
    
def main():
    start = timeit.default_timer()
    # Using the most basic symmetric TSP file: a280.tsp
    # optimal length: 2579
    tsp_file = '../testCases/a280.tsp'
    nodes, X = preprocess_data_from_file(tsp_file)
    
    # y, z, m = UFL_FCM_VAL(X)
    #print("Printing the labels (probability of cluster membership):\n")
    #print(y)
    #print("\nPrinting the centers:\n")
    #print(z)

    # Number of cluster is the same as the number of centers
    # num_clusters = len(z)

    # Visualize the data before & after our algorithms have run
    
    finalMemDegree, finalClusters, finalM, finalNumClusters = UFL_FCM_VAL(X)
    
    # num_clusters = 4
    # m = 1.1
    # visualize_data(X, num_clusters, m) 
    visualize_data(X, finalNumClusters, finalM) 

    stop = timeit.default_timer()
    print('Time: ', stop - start)

main()
# %%
