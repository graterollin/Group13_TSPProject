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
    

# Function for visualizing data 
def visualize_data(X, labels, centers):
        # Visualize data without clusters...
        plt.scatter(X[:,0], X[:,1])    
        plt.show()

        # ... Now visualize with clusters
        f, axes = plt.subplots(1, 2, figsize=(11,5))
        axes[0].scatter(X[:,0], X[:,1], alpha=1)
        axes[1].scatter(X[:,0], X[:,1], c=labels, alpha=1)
        axes[1].scatter(centers[:,0], centers[:,1], marker="+", s=500, c='black')
        plt.show()
        
        return None
        
         
def fcm(X, num_clusters, m):
        my_model = FCM(n_clusters=num_clusters, m=m)
        my_model.fit(X)
        memDegree = my_model.soft_predict(X)
        centers = my_model.centers

        return memDegree, centers
    
    
# H(U) Function from figure 2
def entropy(U):
    c = len(U[1]) # number of clusters
    n = len(U[0]) # number of cities
    x = 0
    i = 0

    for i in range(n):
        for j in range(c):
            x += U[i][j] * math.log(U[i][j])
            
    return -(1/math.log(c)) * 1/n * x
    
    
# Figure 2 from Paper
def UFL_FCM_VAL(X):
    # TODO: revert steps to normal, reduced because runtime is VERY LONG
    m_min = 1.1
    m_max = 3.1
    m_step = 0.5
    finalNumClusters = 2
    h_min = 1
    S_min = 0.09
    S_max = 0.99
    S_step = 0.1
    n = len(X) # number of cities
    
    c = 5 # This is a temp c; c will be created by UFL 
    
    # For loop doesnt work for floats :(
    # TODO: Find better iteration method, this might be good enough
    # for m in range(m_min, m_max, m_step):
    m = m_min
    while m < m_max:
        # TODO same.
        # for S in range(S_min, S_min, S_step):
        S = S_min
        while S < S_max:
            # TODO : Apply UFL
            # UFL will be passed S and X and will return c which is the optimal number of clusters for that S
            c, C, U = UFL(X, S_min, m)
            
            # Apply FCM
            U , centers = fcm(X, c, m)
            
            # Calculate Entropy
            h = entropy(U)
            
            if h_min > h:
                h_min = h
                finalNumClusters = c
                finalClusters = centers
                finalMemDegree = U

            S += S_step
        m += m_step
    
    return finalMemDegree, finalClusters

'''
Function e_dist()
Calculates euclidean distance between two given points a and b with [x,y] coordinates

Params (a, b): points  with shape [x,y]
Returns: dist(euclidean distance between points)
'''
def e_dist(a, b):
    dist = np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    return dist

'''
FIGURE 1
Function ufl()

Determines ideal number of clusters c for given Xs, S_min, and m
Also creates a U matrix of membership degrees, with shape [c X n], c = clusters, n = # of nodes

Params (nodes, X, S_min, m):
        node names, x&y coords of nodes, minimum S(threshold) for new cluster, fuzziness variable
Returns: c(# of clusters),  C(cluster centers X), U(membership matrix of clusters)
'''
def UFL(X, S_min, m):
    np.set_printoptions(threshold=np.inf) # Print everything in big matrices :)
    n = len(X) # Number of cities
    c = 1 # Number of clusters, start with 1
    C = np.zeros((1,2)) # Location of cluster centers
    C[0] = X[0] + (1,1) # Start with center 1 = X_0 + 1 to avoid issues with diving by 0 later

    # First, get the max distance to calculate ratios later. Ratios need to be between 0 and 1
    initial_dist = np.zeros((n,1))
    for i in range(n-1):
        initial_dist[i] = e_dist(X[0], X[i])
    max_dist = np.amax(initial_dist)
    
    # Initialize U matrix with a single row for 1 cluster
    U = np.zeros((1,n))

    # Next, calculate S for each city to each cluster
    for i in range(n-1):
        S = np.zeros(c) # Initialize S with a slot for each cluster
        for k in range(c):
            S[k] = 1 - (e_dist(X[i], C[k])/max_dist)**2

        # If max S for current city is < S_min (threshold)
        # (aka, if this city's similarity to all clusters < S_min)
        # Then create a new cluster centered on current city
        if np.amax(S) < S_min:
            c = c + 1
            U = np.vstack((U, np.zeros((1, n)))) # Append new cluster to U
            C = np.vstack((C, X[i])) # Apend new center to C
        else:
            # If max S > threshold, then update similarity score for each cluster
            # TODO: Maybe this can be skipped if it was updated in previous iteration 
            #       without any new centers being added to reduce runtime
            for j in range(c):
                for k in range(n-1):
                    u_jk = 0
                    for l in range(c):
                        # TODO: Store distances in an array each time center is updated for better runtime
                        #       The above todo might do the same, maybe this is still worth it
                        dist_to_j_cluster = e_dist(X[k],C[j])/max_dist
                        dist_to_l_cluster = e_dist(X[k],C[l])/max_dist
                        
                        if dist_to_l_cluster == 0: # Don't divide by 0 (if X = C, e_dist = 0, but "similarity score" = 1)
                            temp = 1
                        elif dist_to_j_cluster == 0: # Don't set similarity scores to 0
                            temp = 0.00001
                        else:
                            temp = (dist_to_j_cluster)/(dist_to_l_cluster)
                        # TODO: Maybe above if statements can set entire value of u_jk instead?
                        u_jk = u_jk + (temp**(2/m-1))
                    U[j][k] = u_jk**(-1)

                # TODO:
                # Now update the center based on memberships
                # C[j] = C[j] + 

                # This is old code, fixing with todo above. Left in to remember what was happening.
                # u_sum = np.sum(np.power(U[j,:],m))
                # u_sum_x = np.sum(np.power(U[j,:],m)*X[:,0])
                # u_sum_y = np.sum(np.power(U[j,:],m)*X[:,1])
                # C[j] = np.array([u_sum_x/u_sum, u_sum_y/u_sum])

    # print("C: ", C)
    # print("C shape: ", C.shape)
    # print("U: ", U)
    # print("U shape: ", U.shape)

    return c, C, U
    
def main():
    start = timeit.default_timer()
    # Using the most basic symmetric TSP file: a280.tsp
    # optimal length: 2579
    tsp_file = '../testCases/a280.tsp'
    nodes, X = preprocess_data_from_file(tsp_file)
    
    y , z = UFL_FCM_VAL(X)
    print("Printing the labels (probability of cluster membership):\n")
    print(y)
    print("\nPrinting the centers:\n")
    print(z)

    # Visualize the data before & after our algorithms have run
    #visualize_data(X, y, z) 

    stop = timeit.default_timer()
    print('Time: ', stop - start)

main()
# %%
