# Group 13 
# Andres Graterol
# Christopher Hinkle
# Nicolas Leocadio
# ---------------------------
# Functions for inputting location data and creating clusters
# ---------------------------
# from cProfile import label
import re
import numpy as np
import matplotlib.pyplot as plt
from fcmeans import FCM
import math

# DATA PREPROCESSING
def preprocess_data_from_file(filepath): 
    # Parsing the content from the file         
    file = open(filepath)
    
    content = file.readlines()
    length_of_file = len(content)
    
    entries = content[6:length_of_file-1]

    # Remove leading and trailing characters
    for i in range(len(entries)):
        entries[i] = entries[i].strip()
        
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
        centers = my_model.centers

        return my_model, centers
    
    
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
        ret = -(1/math.log(c)) * 1/n * x
        return ret
    
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
    S_max = 0.95
    S_step = 0.01
    n = len(X) # number of cities
    
    m = m_min
    while m <= m_max:
        S = S_min
        while S <= S_max:
            c, C, U = UFL(X, S, m)
            
            # Calculate Entropy
            h = entropy(U)
            
            if h_min > h:
                print("updating best. C: ", c, ", S: ", S, ", m: ", m, ", entropy: ", h)
                h_min = h
                finalNumClusters = c
                finalCenters = C
                finalMemDegree = U
                finalM = m
                finalS = S

            S += S_step
        m += m_step

    # Output:
    # Labels, Centers, Fuzzy Degree (M)
    print("Final m: ", finalM)
    print("Final C: ", finalCenters)
    print("Final num of clusters: ", finalNumClusters)

    return finalMemDegree, finalCenters, finalM, finalNumClusters

def e_dist(a, b):
    dist = np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    return dist

def UFL(X, S_min, m):
    n = len(X) # Number of cities
    c = 1 # Number of clusters, start with 1
    C = np.zeros((1,2)) # Location of cluster centers
    C_norm = np.zeros((1,2)) # Location of cluster centers
    
    X_max = np.amax(X, axis=0)
    X_norm = X[:]/X_max
    C[0] = X[0]
    C_norm[0] = X_norm[0]
    
    # Next, calculate S for each city to each cluster so that we can normalize the data from 0 to 1
    for i in range(n):
        S = np.zeros(c) # Initialize S with a slot for each cluster
        for k in range(c):
            S[k] = 1 - (np.linalg.norm(X_norm[i] - C_norm[k])**2)/2
            
        # If this city's similarity to all clusters < S_min
        # Create a new cluster centered on current city
        if np.amax(S) < S_min:
            c = c + 1
            C = np.vstack((C, X[i])) # Apend new center to C
        
            # Get centers to keep calculating thresholds 
            fcm_model, C = fcm(X, c, m, C)
            C_norm = C[:]/X_max
            
    if c == 1:
        fcm_model, C = fcm(X, c, m, C)
    U = fcm_model.soft_predict(X)
    
    return c, C, U
