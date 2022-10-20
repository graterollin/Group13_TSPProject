# Group 13 
# Andres Graterol
# Christopher Hinkle
# TODO: INSERT NAME HERE
# ---------------------------
# %% 
from cProfile import label
import re
import numpy as np
import matplotlib.pyplot as plt
from fcmeans import FCM
import math

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
    plt.scatter(X[:,0], X[:,1])    
    
    return nodes, X
    
    
# FUZZY CLUSTERING
# Create our cluster centers
def create_cluster_centers(X, num_clusters):
        # Try three just to start 
        my_model = FCM(n_clusters=num_clusters, m=2)
        my_model.fit(X)
        centers = my_model.centers
        labels = my_model.predict(X)
        print()

        # plot result
        f, axes = plt.subplots(1, 2, figsize=(11,5))
        axes[0].scatter(X[:,0], X[:,1], alpha=1)
        axes[1].scatter(X[:,0], X[:,1], c=labels, alpha=1)
        axes[1].scatter(centers[:,0], centers[:,1], marker="+", s=500, c='black')
        plt.show()
        
        soft = my_model.soft_predict(X)
        print(soft.shape)
        print(soft)
        
        return centers
        
         
def fcm(X, num_clusters):
        my_model = FCM(n_clusters=num_clusters, m=2)
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
    while i < n:
        j = 0
        while j < c:
            x += U[i][j] * math.log(U[i][j])
            
            j += 1
        i += 1
    return -(1/math.log(c)) * 1/n * x
    
    
# Figure 2 from Paper
def UFL_FCM_VAL(X):   
    m_min = 1.1
    m_max = 3.1
    m_pas = 0.1
    pas = 0.01
    finalNumClusters = 2
    h_min = 1
    S_min = 0.1
    S_max = 0.99
    n = len(X) # number of cities
    
    
    c = 5 # This is a temp c; c will be created by UFL 
    
    
    m = m_min
    while m < m_max:
        seuil = S_min
        while seuil < S_max:
            
            # TODO : Apply UFL
            
            # Apply FCM
            U , centers = fcm(X, c)
            
            # Calculate Entropy
            h = entropy(U)
            
            if h_min > h:
                h_min = h
                finalNumClusters = c
                finalClusters = centers
                finalMemDegree = U
            
            
            
            
            seuil += pas
        
        m += m_pas
    
    return finalMemDegree, finalClusters
    
    
# TODO: Implement the objective function from:
# https://towardsdatascience.com/fuzzy-c-means-clustering-with-python-f4908c714081   
    
def main():
    # Using the most basic symmetric TSP file: a280.tsp
    # optimal length: 2579
    tsp_file = '../testCases/a280.tsp'
    nodes, X = preprocess_data_from_file(tsp_file)
    
    # TODO: Experiment with different number of clusters
    num_clusters = 5
    #centers = create_cluster_centers(X, num_clusters)
    # print(centers)
    
    y , z = UFL_FCM_VAL(X)
    
    print(y)
    print(z)
main()

# %%
