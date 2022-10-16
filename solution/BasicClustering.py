# Group 13 
# Andres Graterol
# TODO: INSERT NAME HERE
# TODO: INSERT NAME HERE
# ---------------------------
import re
import numpy as np
import matplotlib.pyplot as plt
from fcmeans import FCM

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
    print(X.shape)
    print(X)
    plt.scatter(X[:,0], X[:,1])    
    
    return nodes, X
    
# FUZZY CLUSTERING
# Create our cluster centers
def create_cluster_centers(X, num_clusters):
        # Try three just to start 
        my_model = FCM(n_clusters=num_clusters)
        my_model.fit(X)
        
        return my_model.centers
        
# TODO: Implement the objective function from:
# https://towardsdatascience.com/fuzzy-c-means-clustering-with-python-f4908c714081   
    
def main():
    # Using the most basic symmetric TSP file: a280.tsp
    # optimal length: 2579
    tsp_file = '../testCases/a280.tsp' 
    nodes, X = preprocess_data_from_file(tsp_file)
    
    # TODO: Experiment with different number of clusters
    num_clusters = 3
    centers = create_cluster_centers(X, num_clusters)
    print(centers)
    
main()