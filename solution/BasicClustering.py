# Group 13 
# Andres Graterol
# Nicolas Leocadio
# Christopher Hinkle
# ---------------------------
import re
import numpy as np
import matplotlib.pyplot as plt

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
        
    print(entries)
    
    nodes = np.zeros(len(entries))
    X = np.zeros(len(entries))
    Y = np.zeros(len(entries))
    
    # Split each string into 3 substrings: nodes, X, and Y
    index = 0
    for entry in entries:
        # Remove extra whitespace just in case
        entry = re.sub(' +', ' ', entry)
        data = entry.split(' ')
        
        nodes[index] = int(data[0])
        X[index] = int(data[1])
        Y[index] = int(data[2])
        
        index += 1
        
    plt.scatter(X, Y)    
    
def main():
    # Using the most basic symmetric TSP file: a280.tsp
    # optimal length: 2579
    tsp_file = '../testCases/a280.tsp' 
    preprocess_data_from_file(tsp_file)
    
main()