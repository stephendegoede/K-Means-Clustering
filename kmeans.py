# K-Means clustering implementation
# The following algorithms and functions were written from scratch using
# from research into K-Means Clustering 

import math
import csv
import os
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt


# ====
# Compute the distance between two data points
def point_distance(x1, x2, y1, y2):

    distance = math.sqrt((x2-x1)**2+(y2-y1)**2)
    return distance

# ====
# Read the data in from the csv files

def open_files():
    current_path = os.path.dirname(__file__)
    
    with open(os.path.join(current_path, "data1953.csv"), 'r+') as csvfile:
        files = csv.reader(csvfile, delimiter=',', quotechar='|')
        data1953 = [row for row in files]
        data1953 = np.array(data1953)
        items1953 = data1953[1:, 1:]
        items1953 = items1953.astype('float64')


    with open(os.path.join(current_path, "data2008.csv"), 'r+') as csvfile:
        files = csv.reader(csvfile, delimiter=',', quotechar='|')
        data2008 = [row for row in files]
        data2008 = np.array(data2008)
        items2008 = data2008[1:, 1:]
        items2008 = items2008.astype('float64')


    with open(os.path.join(current_path, "dataBoth.csv"), 'r+') as csvfile:
        files = csv.reader(csvfile, delimiter=',', quotechar='|')
        dataBoth = [row for row in files]
        dataBoth = np.array(dataBoth)
        itemsBoth = dataBoth[1:, 1:]
        itemsBoth = itemsBoth.astype('float64')
        
    return data1953, data2008, dataBoth, items1953, items2008, itemsBoth


# ====
# Find the closest centroid to each point out of all the centroids

def calc_ED(items, centroids, k):

    ED = np.empty([0, k])
    row = []

    for x in range(0, len(items)):
        for y in range(0, k):
            row.append(point_distance(
                items[x][0], centroids[y][0], items[x][1], centroids[y][1]))

        full_row = np.array(row)
        ED = np.append(ED, [full_row], axis=0)
        row.clear()

    return ED

# ====
# Visualise the clusters

def plot_scatter(items, centroids, clusters, fig, ax, title):
    LABEL_COLOR_MAP = {0: 'g',
                       1: 'b',
                       2: 'y',
                       3: 'brown',
                       }

    label_color = [LABEL_COLOR_MAP[l] for l in clusters]

    ax[fig].scatter(items[:, 0], items[:, 1], c=label_color)
    ax[fig].scatter(centroids[:, 0], centroids[:, 1], c='red')
    ax[fig].set_xlabel('Birth Rate')
    ax[fig].set_ylabel('Life Expectancy')
    ax[fig].set_title(title)


# ====
# Initialisation procedure
def init(items, k):
    
    centroids = []
    m = np.shape(items)[0]

    for x in range(k):
        r = np.random.randint(0, m-1)
        centroids.append(items[r])

    centroids = np.array(centroids)

    centroids = centroids[np.argsort(centroids[:, 0])]

    return centroids


# ====
# Implement the k-means algorithm, using appropriate looping for the number of iterations
# --- find the closest centroid to each point and assign the point to that centroid's cluster
# --- calculate the new mean of all points in that cluster
# --- visualise
#---- repeat
def kmeans(data, items, k, iterations):
    stop = 0
    centroids = init(items, k)

    while (stop < iterations):
        ED = calc_ED(items, centroids, k)
        clusters = np.argmin(ED, axis=1)
        convergence = 0

        for track in range(0, k):
            index = 0
            x = 0
            y = 0
            count = 0

            for cluster in clusters:

                if cluster == track:
                    x += items[index][0]
                    y += items[index][1]
                    count += 1

                convergence += ED[index][cluster]**2
                index += 1

            x = x / count
            y = y / count

            centroids[track][0] = x
            centroids[track][1] = y

        print(f"Convergence {stop+1}: {convergence}")

        stop += 1

    print("\nNumber of Countries per Cluster")

    for x in range(0, k):
        print(f"Cluster {x+1}: {np.count_nonzero(clusters == x)}")

    print("")

    for x in range(0, k):
        print(
            f"Cluster {x+1}\nMean Birth Rate: {centroids[x][0]}\nMean Life Expectancy: {centroids[x][1]}\n")

    print("List of Countries per Cluster")

    for track in range(0, k):
        index = 1

        print(f"Cluster {track+1}\n")

        for cluster in clusters:

            if cluster == track:
                print(f"{data[index][0]}")

            index += 1

        print("")

    return items, centroids, clusters

# ====
# Print out the results for questions
# 1.) The number of countries belonging to each cluster
# 2.) The list of countries belonging to each cluster
# 3.) The mean Life Expectancy and Birth Rate for each cluster

def main():

    # k = input('Please specify value k: ')
    # iterations = input('Please specify the number of iterations: ')

    k = 4
    iterations = 6
    fig = 0

    data1953, data2008, dataBoth, items1953, items2008, itemsBoth = open_files()

    figure = plt.figure(num=None, figsize=(8,9), dpi = 80, facecolor = 'w', edgecolor = 'k')
    ax = figure.subplots(3, 1)
    figure.tight_layout(pad=4.0)
    
    items, centroids, clusters = kmeans(data1953, items1953, k, iterations)
    plot_scatter(items, centroids, clusters, fig, ax, "1953")
    fig +=1

    items, centroids, clusters = kmeans(data2008, items2008, k, iterations)
    plot_scatter(items, centroids, clusters, fig, ax, "2008")
    fig +=1

    items, centroids, clusters = kmeans(dataBoth, itemsBoth, k, iterations)
    plot_scatter(items, centroids, clusters, fig, ax, "Both")

    plt.show()

if __name__ == "__main__":
    main()