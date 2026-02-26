import pickle
import os
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score



def gmm_clustering(x_set, y_set, gt=None, max_clusters=3):
    x_set_norm = np.array(x_set)/ 180
    y_set_norm = np.array(y_set)/ 7
    num_samples = len(x_set)
    # Step 1: Stack x and y to get trajectory vectors of shape (8, 150)
    trajectoriesp = np.stack([x_set, y_set], axis=2)  # Shape: (8, 75, 2)
    trajectories = np.stack([x_set_norm, y_set_norm], axis=2)  # Shape: (8, 75, 2)
    X = trajectories.reshape(num_samples, -1)  # Shape: (8, 150)

    pca = PCA(n_components=5)  # Yes, 2–3 max!
    X_pca = pca.fit_transform(X)  # (8, 2)

    # Step 2: Try GMMs with different K values
    lowest_bic = np.inf
    best_gmm = None
    bic_scores = []
    K_range = range(1, max_clusters+1)

    for k in K_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, reg_covar=0.9e-4)
        gmm.fit(X_pca)
        bic = gmm.bic(X_pca)
        bic_scores.append(bic)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    # Step 3: Predict cluster labels using the best model
    labels = best_gmm.predict(X_pca)  
    n_clusters = best_gmm.n_components
    #print(f"Best number of clusters: {n_clusters}")
    return trajectoriesp, labels, n_clusters, K_range, bic_scores
    
  
def agg_clustering(x_set, y_set, max_clusters=6):
    x_set_norm = np.array(x_set)/ 400
    y_set_norm = np.array(y_set)/ 6
    # Step 1: Stack x and y to get trajectory vectors of shape (8, 150)
    trajectoriesp = np.stack([x_set, y_set], axis=2)  # Shape: (8, 75, 2)
    trajectories = np.stack([x_set_norm, y_set_norm], axis=2)  # Shape: (8, 75, 2)
    # Assume trajectories is shape (8, 75, 2)
    X = np.stack([traj.reshape(-1) for traj in trajectories])  # Shape: (8, 150)

    sil_scores = []
    K_range = range(2, max_clusters+1)  # Can't have more clusters than samples

    best_score = -1
    best_k = 2
    best_labels = None

    for k in K_range:
        clustering = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
        labels = clustering.fit_predict(X)
        score = silhouette_score(X, labels)
        sil_scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    

    print(f"Best number of clusters: {best_k}, silhouette score: {best_score:.3f}")
   
    # Plot Silhouette Scores
    plt.plot(K_range, sil_scores, marker='o')
    plt.title("Silhouette Score vs. Number of Clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()


    # Step 5: Plot clustered trajectories
    colors = plt.cm.get_cmap('tab10', best_k)

    plt.figure(figsize=(10, 6))
    for i in range(len(trajectories)):
        cluster_id = best_labels[i]
        traj = trajectoriesp[i]  # shape: (75, 2)
        plt.plot(traj[:, 0], traj[:, 1], color=colors(cluster_id), label=f"Cluster {cluster_id}")
    plt.ylim(6,-6)
    plt.xlabel("Normalized x")
    plt.ylabel("Normalized y")
    plt.axhline(y=-1.75, color='r', linestyle='-')
    plt.axhline(y=1.75, color='r', linestyle='-')
    plt.title("Agglomerative-Clustering of Trajectories")
    plt.grid(True)
    plt.show()



