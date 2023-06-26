from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

#-------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
import math
import plotly.express as px
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MiniBatchKMeans

#-------------------------------

def compute_AIC_BIC(data, maxclusters): # plot AIC and BIC curves
    n_components = np.arange(1, maxclusters)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(data) for n in n_components]

    plt.plot(n_components, [m.bic(data) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(data) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components');
    plt.show()
    plt.close()

#compute_AIC_BIC(data, 10)

#-------------------------------

# Davies-Bouldin score for K means
from sklearn.metrics import davies_bouldin_score
def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding Davies Bouldin for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the Davies Bouldin score for the kmeans model fit to the data
    '''
    # instantiate kmeans
    kmeans = KMeans(n_clusters=center)

    # Then fit the model to your data using the fit method
    model = kmeans.fit_predict(data)

    # Calculate Davies Bouldin score
    score = davies_bouldin_score(data, model)

    return score

#-------------------------------

def find_optimal_k(data):
    score_g, df = optimalK(data, nrefs=5, maxClusters=10)

    plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b');
    plt.xlabel('K');
    plt.ylabel('Gap Statistic');
    plt.title('Gap Statistic vs. K');

#-------------------------------

def plot_kmeans_score(data):

    scores = []
    centers = list(range(2,10))

    for center in centers:
        scores.append(get_kmeans_score(data, center))

    plt.plot(centers, scores, linestyle='--', marker='o', color='b');
    plt.xlabel('K');
    plt.ylabel('Davies Bouldin score');
    plt.title('Davies Bouldin score vs. K');

#-------------------------------

def tricontourf(feature_data):
    for column in feature_data:
        print(column)
        fig = plt.figure(figsize=(15,15))
        ax1 = fig.add_subplot(111)
        f = ax1.tricontourf(df['Cx'], df['Cz'], df[column], levels=50)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_aspect(1.5)
        plt.title(column)
        #plt.savefig(column+".png", dpi=150)
        plt.show()
        plt.close()

#-------------------------------

def plot_yellowbrick(data):

    # Elbow Method for K means
    # Import ElbowVisualizer
    from yellowbrick.cluster import KElbowVisualizer
    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2,10), timings=True)
    visualizer.fit(data) # Fit data to visualizer
    visualizer.show() # Finalize and render figure

    # Silhouette Score for K means
    # Import ElbowVisualizer
    from yellowbrick.cluster import KElbowVisualizer
    model = KMeans()
    # k is range of number of clusters
    visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette', timings=True)
    visualizer.fit(data) # Fit the data to the visualizer
    visualizer.show()

    from yellowbrick.cluster import SilhouetteVisualizer
    model_3clust = KMeans(n_clusters=3, random_state=42)
    sil_visualizer = SilhouetteVisualizer(model_3clust)
    sil_visualizer.fit(data)
    sil_visualizer.show()

    # Calinski-Harabasz Score for K means
    # Import ElbowVisualizer
    from yellowbrick.cluster import KElbowVisualizer
    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2,10), metric='calinski_harabasz', timings=True)
    visualizer.fit(data) # Fit the data to the visualizer
    visualizer.show() # Finalize and render the figure

    #from yellowbrick.cluster import InterclusterDistance
    # Instantiate the clustering model and visualizer
    #model = KMeans(3)
    #visualizer = InterclusterDistance(model)
    #visualizer.fit(data) # Fit the data to the visualizer
    #visualizer.show() # Finalize and render the figure

#-------------------------------

# Gap Statistic for K means
def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):

        print(k)

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):

            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)

#-------------------------------


#-------------------------------


#-------------------------------


#-------------------------------


#-------------------------------


#-------------------------------

LABEL_COLOR_MAP = {0 : 'red',
                   1 : 'green',
                   2 : 'lime',
                   3 : 'yellow',
                   4 : 'maroon',
                   5 : 'pink',
                   6 : 'blue',
                   7 : 'black',
                   8 : 'magenta'}

LABEL_COLOR_MAP = {0 : 'cluster 0',
                   1 : 'cluster 1',
                   2 : 'cluster 2',
                   3 : 'cluster 3',
                   4 : 'cluster 4',
                   5 : 'cluster 5',
                   6 : 'cluster 6',
                   7 : 'cluster 7',
                   8 : 'cluster 8'}

#-------------------------------


#-------------------------------

solution = pd.read_csv('solution.csv')

#-------------------------------

solution['label'] = 9999

#-------------------------------

solution

#| # Velocity field

df = solution[['U:0', 'U:1', 'U:2']]
compute_AIC_BIC(df, 10)



nclusters = 3

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_U.pdf")



nclusters = 4

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_U.pdf")



nclusters = 5

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_U.pdf")

#| # Eta field

df = solution[['Eta1', 'Eta2', 'Eta3', 'Eta4', 'Eta5']]
compute_AIC_BIC(df, 10)



nclusters = 3

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Eta.pdf")



nclusters = 4

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Eta.pdf")




nclusters = 5

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Eta.pdf")




nclusters = 6

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Eta.pdf")

#| # Reynolds stress field

df = solution[['R:0', 'R:1', 'R:2', 'R:3', 'R:4', 'R:5']]
#compute_AIC_BIC(df, 10)
#
#
#
#nclusters = 3
#
#algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
#labels = algo.fit(df).predict(df)
#print(labels.shape)
#label_color = [LABEL_COLOR_MAP[l] for l in labels]
#
#fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
#fig.update_traces(marker_size = 10)
#fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig.show()
#fig.write_image("clusters_"+str(nclusters)+"_R.pdf")


### THIS  IS THE CHOOSEN ONE ###
nclusters = 4

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

#fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
#fig.update_traces(marker_size = 10)
#fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig.show()
#fig.write_image("clusters_"+str(nclusters)+"_R.pdf")

print("Labels:", np.unique(labels))
solution['label'] = labels
###################################


#nclusters = 5
#
#algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
#labels = algo.fit(df).predict(df)
#print(labels.shape)
#label_color = [LABEL_COLOR_MAP[l] for l in labels]
#
#fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
#fig.update_traces(marker_size = 10)
#fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig.show()
#fig.write_image("clusters_"+str(nclusters)+"_R.pdf")

#-------------------------------

plt.scatter(solution['Cx'], solution['Cy'], c=label_color)
plt.show()
plt.close()

#-------------------------------

solution['label'].to_csv('labels.txt', sep='\t', encoding='utf-8')
#
labels = solution['label'].to_numpy()
np.savetxt('labels.out', labels, fmt='%i')   #

#| # aij field

df = solution[['aij:0', 'aij:1', 'aij:2', 'aij:3', 'aij:4', 'aij:5']]
compute_AIC_BIC(df, 10)



nclusters = 2

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_aij.pdf")



nclusters = 3

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_aij.pdf")



nclusters = 5

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_aij.pdf")

#| # Turbulent viscosity field

df = solution[['nut']]
compute_AIC_BIC(df, 10)



nclusters = 2

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_nut.pdf")



nclusters = 4

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_nut.pdf")

#| # Turbulent kinetic energy field

df = solution[['k']]
compute_AIC_BIC(df, 10)



nclusters = 3

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_k.pdf")



nclusters = 4

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_k.pdf")



nclusters = 5

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_k.pdf")



nclusters = 6

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_k.pdf")

#| # added feature Q2

df = solution[['Q2']]
compute_AIC_BIC(df, 10)



nclusters = 2

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Q2.pdf")



nclusters = 4

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Q2.pdf")

#-------------------------------

solution = pd.read_csv('added_Qi_features.csv')

#-------------------------------

solution['label'] = 9999

#-------------------------------

solution

#-------------------------------

df = solution[['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10',]]

compute_AIC_BIC(df, 10)

nclusters = 2
#############

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Qi.pdf")

solution['label'] = labels

#solution[solution['label']==0]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_0.csv")
#solution[solution['label']==0]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_1.csv")

#np.savetxt("clusters_Qi_"+str(nclusters)+"_0.csv", (solution[solution['label']==0]['label']).to_numpy(), fmt='%i')
#np.savetxt("clusters_Qi_"+str(nclusters)+"_1.csv", solution[solution['label']==1].index.tolist(), fmt='%i')

np.savetxt("clusters_Qi_"+str(nclusters)+"_labels.csv", solution['label'].to_numpy(), fmt='%i')

nclusters = 3
#############

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Qi.pdf")

solution['label'] = labels

#solution[solution['label']==0]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_0.csv")
#solution[solution['label']==1]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_1.csv")
#solution[solution['label']==2]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_2.csv")

np.savetxt("clusters_Qi_"+str(nclusters)+"_labels.csv", solution['label'].to_numpy(), fmt='%i')

nclusters = 4
#############

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Qi.pdf")

solution['label'] = labels

#solution[solution['label']==0]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_0.csv")
#solution[solution['label']==1]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_1.csv")
#solution[solution['label']==2]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_2.csv")
#solution[solution['label']==3]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_3.csv")

np.savetxt("clusters_Qi_"+str(nclusters)+"_labels.csv", solution['label'].to_numpy(), fmt='%i')

nclusters = 5
#############

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Qi.pdf")

solution['label'] = labels

#solution[solution['label']==0]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_0.csv")
#solution[solution['label']==1]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_1.csv")
#solution[solution['label']==2]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_2.csv")
#solution[solution['label']==3]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_3.csv")
#solution[solution['label']==4]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_4.csv")

np.savetxt("clusters_Qi_"+str(nclusters)+"_labels.csv", solution['label'].to_numpy(), fmt='%i')

nclusters = 6
#############

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Qi.pdf")

solution['label'] = labels

#solution[solution['label']==0]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_0.csv")
#solution[solution['label']==1]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_1.csv")
#solution[solution['label']==2]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_2.csv")
#solution[solution['label']==3]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_3.csv")
#solution[solution['label']==4]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_4.csv")
#solution[solution['label']==5]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_5.csv")

np.savetxt("clusters_Qi_"+str(nclusters)+"_labels.csv", solution['label'].to_numpy(), fmt='%i')

nclusters = 7
#############

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Qi.pdf")

solution['label'] = labels

#solution[solution['label']==0]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_0.csv")
#solution[solution['label']==1]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_1.csv")
#solution[solution['label']==2]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_2.csv")
#solution[solution['label']==3]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_3.csv")
#solution[solution['label']==4]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_4.csv")
#solution[solution['label']==5]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_5.csv")
#solution[solution['label']==6]['label'].to_csv("clusters_Qi_"+str(nclusters)+"_6.csv")

np.savetxt("clusters_Qi_"+str(nclusters)+"_labels.csv", solution['label'].to_numpy(), fmt='%i')

#-------------------------------


#-------------------------------

#c20 = pd.read_csv('clusters_2_0.csv')
#c21 = pd.read_csv('clusters_2_1.csv')

#-------------------------------

#c20.shape, c21.iloc[:,1]

#-------------------------------

#plt.scatter(solution['Cx'].iloc[c20.iloc[:,0]], solution['Cy'].iloc[c20.iloc[:,0]])
#plt.scatter(solution['Cx'].iloc[c21.iloc[:,0]], solution['Cy'].iloc[c21.iloc[:,0]])

#-------------------------------


#-------------------------------

df = solution[['Q2','Q3','Q5','Q6','Q10',]]

compute_AIC_BIC(df, 10)

nclusters = 2
#############

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Q235610.pdf")

solution['label'] = labels

#solution[solution['label']==0]['label'].to_csv("clusters_"+str(nclusters)+"_0.csv")
#solution[solution['label']==1]['label'].to_csv("clusters_"+str(nclusters)+"_1.csv")

np.savetxt("clusters_Q235610_"+str(nclusters)+"_labels.csv", solution['label'].to_numpy(), fmt='%i')

nclusters = 3
#############

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Q235610.pdf")

solution['label'] = labels

#solution[solution['label']==0]['label'].to_csv("clusters_"+str(nclusters)+"_0.csv")
#solution[solution['label']==1]['label'].to_csv("clusters_"+str(nclusters)+"_1.csv")
#olution[solution['label']==2]['label'].to_csv("clusters_"+str(nclusters)+"_2.csv")

np.savetxt("clusters_Q235610_"+str(nclusters)+"_labels.csv", solution['label'].to_numpy(), fmt='%i')

nclusters = 4
#############

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Q235610.pdf")

solution['label'] = labels

#solution[solution['label']==0]['label'].to_csv("clusters_"+str(nclusters)+"_0.csv")
#solution[solution['label']==1]['label'].to_csv("clusters_"+str(nclusters)+"_1.csv")
#solution[solution['label']==2]['label'].to_csv("clusters_"+str(nclusters)+"_2.csv")
#solution[solution['label']==3]['label'].to_csv("clusters_"+str(nclusters)+"_3.csv")

np.savetxt("clusters_Q235610_"+str(nclusters)+"_labels.csv", solution['label'].to_numpy(), fmt='%i')

nclusters = 5
#############

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Q235610.pdf")

solution['label'] = labels

#solution[solution['label']==0]['label'].to_csv("clusters_"+str(nclusters)+"_0.csv")
#solution[solution['label']==1]['label'].to_csv("clusters_"+str(nclusters)+"_1.csv")
#solution[solution['label']==2]['label'].to_csv("clusters_"+str(nclusters)+"_2.csv")
#solution[solution['label']==3]['label'].to_csv("clusters_"+str(nclusters)+"_3.csv")
#solution[solution['label']==4]['label'].to_csv("clusters_"+str(nclusters)+"_4.csv")

np.savetxt("clusters_Q235610_"+str(nclusters)+"_labels.csv", solution['label'].to_numpy(), fmt='%i')

nclusters = 6
#############

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Q235610.pdf")

solution['label'] = labels

#solution[solution['label']==0]['label'].to_csv("clusters_"+str(nclusters)+"_0.csv")
#solution[solution['label']==1]['label'].to_csv("clusters_"+str(nclusters)+"_1.csv")
#solution[solution['label']==2]['label'].to_csv("clusters_"+str(nclusters)+"_2.csv")
#solution[solution['label']==3]['label'].to_csv("clusters_"+str(nclusters)+"_3.csv")
#solution[solution['label']==4]['label'].to_csv("clusters_"+str(nclusters)+"_4.csv")
#solution[solution['label']==5]['label'].to_csv("clusters_"+str(nclusters)+"_5.csv")

np.savetxt("clusters_Q235610_"+str(nclusters)+"_labels.csv", solution['label'].to_numpy(), fmt='%i')

nclusters = 7
#############

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Q235610.pdf")

solution['label'] = labels

#solution[solution['label']==0]['label'].to_csv("clusters_"+str(nclusters)+"_0.csv")
#solution[solution['label']==1]['label'].to_csv("clusters_"+str(nclusters)+"_1.csv")
#solution[solution['label']==2]['label'].to_csv("clusters_"+str(nclusters)+"_2.csv")
#solution[solution['label']==3]['label'].to_csv("clusters_"+str(nclusters)+"_3.csv")
#solution[solution['label']==4]['label'].to_csv("clusters_"+str(nclusters)+"_4.csv")
#solution[solution['label']==5]['label'].to_csv("clusters_"+str(nclusters)+"_5.csv")
#solution[solution['label']==6]['label'].to_csv("clusters_"+str(nclusters)+"_6.csv")

np.savetxt("clusters_Q235610_"+str(nclusters)+"_labels.csv", solution['label'].to_numpy(), fmt='%i')

nclusters = 8
#############

algo = GaussianMixture(n_components=nclusters, init_params='kmeans', covariance_type="full", random_state=23, verbose=2)
labels = algo.fit(df).predict(df)
print(labels.shape)
label_color = [LABEL_COLOR_MAP[l] for l in labels]

fig = px.scatter(df, x=solution["Cx"], y=solution["Cy"], opacity = 1, color=label_color)
fig.update_traces(marker_size = 10)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image("clusters_"+str(nclusters)+"_Q235610.pdf")

solution['label'] = labels

#solution[solution['label']==0]['label'].to_csv("clusters_"+str(nclusters)+"_0.csv")
#solution[solution['label']==1]['label'].to_csv("clusters_"+str(nclusters)+"_1.csv")
#solution[solution['label']==2]['label'].to_csv("clusters_"+str(nclusters)+"_2.csv")
#solution[solution['label']==3]['label'].to_csv("clusters_"+str(nclusters)+"_3.csv")
#solution[solution['label']==4]['label'].to_csv("clusters_"+str(nclusters)+"_4.csv")
#solution[solution['label']==5]['label'].to_csv("clusters_"+str(nclusters)+"_5.csv")
#solution[solution['label']==6]['label'].to_csv("clusters_"+str(nclusters)+"_6.csv")
#solution[solution['label']==7]['label'].to_csv("clusters_"+str(nclusters)+"_7.csv")

np.savetxt("clusters_Q235610_"+str(nclusters)+"_labels.csv", solution['label'].to_numpy(), fmt='%i')

#-------------------------------


#-------------------------------

#print(solution['label'].index.tolist())

#-------------------------------


