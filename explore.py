from scipy.sparse.construct import random
import streamlit as st
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

CANCER = datasets.load_breast_cancer()
IRIS = datasets.load_iris()
DIGITS = datasets.load_digits()
WINE = datasets.load_wine()

CALIFORNIA = datasets.fetch_california_housing()
DIABETES = datasets.load_diabetes()

data = None

st.set_page_config(page_title="AI Algorithms", page_icon=":shark:")

#Removes the Right Hand Side Navigation Bar during deploy
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# CLASSIFICATION PAGE INPUT FUNCTIONS
def kNN_Classifier_inputs():
    col1, col2 = st.columns(2)
    n_neighbors = col1.number_input('Number of K-Nearest Neighbours', 1, 10)
    metric = col2.selectbox(
        'Metric',
        ('euclidean', 'manhattan', 'chebyshev', 'minkowski', 'seuclidian', 'mahalanobis'))
    return n_neighbors, metric

    #2. MLP
def MLP_Classifier_inputs():
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    hidden_layer_sizes = col1.number_input('Size of Hidden Layer', 1, 10)
    max_iter = col2.number_input('Maximum Iteration', 1, 10)
    random_state = col3.number_input('Random State', 0, 10, 1)
    activation = col4.selectbox(
        'Activation Function',('relu', 'identity', 'logistic', 'tanh'))
    solver = st.selectbox(
        'Solver Function',['adam','lbfgs','sgd'])
    return hidden_layer_sizes, max_iter, activation, solver, random_state

#3. Decision Tree Classifier
def Decision_Tree_Classifier_inputs():
    st.write("This is a Decision Tree")

    #  4. Support Vector Classifier
def Support_Vector_Classifier_inputs():
    kernel = st.selectbox('Choose the Support Vector Classifier Function', ('linear', 'poly', 'rbf', 'sigmoid'))
    return kernel

#CLASSIFICATION ALGORITHM FUNCTIONS
def kNN_Classifier(n_neighbors, metric='euclidean',data = data):
    # metric : euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis
    # Store the feature and target data
    X = data.data
    y = data.target
    # Split the data using Scikit-Learn's train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric = metric)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score

def MLP_Classifier(hidden_layer_sizes, max_iter, activation = "relu", solver = "adam",random_state=1, data = data):
    # activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
    # solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
    # Store the feature and target data
    X = data.data
    y = data.target

    # Split the data using Scikit-Learn's train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter = max_iter,activation = activation,solver=solver,random_state=random_state)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score

def Decision_Tree_Classifier(data = data):
    # Store the feature and target data
    X = data.data
    y = data.target

    # Split the data using Scikit-Learn's train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score

def Support_Vector_Classifier(data = data, kernel = 'linear'):
    #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}
    # Store the feature and target data
    X = data.data
    y = data.target
    # Split the data using Scikit-Learn's train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score

def classification_page(dataset_choice, data):
    st.title("Classification Algorithms")
    st.header(f"The {dataset_choice} dataset")
    st.write('---')
    class_alg_choice = st.selectbox("Select a classification algorithm: ", ['KNN', 'MLP', 
                                                                                                      'Decision Tree Classifier',
                                                                                                      'Support Vector Classifier'])
    if class_alg_choice == 'KNN':
        n_neighbors, metric = kNN_Classifier_inputs()
        score = kNN_Classifier(n_neighbors, metric, data)
    elif class_alg_choice == 'MLP':
        hidden_layer_sizes, max_iter, activation, solver, random_state = MLP_Classifier_inputs()
        score = MLP_Classifier(hidden_layer_sizes, max_iter, activation, solver, random_state, data)
    elif class_alg_choice == 'Decision Tree Classifier':
        # Decision_Tree_Classifier_inputs()        -- No inputs
        score = Decision_Tree_Classifier(data)
    elif class_alg_choice == 'Support Vector Classifier':
        kernel = Support_Vector_Classifier_inputs()
        score = Support_Vector_Classifier(data, kernel)

    st.write('---')
    st.write("Accuracy : ", score)
    st.write('---')
    st.write("A sample of 10 rows from the dataset")
    data_frame = np.c_[data.data, data.target]
    columns = np.append(data.feature_names, ["target"])
    data_frame = pd.DataFrame(data_frame, columns=columns)
    st.write(data_frame.head(10))
    st.write('---')
    st.write("Visualization of the first 20 dataset entries")
    st.bar_chart(data_frame[:20])
    
    if dataset_choice == 'Cancer':
        st.subheader('Correlation matrix of the target with some features')
        corr_matrix = data_frame[
            ['mean radius', 'radius error', 'worst radius',
            'mean perimeter', 'perimeter error', 'worst perimeter',
            'mean smoothness', 'smoothness error', 'worst smoothness', 'target'
            ]].corr()
    elif dataset_choice == 'Digits':
        st.subheader('Correlation matrix of the target with some features')
        corr_matrix = data_frame[
            ['pixel_0_0', 'pixel_0_2', 'pixel_1_4', 'pixel_2_6',
            'pixel_3_2', 'pixel_4_4', 'pixel_5_6',
            'pixel_6_2', 'pixel_7_4', 'pixel_7_6', 'target'
            ]].corr()
    else:
        st.subheader('Correlation matrix')
        corr_matrix = data_frame.corr()

    corr_fig = plt.figure(figsize=(14,7))
    sns.heatmap(corr_matrix, cmap=plt.cm.CMRmap_r, annot=True)
    st.pyplot(corr_fig)
    # Descriptive statistics
    st.write('---')
    st.subheader('Descriptive statistics')
    st.write(data_frame.describe())
    

# REGRESSION PAGE INPUTS
#1.	Linear Regression
def Linear_Regression_inputs():
    st.write("This is the linear regression model")

#2. Ridge Regression
def Ridge_Regression_inputs():
    alpha = st.slider('Enter the Alpha', 0.1, 1.0, 0.1)
    #Might add an onchanged event to scroll to accuracy
    return alpha

#  3. Support Vector Regression
def Support_Vector_Regression_inputs():
    kernel = st.selectbox('Support Vector Regression Function', ['linear', 'poly', 'rbf', 'sigmoid'])   #Removed precomputed
    return kernel

# REGRESSION ALGORITHM FUNCTIONS
def Linear_Regression(data = data):
	# Store the feature and target data
	X = data.data
	y = data.target

	# Split the data using Scikit-Learn's train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	reg = LinearRegression()
	reg.fit(X_train, y_train)
	score = reg.score(X_test, y_test)
	return(score)

def Ridge_Regression(data = data, alpha = 0.1):
	# Store the feature and target data
	X = data.data
	y = data.target

	# Split the data using Scikit-Learn's train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	reg = Ridge(alpha = alpha)
	reg.fit(X_train, y_train)
	score = reg.score(X_test, y_test)
	return(score)

def Support_Vector_Regression(data = data, kernel = 'linear'):
	#kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}

	# Store the feature and target data
	X = data.data
	y = data.target

	# Split the data using Scikit-Learn's train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	reg = SVR(kernel=kernel)
	reg.fit(X_train, y_train)
	score = reg.score(X_test, y_test)
	return(score)

# REGRESSION INPUTS + ALGORITHMS
def regression_page(dataset_choice, data):
    st.title("Regression Algorithms")
    st.header(f"The {dataset_choice} dataset")
    st.write('---')
    reg_alg_choice = st.selectbox("Select a Regression Algorithm: ", ['Linear Regression', 'Ridge Regression',
                                                                                       'Support Vector Regression'])
    if reg_alg_choice == 'Linear Regression':
        #Linear_Regression_inputs()          --> Can be left out because there are no  inputs
        score = Linear_Regression(data)
    elif reg_alg_choice == 'Ridge Regression':
        alpha = Ridge_Regression_inputs()   
        score = Ridge_Regression(data, alpha)
    elif reg_alg_choice == 'Support Vector Regression':
        kernel = Support_Vector_Regression_inputs() 
        score = Support_Vector_Regression(data, kernel)

    st.write('---')
    st.write("Accuracy : ", score) #For training
    st.write('---')
    st.write("A sample of 10 rows from the dataset")
    # st.write(data.data)
    data_frame = np.c_[data.data, data.target]
    columns = np.append(data.feature_names, ["target"])
    data_frame = pd.DataFrame(data_frame, columns=columns)
    st.write(data_frame.head(10))
    st.write("---")
    st.write("Visualization of the first 20 dataset entries")
    st.bar_chart(data_frame[:20])
    
    #Plot the correlation matrix
    st.write('---')
    st.subheader('Correlation matrix')
    corr_matrix = data_frame.corr()
    corr_fig = plt.figure(figsize=(14,7))
    sns.heatmap(corr_matrix, cmap=plt.cm.CMRmap_r, annot=True)
    st.pyplot(corr_fig)
    # Descriptive statistics
    st.write('---')
    st.subheader('Descriptive statistics')
    st.write(data_frame.describe())

#CLUSTERING PAGE INPUTS
#1. K-Means Clustering
def kMeans_Clustering_inputs():
    col1, col2 = st.columns(2)
    n_clusters = col1.number_input('Number of clusters', 1, 10)
    n_init = col2.number_input('Initial Value', 1, 20, 10)
    random_state = st.number_input('Random State', 0, 10, 1, key="random_state",)
    return n_clusters, n_init, random_state
    

# 2. DBSCAN Clustering
def DBSCAN_Clustering_inputs():
    col1, col2 = st.columns(2)
    eps = st.slider('Eps value', 0.1, 1.0)
    min_samples = col1.number_input('Minimum Number Of Samples', 1, 10, 5)
    metric = col2.selectbox(
        'Metric',
        ('euclidean', 'manhattan', 'chebyshev', 'minkowski', 'seucledian', 'mahalanobis'), key="DBSCAN metric")
    return eps, min_samples, metric

#3 Birch Clustering
def Birch_Clustering_inputs():
    col1, col2 = st.columns(2)
    n_clusters = st.number_input('Number of Clusters', 1, 10, 5)
    threshold = col1.slider('Threshhold Value', 0.1, 1.0, 0.5)
    branching_factor = col2.slider('Branching factor', 1, 100, 50)
    return n_clusters, threshold, branching_factor

# CLUSTERING ALGORITHMS
def kMeans_Clustering(n_clusters = 3,random_state = 1,n_init=10, data = data):
    X = data.data
    clustering = KMeans(n_clusters=n_clusters,random_state=random_state,n_init=n_init)
    clustering.fit(X)

def DBSCAN_Clustering(eps=0.5, min_samples=5, metric='euclidean', data = data):
    X = data.data
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    clustering.fit(X)

def Birch_Clustering( n_clusters, threshold=0.5, branching_factor=50, data = data):
    X = data.data
    clustering = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=n_clusters)
    clustering.fit(X)

def clustering_page(dataset_choice, data):
    st.title("Clustering Algorithms")
    st.header(f"The {dataset_choice} dataset")
    st.write('---')
    clust_alg_choice = st.selectbox("Clustering algorithm: ", ['k-Means', 'DBSCAN',
                                                                                       'Birch'])
    if clust_alg_choice == 'k-Means':
        n_clusters, n_init, random_state = kMeans_Clustering_inputs()
        kMeans_Clustering(n_clusters, random_state, n_init, data)

    elif clust_alg_choice == 'DBSCAN':
        eps, min_samples, metric = DBSCAN_Clustering_inputs()
        DBSCAN_Clustering(eps, min_samples, metric, data)

    elif clust_alg_choice == 'Birch':
        n_clusters, threshold, branching_factor = Birch_Clustering_inputs()
        Birch_Clustering(n_clusters, threshold, branching_factor, data)

    st.write('---')
    st.write("A sample of 10 rows from the dataset")
    data_frame = np.c_[data.data, data.target]
    columns = np.append(data.feature_names, ["target"])
    data_frame = pd.DataFrame(data_frame, columns=columns)
    st.write(data_frame.head(10))
    st.write('---')
    st.write("Visualization of the first 20 dataset entries")
    st.bar_chart(data_frame[:20])
    st.write('---')
    if dataset_choice == 'Cancer':
        st.subheader('Correlation matrix of the target with some features')
        corr_matrix = data_frame[
            ['mean radius', 'radius error', 'worst radius',
            'mean perimeter', 'perimeter error', 'worst perimeter',
            'mean smoothness', 'smoothness error', 'worst smoothness', 'target'
            ]].corr()
    elif dataset_choice == 'Digits':
        st.subheader('Correlation matrix of the target with some features')
        corr_matrix = data_frame[
            ['pixel_0_0', 'pixel_0_2', 'pixel_1_4', 'pixel_2_6',
            'pixel_3_2', 'pixel_4_4', 'pixel_5_6',
            'pixel_6_2', 'pixel_7_4', 'pixel_7_6', 'target'
            ]].corr()
    else:
        st.subheader('Correlation matrix')
        corr_matrix = data_frame.corr()

    corr_fig = plt.figure(figsize=(14,7))
    sns.heatmap(corr_matrix, cmap=plt.cm.CMRmap_r, annot=True)
    st.pyplot(corr_fig)
    # Descriptive statistics
    st.write('---')
    st.subheader('Descriptive statistics')
    st.write(data_frame.describe())

# Main Pages
def home_page():
    st.title("Welcome to AI :bulb:")

    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        st.image("https://media4.giphy.com/media/7VzgMsB6FLCilwS30v/200w.webp?cid=ecf05e470qdohkjl35f3tia0r1sz5xs67chf2zfxmq0sm4an&rid=200w.webp&ct=g")

    with col3:
        st.write("")

    st.header("The AI Algorithm Exploration Tool:fire:")
    st.write("""
    This tool has been designed as an exploratory space for Computer Science students\
    to have some intuition on algorithms\
    <span style='color: yellowgreen;'>**without having to write code!**</span>
    <br>
    
    """, unsafe_allow_html=True)
    st.write("The tool also allows the students to try out different datasets such as `California`, `Iris` and `Wine`")
    st.write("<b> Start Exploring Now! </b>", unsafe_allow_html=True)
    
def models_page():
    with st.sidebar.expander("Algorithmic Problems"):
        alg_prob_choice = st.selectbox("Select the algorithmic problem: ", ['Classification', 'Regression', 'Clustering'],)
    if alg_prob_choice == 'Classification':
        dataset_choice = st.sidebar.selectbox("Select a Dataset for Classification", ['Cancer', 'Iris', 'Digits','Wine'])
        if dataset_choice == 'Cancer':
            data = CANCER
        elif dataset_choice == 'Iris':
            data = IRIS
        elif dataset_choice == 'Digits':
            data = DIGITS
        elif dataset_choice == 'Wine':
            data = WINE
        classification_page(dataset_choice, data)
    elif alg_prob_choice == 'Regression':
        dataset_choice = st.sidebar.selectbox("Select a Dataset for Regression", ['California', 'Diabetes'])
        if dataset_choice == 'California':
            data = CALIFORNIA
        elif dataset_choice == 'Diabetes':
            data = DIABETES
        regression_page(dataset_choice, data)
    else:
        dataset_choice = st.sidebar.selectbox("Select a Dataset for Clustering", ['Cancer', 'Iris', 'Digits','Wine'])
        if dataset_choice == 'Cancer':
            data = CANCER
        elif dataset_choice == 'Iris':
            data = IRIS
        elif dataset_choice == 'Digits':
            data = DIGITS
        elif dataset_choice == 'Wine':
            data = WINE
        clustering_page(dataset_choice, data)

st.sidebar.title("EXPLORE")

page_choice = st.sidebar.radio(label="",options=["Home", "Algorithm Tool"])

if page_choice == "Home":
    home_page()
else:
    models_page()
