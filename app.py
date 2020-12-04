import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
from joblib import dump, load
import pickle
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve, precision_score, auc, f1_score,classification_report
import dash_table

from dash.dependencies import Input, Output


global model1,model2,model3,model4,X_train,X_test,y_train,y_test
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

plot = "figure1.pickle"
with open(plot, 'rb') as file:
    plot1 = pickle.load(file)
    
plot = "figure2.pickle"
with open(plot, 'rb') as file:
    plot2 = pickle.load(file)
    
plot = "figure3.pickle"
with open(plot, 'rb') as file:
    plot3 = pickle.load(file)
    
plot = "figure4.pickle"
with open(plot, 'rb') as file:
    plot4 = pickle.load(file)
    

def ROC_AUC_curve(model,model_name="Model"):
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]

    # predict probabilities
    lr_probs = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Model: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.figure(figsize=(8,4))
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label=model_name)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
