import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
from joblib import dump, load
import pickle
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve, precision_score, auc, f1_score
import dash_table
from dash.dependencies import Input, Output

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
	
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    html.Header([
                html.H2("IST707 Final Project Dashboard", style={'height': '70px','width':'95%', 'display': 'inline-block'}),
                html.Img(src="/assets/syrlogo2.jpg", style={'height': '70px','width':'5%', 'display': 'inline-block','textAlign': 'right'})],
                style={'backgroundColor':'orange','height': '70px'}, className="banner"),
    dcc.Tabs([
        dcc.Tab(label='About', children=[
            dcc.Markdown('''
            ### Home Credit Default Risk Prediction
            #### Context:
            **In finance, a loan is the lending of money by one or more individuals, organizations, or other entities to other individuals, organizations etc.
            The recipient (i.e., the borrower) incurs a debt and is usually liable to pay interest on that debt until it is repaid as well as to repay the 
            principal amount borrowed.**

            #### Problem:
            **Evaluating and predicting the repayment ability of the borrowers is important for the banks to minimize the risk of loan payment default. 
            For this reason, there is a system created by the banks to process the loan request based on the borrowers’ credentials, such as employment 
            status, credit history, etc. However, the current system might not be appropriate to evaluate some borrowers’ repayment ability, such as students 
            or people without credit histories. The major goal of the project is to find a more robust way to assess whether a candidate will default 
            on a loan.**

            #### Creators:
            - **Aneesh Phatak**
            - **Sankalp Singh**
            '''
            )
            ]),
        dcc.Tab(label='Exploratory Data Analysis', children=[
            html.H1(children='Exploratory Data Analysis'),
            html.P('Visualization of the target variable. The dataset is quite imbalanced as can be seen below'),
            dcc.Graph(id="Target", style={"width": "75%", "display": "inline-block"},
                figure= plot1),
            dcc.Graph(id="plot1", style={"width": "75%", "display": "inline-block"},
                figure= plot2),
            dcc.Graph(id="plot2", style={"width": "75%", "display": "inline-block"},
                figure= plot3),
            dcc.Graph(id="plot3", style={"width": "75%", "display": "inline-block"},
                figure= plot4)
        ])

		
if __name__ == '__main__':
    app.run_server(debug=True)
