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

X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')
gb_top_features = pd.read_csv('lgb_imp_feat.csv')
rf_top_features = pd.read_csv('rf_imp_feat.csv')
df_gb = pd.read_csv('Comp_Table.csv')
df_rf = pd.read_csv('Comp_Table_RF.csv')
target = "Target_count.pkl"
with open(target, 'rb') as file:
    target_vis = pickle.load(file)

target = "plot1.pkl"
with open(target, 'rb') as file:
    plot1_vis = pickle.load(file)

target = "plot2.pkl"
with open(target, 'rb') as file:
    plot2_vis = pickle.load(file)

target = "plot3.pkl"
with open(target, 'rb') as file:
    plot3_vis = pickle.load(file)

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
            - **Sarthak Tandon**
            - **Tanishk Parihar**
            - **Pranav Kottoli Radhakrishna**
            '''
            )
            ]),
        dcc.Tab(label='Exploratory Data Analysis', children=[
            html.H1(children='Exploratory Data Analysis'),
            html.P('Visualization of the target variable. The dataset is quite imbalanced as can be seen below'),
            dcc.Graph(id="Target", style={"width": "75%", "display": "inline-block"},
                figure= target_vis),
            dcc.Graph(id="plot1", style={"width": "75%", "display": "inline-block"},
                figure= plot1_vis),
            dcc.Graph(id="plot2", style={"width": "75%", "display": "inline-block"},
                figure= plot2_vis),
            dcc.Graph(id="plot3", style={"width": "75%", "display": "inline-block"},
                figure= plot3_vis)
        ]),
        dcc.Tab(label='Random Forest', children=[
            html.H1(children='Random Forest'),
            html.P('Select Class Weight'),
            dcc.Dropdown(
                id='RF_Class_Weight',
                options=[{'label': i, 'value': i} for i in ['{0:1,1:17}','{0:1,1:5}','{0:1,1:1}']],
                value='{0:1,1:17}',
                style={'width': '160px'}
            ),
            html.P('Select Max Depth'),
            dcc.Dropdown(
                id='RF_Max_Depth',
                options=[{'label': i, 'value': i} for i in [50,25]],
                value= 50,
                style={'width': '160px'}
            ),
            html.P('Select Min Samples Split'),
            dcc.Dropdown(
                id='RF_Min_Samples_Split',
                options=[{'label': i, 'value': i} for i in [100,50,25]],
                value= 100,
                style={'width': '160px'}
            ),
            html.Div(id='rf_evaluation'),
            dcc.Graph(id='rf-features')
        ]),
        dcc.Tab(label='Light Gradient Boosting', children=[
            html.H1(children='Light Gradient Boosting'),
            html.P('Select Lambda Regularization Parameter'),
            dcc.Dropdown(
                id='LGB_Lambda',
                options=[{'label': i, 'value': i} for i in [0.1,0.8]],
                value=0.8,
                style={'width': '160px'}
            ),
            html.P('Select Alpha Regularization Parameter'),
            dcc.Dropdown(
                id='LGB_Alpha',
                options=[{'label': i, 'value': i} for i in [0.1,0.5,1.0]],
                value= 1.0,
                style={'width': '160px'}
            ), 
            html.P('Select Learning Rate'),
            dcc.Dropdown(
                id='LGB_LearningRate',
                options=[{'label': i, 'value': i} for i in [0.02,0.05,0.08]],
                value= 0.02,
                style={'width': '160px'}
            ), 
            html.Div(id='lgb_evaluation'),
            dcc.Graph(id='lgb-features')
        ]),
        dcc.Tab(label='Logistic Regression', children=[
            html.H1(children='Logistic Regression'),
            html.P('Select C'),
            dcc.Dropdown(
                id='LR_C',
                options=[{'label': i, 'value': i} for i in [0.001,0.01,0.1]],
                value=0.001,
                style={'width': '160px'}
            ),
            html.P('Select Class Weight'),
            dcc.Dropdown(
                id='LR_Class_Weight',
                options=[{'label': i, 'value': i} for i in ['{0:1,1:11}','{0:1,1:5}','{0:1,1:1}']],
                value= '{0:1,1:11}',
                style={'width': '160px'}
            ),
            html.Div(id='lr_evaluation'),
            dcc.Graph(id='lr-roc')
        ]),
    ])
])
@app.callback([Output('rf_evaluation', 'children'),
               Output('rf-features', 'figure')],
              [Input('RF_Class_Weight', 'value'),
               Input('RF_Max_Depth', 'value'),
               Input('RF_Min_Samples_Split', 'value')]
)
def rand_for(class_weight,max_depth,min_samples):
    tab = df_rf[(df_rf['max_depth']==max_depth) & (df_rf['min_samples_split']==min_samples) & (df_rf['class_weight']==class_weight)]
    tab.drop(['max_depth','min_samples_split','class_weight'],axis = 1,inplace = True)

    fig = px.bar(rf_top_features, x="index",y="Importance")
     
    return [html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in tab.columns],
                data=tab.to_dict("rows"),
                style_cell={'width': '300px',
                'height': '60px',
                'textAlign': 'left'})
            ]),fig]

@app.callback([Output('lgb_evaluation', 'children'),
                Output('lgb-features', 'figure')],
              [Input('LGB_Lambda', 'value'),
              Input('LGB_Alpha', 'value'),
              Input('LGB_LearningRate', 'value')]
)
def grad_boost(lamda,alpha,learning_rate):

    tab = df_gb[(df_gb['reg_lambda']==lamda) & (df_gb['reg_alpha']==alpha) & (df_gb['learning_rate']==learning_rate)]
    tab.drop(['reg_lambda','reg_alpha','learning_rate'],axis = 1,inplace = True)

    fig = px.bar(gb_top_features, x="index",y="Importance")

    return [html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in tab.columns],
                data=tab.to_dict("rows"),
                style_cell={'width': '300px',
                'height': '60px',
                'textAlign': 'left'})
            ]),fig]

@app.callback([Output('lr_evaluation', 'children'),
                Output('lr-roc', 'figure')],
              [Input('LR_C', 'value'),
              Input('LR_Class_Weight', 'value')]
)
def log_reg(c_par,class_weight):

    if(c_par == 0.001 and class_weight == '{0:1,1:11}'):
        lr = load('lr1.joblib') 
    elif(c_par == 0.01 and class_weight == '{0:1,1:11}'):
        lr = load('lr2.joblib')
    elif(c_par == 0.1 and class_weight == '{0:1,1:11}'):
        lr = load('lr3.joblib')
    elif(c_par == 0.001 and class_weight == '{0:1,1:5}'):
        lr = load('lr4.joblib')
    elif(c_par == 0.01 and class_weight == '{0:1,1:5}'):
        lr = load('lr5.joblib')
    elif(c_par == 0.1 and class_weight == '{0:1,1:5}'):
        lr = load('lr6.joblib')
    elif(c_par == 0.001 and class_weight == '{0:1,1:1}'):
        lr = load('lr7.joblib')
    elif(c_par == 0.01 and class_weight == '{0:1,1:1}'):
        lr = load('lr8.joblib')
    elif(c_par == 0.1 and class_weight == '{0:1,1:1}'):
        lr = load('lr9.joblib')

    pred = lr.predict(X_test)
    model_eval = {'Recall': [recall_score(y_test,pred)],
        'Precision': [precision_score(y_test,pred)],
        'Accuracy': [accuracy_score(y_test,pred)],
        'AUC': [roc_auc_score(y_test,pred)],
        'F1-Score': [f1_score(y_test,pred)]
       }
    tab = pd.DataFrame(model_eval)

    ns_probs = [0 for _ in range(len(y_test))]

    # predict probabilities
    lr_probs = lr.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)

    # calculate roc curves
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

    df_roc = pd.DataFrame({'fpr': lr_fpr, 'tpr': lr_tpr}, columns=['fpr', 'tpr'])

    fig = px.line(df_roc, x="fpr", y="tpr", title='ROC Curve')
     
    return [html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in tab.columns],
                data=tab.to_dict("rows"),
                style_cell={'width': '300px',
                'height': '60px',
                'textAlign': 'left'})
            ]),fig]

if __name__ == '__main__':
 app.run_server(debug=True)