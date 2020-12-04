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
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


global model1,model2,model3,model4,X_train,X_test,y_train,y_test
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')
    

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



app.layout = html.Div([
    html.Header([
                html.H2("Data Analytics Project Dashboard - by Sankalp Singh and Aneesh Phatak", style={'height': '100px','width':'100%', 'display': 'inline-block'})],
                style={'backgroundColor':'yellow','height': '70px'}, className="banner"),
    dcc.Tabs([
        dcc.Tab(label='Motivation', children=[
            dcc.Markdown('''
            ### US Visa Application Outcome Prediction
            #### Background:
            **United States of America is known as the land of opportunity. Every year hundreds and thousands of individuals arrive in the United States with a dream to achieve something big in their lives. One of the major problems that every individual who’s not a US citizen faces, is the strenuous process of applying for a permanent work visa. This problem is faced by the international workers as well as the international students. A permanent labor certification issued by the Department of Labor (DOL) allows an employer to hire a foreign worker to work permanently in the U.S. The employer must obtain a certified labor certification application from the DOL’s Employment and Training Administration. After this step, the employer can submit an immigration petition to the Department of Homeland Security’s U.S. Citizenship and Immigration Services (USCIS). This process depends on a lot of different factors and it can take several years before the process completes. This is a major problem which U.S. as well as international employers face today. This problem needs to be addressed carefully in order for an employer to successfully allow an employee to work for their company in the States.**

            #### Approach:
            **To tackle this major societal problem, we will be designing several machine learning models that will help us to predict the visa decisions based on the employer, employee, wage etc. Hopefully, our models will be robust enough and will be able to predict whether a visa application will be approved or denied. This will help us to resolve one of the major societal issue that every company inside or outside of the United States faces today. **

            '''
            )
            ]),
        
        dcc.Tab(label='Logistic Regression', children=[
            html.H1(children='Logistic Regression'),
            html.P('Select C'),
            dcc.Dropdown(
                id='LR_C',
                options=[{'label': i, 'value': i} for i in [.001,.005,.01,0.1,0.3,0.5,0.7,0.9]],
                value=0.001,
                style={'width': '160px'}
            ),
           
            html.Div(id='lr_evaluation'),
            dcc.Graph(id='lr-roc')
        ]),
        
       
        
        dcc.Tab(label='Gradient Boosting Classifier', children=[
            html.H1(children='Gradient Boosting Classifier'),
            html.P('Select max features'),
            dcc.Dropdown(
                id='GBC_max_features',
                options=[{'label': i, 'value': i} for i in ['auto','sqrt']],
                value='auto',
                style={'width': '160px'}
            ),
            html.P('Select no. of estimators'),
            dcc.Dropdown(
                id='GBC_n_estimators',
                options=[{'label': i, 'value': i} for i in [20,50,100,200]],
                value= 20,
                style={'width': '160px'}
            ), 
            html.P('Select subsample'),
            dcc.Dropdown(
                id='GBC_subsample',
                options=[{'label': i, 'value': i} for i in [0.6,0.8,0.9]],
                value= 0.6,
                style={'width': '160px'}
            ),
            html.P('Select Max Depth'),
            dcc.Dropdown(
                id='GBC_maxdepth',
                options=[{'label': i, 'value': i} for i in [1,5,20,10,50,100]],
                value= 1,
                style={'width': '160px'}
            ), 
            html.P('Select min samples split'),
            dcc.Dropdown(
                id='GBC_min_samples_split',
                options=[{'label': i, 'value': i} for i in [25,50,100,200]],
                value=25,
                style={'width': '160px'}
            ), 
            html.P('Select min samples leaf'),
            dcc.Dropdown(
                id='GBC_min_samples_leaf',
                options=[{'label': i, 'value': i} for i in [10,30,50,100]],
                value= 10,
                style={'width': '160px'}
            ), 
            
            html.Div(id='gbc_evaluation'),
            dcc.Graph(id='gbc-roc')
        ]),
        
        dcc.Tab(label='Linear SVM', children=[
            html.H1(children='Linear SVM'),
            html.P('Select C'),
            dcc.Dropdown(
                id='LSVM_C',
                options=[{'label': i, 'value': i} for i in [0.005,.01,.1,.2,.4,.8]],
                value=0.01,
                style={'width': '160px'}
            ),
           
            html.Div(id='lsvm_evaluation')
        ])
        
        
    ])
])


@app.callback([Output('lr_evaluation', 'children'),
              Output('lr-roc','figure')],
              [Input('LR_C', 'value')]
)



def update_lr(x1):
    model1=LogisticRegression(C=x1)
    model1.fit(X_train,y_train)
    pred1 = model1.predict(X_test)
    model_eval = {'Recall': [recall_score(y_test,pred1)],
        'Precision': [precision_score(y_test,pred1)],
        'Accuracy': [accuracy_score(y_test,pred1)],
        'AUC': [roc_auc_score(y_test,pred1)],
        'F1-Score': [f1_score(y_test,pred1)]
       }
    tab = pd.DataFrame(model_eval)

    ns_probs = [0 for _ in range(len(y_test))]

    # predict probabilities
    lr_probs = model1.predict_proba(X_test)
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
 

        


       

@app.callback([Output('gbc_evaluation', 'children'),
               Output('gbc-roc','figure')],
              [Input('GBC_max_features', 'value'),
               Input('GBC_n_estimators', 'value'),
               Input('GBC_subsample', 'value'),
               Input('GBC_maxdepth', 'value'),
               Input('GBC_min_samples_split', 'value'),
               Input('GBC_min_samples_leaf', 'value')]
)
        
def update_gbc(x8,x9,x10,x11,x12,x13):
    model3=GradientBoostingClassifier(max_features=x8,n_estimators=x9,subsample=x10,max_depth=x11,min_samples_split=x12,min_samples_leaf=x13)
    model3.fit(X_train,y_train)
    pred3 = model3.predict(X_test)
    model_eval = {'Recall': [recall_score(y_test,pred3)],
        'Precision': [precision_score(y_test,pred3)],
        'Accuracy': [accuracy_score(y_test,pred3)],
        'AUC': [roc_auc_score(y_test,pred3)],
        'F1-Score': [f1_score(y_test,pred3)]
       }
    tab = pd.DataFrame(model_eval)

    ns_probs = [0 for _ in range(len(y_test))]

    # predict probabilities
    lr_probs = model3.predict_proba(X_test)
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
    
  
@app.callback([Output('lsvm_evaluation', 'children')],
              [Input('LSVM_C', 'value')]
              
)

def update_lsvm(x14):
    model4=LinearSVC(C=x14)
    model4.fit(X_train,y_train)
    pred4 = model4.predict(X_test)
    model_eval = {'Recall': [recall_score(y_test,pred4)],
        'Precision': [precision_score(y_test,pred4)],
        'Accuracy': [accuracy_score(y_test,pred4)],
        'AUC': [roc_auc_score(y_test,pred4)],
        'F1-Score': [f1_score(y_test,pred4)]
       }
    tab = pd.DataFrame(model_eval)

    
     
    return [html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in tab.columns],
                data=tab.to_dict("rows"),
                style_cell={'width': '300px',
                'height': '60px',
                'textAlign': 'left'})
            ])]

  
if __name__ == '__main__':
 app.run_server(debug=False)
