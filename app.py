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
            dcc.Graph(id="plot1", style={"width": "75%", "display": "inline-block"},
                figure= plot1),
            dcc.Graph(id="plot2", style={"width": "75%", "display": "inline-block"},
                figure= plot2),
            dcc.Graph(id="plot3", style={"width": "75%", "display": "inline-block"},
                figure= plot3),
            dcc.Graph(id="plot4", style={"width": "75%", "display": "inline-block"},
                figure= plot4)
        ]),
        
        dcc.Tab(label='Logistic Regression', children=[
            html.H1(children='Logistic Regression'),
            html.P('Select C'),
            dcc.Dropdown(
                id='LR_C',
                options=[{'label': i, 'value': i} for i in [.001,.005,.01,0.1,0.3,0.5,1,3,5,10]],
                value=0.001,
                style={'width': '160px'}
            ),
            html.P('Select penalty'),
            dcc.Dropdown(
                id='LR_Loss',
                options=[{'label': i, 'value': i} for i in ['l1','l2']],
                value= 'l1',
                style={'width': '160px'}
            ),
            html.Div(id='lr_evaluation'),
            dcc.Graph(id='lr-roc')
        ]),
        
        dcc.Tab(label='Random Forest', children=[
            html.H1(children='Random Forest'),
            html.P('Select Max Depth'),
            dcc.Dropdown(
                id='RF_Max_Depth',
                options=[{'label': i, 'value': i} for i in [1,25,10, 20, 100]],
                value=10,
                style={'width': '160px'}
            ),
            html.P('Select Max Features'),
            dcc.Dropdown(
                id='RF_Max_Features',
                options=[{'label': i, 'value': i} for i in ['auto', 'sqrt']],
                value= 'auto',
                style={'width': '160px'}
            ),
            html.P('Select Min Samples Leaf'),
            dcc.Dropdown(
                id='RF_Min_Samples_Leaf',
                options=[{'label': i, 'value': i} for i in [10,30,50,100]],
                value= 30,
                style={'width': '160px'}
            ),
            html.P('Select Min Samples Split'),
            dcc.Dropdown(
                id='RF_Min_Samples_Split',
                options=[{'label': i, 'value': i} for i in [10,50,100,200]],
                value= 50,
                style={'width': '160px'}
            ),
            html.P('Select No. of estimators'),
            dcc.Dropdown(
                id='RF_N_Estimators',
                options=[{'label': i, 'value': i} for i in [25,50,100,200]],
                value= 50,
                style={'width': '160px'}
            ),
            html.Div(id='rf_evaluation'),
            dcc.Graph(id='rf-roc')
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
            html.P('Select penalty'),
            dcc.Dropdown(
                id='LSVM_penalty',
                options=[{'label': i, 'value': i} for i in ['l1', 'l2']],
                value= 'l1',
                style={'width': '160px'}
            ), 
            html.P('Select loss'),
            dcc.Dropdown(
                id='LSVM_loss',
                options=[{'label': i, 'value': i} for i in ['hinge', 'square_hinge']],
                value= 'hinge',
                style={'width': '160px'}
            ), 
            html.Div(id='lsvm_evaluation'),
            dcc.Graph(id='lsvm-roc')
        ])
        
        
    ])
])


@app.callback([Output('lr_evaluation', 'children'),
              Output('lr-roc','figure')],
              [Input('LR_C', 'value'),
              Input('LR_Loss', 'value')]
)

def update_lr(x1,x2):
    model1=LogisticRegression(C=x1,penalty=x2)
    model1.fit(X_train,y_train)
    cr1=classification_report(model1.predict(X_test),y_test)
    roc1=ROC_AUC_curve(model=model1,model_name="Logistic Regression")
    return cr1,roc1
        

@app.callback([Output('rf_evaluation', 'children'),
               Output('rf-roc','figure')],
              [Input('RF_Max_Depth', 'value'),
               Input('RF_Max_Features', 'value'),
               Input('RF_Min_Samples_Leaf', 'value'),
               Input('RF_Min_Samples_Split', 'value'),
               Input('RF_N_estimators', 'value')]
)

def update_rf(x3,x4,x5,x6,x7):
    model2=RandomForestClassifier(max_depth=x3,max_features=x4,min_samples_leaf=x5,min_samples_split=x6,n_estimators=x7)
    model2.fit(X_train,y_train)
    cr2=classification_report(model2.predict(X_test),y_test)
    roc2=ROC_AUC_curve(model=model2,model_name="Random Forest")
    return cr2,roc2        

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
    cr3=classification_report(model3.predict(X_test),y_test)
    roc3=ROC_AUC_curve(model=model2,model_name="Random Forest")
    return cr3,roc3 

@app.callback([Output('lsvm_evaluation', 'children'),
              Output('lsvm-roc','figure')],
              [Input('LSVM_C', 'value'),
              Input('LSVM_penalty', 'value'),
              Input('LSVM_loss', 'value')]
)

def update_lsvm(x14,x15,x16):
    model4=LinearSVC(C=x14,penalty=x15,loss=x16)
    model4.fit(X_train,y_train)
    cr4=classification_report(model3.predict(X_test),y_test)
    roc4=ROC_AUC_curve(model=model2,model_name="Random Forest")
    return cr4,roc4
        

  
if __name__ == '__main__':
    app.run_server(debug=True)
