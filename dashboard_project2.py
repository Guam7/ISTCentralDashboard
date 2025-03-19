import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pickle

# Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Load data

df_real = pd.read_csv(r"real_data.csv")
df_real['Date'] = pd.to_datetime (df_real['Date']) # create a new column 'data time' of datetime type
df0=df_real.iloc[:,2:]
X0=df0.values
df1=df_real.iloc[:,[9, 8, 7, 2]]
X1=df1.values
fig1 = px.line(df_real, x="Date", y=df_real.columns[1:11])# Creates a figure with the raw data
y2=df_real['Power (kW)'].values


# Load RF model 0
with open(r"RF_model_0.pkl", 'rb') as file:
    RF_model_0 = pickle.load(file)

y2_pred_RF_0 = RF_model_0.predict(X0)

# Evaluate errors
MAE_RF_0=metrics.mean_absolute_error(y2,y2_pred_RF_0)
MBE_RF_0=np.mean(y2-y2_pred_RF_0) 
MSE_RF_0=metrics.mean_squared_error(y2,y2_pred_RF_0)  
RMSE_RF_0= np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF_0))
cvRMSE_RF_0=RMSE_RF_0/np.mean(y2)
NMBE_RF_0=MBE_RF_0/np.mean(y2)

# Load RF model 2
with open(r"RF_model_1.pkl", 'rb') as file:
    RF_model_1 = pickle.load(file)

y2_pred_RF_1 = RF_model_1.predict(X1)

# Evaluate errors
MAE_RF_1=metrics.mean_absolute_error(y2,y2_pred_RF_1)
MBE_RF_1=np.mean(y2-y2_pred_RF_1) 
MSE_RF_1=metrics.mean_squared_error(y2,y2_pred_RF_1)  
RMSE_RF_1= np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF_1))
cvRMSE_RF_1=RMSE_RF_1/np.mean(y2)
NMBE_RF_1=MBE_RF_1/np.mean(y2)

# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    style={'margin': '30px', 'padding': '20px'},

    children = [
    html.H1('IST Central Building Energy Forecast tool (kWh)'),
    html.P('Representing data, forecasting and error metrics for 2017, 2018 and the begining of 2019.'),

    # Date Picker
    html.Label("Select Date Range (you can choose any date between 2017-01-01 and 2019-04-11):"),
    dcc.DatePickerRange(
        id='date-picker',
        start_date='2019-01-01',
        end_date='2019-04-01',
        display_format='YYYY-MM-DD',
        style={'margin-bottom': '20px'}  # Adds space below
    ),

    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Custom Model', value='tab-2'),
        dcc.Tab(label='Error Metrics and Forecast', value='tab-3'),
    ]),
    html.Div(id='tabs-content'),

    # Hidden storage for dynamic forecast & metrics
    dcc.Store(id='custom-forecast-store'),
    dcc.Store(id='custom-metrics-store')
])

@app.callback(
    [Output('model-results', 'children'),
     Output('custom-forecast-store', 'data'),
     Output('custom-metrics-store', 'data')],
    [Input('train-button', 'n_clicks')],
    [State('feature-selector', 'value'),
     State('num-trees', 'value'),
     State('max-depth', 'value')]
)
def train_model(n_clicks, selected_features, num_trees, max_depth):
    if n_clicks > 0 and selected_features:
        X = df_real[selected_features].values
        y = df_real['Power (kW)'].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model_custom = RandomForestRegressor(n_estimators=num_trees, max_depth=max_depth, random_state=42)
        model_custom.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model_custom.predict(X_test)
        y2_pred_custom = model_custom.predict(X)

        # Calculate errors
        MAE_RF_2 = metrics.mean_absolute_error(y_test, y_pred)
        MBE_RF_2 = np.mean(y_test - y_pred)
        MSE_RF_2 = metrics.mean_squared_error(y_test, y_pred)
        RMSE_RF_2 = np.sqrt(MSE_RF_2)
        cvRMSE_RF_2 = RMSE_RF_2 / np.mean(y_test)
        NMBE_RF_2 = MBE_RF_2 / np.mean(y_test)

        # ASHRAE Standard Checks
        ASHRAE_cvRMSE_status = "✅ Pass" if cvRMSE_RF_2 <= 0.3 else "❌ Fail"
        ASHRAE_NMBE_status = "✅ Pass" if abs(NMBE_RF_2) <= 0.1 else "❌ Fail"

        # IPMVP Standard Checks
        IPMVP_cvRMSE_status = "✅ Pass" if cvRMSE_RF_2 <= 0.2 else "❌ Fail"
        IPMVP_NMBE_status = "✅ Pass" if abs(NMBE_RF_2) <= 0.05 else "❌ Fail"

        # Store results
        forecast_data = {'Date': df_real['Date'].astype(str).tolist(), 'CustomRandomForest': y2_pred_custom.tolist()}
        metrics_data = {
            'Methods': 'Random Forest (Custom)',
            'MAE': MAE_RF_2,
            'MBE': MBE_RF_2,
            'MSE': MSE_RF_2,
            'RMSE': RMSE_RF_2,
            'cvMSE': cvRMSE_RF_2,
            'NMBE': NMBE_RF_2
        }

        # Table Displaying Model Metrics and Pass/Fail Status for ASHRAE & IPMVP
        table = html.Table([
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Value"), html.Th("ASHRAE (≤30% & ±10%)"), html.Th("IPMVP (≤20% & ±5%)")])),
            html.Tbody([
                html.Tr([html.Td("cvRMSE"), html.Td(f"{cvRMSE_RF_2:.4f}"), html.Td(ASHRAE_cvRMSE_status), html.Td(IPMVP_cvRMSE_status)]),
                html.Tr([html.Td("NMBE"), html.Td(f"{NMBE_RF_2:.4f}"), html.Td(ASHRAE_NMBE_status), html.Td(IPMVP_NMBE_status)])
            ])
        ])

        return html.Div([
            html.H5("Model Performance Based on Standards"),
            table
        ]), forecast_data, metrics_data

    return "Select features and click 'Train Model' to see results.", None, None

@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')],
     [State('custom-forecast-store', 'data'),
     State('custom-metrics-store', 'data')]
)

def render_content(tab, start_date, end_date, custom_forecast, custom_metrics):
    filtered_df = df_real[(df_real['Date'] >= start_date) & (df_real['Date'] <= end_date)]
    
    # Update forecast results dynamically
    df_forecast = pd.DataFrame({'Date': df_real['Date'], 'RandomForest_0': y2_pred_RF_0, 'RandomForest_1': y2_pred_RF_1})

    if custom_forecast:
        df_forecast['CustomRandomForest'] = custom_forecast['CustomRandomForest']
    
    df_results = pd.merge(df_real[['Date', 'Power (kW)']], df_forecast, on='Date')

    filtered_results = df_results[(df_results['Date'] >= start_date) & (df_results['Date'] <= end_date)]

    filtered_results = filtered_results.rename(columns={
        "RandomForest_0": "Random Forest (All Features)",
        "RandomForest_1": "Random Forest (4 Features)",
        "CustomRandomForest": "Random Forest (Custom Features)"
    })

    
    # Update error metrics dynamically
    df_metrics = pd.DataFrame({
        'Methods': ['Random Forest (All)', 'Random Forest (Power-1, Day Type, Hour, Temperature)'],
        'MAE': [MAE_RF_0, MAE_RF_1],
        'MBE': [MBE_RF_0, MBE_RF_1],
        'MSE': [MSE_RF_0, MSE_RF_1],
        'RMSE': [RMSE_RF_0, RMSE_RF_1],
        'cvRMSE': [cvRMSE_RF_0, cvRMSE_RF_1],
        'NMBE': [NMBE_RF_0, NMBE_RF_1]
    })

    if custom_metrics:
        df_metrics = pd.concat([df_metrics, pd.DataFrame([custom_metrics])], ignore_index=True)

    # Create the tabs

    if tab == 'tab-1':
        fig1_filtered = px.line(filtered_df, x="Date", y=filtered_df.columns[1:])
        return html.Div([
            html.H4('IST Central Building Raw Data'),
            html.P('You can change the represented timespan above. To isolate a specific variable, double-click on the legend. All units are specified in the legend.'),
            dcc.Graph(id='filtered-raw-data', figure=fig1_filtered),

            # Bullet points explaining each variable
            html.H5("Feature Descriptions:"),
            html.Ul([
                html.Li("Power (kW): The actual energy consumption of the building."),
                html.Li("Temperature (°C): Outside temperature."),
                html.Li("Humidity (%): Relative humidity level."),
                html.Li("Pressure (mbar): Atmospheric pressure level."),
                html.Li("Rain (mm/h): Rainfall intensity."),
                html.Li("Day of Week: Numeric representation of the day (e.g., Monday = 0)."),
                html.Li("Hour: The hour of the day (0-23) when the measurement was taken."),
                html.Li("Day Type: The same as Day of Week, but holidays have value 7."),
                html.Li("Power-1 (kW): Power consumption from the previous hour."),
                html.Li("Power-2 (kW): Power consumption from two hours earlier."),
                html.Li("HDH (h°C): Heating Degree Hours, an indicator of heating demand."),
            ])
        ])
    
    elif tab == 'tab-2':
        return html.Div([
            html.H4('Train Your Own RandomForest Model'),
            
            html.Label("Select Features for Training:"),
            dcc.Dropdown(
                id='feature-selector',
                options=[{'label': col, 'value': col} for col in df_real.columns[2:]],  # Feature selection
                multi=True,
                placeholder="Select features...",
                style={'margin-bottom': '20px'}  # Adds space below
            ),

            html.Div([
                # Left side: Inputs stacked vertically
                html.Div([
                    html.Label("Number of Trees (n_estimators):"),
                    dcc.Input(id='num-trees', type='number', value=200, step=10, style={'margin-bottom': '10px'}),

                    html.Label("Max Depth:"),
                    dcc.Input(id='max-depth', type='number', value=20, step=1)
                ], style={'display': 'flex', 'flex-direction': 'column', 'margin-right': '20px'}),

                # Right side: Centered button
                html.Div([
                    html.Button(
                        'TRAIN MODEL', 
                        id='train-button', 
                        n_clicks=0, 
                        style={
                            'width': '200px',
                            'height': '60px',
                            'font-size': '20px',
                            'font-weight': 'bold',
                            'border-radius': '10px',
                            'background-color': '#A9A9A9',  # Grey when disabled
                            'color': 'white',
                            'border': 'none',
                            'cursor': 'not-allowed',  # Greyed-out cursor
                            'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.2)'
                        }
                    )
                ],
                id='button-container',
                title="You must select at least one feature before training your model",
                style={
                    'display': 'flex', 
                    'align-items': 'center', 
                    'justify-content': 'center', 
                    'flex-grow': '1'
                })
                
            ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between', 'width': '100%'}),

            html.Br(),
            html.H5("Model Results:"),
            html.Div(id='model-results'),  # Placeholder for model output

            html.P('WARNING: changing to another tab will reset the model. To see the full model metrics check "Error Metrics and Forecast", but keep in mind that returning to this tab will reset the model.')
        ])
    
    elif tab == 'tab-3':
        return html.Div([
            html.H4('IST Central Building Electricity Forecast Error Metrics and Forecast'),
            html.P('You can create your own model on "Custom Model". Keep in mind that the other models were trained using only 2017 and 2018 data, while the custom model is trained with all the data. Some parameters, such as the random seed, are also different, which is why it may not wield the same results.'),
            generate_table(df_metrics),
            dcc.Graph(
                id='forecast-graph',
                figure=px.line(
                    filtered_results,
                    x="Date",
                    y=filtered_results.columns[1:],  # Your Y-axis columns
                    labels={
                        "RandomForest_0": "Random Forest (All Features)",
                        "RandomForest_1": "Random Forest (Selected Features)",
                        "CustomRandomForest": "Custom Model Prediction",
                        "Power (kW)": "Actual Power Consumption"
                    }
                )
            )
        ])

# Callback to enable/disable the button
@app.callback(
    Output('train-button', 'disabled'),
    Output('train-button', 'style'),
    Output('button-container', 'title'),
    Input('feature-selector', 'value')
)
def update_button(features):
    if not features:  # If no features selected, disable button
        return True, {
            'width': '200px',
            'height': '60px',
            'font-size': '20px',
            'font-weight': 'bold',
            'border-radius': '10px',
            'background-color': '#A9A9A9',  # Grey
            'color': 'white',
            'border': 'none',
            'cursor': 'not-allowed',
            'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.2)',
        }, "You must select at least one feature"
    
    else:  # Enable button when features are selected
        return False, {
            'width': '200px',
            'height': '60px',
            'font-size': '20px',
            'font-weight': 'bold',
            'border-radius': '10px',
            'background-color': '#007BFF',  # Blue when enabled
            'color': 'white',
            'border': 'none',
            'cursor': 'pointer',
            'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.2)'
        }, ""


if __name__ == '__main__':
    app.run_server()