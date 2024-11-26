from django.shortcuts import render
from dotenv import load_dotenv
import os
import pandas as pd
import psycopg2 as DB
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime

# reservoirs = {
#         'LAKE MALOYA NR RATON NM': '07199450',
#         'EAGLE NEST LAKE NR EAGLE NEST, NM': '07205500',
# }

reservoirs = {
    "LAKE MALOYA NR RATON, NM": "07199450",
    "EL CAPITAN RES NR LAKESIDE, CA": "11020600",
    "INDEPENDENCE LK NR TRUCKEE, CA": "10342900",
    "LK PIRU NR PIRU, CA": "11109700",
    "STAMPEDE RES NR BOCA, CA": "10344300",
    "BLUEWATER LAKE NEAR BLUEWATER, NM": "08341400",
    "EAGLE NEST LAKE NR EAGLE NEST, NM": "07205500",
    "PROSSER C RES NR TRUCKEE, CA": "10340300",
    "NAMBE FALLS RESERVOIR NEAR NAMBE, NM": "08294200",
    "SAN VICENTE RES NR LAKESIDE, CA": "11022100",
    "VAIL LK NR TEMECULA, CA": "11042510",
    "DONNER LK NR TRUCKEE, CA": "10338400",
    "SANTA YNEZ R AB GIBRALTAR DAM NR SANTA BARBARA, CA": "11122000",
    "ALISAL RES NR SOLVANG, CA": "11128300",
    "BOCA RES NR TRUCKEE, CA": "10344490",
    "LAKE MOHAVE AT DAVIS DAM, AZ-NV": "09422500"
}



def charts(request):
    charts = {}
    load_dotenv()
    conn = DB.connect(f'host={os.getenv("HOST")} port={os.getenv("PORT")} dbname={os.getenv("DBNAME")} user={os.getenv("DBUSER")}')
    res_data = get_data(conn)
    data_dict = process_data(res_data)
    charts['reservoirs'] = reservoirs
    if request.method == 'POST':
        charts['selected_reservoir'] = request.POST.get('reservoir')
        stn_name = request.POST.get('reservoir')
        if stn_name not in reservoirs:
            return render(request, 'predictor/charts.html', charts)
        stn_id = reservoirs[stn_name]
        df = data_dict[stn_id]
        df.reset_index(inplace=True)
        combined_plot = get_combined_plot(df, stn_name)
        daily_plot = get_plot_daily(df, stn_name)
        charts['combined_plot'] = combined_plot.to_html(full_html=False, include_plotlyjs='cdn', default_height=500, default_width=1000)
        charts['daily_plot'] = daily_plot.to_html(full_html=False, include_plotlyjs='cdn', default_height=500, default_width=1000)
        return render(request, 'predictor/charts.html', charts)
    return render(request, 'predictor/charts.html', charts)

def get_combined_plot(df, stn_name):
    plot1 = get_plot_1(df)
    plot2 = get_plot_2(df)

    fig = go.Figure()
    fig.add_trace(plot1)
    fig.add_trace(plot2)
    fig.update_layout(
        title=f'{stn_name}\'s Water Reservoir Storage Values',
        xaxis_title='Month',
        yaxis_title='Average Storage Value',
        showlegend=True,
        barmode='group'
    )
    return fig

def get_plot_1(df):
    df_2023 = df[df['datetime'].dt.year == 2023]
    df_2023['month'] = df_2023['datetime'].dt.month
    df_2023 = df_2023.groupby('month').mean().reset_index()
    plot = go.Bar(x=df_2023['month'], y=df_2023['res_storage'], name='2023 Monthly Average')
    return plot

def get_plot_2(df):
    df_long_average = df
    df_long_average['month'] = df_long_average['datetime'].dt.month
    df_long_average = df_long_average.groupby('month').mean().reset_index()
    plot = go.Bar(x=df_long_average['month'], y=df_long_average['res_storage'], name='Long Monthly Average')
    return plot

def get_plot_daily(df, stn_name):
    df['date'] = df['datetime'].dt.date
    df = df.groupby('date').mean().reset_index()
    df['rolling_average'] = df['res_storage'].rolling(7).mean()
    plot = go.Bar(x=df['date'], y=df['res_storage'], name='Daily Average')
    plot2 = go.Scatter(x=df['date'], y=df['rolling_average'], name='7-day Rolling Average')
    fig = go.Figure()
    fig.add_trace(plot)
    fig.add_trace(plot2)
    fig.update_layout(
        title=f'{stn_name}\'s Daily Average Storage Values',
        xaxis_title='Date',
        yaxis_title='Average Storage Value',
        showlegend=True,
        barmode='group'
    )
    return fig


def home(request):
    content = {}
    load_dotenv()
    conn = DB.connect(f'host={os.getenv("HOST")} port={os.getenv("PORT")} dbname={os.getenv("DBNAME")} user={os.getenv("DBUSER")}') 
    res_data = get_data(conn)
    data_dict = process_data(res_data)
    content['real_time_plot'] = None
    models = {
        'Support Vector Regression': 'SVR',
        'Neareset Neighbors Regressor': 'KNeighborsRegressor',
        'Gaussian Process Regressor': 'GaussianProcessRegressor',
        'Decision Tree Regressor': 'DecisionTreeRegressor',
        'Random Forest Regressor': 'RandomForestRegressor',
        'Neural Networks (MLPRegressor)': 'MLPRegressor'
    }
    content['reservoirs'] = reservoirs
    content['models'] = models
    if request.method == 'POST':
        content['selected_reservoir'] = request.POST.get('reservoir')
        content['selected_model'] = request.POST.get('model')
        stn_name = request.POST.get('reservoir')
        if stn_name not in reservoirs:
            return render(request, 'predictor/home.html', content)
        stn_id = reservoirs[stn_name]
        training_data, testing_data = sort_data(data_dict[stn_id])
        content['real_time_plot'] = real_time_plot(data_dict[stn_id], stn_name)

        plots, results = run_machine_learning(stn_name, training_data, testing_data, content['selected_model'])
        content['plots'] = plots
        content['results'] = results
        return render(request, 'predictor/home.html', content)
    return render(request, 'predictor/home.html', content)

def real_time_plot(df, stn_name):
    train_start_dt = '2010-01-01 00:00:00'
    test_start_dt = '2023-01-01 00:00:00'
    train_start_dt = datetime.strptime(train_start_dt, '%Y-%m-%d %H:%M:%S')
    test_start_dt = datetime.strptime(test_start_dt, '%Y-%m-%d %H:%M:%S')
    train_df = df[(df.index < test_start_dt) & (df.index >= train_start_dt)][['res_storage']].rename(columns={'res_storage':'train'})
    test_df = df[test_start_dt:][['res_storage']].rename(columns={'res_storage':'test'})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df['train'], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['test'], mode='lines', name='Test'))

    fig.update_layout(
        title=f'Real-time Reservoir Storage for {stn_name}',
        xaxis_title='Timestamp',
        yaxis_title='Reservoir Storage',
        font=dict(size=12)
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500, default_width=1000)

def run_machine_learning(stn_name, training_data, testing_data, selected_model):
    
    plots = {}
    for i in range(1, 16):
        training_data[f'res_storage_lag_{i}'] = training_data['res_storage'].shift(i)
        testing_data[f'res_storage_lag_{i}'] = testing_data['res_storage'].shift(i)

    training_data = training_data[15:]
    testing_data = testing_data[15:]

    x_train = training_data[['pcpn', 'avgt', 'res_elevation'] + [f'res_storage_lag_{i}' for i in range(1, 16)]]
    y_train = training_data[['res_storage']]
    x_test = testing_data[['pcpn', 'avgt', 'res_elevation'] + [f'res_storage_lag_{i}' for i in range(1, 16)]]
    y_test = testing_data[['res_storage']]

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)
    models_and_parameters = [
        ('SVR', SVR(kernel='rbf'), {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}),
        ('KNeighborsRegressor', KNeighborsRegressor(), {'n_neighbors': [3, 5, 7, 9]}),
        ('GaussianProcessRegressor', GaussianProcessRegressor(), {}),
        ('DecisionTreeRegressor', DecisionTreeRegressor(), {'max_depth': [5, 10], 'min_samples_split': [10, 50]}),
        ('RandomForestRegressor', RandomForestRegressor(), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}),
        ('MLPRegressor', MLPRegressor(), {"hidden_layer_sizes": [(1,),(50,)], "activation": ["identity", "logistic", "tanh", "relu"], "solver": ["lbfgs", "adam"], "alpha": [0.00005,0.0001]})
    ]

    grid_searches = []

    for model_name, model, param_grid in models_and_parameters:
        if model_name == selected_model:
            grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2, error_score='raise')
            grid_search.fit(x_train_scaled, y_train_scaled)
            grid_searches.append(grid_search)
            best_model = grid_search.best_estimator_
            predictions = best_model.predict(x_test_scaled)        
            predictions = pd.DataFrame(predictions, columns=['res_storage'])
            score = best_model.score(x_test_scaled, y_test_scaled)
            predictions_inversed = scaler.inverse_transform(predictions)
            mae = mean_absolute_error(y_test, predictions_inversed)
            mape = mean_absolute_percentage_error(y_test, predictions_inversed)
            results = {'Model': [model.__class__.__name__], 'Best Parameters': [grid_search.best_params_], 'Score': [score], 'MAE': [mae], 'MAPE': [mape]}

            timestamps = x_test.index
            predictions = scaler.inverse_transform(predictions)
            fig = go.Figure()
            predictions = pd.DataFrame(predictions, columns=['res_storage'])
            fig.add_trace(go.Scatter(x=timestamps, y=y_test['res_storage'], mode='lines', name='Actual', line=dict(color='red', width=2)))
            fig.add_trace(go.Scatter(x=timestamps, y=predictions['res_storage'], mode='lines', name='Predicted', line=dict(color='blue', width=0.8)))

            fig.update_layout(
                title=f'Reservoir Storage Predictions for {stn_name} ({model.__class__.__name__})',
                xaxis_title='Timestamp',
                yaxis_title='Reservoir Storage',
                font=dict(size=12),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            fig.add_annotation(
                x=max(timestamps),
                y=max(y_test),
                text=f'MAE: {mae:.4f}<br>MAPE: {mape:.4f}',
                showarrow=False,
                font=dict(
                    size=10
                )
            )
            plots[model.__class__.__name__] = fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500, default_width=1000)
    return plots, results

def get_data(conn):
    cur = conn.cursor()
    cur.execute('SELECT * FROM res_data')
    res_data = cur.fetchall()
    cur.close()
    conn.close()
    res_data = pd.DataFrame(res_data, columns=['stn_id', 'datetime', 'pcpn', 'avgt', 'res_elevation', 'res_storage'])
    res_data['datetime'] = pd.to_datetime(res_data['datetime'])
    return res_data

def sort_data(res_data):
    training_data = res_data.loc[:'2022-12-31']
    testing_data = res_data.loc['2023-01-01':]
    return training_data, testing_data

def process_data(res_data):
    data_dict = {}
    reservoirs = res_data['stn_id'].unique()
    for stn_id in reservoirs:
        data = res_data[res_data['stn_id'] == stn_id]
        data.drop(['stn_id'], axis=1, inplace=True)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')
        data_dict[stn_id] = data
    return data_dict