[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/O6idXAzd)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13072810&assignment_repo_type=AssignmentRepo)

# Final Project Report

## Running the program

To run the program you must first install the requirements. To do this, run the following command:

```
pip install -r requirements.txt
```

Then you will need to add in your own data into the .env folder to ensure that you don't have any problems connecting to your database.
You will probably need to change the HOST, PORT, DEFAULT_DBNAME, and DBUSER. Ensure that you do not change the DBNAME as the program will
create the database for you. Then you must run the the data_ingestor.py file. This will collect the data and save it to a database. To do this, run the following command:

```
python3 data_ingestor.py
```

After this step is complete and the data is in the database, you can start the django server. To do this, run the following command:

```
python3 manage.py runserver
```

Once you are in the website, you can select a reservoir and model to view the results of that ML model. If you select only a reservoir
and hit the submit button, then you will only be able to see the Real-Time Reservoir Storage plot which shoes the timeframe for the training and testing data.
If you select a reservoir and a model, then you will be able to see the Real-Time Reservoir Storage plot, the Reservoir Storage Prediction plot, and the stats table for that model.

## Data Collection

Data was collected through two sites:
'''
https://waterservices.usgs.gov/nwis/dv/?format=rdb&stateCd=nm&startDT={}&endDT={}&parameterCd={}&siteStatus=active
https://data.rcc-acis.org/GridData
'''
I utilized the https://builder.rcc-acis.org/ website in order to generate the API call for the data. Once this data was collected,
I added the data to a dataframe which held the data from both sites. This dataframe was then saved into a database.

## Preprocessing the data

The data was preprocessed by first spliting the data into two dataframes, one for each site. Then the data was split into training and testing data. The training data was used to train the models and the testing data was used to test the models. The data was then scaled using the MinMaxScaler. The data was then split into X and y values. The X values were the features and the y values were the target values. The training data was used to train the models and the testing data was used to test the models. I also added 15 columns which held the res_storage values for the previous 15 days.

## Training the models

The models were trained using the training data. The models were trained using the following parameters:
'''
('SVR', SVR(kernel='rbf'), {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}),
('KNeighborsRegressor', KNeighborsRegressor(), {'n_neighbors': [3, 5, 7, 9]}),
('GaussianProcessRegressor', GaussianProcessRegressor(), {}),
('DecisionTreeRegressor', DecisionTreeRegressor(), {'max_depth': [5, 10], 'min_samples_split': [10, 50]}),
('RandomForestRegressor', RandomForestRegressor(), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}),
('MLPRegressor', MLPRegressor(), {"hidden_layer_sizes": [(1,),(50,)], "activation": ["identity", "logistic", "tanh", "relu"], "solver": ["lbfgs", "adam"], "alpha": [0.00005,0.0001]})
'''
The models were trained using the GridSearchCV function. The best parameters were then used to train the models.

## Test/Predict the results

After finding the best parameters for each model, the models were then used to predict the res_storage values of the testing data. The models were then scored using the testing data. The models were scored using the following metrics: MAE, and MAPE.

## ML Results

'''
LAKE MALOYA NR RATON NM

                   Model                                                                             Best Parameters     Score       MAE      MAPE
                     SVR                                                                    {'C': 10, 'gamma': 0.01}  0.928684  0.029802   0.05093
     KNeighborsRegressor                                                                          {'n_neighbors': 5}  0.988091  0.009405  0.016399

GaussianProcessRegressor {} 0.999721 0.001143 0.001964
DecisionTreeRegressor {'max_depth': 10, 'min_samples_split': 10} 0.999922 0.00051 0.001007
RandomForestRegressor {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 50} 0.999958 0.000365 0.000735
MLPRegressor {'activation': 'identity', 'alpha': 0.0001, 'hidden_layer_sizes': (1,), 'solver': 'lbfgs'} 0.995796 0.004954 0.008148

EAGLE NEST LAKE NR EAGLE NEST NM

                   Model                                                                             Best Parameters     Score       MAE      MAPE
                     SVR                                                                    {'C': 10, 'gamma': 0.01}  0.973591  0.013467  0.021236
     KNeighborsRegressor                                                                          {'n_neighbors': 3}  0.988198  0.008962  0.015093

GaussianProcessRegressor {} 0.999962 0.000345 0.000572
DecisionTreeRegressor {'max_depth': 10, 'min_samples_split': 50} 0.998158 0.002965 0.004738
RandomForestRegressor {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 50} 0.999966 0.000385 0.000612
MLPRegressor {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50,), 'solver': 'lbfgs'} 0.99907 0.00254 0.004143
'''
