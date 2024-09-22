from sklearn.base import BaseEstimator, TransformerMixin
import re
from utils.training.functions import range_number_finder, rep_number_finder

class DataProcessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # No need to fit anything for this transformer
        return self

    def transform(self, X):
        # Check if X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        
        X_transformed = X.copy()

        ids_to_delete = [i for i, c in zip(X_transformed['id'], X_transformed['price'].astype(str)) if (c == rep_number_finder(c)) | (c==range_number_finder(c))] 

        X_transformed = X_transformed[~X_transformed.index.isin(ids_to_delete)]

        X_transformed = X_transformed[X_transformed['currency_id'] == 'USD']
        X_transformed = X_transformed[X_transformed['city'] == 'Capital Federal']

        X_transformed.drop(['currency_id', 'operation', 'city'], axis=1, inplace=True)
        
        for col in X_transformed.columns:
            
            if col in ['covered_area', 'total_area']:
                X_transformed[col] = X_transformed[col].apply(lambda x: \
                                                              float(re.sub(r'[^\d.]', '', x)) if pd.notna(x) else x)
                
            
            elif col in ['has_air_conditioning', 'with_virtual_tour', 'item_condition']:
                X_transformed[col] = X_transformed[col].apply(lambda x:\
                                                              1 if (x=='Sí') | (x=='Nuevo') else 0)
                                                                
            elif col in ['bedrooms']:
                X_transformed[col] = X_transformed[col].apply(lambda x:\
                                                         np.nan if x<0 else x)
               
            elif col in ['rooms', 'full_bathrooms', 'total_area', 'covered_area']:
                X_transformed[col] = X_transformed[col].apply(lambda x:\
                                                         np.nan if x<=0 else x)
                if col == 'rooms':
                    X_transformed[col] = X_transformed.apply(lambda x:\
                                                                x['rooms'] if ((x['rooms']>=x['bedrooms']) | pd.isna(x['rooms'])) else x['bedrooms'],
                                                                axis=1)

                if col == 'full_bathrooms':
                    X_transformed[col] = X_transformed[col].apply(lambda x: 1 if x<1 else x)

                    X_transformed[col] = np.where((X_transformed[col]>X_transformed['bedrooms']),
                                                   X_transformed['bedrooms'],
                                                   X_transformed[col])

            elif col == 'parking_lots':
                X_transformed[col] = X_transformed[col].fillna('[0-0]')
                #X_transformed[col] = X_transformed[col].str.replace('[0-0]','0').replace('[1-1]','1').replace('[2-2]','2').replace('[3-*]','3').astype(int)
                X_transformed[col] = X_transformed[col].apply(lambda x: 0 if x == '[0-0]' else 1)


        X_transformed['total_area'] = np.where((X_transformed['total_area']<X_transformed['covered_area']) | (~X_transformed['total_area'].isna()),
                                                X_transformed['covered_area'],
                                                X_transformed['total_area'])

        X_transformed['covered_area'] = np.where(X_transformed['covered_area'] < 20,
                                                 np.nan,
                                                 X_transformed['covered_area'])

        
        return X_transformed
    
    
class RemoveOutliers(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        
        X_transformed = X.copy()

        max_values = {
            'latitude': -34.1,
            'longitude': -58.1,
            'rooms': 10,
            'bedrooms': 9,
            'full_bathrooms': 9,
            'total_area': 1500,
            'covered_area': 1000,
            'price': 5000000
                      }

        min_values = {
            'latitude': -34.9,
            'longitude': -58.8,
            'rooms': 1,
            'bedrooms': 0,
            'full_bathrooms': 1,
            'total_area': 20,
            'covered_area': 20,
            'price': 40000
                      }

        for col in min_values.keys():
            min_value = min_values.get(col)
            max_value = max_values.get(col)

            mask = ((X_transformed[col] >= min_value) & (X_transformed[col] <= max_value)) |\
                   (X_transformed[col].isna())
            
            X_transformed = X_transformed[mask]
        
        return X_transformed
    
from utils.training.functions import calculate_density_vectorized
import os

class DensityEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, stage='deploy'):
        self.stage= stage
        pass
    
    def fit(self, X, y=None):
        # No need to fit anything for this transformer
        return self

    def transform(self, X):
          
        X_transformed = X.copy()
        
        for file in ['bus_stop.csv', 'subway_stop.csv', 'train_stop.csv', 'places.csv']:
            file_name = file.split('.')[0]
            
            if self.stage=='train':
                data_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'utils\\training\\files')
                
            else:
                data_path = os.path.join(os.getcwd(), 'utils', 'training', 'files')
            file_df = pd.read_csv(os.path.join(data_path, file))

            radius_km = 0.1

            if file_name == 'places':
                for place in file_df['place_name'].unique():
 
                    file_df_filter = file_df[file_df['place_name'] == place]

                    X_transformed[place+"_density"] = calculate_density_vectorized(
                        X_transformed,
                        file_df_filter,
                        radius_km
                    )

            else:
                X_transformed[file_name+"_density"] = calculate_density_vectorized(
                        X_transformed,
                        file_df,
                        radius_km
                    )
        
        return X_transformed

import numpy as np
import pandas as pd

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, correlation_threshold=0.9):
        self.correlation_threshold = correlation_threshold 
        self.categorical_columns = None
        self.numerical_columns = None
        self.columns_to_drop = None

    def fit(self, X, y=None):

        self.numerical_columns = X.select_dtypes(include=['number']).columns.drop('price')
        
        ##### DROP CORRELATED COLUMNS ######
        correlation_matrix = X[self.numerical_columns].corr().abs()

        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        self.columns_to_drop = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > self.correlation_threshold)
        ]

        ##### CREATE NEW COLUMNS ######
        X_2 = X.copy()

        X_2['m2_price_median'] = X_2['price'] / X_2['covered_area']
        X_2['m2_price_median'] = X_2['m2_price_median'].replace([np.inf, -np.inf], np.nan)

        self.m2_prices_dict = X_2.groupby(['neighborhood', 'property_type', 'rooms']).agg({'m2_price_median': 'median'}).reset_index()

        X_2['m2_per_room'] = X_2['covered_area'] / X_2['rooms']
        X_2['m2_per_room'] = X_2['m2_per_room'].apply(lambda x: x if x!=np.inf else X_2['m2_per_room'].median())

        self.m2_room_dict = X_2.groupby(['neighborhood', 'property_type', 'rooms']).agg({'m2_per_room': 'median'}).reset_index()
        return self

    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_transformed = X.copy()
        
        ##### ADD NEW COLUMNS #####
        X_transformed = pd.merge(X_transformed, self.m2_prices_dict, how='left', on =['neighborhood', 'property_type', 'rooms'])
        X_transformed = pd.merge(X_transformed, self.m2_room_dict, how='left', on =['neighborhood', 'property_type', 'rooms'])
        
        ##### DROP CORRELATED COLUMNS ######
        X_transformed.drop(columns=self.columns_to_drop, inplace=True)
        X_transformed.index = X.index
        
        return X_transformed



from utils.training.functions import agg_values, null_impute_1, null_impute_2

class DataImputer(BaseEstimator, TransformerMixin):
    def __init__(self, column_dict):
        self.column_dict = column_dict

    def fit(self, X, y=None):
        
        median_values =  X.select_dtypes(include='number').median()
        
        mode_values = X.select_dtypes(include='object').mode().iloc[0]
        
        self.median_mode_dict = {**median_values.to_dict(), **mode_values.to_dict()}
        self.numeric_columns = X.select_dtypes('number').columns
        self.median = X[self.numeric_columns].median()

        self.agrup_dict = {}

        for col in self.column_dict.keys():
            agrup_columns = self.column_dict[col]
            for agrup_col in agrup_columns:
                if (col in X.columns) & (agrup_col in X.columns): 
                    agrup_values = agg_values(X, col, agrup_col)

                    if col not in self.agrup_dict.keys():            
                        self.agrup_dict[col] = {agrup_col:agrup_values}
                    else:
                        self.agrup_dict.get(col).update({agrup_col:agrup_values})
        
        
        return self

    def transform(self, X):
        X_transformed = X.copy()

        for col in self.column_dict.keys():
            if (col in X.columns):
                if X_transformed[col].isna().sum()>0:
                    agrup_columns = self.column_dict[col]

                    if col in ['rooms', 'full_bathrooms', 'total_area']:
                        for agrup_col in agrup_columns:
                            if (agrup_col in X.columns):

                                agrup_values = self.agrup_dict.get(col).get(agrup_col)
                                X_transformed[col] = null_impute_1(X_transformed, col, agrup_col, agrup_values)
                            
                        X_transformed[col] = X_transformed[col].fillna(X_transformed[col].dropna().median())
                    
                        
                    else: # col in ['city', 'neighborhood']:
                        lat_dict =  self.agrup_dict.get(col).get('latitude')
                        lon_dict =  self.agrup_dict.get(col).get('longitude')

                        if ((lat_dict is not None) and (lon_dict is not None)):
                            X_transformed[col] = X_transformed.\
                                apply(lambda row: null_impute_2(row['latitude'], row['longitude'], lat_dict, lon_dict), axis=1)
                        
                    if X_transformed[col].isna().sum()>0:
                        X_transformed[col] = X_transformed[col].fillna(self.median_mode_dict[col])
            
        X_transformed[self.numeric_columns] = X_transformed[self.numeric_columns].fillna(self.median)

        
        return X_transformed    

from sklearn.preprocessing import OneHotEncoder

class OneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        for col in ['id']:
            if col in X.columns:
                X.drop(col, axis=1, inplace=True)
        
        self.categorical_columns = X.select_dtypes(include=['object']).columns.drop('neighborhood')
        self.numerical_columns = X.select_dtypes(include=['number']).columns
        self.ohe = OneHotEncoder()
        self.ohe.fit(X[self.categorical_columns])
        
        return self

    def transform(self, X):
        
        for col in ['id']:
            if col in X.columns:
                X.drop(col, axis=1, inplace=True)

        X_transformed = X.copy()
        X_cat = self.ohe.transform(X_transformed[self.categorical_columns]).toarray()
        X_cat = pd.DataFrame(X_cat,
                             columns=self.ohe.get_feature_names_out(self.categorical_columns),
                             index=X_transformed.index)

        X_num = X_transformed[self.numerical_columns]
        X_transformed = pd.concat([X_num, X_cat], axis=1)
        
        return X_transformed

class LogScale(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):

        columns = X.select_dtypes('number').columns
        skew = X[columns].skew().abs()
        skew_log = X[columns].abs().apply(np.log1p).skew().abs()

        self.col_to_log = (skew_log < skew)[lambda x:x==True].index
        
        return self

    def transform(self, X):
        
        X_transformed = X.copy()
        X_scaled = X_transformed[self.col_to_log].abs().apply(np.log1p)
        
        col_not_log = X_transformed.columns[~X_transformed.columns.isin(X_scaled.columns)]

        X_transformed = pd.concat([X_scaled, X_transformed[col_not_log]],
                                  axis=1)
        
        return X_transformed

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_name):
        self.scaler_name=scaler_name
        pass
    
    def fit(self, X, y=None):
        
        self.numeric_columns = X.select_dtypes('number').columns
        self.obj_columns = X.select_dtypes('object').columns

        if self.scaler_name=='MinMaxScaler':
            self.scaler = MinMaxScaler()
        elif self.scaler_name=='StandardScaler':
            self.scaler = StandardScaler()
        elif self.scaler_name=='RobustScaler':
            self.scaler = RobustScaler()
        self.scaler.fit(X[self.numeric_columns].drop('price', axis=1))
        
        return self

    def transform(self, X):
        
        X_transformed = X.copy()
        X_scaled = self.scaler.transform(X_transformed[self.numeric_columns].drop('price', axis=1))
        X_scaled = pd.DataFrame(X_scaled,
                                columns=self.numeric_columns.drop('price'),
                                index = X_transformed.index)

        X_transformed = pd.concat([X_scaled, X_transformed[self.obj_columns] ,X['price']], axis=1)
        
        return X_transformed
    
    def inverse_transform(self, X):
        # Check if X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        X_transformed = X.copy()
        X_scaled = self.scaler.inverse_transform(X_transformed.drop('price', axis=1))
        X_scaled = pd.DataFrame(X_scaled,
                                     columns=X.columns.drop('price'),
                                     index = X.index)
        
        X_transformed = pd.concat([X_scaled, X['price']], axis=1)
        
        return X_transformed


from sklearn.decomposition import PCA

class PCA_(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components=n_components
        pass
    
    def fit(self, X, y=None):
        
        self.columns = X.columns.drop('price')
        self.pca = PCA(self.n_components)
        self.pca.fit(X.drop('price', axis=1))
        
        return self

    def transform(self, X):
        
        X_transformed = X.copy()

        X_pca = self.pca.transform(X_transformed.drop('price', axis=1))
        n_col = X_pca.shape[1]
        column_name = [f"PCA_{i}" for i in range(1, n_col+1)]

        X_pca = pd.DataFrame(X_pca,
                             columns=column_name,
                             index = X.index)
        
        X_transformed = pd.concat([X_pca, X['price']], axis=1)
    
        return X_transformed

    def inverse_transform(self, X):
        # Check if X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        X_transformed = X.copy()
        X_pca = self.pca.inverse_transform(X_transformed.drop('price', axis=1))
        X_pca = pd.DataFrame(X_pca,
                             columns=self.columns,
                             index = X_transformed.index)
        
        X_transformed = pd.concat([X_pca, X_transformed['price']], axis=1)
        
        return X_transformed


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

class Trainer(BaseEstimator, TransformerMixin):
    def __init__(self, test_dataset, model_name='GBM', cv=5, param_grids=None):
        self.test = test_dataset
        self.cv = cv
        self.param_grids = param_grids if param_grids is not None else {}
        
        if model_name == 'GBM':
            self.model = GradientBoostingRegressor(random_state=42)
            self.param_grid = self.param_grids.get('GBM', {})
        elif model_name == 'RandomForest':
            self.model = RandomForestRegressor(random_state=42)
            self.param_grid = self.param_grids.get('Random Forest', {})
        elif model_name == 'LinearRegression':
            self.model = LinearRegression()
            self.param_grid = self.param_grids.get('Ridge', {})
                
        else:
            raise ValueError("Model name not recognized. Choose from 'GBM', 'RandomForest', 'LinearRegression', or 'XGB'")
        
        self.grid_search = GridSearchCV(self.model, self.param_grid, cv=self.cv, scoring='neg_mean_absolute_error',n_jobs=-1)
    
    def train(self, train):

        if 'id' in train.columns:
            train.drop('id', axis=1, inplace=True)

        if 'id' in self.test.columns:
            self.test.drop('id', axis=1, inplace=True)

        y_train = train['price']
        X_train = train.drop('price', axis=1)

        y_test = self.test['price']
        X_test = self.test.drop('price', axis=1)

        self.grid_search.fit(X_train, y_train)

        y_pred_test = self.grid_search.predict(X_test)
        
        test_mae = mean_absolute_error(y_test, y_pred_test)

        result_dict = {
            'best_params': self.grid_search.best_params_,
            'train_cv_mae': round(-self.grid_search.best_score_, 0),
            'test_mae': round(test_mae,0)
        }

        return result_dict