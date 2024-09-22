def mysql_conn():
    from utils.data_collection.config import username, password, host, database
    from sqlalchemy import create_engine

    connection_string = f"mysql+mysqlconnector://{username}:{password}@{host}/{database}"
    engine = create_engine(connection_string)
    return engine
    
def range_number_finder(number):
    if not isinstance(number, str):
        number = str(number)

    num=[]
    secuence = int(number[0])
    i=0
    while i<len(number):
        num.append(str(secuence))
        secuence+=1
        i+=1

    return "".join(num)

def rep_number_finder(number):
    if not isinstance(number, str):
        number = str(number)

    firts_char = number[0]
    rep_number = firts_char * len(number)
    return rep_number


import numpy as np
def train_test_split(df, train_size=0.8, random_state=None):
    # Opcional: Fijar una semilla para la aleatoriedad
    if random_state is not None:
        np.random.seed(random_state)
    
    # Mezcla las filas del DataFrame
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    
    # Calcula el número de filas para el conjunto de entrenamiento
    train_length = int(len(shuffled_df) * train_size)
    
    # Divide el DataFrame en train y test
    train_df = shuffled_df.iloc[:train_length]
    test_df = shuffled_df.iloc[train_length:]
    
    return train_df, test_df


import numpy as np

def agg_values(data, agg_col, agrup_col):
    if agg_col == 'neighborhood':
        agrup_values = data.dropna().groupby(agg_col).agg({agrup_col: 'median'}).squeeze()
    else:
        agrup_values = data.dropna().groupby(agrup_col).agg({agg_col: 'median'}).squeeze()

    return agrup_values


def null_impute_1(data, col, agrup_col, agrup_values):    
    new_col = data[agrup_col].map(agrup_values)
    data_2 = pd.DataFrame({col: data[col], agrup_col: new_col},
                          index=data.index)

    imputed_col = data_2.apply(lambda x: x[agrup_col] if pd.isna(x[col]) else x[col],
                              axis=1)

    return imputed_col

from geopy.distance import distance
import pandas as pd

def null_impute_2(lat, lon, lat_dict, lon_dict):
    if all(pd.notna([lat, lon])):
    
        min_distance = float('inf')
        closest_suburb = None
        
        for suburb, avg_lat in lat_dict.items():
            avg_lon = lon_dict.get(suburb)
            if avg_lon is None:
                continue
            
            dist = distance((lat, lon), (avg_lat, avg_lon)).km
            
            if dist < min_distance:
                min_distance = dist
                closest_suburb = suburb

    else:
        closest_suburb = np.nan
    
    return closest_suburb



def calculate_density_vectorized(df, df_places, radius_km):
    latitudes = df['latitude'].values
    longitudes = df['longitude'].values
    
    # Convertir a radianes
    lat1 = np.radians(latitudes)
    lon1 = np.radians(longitudes)
    lat2 = np.radians(df_places['latitude'].values)
    lon2 = np.radians(df_places['longitude'].values)
    
    # Radio de la Tierra
    R = 6371
    
    # Inicializar una lista para los resultados
    counts = np.zeros(len(latitudes))
    
    # Calcular distancias para cada punto en transformed_data_pipeline_2
    for i in range(len(latitudes)):
        dlat = lat2 - lat1[i]
        dlon = lon2 - lon1[i]
        
        a = np.sin(dlat / 2)**2 + np.cos(lat1[i]) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dist = R * c
        
        counts[i] = np.sum(dist <= radius_km)
    
    return counts

