from utils.data_collection.functions import get_data
from utils.data_collection.config import ACCESS_TOKEN
from utils.training.functions import mysql_conn
# Will extract historic data from the API and the transform it as a SQL table
all_results = get_data(acces_token=ACCESS_TOKEN,
                            date='')

# Convert results to a DataFrame
import pandas as pd
data = pd.DataFrame.from_dict(all_results, orient='index')

# Load data into a SQL table
engine = mysql_conn()

# Get id already stored
query = f"SELECT id FROM properties"
id_stored = pd.read_sql(query, engine)

# filter data
filtered_data = data[~data.index.isin(id_stored)]

# Delete duplicated: The ids could be extracted more than once or the same sample could have different id
filtered_data = filtered_data[~filtered_data.index.duplicated(keep='first')]
filtered_data.drop_duplicates(inplace=True)

from datetime import datetime
# Add date to the DF
filtered_data['date'] = [datetime.today().date()] * len(filtered_data)
# Sent data to MySQL
filtered_data.reset_index().rename(columns={'index':'id'}).to_sql('properties', 
                                                                  con=engine, 
                                                                  if_exists='append', 
                                                                  index=False)
# Prints
if len(filtered_data)>0:

    print(f"{len(filtered_data)} samples have been stored!!")
else:
    print("No samples to add")