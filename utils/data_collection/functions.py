from utils.data_collection.classes import PropertyModel

def data_process(result, add_on_dict):
    """ Function to convert the JSON obtained from Mercado Libre into SQL format table"""
    
    # Field names
    fields = ['currency_id', 'price', 'attributes', 'location']
    location_fields = ['city','neighborhood', 'latitude', 'longitude']
    attributes_fields = ['OPERATION', 'PROPERTY_TYPE', 'ITEM_CONDITION', 'ROOMS',\
                          'BEDROOMS', 'FULL_BATHROOMS', 'TOTAL_AREA', 'COVERED_AREA',\
                              'HAS_AIR_CONDITIONING']
    
    # Dictionary to save all data
    data = {}
    # Dictionary to save each item
    item = {}
    # Get the Item ID
    id = result.get('id')

    for field in fields:
        if field == 'location':
            location = result.get(field)
            for loc_field in location_fields:
                if loc_field not in ['neighborhood', 'city']:
                    item[loc_field] = location.get(loc_field)
                else:
                    item[loc_field] = location.get(loc_field).get('name')
        
        elif field == 'attributes':
            attributes = result.get(field)

            for i in range(len(attributes)):
                att_field = attributes[i].get('id')
                if att_field in attributes_fields:
                    att_value = attributes[i].get('values')[0].get('name')
                    item[att_field.lower()] = att_value
    

        elif field == 'price':
            item[field] = int(result.get(field))
        
        elif field == 'currency_id':
            item[field] = result.get(field)

    for att in ['furnished', 'has_multipurpose_room', 'has_swimming_pool', 'has_gym']:
        value = add_on_dict[att]
        item[att] = False if value=='' else True

    for att in [ 'parking_lots']:
        value = add_on_dict[att]
        if value=='':
            item[att] = '[0-0]'
        else:
            item[att] = value
    
    
    data[id] = PropertyModel(**item).model_dump()
    return data



import requests
from utils.data_collection.attributes import att_dict

def get_data(acces_token, date=''):
    all_results = {}

    category      = 'MLA1459' # Properties
    state         = 'TUxBUENBUGw3M2E1'  # Capital Federal
    property_type = "242062,242069,242060"  # Appartments, PH, House
    operation     = "242075"  # Sell
    currency      = "USD"

    neighborhood_dict = att_dict['neighborhood']
    furnished_dict    = att_dict['furnished']
    multipurpose_dict = att_dict['has_multipurpose_room']
    swim_dict         = att_dict['has_swimming_pool']
    gym_dict          = att_dict['has_gym']
    parking_dict      = att_dict['parking_lots']

    for neighborhood in list(neighborhood_dict.keys()):
        print(neighborhood_dict.get(neighborhood))
        for furnished in furnished_dict:
            for multipurpose in multipurpose_dict:
                for swim in swim_dict:
                    for gym in gym_dict:
                        for parking in parking_dict:
                            
                            offset = 0    

                            while offset < 4000:
                                # Parámetros de búsqueda
                                params = {
                                    "access_token": acces_token,
                                    "since": date,
                                    "category": category,
                                    "state": state,
                                    "property_type": property_type,
                                    "neighborhood": neighborhood,                                    
                                    "currency": currency,
                                    "operation": operation,
                                    "furnished": furnished,
                                    "has_multipurpose_room": multipurpose,
                                    "has_swimming_pool": swim,
                                    'has_gym': gym,
                                    "parking_lots": parking,
                                    "offset": offset,
                                    "limit": 50
                                }

                                url = f"https://api.mercadolibre.com/sites/MLA/search"

                                response = requests.get(url, params=params)

                                if response.status_code == 200:
                                    try:
                                        data = response.json()
                                        results = data.get('results')

                                        add_on_dict = {
                                            'furnished': furnished,
                                            'has_multipurpose_room': multipurpose,
                                            'has_swimming_pool': swim,
                                            'has_gym': gym,
                                            'parking_lots': parking
                                        }

                                        if not results:
                                            break
                                        else:
                                            for result in results:
                                                results_proc = data_process(result, add_on_dict)
                                                
                                                id=list(results_proc.keys())

                                                if not all_results.get(id[0]):
                                                  
                                                    all_results.update(results_proc)
                                                
                                            offset += 50

                                    except Exception as e:
                                        print(e)
                                else:
                                    print(f"Error: {response.status_code} - {response.text}")
    
    return all_results


