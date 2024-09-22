import streamlit as st
from utils.deployment.geo_codification import get_lat_lon
from utils.deployment.model_loader import LoadNeighborhoodVar
import requests

neighborhood_var = LoadNeighborhoodVar()

# Sorted list of neighborhoods
neighborhoods = sorted(['San Cristóbal', 'Versalles', 'Villa Real', 'Velez Sarsfield',
                        'La Boca', 'Colegiales', 'Santa Rita', 'Nueva Pompeya',
                        'Balvanera', 'Palermo Soho', 'Paternal', 'Constitución',
                        'Villa Gral. Mitre', 'Coghlan', 'Villa Soldati', 'Villa Ortúzar',
                        'Chacarita', 'Parque Chas', 'Palermo Hollywood', 'Agronomía',
                        'Monserrat', 'Las Cañitas', 'Barrio Norte', 'Belgrano R',
                        'Recoleta', 'Villa Riachuelo', 'Botánico', 'San Nicolás',
                        'Congreso', 'Palermo Viejo', 'Palermo Chico', 'Once',
                        'Belgrano Chico', 'Belgrano C', 'Retiro', 'Palermo Nuevo',
                        'Mataderos', 'Liniers', 'Villa Devoto', 'Villa Urquiza',
                        'Villa Lugano', 'Flores', 'Villa Luro', 'Villa Pueyrredón',
                        'Caballito', 'Barracas', 'Floresta', 'Villa del Parque',
                        'Saavedra', 'Palermo', 'Parque Avellaneda', 'Parque Chacabuco',
                        'Monte Castro', 'Almagro', 'Boedo', 'Parque Patricios',
                        'Villa Crespo', 'Núñez', 'San Telmo', 'Belgrano'])

# Main function for Streamlit UI
def main():
    st.title('Property Price Prediction')

    # Fixed field: City
    st.write('**City:** Capital Federal')

    col1, col2 = st.columns(2)

    # Contenido de la primera columna
    with col1:
        neighborhood = st.selectbox('Neighborhood', neighborhoods)
        street = st.text_input('Street')
        number = st.number_input('Number', min_value=1, max_value=10000, step=1)
        property_type = st.selectbox('Property Type', ['Apartment', 'House', 'PH'])
        item_condition = st.selectbox('Condition', ['New', 'Used'])
        
    # Contenido de la segunda columna
    with col2:
        rooms = st.number_input('Number of Rooms', min_value=1, max_value=10, step=1)
        bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=9, step=1)
        full_bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=9, step=1)
        total_area = st.number_input('Total Area (m²)', min_value=20, max_value=1500, step=1)
        covered_area = st.number_input('Covered Area (m²)', min_value=20, max_value=1000, step=1)
        parking_lots = st.number_input('Parking Lots', min_value=0, max_value=5, step=1)
    
    has_air_conditioning = st.checkbox('Has Air Conditioning?')
    furnished = st.checkbox('Is Furnished?')
    has_multipurpose_room = st.checkbox('Has Multipurpose Room?')
    has_swimming_pool = st.checkbox('Has Swimming Pool?')
    has_gym = st.checkbox('Has Gym?')
    
    # Prediction button
    if st.button('Predict Price'):
        # Prepare the input data as a dictionary
        try:
            longitude, latitude = get_lat_lon(street.strip().lower(), str(number).strip())
            is_correct_address = True
        except:
            is_correct_address = False

        if is_correct_address:

            property_type_dict = {'Apartment': 'Departamento',
                                'House': 'Casa',
                                'PH': 'Ph'}
            input_data = {
                "id": "1",
                "city": "Capital Federal",
                "neighborhood": neighborhood,
                "latitude": latitude,
                "longitude": longitude,
                "operation": "Venta",
                "property_type": property_type_dict[property_type],
                "item_condition": "Nuevo" if item_condition=="New" else "Usado",
                "rooms": str(rooms).strip(),
                "bedrooms": str(bedrooms).strip(),
                "full_bathrooms": str(full_bathrooms).strip(),
                "total_area": f"{str(total_area).strip()} m²",
                "covered_area": f"{str(covered_area).strip()} m²",
                "has_air_conditioning": "Sí" if has_air_conditioning else "No",
                "furnished": furnished,
                "has_multipurpose_room": has_multipurpose_room,
                "has_swimming_pool": has_swimming_pool,
                "has_gym": has_gym,
                "parking_lots": f"[{parking_lots}-{parking_lots}]",
                "currency_id": "USD",
                "price": 40000
            }

            
            backend_url = "http://fastapi_service:8000/model/predict"

            # Hacer la solicitud POST
            response = requests.post(backend_url, json=input_data)

            # Mostrar la respuesta del servidor
            if response.status_code == 200:
                pred = response.json()
                print("Prediction Response:", pred)

                factor=neighborhood_var[neighborhood]
                price_min = pred*(1-factor)
                price_max = pred*(1+factor)

            else:
                print(f"Request failed with status code {response.status_code}: {response.text}")
                pred = 'fail'
            
            st.success(f'**Estimated Price: ${pred:,.2f}**')
            st.info(f'**Price Range: ${price_min:,.2f} - ${price_max:,.2f}**')

        else:
            st.write('Address not found. Try again, please.')

# Run the main function
if __name__ == '__main__':
    main()
