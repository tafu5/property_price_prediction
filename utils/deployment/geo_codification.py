
import json
import geopandas as gpd
from shapely.geometry import Point
from usig_normalizador_amba import NormalizadorAMBA
import requests

def get_lat_lon(street: str, number: str):
    nd = NormalizadorAMBA(include_list=['caba'])
    res = nd.normalizar(street + ' ' + number)
    street=res[0].calle.nombre
    number=res[0].altura
    
    url=f'https://datosabiertos-usig-apis.buenosaires.gob.ar/geocoder/2.2/geocoding?cod_calle={street}&altura={number}'
    response = requests.get(url)
    lat_lon = response.text

    lat_lon = json.loads(lat_lon.strip('()'))

    lat = float(lat_lon['x'])
    lon = float(lat_lon['y'])

    gdf = gpd.GeoDataFrame(
        {'geometry': [Point(lat, lon)]},
        crs='EPSG:9498' 
    )

    # Convert to EPSG:4326 (lat/lon)
    gdf = gdf.to_crs(epsg=4326)

    # get coordinates
    longitude, latitude = gdf.geometry.x[0], gdf.geometry.y[0]

    factor_latitude = 1.0079866966323796
    factor_longitude = 1.015116062181048

    longitude, latitude = longitude*factor_longitude, latitude*factor_latitude
    
    return longitude, latitude