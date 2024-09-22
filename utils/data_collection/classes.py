from typing import Optional
from pydantic import BaseModel

# Data Validator
class PropertyModel(BaseModel):
    city: Optional[str] = None
    neighborhood: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    operation: Optional[str] = None
    property_type: Optional[str] = None
    item_condition: Optional[str] = None
    rooms: Optional[int] = None
    bedrooms: Optional[int] = None
    full_bathrooms: Optional[int] = None
    total_area: Optional[str] = None
    covered_area: Optional[str] = None
    has_air_conditioning: Optional[str] = None
    furnished: Optional[bool] = None
    has_multipurpose_room: Optional[bool] = None
    has_swimming_pool: Optional[bool] = None
    has_gym: Optional[bool] = None
    parking_lots: Optional[str] = None
    currency_id: Optional[str] = None
    price: Optional[int] = None


