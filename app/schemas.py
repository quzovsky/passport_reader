from pydantic import BaseModel,model_validator,validator
from datetime import date,datetime
from typing import Optional

class result(BaseModel):
    FathersName: Optional[str]
    DateOfIssue: Optional[date]
    PlaceOfBirth: Optional[str]
    Type: Optional[str]
    Country: Optional[str]
    PassportNumber: Optional[str]
    DateOfBirth: Optional[date]
    DateOfExpiry: Optional[date]
    Nationality: Optional[str]
    Sex: Optional[str]
    Name: Optional[str]
    SurName: Optional[str]

    @validator('DateOfIssue', 'DateOfBirth', 'DateOfExpiry', pre=True, always=True)
    def validate_dates(cls, value):
        if not value:
            return None
        try:
            return datetime.strptime(value, "%d/%m/%Y").date()
        except ValueError:
            return None

    @validator('FathersName', pre=True, always=True)
    def validate_fname(cls, value):
        if not value:
            return ''
        return ' '.join(value)
    @validator('Type',pre=True,always=True)
    def validate_type(cls,value):
        if not value:
            return None
        return value[0]
    @validator('Sex',pre=True,always=True)
    def validate_sex(cls,value):
        if not value: 
            return None
        if value!='F' and value!='M':
            return None
        else:
            return value