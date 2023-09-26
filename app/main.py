from fastapi import FastAPI,Header,Body,HTTPException,Depends, File
from . import crud, schemas
import cv2
import numpy as np



app = FastAPI()


async def get_api_key(api_key: str = Header(...)):
    return api_key


def validate_api_key(api_key: str = Depends(get_api_key)):
    if api_key != "fake_secret_apikey":
        raise HTTPException(status_code=403, detail="Invalid API key")

async def read_passport(img):
    im=crud.remove_face(img)
    img2=crud.procss_img(crud.brightness_adjust(im,'b'),'base')
    results=crud.find_cnts(im,img2,0)
    results=crud.extractor(results)
    results=crud.mrz_reader(img,results)
    results=crud.str_to_none(results)
    results=crud.refactor(results)
    return results


@app.post('/read-passport/',dependencies=[Depends(validate_api_key)])
async def read_pass(img :bytes=File(...)):
    image_array = np.frombuffer(img, np.uint8)
    cv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    res=await read_passport(cv_image)
    return res
