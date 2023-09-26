import  cv2
from PIL import Image,ImageOps
import numpy as np
from statistics import mean
import pytesseract
import math
from passporteye import read_mrz
import datetime

from .schemas import result



def remove_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    for (x,y,w,h) in faces:
        if x<200:
            cv2.rectangle(image, (x-40, y-60), (x + w + 30, y + h + 100), (255, 255, 255), -1)
    return(image)

def no_noise(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)


def procss_img(img,i):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    blur=cv2.GaussianBlur(thresh,(7,7),0)
    blur=no_noise(blur)
    thresh=cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    nonoise=no_noise(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18  ,1))#(18,1) for iranian (12,1)foreign
    morph = cv2.morphologyEx(nonoise, cv2.MORPH_CLOSE, kernel)
    morph=no_noise(morph)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))#(1,2) for iranian (1,1)foreign
    dilate= cv2.dilate(morph, kernel, iterations=2)
    dilate=cv2.threshold(dilate,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    dilate=cv2.copyMakeBorder(dilate,200,200,200,200,cv2.BORDER_CONSTANT,value=(0,0,0))
    return dilate


def brightness_adjust(img,flag):
    im=Image.fromarray(img)
    im=ImageOps.grayscale(im)
    pix_val = list(im.getdata())
    m=mean(pix_val)
    if flag=='b':
        if m<=180 and m>160:
            return(cv2.convertScaleAbs(img,20,2))
        elif m>180 and m<=210:
            return(cv2.convertScaleAbs(img,10000,1.7))
        elif m>210:
            return(cv2.convertScaleAbs(img,10000,1.5))
        elif m<=160 and m>140:
            return(cv2.convertScaleAbs(img,500,2.5))
        else:
            return(cv2.convertScaleAbs(img,500,3))#(2.7)
    if flag=='r':
        if m<=180 and m>160:
            return(cv2.convertScaleAbs(img,1000,1))#(1000,1.5)
        elif m>180 and m<=210:
            return(cv2.convertScaleAbs(img,1000,1.5))#(1.5,2)
        elif m>210:
            return(cv2.convertScaleAbs(img,1000,0.5))
        elif m<=160 and m>140:
            return(cv2.convertScaleAbs(img,1000,2))#(1000,2)
        else:
            return(cv2.convertScaleAbs(img,1000,4))#(1000,3)
        






def find_cnts(image,processed,m):
    image=cv2.copyMakeBorder(image,200,200,200,200,cv2.BORDER_CONSTANT,value=(0,0,0))
    cnts= cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts=cnts[0] if len(cnts)==2 else cnts[1]
    cnts= sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
    i=0
    results=[]
    dpbirth=''
    fname=''
    issue=''
    im2=image.copy()
    for c in cnts:
        x, y, w, h= cv2.boundingRect(c)
        if h<50 :#and h>15 and w>20 :
            roi= image[y-5:y+h+5, x-10:x+w+10]
            roi=brightness_adjust(roi,'r')
            if len(roi)==0:
                i+=1
            else:
                ocr_result=pytesseract.image_to_string(roi, lang='eng', config=' --psm 7')
                if 'Place' in ocr_result or 'Piace' in ocr_result or 'Birth' in ocr_result or 'Bith' in ocr_result or ('Date' in ocr_result and '&' in ocr_result):#('P' in ocr_result and 'B' in ocr_result and 'D' in ocr_result and 'a' in ocr_result) or ('place' in ocr_result or 'Birth' in ocr_result):
                    if w>(15*h):
                        dpbirth=ocr_result
                    elif w<=(5*h):#4?
                        roi=brightness_adjust(image[y-13:y+h+13,x-13:x+(5*w)],'r')#5
                        dpbirth=pytesseract.image_to_string(roi,lang='eng',config='--psm 7')
                    else:
                        roi=brightness_adjust(image[y-13:y+h+13,x-13:x+(4*w)],'r')
                        dpbirth=pytesseract.image_to_string(roi,lang='eng',config='--psm 7')
                if ('ss' in ocr_result and 'Date' in ocr_result) or ('I' in ocr_result and 'Date' in ocr_result):#('ss' in ocr_result and 'Date' in ocr_result) or ('I' in ocr_result and 'D' in ocr_result and 'a' in ocr_result):
                    if w>(8*h):
                        issue=ocr_result
                    else:
                        roi=brightness_adjust(image[y-13:y+h+13,x-13:x+(3*w)],'r')
                        issue=pytesseract.image_to_string(roi,lang='eng',config='--psm 7')
                if 'Father' in ocr_result or 'ther' in ocr_result:#'Father' in ocr_result or ('F' in ocr_result and 'Name' in ocr_result and 'a' in ocr_result):
                    if w>(10*h):#8
                        fname=ocr_result
                    elif w<=(4*h):#4
                        roi=brightness_adjust(image[y-13:y+h+13,x-13:x+(7*w)],'r')
                        fname=pytesseract.image_to_string(roi,lang='eng',config='--psm 7')

                    else:
                        roi=brightness_adjust(image[y-13:y+h+13,x-13:x+(3*w)],'r')
                        fname=pytesseract.image_to_string(roi,lang='eng',config='--psm 7')
                i+=1
    results.append(fname)
    results.append(issue)
    results.append(dpbirth)
    return(results)




def extractor(input):
    output={"FathersName":'',"DateOfIssue":'',"PlaceOfBirth":''}
    valid_chars=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
    valid_dates=['1','2','3','4','5','6','7','8','9','0','/']
    for i in range(len(input)):#fathername
        input[i]=input[i].split('\n',1)[0]#stritp('\n')
        if i==0 and input[i]!='':
            for j in input[i]:
                if j not in valid_chars:
                    input[i]=input[i].replace(j,' ')
                    #input[i]=input[i].translate({ord(j): None})
            temp=input[i].split(' ')
            temp=[x for x in temp if x!='']
            j=0
            k=len(temp)
            while j<k:
                d=0
                if 'N' in temp[j] or 'm' in temp[j]:
                    if j+1<k:
                        if temp[j+1][0] in valid_chars:
                            temp=temp[j+1:]
                            k=len(temp)
                            j=k
                            d=1
                            temp=[x for x in temp if (len(x)>=3 and not x.islower())] ###
                if j+1==k and d==0:
                    temp=[]
                j+=1
            output["FathersName"]=temp
        if i==1 and input[i]!='':#issue date
            for j in input[i]:
                if j not in valid_chars and j not in valid_dates:
                    input[i]=input[i].replace(j,' ')
                    #input[i]=input[i].translate({ord(j): None})
            temp=input[i].split(' ')
            temp=[x for x in temp if x!='']
            j=0
            k=len(temp)
            while j<k:
                d=0
                if 'ss' in temp[j]:
                    if j+1<k:
                        if temp[j+1][0] in valid_dates:
                            temp=temp[j+1:]
                            k=len(temp)
                            j=k
                            d=1
                            temp=[x for x in temp if (len(x)>=3 and not x.islower())]
                            if temp:
                                temp=temp[0]
                if j+1==k and d==0:
                    temp=[]
                j+=1
            output["DateOfIssue"]=temp
        if i==2 and input[i]!='':#dpbirth
            for j in input[i]:
                if j not in valid_chars and j not in valid_dates:
                    input[i]=input[i].replace(j,' ')
                    #input[i]=input[i].translate({ord(j): None})
            temp=input[i].split(' ')
            temp=[x for x in temp if x!='']
            j=0
            k=len(temp)
            while j<k:
                d=0
                if 'h' in temp[j] or 'B' in temp[j] or 'r' in temp[j]:
                    if j+1<k:
                        if temp[j+1][0] in valid_dates:
                            temp=temp[j+1:]
                            k=len(temp)
                            j=k
                            d=1
                            temp=[x for x in temp if (len(x)>=3 and not x.islower())]
                if j+1==k and d==0:
                    temp=[]
                j+=1
            if temp!=[]:
                try:
                    output["PlaceOfBirth"]=temp[1]
                except:
                    if temp[0][0] not in valid_dates:
                        output["PlaceOfBirth"]=temp[0]
                    else: output["DateOfIssue"]=temp[0]

    return(output)


def mrz_reader(img,results):
    today=datetime.date.today()
    year=today.year-int(today.year/100)*100
    h,w,c=img.shape#c
    y1=math.floor((0.6)*h)
    img=img[y1:h]
    _, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
    img = im_arr.tobytes()
    mrz=read_mrz(img)
    if mrz:
        r2=[mrz.type,mrz.country,mrz.number,mrz.date_of_birth,mrz.expiration_date,mrz.nationality,mrz.sex,mrz.names,mrz.surname]
        for j in range(len(r2)):
            r2[j]=r2[j].split('  ',1)[0]
            r2[j]=r2[j].split('<',1)[0]
            r2[j]=r2[j].split('KK',1)[0]
            if j==3:
                try:
                    r2[j]=int(r2[j])
                    year2=int(r2[j]/10000)
                    month=int(r2[j]/100-year2*100)
                    if month<10:
                            month='0{}'.format(month)
                    day=r2[j]-((year2*10000)+(int(month)*100))
                    if year2<=year:
                        r2[j]='{d}/{m}/20{y}'.format(y=year2,m=month,d=day)
                    elif year2>year:
                        r2[j]='{d}/{m}/19{y}'.format(y=year2,m=month,d=day)
                except:
                    r2[j]=''
            if j==4: 
                try:
                    r2[j]=int(r2[j])               
                    year2=int(r2[j]/10000)
                    month=int(r2[j]/100-year2*100)
                    if month<10:
                            month='0{}'.format(month)
                    day=r2[j]-((year2*10000)+(int(month)*100))
                    r2[j]='{d}/{m}/20{y}'.format(y=year2,m=month,d=day)
                except:
                    r2[j]=''
        results["Type"]=r2[0]
        results["Country"]=r2[1]
        results["PassportNumber"]=r2[2]
        results["DateOfBirth"]=r2[3]
        results["DateOfExpiry"]=r2[4]
        results["Nationality"]=r2[5]
        results["Sex"]=r2[6]
        results["Name"]=r2[7]
        results["SurName"]=r2[8]
        #print('{}  =   {}'.format(i,results))
        #results.clear()
        #i+=2
        return(results)
    else:
        results["Type"]=''
        results["Country"]=''
        results["PassportNumber"]=''
        results["DateOfBirth"]=''
        results["DateOfExpiry"]=''
        results["Nationality"]=''
        results["Sex"]=''
        results["Name"]=''
        results["SurName"]=''
        return results
    
def refactor(results:dict):
    temp=result(FathersName=results['FathersName'],DateOfIssue=results['DateOfIssue'],PlaceOfBirth=results['PlaceOfBirth'],Type=results['Type'],Country=results['Country'],
                PassportNumber=results['PassportNumber'],DateOfBirth=results['DateOfBirth'],DateOfExpiry=results['DateOfExpiry'],Nationality=results['Nationality'],Sex=results['Sex'],Name=results['Name'],SurName=results['SurName'])
    return temp

def str_to_none(input: dict):
    result_dict = {} 
    
    for key, value in input.items():
        if value == '' or value == []:
            result_dict[key] = None
        else:
            result_dict[key] = value 
    
    return result_dict 
