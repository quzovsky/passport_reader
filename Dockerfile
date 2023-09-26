FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

RUN apt-get update && apt-get install -y libgl1-mesa-glx libgl1-mesa-dri tesseract-ocr


RUN mkdir -p /home/pass_app


COPY . /home/pass_app

WORKDIR /home/pass_app

RUN pip install --no-cache-dir --upgrade -r requirements.txt


EXPOSE 8000

ENV PATH="${PATH}:/usr/bin/tesseract"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]