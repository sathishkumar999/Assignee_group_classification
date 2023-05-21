FROM python:3.8
COPY . /ML_app
WORKDIR /ML_app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT classifeir_app:app