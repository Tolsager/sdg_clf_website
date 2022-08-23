FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code

RUN pip install --no-cache-dir -r requirements.txt

COPY . /code

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]