FROM python:3.9.11

WORKDIR /code

RUN pip install --upgrade pip
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /code

EXPOSE 8000

ENTRYPOINT ["bash", "/code/run.sh"]