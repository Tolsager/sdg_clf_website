FROM 3.9.11-alpine

RUN pip install --no-cache-dir --upgrade pip --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt