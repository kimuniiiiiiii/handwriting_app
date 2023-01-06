FROM python:3.9

COPY requirements.txt /app/requirements.txt

# RUN mkdir /app
WORKDIR /app

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . /app/.