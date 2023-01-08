FROM python:3.9

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" , "--reload"]