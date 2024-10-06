FROM python:3.7

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN python train_model.py

EXPOSE 5000

CMD ["python", "app.py"]