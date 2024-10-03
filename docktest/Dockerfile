FROM python:3.12

WORKDIR /docktest/app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]