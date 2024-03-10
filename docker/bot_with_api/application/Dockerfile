FROM python:3.10-slim-bullseye
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 sudo
RUN sudo rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip \
    -r requirements.txt
EXPOSE 5001
COPY . .
CMD ["python", "app.py"]