FROM python:3.10-slim-bullseye
WORKDIR /bot
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 sudo
RUN sudo rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip \
    -r requirements.txt
COPY . .
CMD python bot.py