FROM python:3.7

WORKDIR /usr/src/app

ARG SUBSTRA_GIT_REPO
ARG SUBSTRA_GIT_REF
RUN pip install --no-cache-dir "git+${SUBSTRA_GIT_REPO}@${SUBSTRA_GIT_REF}"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
