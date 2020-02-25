ARG SUBSTRA_VERSION=latest
FROM substra:$SUBSTRA_VERSION

WORKDIR /usr/src/app/substra-tests

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
