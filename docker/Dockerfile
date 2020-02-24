ARG SUBSTRA_VERSION=latest
FROM substra:$SUBSTRA_VERSION

RUN mkdir -p /usr/src/app/substra-tests
COPY ./requirements.txt /usr/src/app/substra-tests
COPY ./*.yaml /usr/src/app/substra-tests/
COPY ./Makefile /usr/src/app/substra-tests/
COPY ./tests /usr/src/app/substra-tests/tests/
COPY ./substratest /usr/src/app/substra-tests/substratest/

WORKDIR /usr/src/app/substra-tests
RUN pip3 install -r requirements.txt
