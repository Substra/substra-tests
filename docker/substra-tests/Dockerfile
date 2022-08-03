FROM python:3.8

WORKDIR /usr/src/app

### Dirty build to install docker client ###
# We need to install a docker client because substra debug mode relies on Docker (DinD in a side container)
# and we need a docker client to authenticate this docker against our container registry to download connect-tools.
RUN apt update && apt install --yes apt-transport-https ca-certificates curl gnupg lsb-release
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN apt update && apt install --yes docker-ce-cli


COPY substra ./substra
RUN pip install ./substra

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY requirements-workflows.txt .
RUN pip install --no-cache-dir -r requirements-workflows.txt

COPY . .
