FROM python:3.6

WORKDIR /substra

# Install kubectl
RUN curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl
RUN chmod +x ./kubectl
RUN mv ./kubectl /usr/local/bin/kubectl

# Install helm (v2.14.2)
RUN curl https://get.helm.sh/helm-v2.14.2-linux-amd64.tar.gz -o helm-v2.14.2-linux-amd64.tar.gz
RUN tar xzf helm-v2.14.2-linux-amd64.tar.gz
RUN mv linux-amd64/helm linux-amd64/tiller /usr/local/bin/

# Install skaffold (v1.0.1)
RUN curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/v1.0.1/skaffold-linux-amd64 && chmod +x skaffold && mv skaffold /usr/local/bin

# Copy scripts
COPY bin/* /substra/
