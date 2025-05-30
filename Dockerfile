FROM python:3.12.3-slim-bookworm AS build

RUN apt-get update && \
    apt-get install -y \
    libpq-dev python3-dev build-essential python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --force-reinstall -r requirements.txt

FROM python:3.12.3-slim-bookworm

EXPOSE 8501

RUN apt-get update && \
    apt-get install -y libpq5

ENV PATH="/opt/venv/bin:$PATH"

ENV PYTHONPYCACHEPREFIX="/root/.cache"
COPY --from=build /opt/venv /opt/venv

COPY . /app

WORKDIR /app

ENV STREAMLIT_EMAIL=""

ENTRYPOINT [ "bash", "-c", "streamlit run predicoes.py --server.port=8501 --server.address=0.0.0.0" ]



