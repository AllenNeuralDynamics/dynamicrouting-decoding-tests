# hash:sha256:b806b6e392cc981c071a62dbd81dd778a4793e00f2d4ca79e7bd2daef43c16f9
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV TOKIO_WORKER_THREADS=1
ENV RAYON_NUM_THREADS=1
ENV POLARS_MAX_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip3 install -U --no-cache-dir \
    aind-data-schema==1.1.1 \
    aind-data-schema-models==0.5.7 \
    code-ocean-aux-tools==1.1.3 \
    hdmf==3.14.6 \
    numba==0.60.0 \
    pandas[all]==2.2.3 \
    polars[plot,deltalake]==1.32.2 \
    pydantic==2.7.4 \
    pynwb==2.8.2 \
    s3fs==2024.10.0 \
    scikit-learn==1.5.2 \
    universal-pathlib==0.1.4 \
    zarr==2.17.2 \
    -e git+https://github.com/AllenInstitute/dynamic_routing_analysis@ea8047ee9872af7e43a494a2ea29ed84c9a91007#egg=dynamic-routing-analysis

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
