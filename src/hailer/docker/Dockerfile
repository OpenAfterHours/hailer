# The Hailer kernel image: marimo, Polars, fastexcel, DuckDB, altair and plotly, plus the hailer package
# (the notebook helpers in hailer.periods and the forwarder in hailer._forward; nothing else of
# Hailer runs in here: no LangChain, no keyring, no API key).
#
# Build it with `uvx hailer kernel build`: that copies this file and the installed hailer package
# into a temporary context and passes the build args below from the versions Hailer runs with
# (hailer.kernel_image.prepare_context). The version label must equal the Hailer version, or
# Hailer refuses to use the image.
ARG BASE_IMAGE=python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
FROM ${BASE_IMAGE}

# Use the builder's bundled Dockerfile frontend: no public frontend image to download.
# BuildKit is required for the optional pip configuration and CA secret mounts below.
USER 0
ENTRYPOINT []

ARG MARIMO_VERSION
ARG POLARS_VERSION
ARG FASTEXCEL_VERSION
ARG DUCKDB_VERSION
ARG HAILER_VERSION
ARG ALTAIR_VERSION=6.3.0
ARG PLOTLY_VERSION=7.1.0

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# A venv supports different Python versions/installation layouts and distro-managed Python.
# ensurepip bootstraps pip from Python's bundled wheel, without a separate download.
RUN python3 -c 'import sys; assert sys.version_info >= (3, 12), "Hailer needs Python 3.12 or newer"' \
    && python3 -m venv /opt/hailer-venv
ENV PATH="/opt/hailer-venv/bin:${PATH}"

RUN --mount=type=secret,id=pip_config --mount=type=secret,id=pip_cert \
    test -n "$MARIMO_VERSION" && test -n "$POLARS_VERSION" && test -n "$FASTEXCEL_VERSION" && test -n "$DUCKDB_VERSION" && test -n "$HAILER_VERSION" \
    || { echo "Missing build args: build with uvx hailer kernel build (see hailer.kernel_image)." >&2; exit 1; }; \
    if [ -f /run/secrets/pip_config ]; then \
        unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST; \
        export PIP_CONFIG_FILE=/run/secrets/pip_config; \
    fi; \
    if [ -f /run/secrets/pip_cert ]; then export PIP_CERT=/run/secrets/pip_cert; fi; \
    python -m pip install --no-cache-dir --disable-pip-version-check \
        marimo==${MARIMO_VERSION} \
        polars==${POLARS_VERSION} \
        fastexcel==${FASTEXCEL_VERSION} \
        duckdb==${DUCKDB_VERSION} \
        altair==${ALTAIR_VERSION} \
        plotly==${PLOTLY_VERSION}

COPY hailer/ /opt/hailer-package/hailer/
RUN python -c 'import pathlib, sysconfig; pathlib.Path(sysconfig.get_path("purelib"), "hailer.pth").write_text("/opt/hailer-package\n")'

# /work is the kernel's working directory (read-only, like the rest of the image):
# - .marimo.toml is marimo's user configuration (marimo looks in its working directory first):
#   a notebook's cells run when it opens, so the starter notebook's globals (DATA_DIR,
#   data_files, ...) are there before anyone runs a cell. marimo 0.24's editor takes this
#   setting from the user configuration; a pyproject.toml [tool.marimo] section does not make it
#   run the cells. The user's own marimo configuration is never mounted (it can hold API keys).
# - An empty hailer.toml marks /work as the workspace, so the starter notebook finds
#   WORKSPACE = /work and DATA_DIR = /work/data.
RUN mkdir -p /work /home/analyst \
    && chown 1000:1000 /home/analyst \
    && printf '[runtime]\nauto_instantiate = true\n' > /work/.marimo.toml \
    && touch /work/hailer.toml

# Numeric IDs avoid requiring useradd or an unused account name in a corporate base.
ENV HOME=/home/analyst
USER 1000:1000
WORKDIR /work

LABEL org.opencontainers.image.version=${HAILER_VERSION} \
      org.opencontainers.image.source=https://github.com/OpenAfterHours/hailer
