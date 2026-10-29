# GraXpert's AI denoise, as a tool to run beside quietsky on a linear master.
#
#     docker compose run --rm graxpert IN.fits OUT.fits --strength 0.5
#
# The PyPI package is pinned, the denoise model is downloaded here and kept
# in the image, and nothing is fetched at run time.
FROM python:3.13-slim

# GraXpert imports tkinter even on the command line; no display is needed.
RUN apt-get update && apt-get install -y --no-install-recommends libtk8.6 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Models live here, readable by whoever runs the container; preferences
    # and logs go to /tmp, as the runtime user may not own anything else.
    XDG_DATA_HOME=/opt/graxpert \
    XDG_CONFIG_HOME=/tmp \
    XDG_STATE_HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    HOME=/tmp

RUN pip install "graxpert[cpuonly]==3.2.0a3"

# The denoise model, by version, fetched now so that a run needs no network.
# Versions available on 2026-10-29: 1.0.0, 1.1.0, 2.0.0, 3.0.0, 3.0.1, 3.0.2.
ARG DENOISE_MODEL=3.0.2
RUN python -c "\
from graxpert.ai_model_handling import download_version, denoise_ai_models_dir, list_local_versions; \
from graxpert.s3_secrets import denoise_bucket_name; \
download_version(denoise_ai_models_dir, denoise_bucket_name, '${DENOISE_MODEL}'); \
assert '${DENOISE_MODEL}' in [v['version'] for v in list_local_versions(denoise_ai_models_dir)], 'model not downloaded'" \
    && chmod -R a+rX /opt/graxpert && du -sh /opt/graxpert

COPY entrypoint.sh /usr/local/bin/entrypoint
COPY denoise.py /usr/local/bin/graxpert-denoise
RUN chmod +x /usr/local/bin/entrypoint /usr/local/bin/graxpert-denoise

WORKDIR /app
ENTRYPOINT ["entrypoint", "graxpert-denoise"]
