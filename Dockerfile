FROM ghcr.io/astral-sh/uv:0.9.30-python3.13-bookworm-slim

# The virtualenv lives outside /app so the source bind mount doesn't hide it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    HOME=/tmp

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Editable install: the package resolves to /app/src, which compose bind-mounts.
COPY src ./src
RUN uv sync --frozen

CMD ["quietsky", "--help"]
