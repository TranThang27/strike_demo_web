FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libegl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_PREFERENCE=only-managed

RUN uv python install 3.13

WORKDIR /app

# Sao chép các tệp quản lý môi trường trước
COPY mjlab/uv.lock mjlab/pyproject.toml /app/mjlab/

WORKDIR /app/mjlab
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-editable --no-dev

WORKDIR /app
# Sao chép toàn bộ dự án
COPY . /app

WORKDIR /app/mjlab
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev

WORKDIR /app
ENV MUJOCO_GL=egl
EXPOSE 8080

RUN chmod +x play.sh
CMD ["./play.sh"]
