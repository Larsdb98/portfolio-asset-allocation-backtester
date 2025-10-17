FROM python:3.11-slim

LABEL maintainer="Lars del Bubba" \
      description="Compute an optimal portfolio allocation based on the Makowitz efficient frontier for financial instruments. Allocation weights are computed by maximizing the Sharpe ratio."

# Non-interactive APT & timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Zurich

# ===== System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential tzdata ca-certificates pipx \
 && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && dpkg-reconfigure -f noninteractive tzdata \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Make Poetry available in PATH for non-login shells
ENV PATH=/root/.local/bin:$PATH

# ===== Poetry installation
RUN pipx install poetry
ENV PATH="/root/.local/bin:$PATH"

# ===== Workdir
WORKDIR /app

# ===== Copy only dependency manifests first
COPY pyproject.toml poetry.lock* ./
RUN  poetry install --no-interaction --no-ansi --no-root

COPY . .

# Might be used in the future
RUN mkdir -p /app/data /app/logs

# One can modify the flags here directly
CMD ["poetry", "run", "alloc"]
