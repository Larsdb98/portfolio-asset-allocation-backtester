#!/usr/bin/env bash
set -e

# ------------------------
# Defaults
# ------------------------
IMAGE_NAME="portfolio-asset-allocation-backtester"
CONTAINER_NAME="paab-container"
DOCKERFILE="Dockerfile"
PLATFORM=""           # e.g. "--platform linux/amd64" or "--platform linux/arm64"
BUILD=false
ENV_FILE=".env"       # optional
WORKDIR_MOUNT="$(pwd)"
DATA_DIR="$(pwd)/data"
LOGS_DIR="$(pwd)/logs"

# ------------------------
# Args
# ------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--build) BUILD=true; shift ;;
    -i|--image) IMAGE_NAME="$2"; shift 2 ;;
    -n|--name) CONTAINER_NAME="$2"; shift 2 ;;
    -f|--file) DOCKERFILE="$2"; shift 2 ;;
    -p|--platform) PLATFORM="--platform $2"; shift 2 ;;
    -e|--env) ENV_FILE="$2"; shift 2 ;;
    -w|--workspace) WORKDIR_MOUNT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "  -b, --build           Build the image before running"
      echo "  -i, --image NAME      Docker image name (default: follow-the-leaders)"
      echo "  -n, --name NAME       Container name (default: ftl_container)"
      echo "  -f, --file FILE       Dockerfile (default: Dockerfile)"
      echo "  -p, --platform PLAT   Target platform (e.g. linux/amd64 or linux/arm64)"
      echo "  -e, --env FILE        .env file path (default: .env)"
      echo "  -w, --workspace DIR   Directory to mount at /app (default: current dir)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1 ;;
  esac
done

# ------------------------
# Build image
# ------------------------
if [ "$BUILD" = true ]; then
  echo "🔨 Building image '$IMAGE_NAME' with Dockerfile '$DOCKERFILE'..."
  docker build $PLATFORM -f "$DOCKERFILE" -t "$IMAGE_NAME" .
fi

# Ensure mount dirs exist
mkdir -p "$DATA_DIR" "$LOGS_DIR"

# ------------------------
# Run container
# ------------------------
echo "🚀 Running container '$CONTAINER_NAME' from image '$IMAGE_NAME' ..."
RUN_ARGS=(
  -it --rm
  --name "$CONTAINER_NAME"
  -v "$WORKDIR_MOUNT:/app"
  -v "$DATA_DIR:/app/data"
  -v "$LOGS_DIR:/app/logs"
)

# Load env file if exists
# if [ -f "$ENV_FILE" ]; then
#   RUN_ARGS+=( --env-file "$ENV_FILE" )
# else
#   echo "No .env file found at '$ENV_FILE'.
# fi

# Example: override/append envs on the command line
# RUN_ARGS+=( -e TELEGRAM_BOT_TOKEN=xxx -e TELEGRAM_CHAT_ID=yyy )

# Start container
docker run "${RUN_ARGS[@]}" "$IMAGE_NAME"
