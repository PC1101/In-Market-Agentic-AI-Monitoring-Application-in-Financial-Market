# Reproducible runtime image for monitoring jobs on vast.ai.
# Bakes the Python deps + ollama + the qwen2.5:3b model so a rented instance is
# ready to run without a cold model pull. Build + push to a registry, then set
# `image:` in job.yaml to the pushed tag.
#
#   docker build -t <you>/inmarket-monitor:latest scripts/vast
#   docker push <you>/inmarket-monitor:latest
FROM python:3.12-slim

# System deps for pandas/pyarrow + curl for the ollama installer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# ollama runtime.
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /root/monitoring
COPY monitoring/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt pyyaml

# Pre-pull the model so the instance starts warm (backgrounds ollama, pulls, stops).
RUN ollama serve & \
    sleep 5 && \
    ollama pull qwen2.5:3b && \
    pkill ollama || true

# The repo itself is pushed at run time via `vastai copy`, keeping the image
# stable across code changes. onstart just needs to start ollama.
CMD ["bash", "-lc", "ollama serve"]
