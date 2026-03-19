## ── SRHN v4 Production Dockerfile ──────────────────────────────
## Multi-stage: build → minimal runtime
## Final image: ~15MB, Debian slim, no build tools

## ── Stage 1: Build ──────────────────────────────────────────────
FROM debian:bookworm-slim AS builder

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential gcc make ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .
RUN make release 2>&1 | tail -5

## ── Stage 2: Runtime ────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        libpthread-stubs0-dev python3 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

## Copy only what's needed at runtime
COPY --from=builder /build/build/srhn4_train  /app/srhn4_train
COPY --from=builder /build/build/libsrhn4.a   /app/libsrhn4.a
COPY srhn_v4_visualizer.html                  /app/visualizer.html
COPY srhn_client.py                           /app/srhn_client.py

## Data directory for persistent network snapshots
RUN mkdir -p /data/srhn

## Environment
ENV SRHN_DATA=/data/srhn
ENV SRHN_PORT=8765
ENV SRHN_EPOCHS=10

## Healthcheck
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:${SRHN_PORT}/health || exit 1

## Startup: train then serve
CMD sh -c "\
    echo '[docker] Training SRHN v4 (epochs=${SRHN_EPOCHS})...' && \
    /app/srhn4_train --epochs ${SRHN_EPOCHS} --quiet && \
    echo '[docker] Starting REST API on port ${SRHN_PORT}...' && \
    /app/srhn4_train --serve --port ${SRHN_PORT}"

EXPOSE 8765
VOLUME ["/data/srhn"]
LABEL maintainer="SRHN v4" version="4.0.0" description="Semantic Resonance Hypergraph Network"
