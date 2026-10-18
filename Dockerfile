# Review app container.
#
# The app is pure standard-library Python, so this image is deliberately boring:
# no pip install, no build step, nothing to go stale. That is the point of
# containerising it — a contributor runs one command and gets the same app you
# are looking at, on any machine.
FROM python:3.11-slim

LABEL org.opencontainers.image.title="SSBU label review" \
      org.opencontainers.image.description="Local bounding-box review UI" \
      org.opencontainers.image.source="https://github.com/MicahHeneveld/bros-analysis"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Only the app is baked in. Datasets are mounted, because labels are meant to be
# written back to the host and committed to git.
COPY app/ ./app/

# Reviewers are not running this as root on the host, so match a normal UID and
# keep the mounted label files owned by whoever started the container.
RUN useradd --create-home --uid 1000 reviewer \
 && chown -R reviewer:reviewer /app
USER reviewer

EXPOSE 8770

# 0.0.0.0 rather than the default 127.0.0.1, or nothing outside the container
# could reach it.
ENTRYPOINT ["python", "app/server.py", "--host", "0.0.0.0"]
CMD ["--port", "8770"]
