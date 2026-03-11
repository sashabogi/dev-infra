FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml ./
COPY cli/ cli/
COPY daneel/ daneel/
COPY rescue/ rescue/

RUN pip install --no-cache-dir .


FROM python:3.12-slim

RUN groupadd --gid 1000 daneel \
    && useradd --uid 1000 --gid daneel --create-home daneel

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/dev-infra /usr/local/bin/dev-infra

WORKDIR /app

COPY cli/ cli/
COPY daneel/ daneel/
COPY rescue/ rescue/
COPY pyproject.toml ./

# Config will be bind-mounted at /app/config.yaml
# Symlink it to where the CLI looks: ~/.dev-infra/config.yaml
RUN mkdir -p /home/daneel/.dev-infra \
    && ln -s /app/config.yaml /home/daneel/.dev-infra/config.yaml \
    && chown -R daneel:daneel /home/daneel /app

USER daneel

EXPOSE 8889

CMD ["dev-infra", "start", "--foreground"]
