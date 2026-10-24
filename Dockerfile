# syntax=docker/dockerfile:1
FROM python:3.12.14-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    python -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -c "from pathlib import Path; [p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n')) for p in Path('.').rglob('*') if p.is_file() and p.suffix in {'.py','.json','.txt','.md','.yml','.yaml'}]"
CMD ["python", "run_observed.py", "--out", "/output"]
