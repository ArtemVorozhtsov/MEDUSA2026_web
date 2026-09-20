# MEDUSA 2026 web service — self-contained image.
# Build context: the PARENT of medusa_web/ (contains both medusa_web/ and MEDUSA2026/).
#   docker compose up --build   (from medusa_web/, context set in docker-compose.yml)
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# g++ is required to build the IsoSpecPy sdist (single C++ translation unit).
# libglib2.0 (libglib-2.0.so.0 / libgthread-2.0.so.0) is needed by the pyopenms
# wheel at import time. Name is libglib2.0-0t64 on trixie, libglib2.0-0 on
# bookworm and older.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && (apt-get install -y --no-install-recommends libglib2.0-0t64 \
        || apt-get install -y --no-install-recommends libglib2.0-0) \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Python dependencies (cached layer)
COPY medusa_web/requirements.txt /app/medusa_web/requirements.txt
RUN pip install -r /app/medusa_web/requirements.txt

# 1b) The torch 1.12.0 CPU wheel ships libtorch_cpu.so with an executable-stack
#     (PT_GNU_STACK RWE) flag left over from its build toolchain. Newer glibc
#     (>= 2.41, e.g. the current python:3.9-slim) refuses to honor it and aborts
#     with "cannot enable executable stack as shared object requires". The
#     library does not need an executable stack, so clear the PF_X bit on every
#     PT_GNU_STACK program header under site-packages and /app.
RUN python - <<'PYFIX'
import os, struct

PT_GNU_STACK = 0x6474E551
fixed = []
for root in ('/usr/local/lib/python3.9/site-packages', '/app'):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not (name.endswith('.so') or '.so.' in name):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, 'rb') as fh:
                    head = fh.read(64)
                    if head[:4] != b'\x7fELF' or head[4] != 2:
                        continue
                    e_phoff = struct.unpack_from('<Q', head, 0x20)[0]
                    fh.seek(e_phoff)
                    phdrs = fh.read(struct.unpack_from('<H', head, 0x38)[0]
                                    * struct.unpack_from('<H', head, 0x36)[0])
                if not phdrs:
                    continue
                ent = struct.unpack_from('<H', head, 0x36)[0]
                for i in range(len(phdrs) // ent):
                    p_type, p_flags = struct.unpack_from('<II', phdrs, i * ent)
                    if p_type == PT_GNU_STACK and p_flags & 0x1:
                        with open(path, 'r+b') as fh:
                            fh.seek(e_phoff + i * ent + 4)
                            fh.write(struct.pack('<I', p_flags & ~0x1))
                        fixed.append(path)
                        break
            except (OSError, struct.error):
                continue
print(f'cleared executable-stack flag in {len(fixed)} file(s)')
for path in fixed:
    print(' ', path)
PYFIX

# 2) Web service code
COPY medusa_web/app /app/medusa_web/app
COPY medusa_web/static /app/medusa_web/static
COPY medusa_web/tests /app/medusa_web/tests

# 3) MEDUSA2026 core package (imported as mass_automation)
COPY MEDUSA2026/mass_automation /app/mass_automation

# 4) Pre-trained models, baked into the image (overridable via CGB_MODEL / TRANSFORMER_CKPT)
COPY MEDUSA2026/data/models/charge1_optuna150.pkl /app/models/charge1_optuna150.pkl
COPY MEDUSA2026/nn_models/transfomer_classifier.ckpt /app/models/transfomer_classifier.ckpt

ENV PYTHONPATH=/app:/app/medusa_web \
    MASS_AUTOMATION_PATH=/app \
    CGB_MODEL=/app/models/charge1_optuna150.pkl \
    TRANSFORMER_CKPT=/app/models/transfomer_classifier.ckpt \
    SPECTRA_DIR=/data/spectra \
    UPLOAD_DIR=/data/uploads \
    MAX_UPLOAD_MB=1024 \
    MAX_ACTIVE_SESSIONS=8 \
    MAX_CONCURRENT_CPU_JOBS=2 \
    PORT=8000

RUN useradd --create-home --shell /usr/sbin/nologin medusa \
    && mkdir -p /data/spectra /data/uploads \
    && chown -R medusa:medusa /data /app
USER medusa

EXPOSE 8000
# Single uvicorn worker is mandatory: session state lives in process memory.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1"]
