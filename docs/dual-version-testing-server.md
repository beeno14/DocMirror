# Dual-version DocMirror on the testing server

This layout keeps the vNext and legacy DocMirror processes independent while
preserving the existing `start.sh` and `stop.sh` operational entry points.

## Verified existing layout

The testing server currently runs vNext directly with Uvicorn, not Docker
Compose:

- application home: `/home/docmirror`
- active vNext release: `/home/docmirror/current`
- vNext Python: `/home/docmirror/current/venv/bin/python`
- vNext port: `8000`
- vNext PID/log: `/home/docmirror/api.pid`, `/home/docmirror/api.log`

Do not reuse the old root `stop.sh` after the legacy process is added. It uses a
broad `pkill -f "uvicorn docmirror.server.api"` and can stop both versions.

## Recommended layout

```text
/home/docmirror/
  current -> releases/<vnext-release>
  releases/
  api.pid
  api.log
  .env

  docmirror_old/
    docmirror/
    docmirror_enterprise/
    docmirror_finance/
    venv/
    pyproject.toml
    .env.test
    api.pid
    api.log
```

The two API endpoints are:

- vNext: `http://127.0.0.1:8000/v1/parse`
- legacy bank: `http://127.0.0.1:8002/bankCashflow`
- legacy payment: `http://127.0.0.1:8002/payCashflow`

`docmirror_old` must own its virtual environment. Do not share the vNext virtual
environment with the old version because their dependency sets may differ.

## Management scripts

Install the files from `scripts/testing_server/` under one root-owned operations
directory, make them executable, and source the environment file before use.
The default wrappers manage both processes:

```bash
./start.sh
./stop.sh
./restart.sh
./status.sh
```

Pass a target to operate on one version:

```bash
./start.sh legacy
./stop.sh vnext
./restart.sh legacy
./status.sh all
./docmirror-services.sh logs legacy
```

The manager validates the PID against the configured release path and port
before sending `SIGTERM`. It deliberately does not use `pkill` and does not
force-kill a process after timeout.

## First legacy deployment

Place the old source package in its dedicated directory and create an independent
virtual environment:

```bash
mkdir -p /home/docmirror/docmirror_old
tar -xzf <legacy-archive>.tar.gz -C /home/docmirror/docmirror_old
cd /home/docmirror/docmirror_old
python3.11 -m venv venv
venv/bin/python -m pip install -e '.[all]'
```

Keep the old version's `.env.test` in `/home/docmirror/docmirror_old/.env.test`
and set permissions to `600`. The management script exports `ENVIRONMENT=test`,
so the old version's existing environment loader selects this file. Do not copy
or commit credentials into the vNext release.

Before starting legacy, verify that port `8002` is free:

```bash
ss -lntp | grep ':8002 ' || true
```

Then start only legacy and verify both health endpoints:

```bash
./start.sh legacy
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8002/health
./status.sh all
```

Finally configure FlashVal Agent with the server address reachable from the
Agent host:

```dotenv
DOCMIRROR_VNEXT_URL=http://192.168.1.31:8000/v1/parse
DOCMIRROR_LEGACY_BANK_URL=http://192.168.1.31:8002/bankCashflow
DOCMIRROR_LEGACY_PAYMENT_URL=http://192.168.1.31:8002/payCashflow
```

If a firewall or reverse proxy limits access, expose only the Agent host to
ports `8000` and `8002`; do not expose the services broadly without API
authentication.
