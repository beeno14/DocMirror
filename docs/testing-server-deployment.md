# Testing Server Packaging and Deployment

This guide packages the current DocMirror working tree and deploys it to the
testing server. The recommended server path uses Docker Compose because the
repository already contains a Dockerfile, a Compose service, persistent cache
volumes, and a health check.

## Important review result

Do not run the downloaded `deploy.sh` unchanged. Although both downloaded
scripts pass `bash -n`, the deployment script has functional and safety errors:

- It uses relative paths without first changing to `/home/docmirror`. Running
  it from another directory can back up, delete, and extract the wrong paths.
- The archive expands to `/home/docmirror/docmirror`, but the script runs
  `pip install -e ".[server]"` from `/home/docmirror`. The install therefore
  targets the wrong directory.
- It deletes the active source and the uploaded archive before the replacement
  has been installed or health-checked. A failed install has no automatic
  rollback and leaves the service stopped.
- It installs only the `server` extra. A new environment will be missing the
  PDF, OCR, Office, layout, table, and other dependencies used by the full
  service. The project defines these under the `all` extra.
- `curl -s` treats HTTP 4xx and 5xx responses as success. Health checks should
  use `curl -fsS` and should retry until a timeout rather than sleeping for a
  fixed five seconds.
- The first deployment fails at `cp -r docmirror ...` when there is no active
  `docmirror` directory.
- `ENVIRONMENT=test` is not read by the current DocMirror source. Runtime
  settings use `DOCMIRROR_*` variables instead.
- The copy downloaded with it does not contain the required `start.sh` or
  `stop.sh`, so those scripts and their working-directory assumptions remain
  unverified.
- The archive is not checked for unsafe link targets before extraction.

The downloaded `build.sh` also needs attention if its wheel is intended for a
full deployment:

- It checks and compile-tests Community, Enterprise, and Finance sources, but
  `pip wheel "$SOURCE_DIR"` selects the default Hatch wheel target. The current
  `pyproject.toml` default wheel contains only `docmirror`; it explicitly
  excludes `docmirror_enterprise` and `docmirror_finance`. The resulting wheel
  is therefore not a complete three-edition artifact.
- `deploy.sh` never consumes the build directory or wheel produced by
  `build.sh`; it deploys the original source archive instead.
- A failed wheel build leaves a partial timestamped directory under
  `/home/docmirror/builds`.
- Its archive path checks do not inspect symlink or hard-link targets. Use it
  only with an archive produced through a trusted path.

## 1. Package on the development machine

From the repository root, run:

```bash
bash scripts/package_release.sh
```

On Windows, use the native PowerShell packager (no WSL or Git Bash required):

```powershell
.\scripts\package_release.ps1
```

Pass `-Output` to choose a different archive path. The default is
`docmirror-complete.tar.gz` in the repository root.

The command creates:

```text
docmirror-complete.tar.gz
```

It validates the archive and prints its SHA-256 value without creating a
separate checksum file. Compare that value with `sha256sum
docmirror-complete.tar.gz` after uploading when transfer verification is
required.

The archive has exactly one top-level directory named `docmirror`. It includes
the current core source files from `docmirror`, including current uncommitted
source changes. It omits tests, output data, caches, bytecode, `.env`,
credentials, and secrets.

Validate the result before transfer:

```bash
tar -tzf docmirror-complete.tar.gz | head
```

The first archive member should be `docmirror/`; the package must also contain
`docmirror/pyproject.toml` and `docmirror/docmirror/server/api.py`.

## 2. Upload to the testing server

Replace `TEST_HOST` with the SSH host name or address:

```bash
ssh docmirror@TEST_HOST 'mkdir -p /home/docmirror/incoming'
scp docmirror-complete.tar.gz docmirror@TEST_HOST:/home/docmirror/incoming/
```

On the server, verify the upload before extracting it:

```bash
cd /home/docmirror/incoming
tar -tzf docmirror-complete.tar.gz >/dev/null
```

Stop if the archive validation command fails.

## 3. One-time testing-server configuration

The server needs Docker Engine, the Docker Compose plugin, `curl`, and enough
free disk space to build the OCR dependencies. The `docmirror` account must be
allowed to run Docker.

Keep environment settings outside release directories:

```bash
mkdir -p /home/docmirror/config /home/docmirror/releases
chmod 700 /home/docmirror/config
```

Create `/home/docmirror/config/test.env` with the real test credentials and
restrict its permissions:

```dotenv
DOCMIRROR_API_KEY=replace-with-a-long-random-test-key
```

```bash
chmod 600 /home/docmirror/config/test.env
```

Do not copy `.env.example` directly: it contains placeholder external-OCR
settings. Create `/home/docmirror/config/docker-compose.test.yml` once:

```yaml
services:
  docmirror:
    environment:
      DOCMIRROR_API_KEY: ${DOCMIRROR_API_KEY:?set DOCMIRROR_API_KEY}
      DOCMIRROR_ENHANCE_MODE: standard
      DOCMIRROR_LOG_LEVEL: info
      OMP_NUM_THREADS: "4"
```

If authentication is intentionally disabled on an isolated test network,
remove the `DOCMIRROR_API_KEY` entry from the override instead of committing a
blank or real key to the repository.

## 4. Deploy a release with Docker Compose

Use a versioned release directory so the previous source remains available for
rollback:

```bash
set -Eeuo pipefail
cd /home/docmirror

RELEASE_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RELEASE_DIR="/home/docmirror/releases/$RELEASE_ID"
mkdir -p "$RELEASE_DIR"

tar -xzf /home/docmirror/incoming/docmirror-complete.tar.gz \
  -C "$RELEASE_DIR" \
  --strip-components=1 \
  --no-same-owner \
  --no-same-permissions

test -f "$RELEASE_DIR/pyproject.toml"
test -f "$RELEASE_DIR/Dockerfile"
test -f "$RELEASE_DIR/docmirror/server/api.py"
test -d "$RELEASE_DIR/docmirror_enterprise"
test -d "$RELEASE_DIR/docmirror_finance"

cd "$RELEASE_DIR"
docker compose \
  --project-name docmirror \
  --env-file /home/docmirror/config/test.env \
  -f docker-compose.yml \
  -f /home/docmirror/config/docker-compose.test.yml \
  build --pull docmirror

docker compose \
  --project-name docmirror \
  --env-file /home/docmirror/config/test.env \
  -f docker-compose.yml \
  -f /home/docmirror/config/docker-compose.test.yml \
  up -d --remove-orphans
```

Do not delete the uploaded archive until verification succeeds.

## 5. Verify the deployment

Check container state and logs:

```bash
cd "$RELEASE_DIR"
docker compose --project-name docmirror ps
docker compose --project-name docmirror logs --tail=200 docmirror
```

Poll the health endpoint for up to two minutes:

```bash
for attempt in $(seq 1 24); do
  if curl -fsS http://127.0.0.1:8000/health; then
    printf '\nDocMirror is healthy.\n'
    break
  fi
  if [ "$attempt" -eq 24 ]; then
    echo 'DocMirror did not become healthy within 120 seconds.' >&2
    exit 1
  fi
  sleep 5
done
```

The expected response contains `"status":"ok"` and the deployed version.
Then make one representative parse request using a non-sensitive test file and
the API key:

```bash
set -a
. /home/docmirror/config/test.env
set +a

curl -fsS \
  -H "Authorization: Bearer $DOCMIRROR_API_KEY" \
  -F 'file=@/path/to/test-document.pdf' \
  'http://127.0.0.1:8000/v1/tasks?wait=true'
```

## 6. Roll back

Choose the preceding directory under `/home/docmirror/releases`, then rebuild
and start it with the same fixed Compose project name and configuration:

```bash
PREVIOUS_RELEASE=/home/docmirror/releases/REPLACE_WITH_PREVIOUS_ID
cd "$PREVIOUS_RELEASE"

docker compose \
  --project-name docmirror \
  --env-file /home/docmirror/config/test.env \
  -f docker-compose.yml \
  -f /home/docmirror/config/docker-compose.test.yml \
  up -d --build --remove-orphans

curl -fsS http://127.0.0.1:8000/health
```

Keep at least the previous known-good release and its archive until the new
release has passed the server's smoke tests.
