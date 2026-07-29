# Port optree features into VoxCity — Combined Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This document bundles FOUR independent, sequenced plans; implement one plan fully (through its verification task) before starting the next.

**Goal:** Cherry-pick three capabilities from `optree_voxcity/app` into `VoxCity/app` — Copy Share Link, DXF auxiliary-line import, and UX/UI polish (CSS delta, bundled Inter font, en/ja i18n) — without wholesale-copying optree.

**Architecture:** Keep VoxCity's app shell, tab set, single global `app_state` backend, `#6666FF` indigo palette, and VoxCity logo. Port self-contained optree modules, adapting each `Depends(get_session)` endpoint to VoxCity's global `app_state`. Every VoxCity-only feature (PLATEAU generation, GeoTIFF export, large-area preview-disable) is preserved by construction.

**Tech Stack:** FastAPI + Pydantic + pytest (backend); React 18 + TypeScript + Vite + Vitest (frontend); `ezdxf` (new backend dep); `@fontsource/inter` (new frontend dep).

**Spec:** `docs/superpowers/specs/2026-07-29-port-optree-features-design.md`

---

## Plan set overview & sequencing

Four independent subsystems, ordered low→high risk. Each produces working, testable software on its own; complete each through its final verification task before starting the next.

| # | Plan | Risk | Why this order |
|---|------|------|----------------|
| 1 | Copy Share Link | Low | Smallest, self-contained, no new rendering. |
| 2 | DXF auxiliary-line import | High | New subsystem (backend parser + 3D aux-line rendering). |
| 3 | CSS delta + Inter font | Low | Mechanical delta port; small (2026-06-01 port already did most). |
| 4 | i18n (en/ja) | Med | **Last** — it wraps the English strings Plans 1–2 add. |

### Global conventions (all plans)

- **Backend tests** run from the repo root `C:\Users\kunih\OneDrive\00_Codes\python\VoxCity`:
  `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/<test>.py -v`
  (conda is NOT on PATH — always use the full path. If `backend.*` imports fail, prefix `PYTHONPATH=app`.)
- **Frontend** commands run from `app/frontend`: `npm test -- <name>`, `npx tsc -b --noEmit`, `npm run build`.
- **Commit** after each task with a conventional-commit message; end every commit body with the trailer
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Line numbers** cited in edit steps are from the draft's read of each file; re-confirm each anchor by matching the quoted code (files may have shifted) before editing.
- **Palette guard:** any CSS/color ported from optree must use VoxCity indigo `#6666FF` (never teal `#009999`).

---

## Plan 1 of 4 — Copy Share Link

**Goal:** Add a "Copy Share Link" feature to VoxCity: persist the current session server-side under an unguessable token and let anyone with the `/share/<token>` URL load that snapshot.

**Architecture:** Port optree's `share.py` (session-zip snapshot + token) adapted to VoxCity's single global `app_state` (dropping optree's optimization/per-session-cookie machinery). Two new endpoints (`POST /api/share`, `POST /api/share/{token}/load`) reuse the existing `save_session_to_zip` / `parse_session_zip` / `apply_session_to_state` pipeline. Frontend adds a Share section to the File tab and consumes `/share/<token>` URLs on app load via pathname parsing (no router).

**Tech Stack:** FastAPI + Pydantic (backend), React 18 + TypeScript + Vite + Vitest (frontend), pytest (backend tests).

**Part of:** `docs/superpowers/specs/2026-07-29-port-optree-features-design.md`. Strings added here are plain English; the later i18n plan (4 of 4) wraps them in `t()`.

---

### File Structure

**Backend (`app/backend/`):**
- Modify `config.py` — add `SHARE_DIR`.
- Create `share.py` — token validation, `share_zip_path`, `create_share`.
- Modify `main.py` — import share helpers; add two endpoints.
- Create `test_share.py` — module + endpoint tests.

**Frontend (`app/frontend/src/`):**
- Modify `api.ts` — `ShareCreateResult`, `createShare`, `loadShare`.
- Create `lib/shareLink.ts` — `parseShareToken`.
- Create `lib/shareLink.test.ts`.
- Modify `tabs/ExportTab.tsx` — Share button + link/copy UI.
- Modify `App.tsx` — consume `/share/<token>` on load + loading/error overlay.

**Reference (read-only, do not edit):** optree source at
`C:\Users\kunih\OneDrive\00_Codes\python\optree_voxcity\app\` — `backend/share.py`,
`frontend/src/lib/shareLink.ts`, `frontend/src/tabs/ExportTab.tsx`.

**Test command (backend):** run from repo root
`C:\Users\kunih\OneDrive\00_Codes\python\VoxCity`:
```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_share.py -v
```
(Backend tests import `from backend.main import ...`, so pytest must resolve `backend` as a top-level package — the existing suite already does; run the command exactly as shown from the repo root. If import fails, run with `PYTHONPATH=app`: `PYTHONPATH=app & "..." ... python -m pytest app/backend/test_share.py -v`.)

**Test command (frontend):** run from `app/frontend`:
```
npm test -- shareLink
```

---

### Task 1: Backend config — SHARE_DIR

**Files:**
- Modify: `app/backend/config.py` (after the `OUTPUT_DIR` block, ~line 37)

- [ ] **Step 1: Add the SHARE_DIR constant**

In `app/backend/config.py`, immediately after the `OUTPUT_DIR = _env_or(...)` assignment in the "Base directories" section, add:

```python
# Server-side share snapshots (share-by-URL). Defaults under OUTPUT_DIR so the
# same VOXCITY_OUTPUT_DIR override relocates shares too.
SHARE_DIR = _env_or("VOXCITY_SHARE_DIR", os.path.join(OUTPUT_DIR, "shares"))
```

- [ ] **Step 2: Verify it imports**

Run:
```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -c "from app.backend import config; print(config.SHARE_DIR)"
```
Expected: prints a path ending in `voxcity_output\shares` (or the `VOXCITY_OUTPUT_DIR` override).

- [ ] **Step 3: Commit**

```bash
git add app/backend/config.py
git commit -m "feat(share): add SHARE_DIR config for share snapshots"
```

---

### Task 2: Backend share.py module

**Files:**
- Create: `app/backend/share.py`
- Create: `app/backend/test_share.py`

- [ ] **Step 1: Write the failing tests (token validation + create/lookup)**

Create `app/backend/test_share.py`:

```python
"""Tests for share-by-URL: share module helpers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import config, share


@pytest.fixture(autouse=True)
def _share_dir(tmp_path):
    share.set_share_base_dir(tmp_path / "shares")
    yield
    share.set_share_base_dir(Path(config.SHARE_DIR))


# --- token validation -------------------------------------------------------

def test_is_valid_token_accepts_urlsafe_tokens():
    assert share.is_valid_token("Ab3xK9_qRt2mVw8pLzYh4g")  # 22 chars
    assert share.is_valid_token("a" * 16)
    assert share.is_valid_token("a" * 64)


def test_is_valid_token_rejects_malformed_tokens():
    assert not share.is_valid_token("")
    assert not share.is_valid_token("short")
    assert not share.is_valid_token("a" * 65)
    assert not share.is_valid_token("../" + "a" * 16)
    assert not share.is_valid_token("a" * 16 + "/x")


def test_share_zip_path_rejects_invalid_token_without_touching_fs():
    assert share.share_zip_path("../etc/passwd") is None
    assert share.share_zip_path("short") is None


def test_share_zip_path_none_for_unknown_token():
    assert share.share_zip_path("a" * 20) is None


# --- create_share -----------------------------------------------------------

def _fake_state():
    # save_session_to_zip only needs a state with a scene; monkeypatched below.
    return SimpleNamespace(sim_results_by_type={})


def test_create_share_writes_zip_and_meta(monkeypatch, tmp_path):
    import io

    def fake_save(state, **kwargs):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"ok": True}))
        buf.seek(0)
        return buf

    monkeypatch.setattr(share, "save_session_to_zip", fake_save)

    token = share.create_share(_fake_state(), frontend_state='{"zones":[]}')
    assert share.is_valid_token(token)

    zip_path = share.share_zip_path(token)
    assert zip_path is not None and zip_path.is_file()

    meta_path = zip_path.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["has_sim_results"] is False
    assert meta["size_bytes"] > 0
    assert "created_at_utc" in meta


def test_create_share_leaves_no_staging_dir_on_success(monkeypatch, tmp_path):
    import io

    def fake_save(state, **kwargs):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("x", "y")
        buf.seek(0)
        return buf

    monkeypatch.setattr(share, "save_session_to_zip", fake_save)
    share.create_share(_fake_state())
    base = share.share_base_dir()
    assert not list(base.glob(".tmp-*"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_share.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.share'` (or import error).

- [ ] **Step 3: Write share.py**

Create `app/backend/share.py`:

```python
"""Server-side persisted share snapshots (share-by-URL).

A share is a session zip written under SHARE_DIR/<token>/session.zip with a
meta.json sidecar. Tokens are unguessable URL-safe strings; possession of the
URL is the access model (no auth, no enumeration endpoint).

Adapted from optree_voxcity's share.py for VoxCity's single global app_state:
the optimization-job branch is dropped (VoxCity has no optimizer).
"""

from __future__ import annotations

import datetime
import json
import re
import secrets
import shutil
from pathlib import Path
from typing import Optional

from . import config
from .session_io import _app_version, save_session_to_zip

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")

_share_base_dir = Path(config.SHARE_DIR)


def set_share_base_dir(path) -> None:
    """Test hook to relocate the share directory."""
    global _share_base_dir
    _share_base_dir = Path(path)


def share_base_dir() -> Path:
    return _share_base_dir


def is_valid_token(token: str) -> bool:
    return bool(_TOKEN_RE.fullmatch(token))


def share_zip_path(token: str) -> Optional[Path]:
    """Return the stored session.zip for *token*, or None if invalid/unknown.

    Token validation happens before any filesystem access, so traversal-shaped
    input never touches the disk.
    """
    if not is_valid_token(token):
        return None
    path = _share_base_dir / token / "session.zip"
    return path if path.is_file() else None


def create_share(state, frontend_state: Optional[str] = None) -> str:
    """Persist a share snapshot of *state* and return its token.

    Sim results are embedded iff cached. Creation is atomic: files are staged in
    a sibling ".tmp-<token>" dir and renamed into place, so a concurrent load
    can never observe a half-written share.

    Raises ValueError (from save_session_to_zip) when the state has no scene.
    """
    include_sim_results = bool(getattr(state, "sim_results_by_type", None))

    buf = save_session_to_zip(
        state,
        include_sim_results=include_sim_results,
        frontend_state=frontend_state,
    )
    data = buf.getvalue()

    token = secrets.token_urlsafe(16)
    base = _share_base_dir
    base.mkdir(parents=True, exist_ok=True)
    staging = base / f".tmp-{token}"
    staging.mkdir()
    try:
        (staging / "session.zip").write_bytes(data)
        meta = {
            "created_at_utc": datetime.datetime.now(
                tz=datetime.timezone.utc
            ).isoformat(timespec="seconds"),
            "app_version": _app_version(),
            "has_sim_results": include_sim_results,
            "size_bytes": len(data),
        }
        (staging / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        staging.rename(base / token)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return token
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_share.py -v
```
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add app/backend/share.py app/backend/test_share.py
git commit -m "feat(share): share.py snapshot module + tests"
```

---

### Task 3: Backend endpoints — /api/share and /api/share/{token}/load

**Files:**
- Modify: `app/backend/main.py` (imports near line 78; endpoints after the `/api/session/load` endpoint, ~line 908)
- Modify: `app/backend/test_share.py` (append endpoint tests)

- [ ] **Step 1: Write the failing endpoint tests**

Append to `app/backend/test_share.py`:

```python
# --- endpoints --------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    import io

    def fake_save(state, **kwargs):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"v": 1}))
        buf.seek(0)
        return buf

    # Patch where create_share looks the symbol up.
    monkeypatch.setattr(share, "save_session_to_zip", fake_save)

    from backend.main import app, app_state
    app_state.sim_results_by_type = {}
    # Make the load path deterministic: apply_session_to_state returns a summary.
    monkeypatch.setattr(
        "backend.main.parse_session_zip",
        lambda stream, max_bytes=None: SimpleNamespace(_root="ignored"),
    )
    monkeypatch.setattr(
        "backend.main.apply_session_to_state",
        lambda parsed, state: {"has_voxcity": True, "rectangle_vertices": None,
                               "land_cover_source": "OpenStreetMap",
                               "frontend_state": '{"zones":[]}',
                               "has_sim_results": False, "last_sim_type": None,
                               "sim_result_types": [], "landmark_building_ids": []},
    )
    monkeypatch.setattr("backend.main.parsed_session_temp_root", lambda parsed: None)
    monkeypatch.setattr("backend.main.shutil.rmtree", lambda *a, **k: None)
    return TestClient(app)


def test_post_share_returns_token_and_path(client):
    res = client.post("/api/share", data={"frontend_state": '{"zones":[]}'})
    assert res.status_code == 200
    body = res.json()
    assert share.is_valid_token(body["token"])
    assert body["path"] == f"/share/{body['token']}"


def test_load_unknown_share_is_404(client):
    res = client.post("/api/share/" + "a" * 20 + "/load")
    assert res.status_code == 404


def test_share_roundtrip_creates_then_loads(client):
    token = client.post("/api/share").json()["token"]
    res = client.post(f"/api/share/{token}/load")
    assert res.status_code == 200
    assert res.json()["has_voxcity"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_share.py -k endpoint -v
```
(Or run the whole file.) Expected: FAIL — `404` for `/api/share` (route not defined yet) on `test_post_share_returns_token_and_path`.

- [ ] **Step 3: Add the import**

In `app/backend/main.py`, in the import block near line 78 (right after the `from .session_io import (...)` block), add:

```python
from .share import create_share, share_zip_path
```

- [ ] **Step 4: Add the two endpoints**

In `app/backend/main.py`, immediately after the existing `load_session_endpoint` function (the one ending in the `shutil.rmtree(parsed_session_temp_root(parsed), ...)` finally block, ~line 908), add:

```python
@app.post("/api/share")
async def create_share_endpoint(frontend_state: Optional[str] = Form(None)):
    """Persist a server-side share snapshot; return its token and URL path."""
    try:
        token = create_share(app_state, frontend_state=frontend_state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"token": token, "path": f"/share/{token}"}


@app.post("/api/share/{token}/load")
async def load_share_endpoint(token: str):
    """Load a persisted share snapshot into the global session."""
    zip_path = share_zip_path(token)
    if zip_path is None:
        raise HTTPException(status_code=404, detail="Unknown share link.")
    try:
        with open(zip_path, "rb") as fh:
            parsed = parse_session_zip(fh, max_bytes=_max_session_upload_bytes())
    except SessionLoadError as exc:
        raise HTTPException(
            status_code=500, detail=f"Stored share is unreadable: {exc}"
        ) from exc
    try:
        return apply_session_to_state(parsed, app_state)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to apply shared session: {exc}"
        ) from exc
    finally:
        shutil.rmtree(parsed_session_temp_root(parsed), ignore_errors=True)
```

Note: `Form`, `Optional`, `HTTPException`, `shutil`, `SessionLoadError`, `parse_session_zip`, `apply_session_to_state`, `parsed_session_temp_root`, and `_max_session_upload_bytes` are all already imported/defined in `main.py` (used by the session endpoints). Do not re-import them.

- [ ] **Step 5: Run tests to verify they pass**

Run:
```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_share.py -v
```
Expected: PASS (all tests, including the 3 endpoint tests).

- [ ] **Step 6: Commit**

```bash
git add app/backend/main.py app/backend/test_share.py
git commit -m "feat(share): /api/share create + load endpoints"
```

---

### Task 4: Frontend api.ts — createShare / loadShare

**Files:**
- Modify: `app/frontend/src/api.ts` (after the `loadSession` function, ~line 100 region; place near the existing session functions)

- [ ] **Step 1: Add the share client functions**

In `app/frontend/src/api.ts`, directly after the existing `loadSession` function, add:

```typescript
export interface ShareCreateResult {
  token: string;
  path: string;
}

/** Persist the current session server-side; returns the share token and URL path. */
export async function createShare(frontendState: string): Promise<ShareCreateResult> {
  const form = new FormData();
  form.append('frontend_state', frontendState);
  const res = await fetch(`${BASE}/share`, { method: 'POST', body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/** Load a shared snapshot into the current session. */
export async function loadShare(token: string): Promise<SessionLoadSummary> {
  const res = await fetch(`${BASE}/share/${encodeURIComponent(token)}/load`, {
    method: 'POST',
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}
```

(`SessionLoadSummary` and `BASE` already exist in this file.)

- [ ] **Step 2: Verify it typechecks**

Run from `app/frontend`:
```
npx tsc -b --noEmit
```
Expected: no errors (exit 0).

- [ ] **Step 3: Commit**

```bash
git add app/frontend/src/api.ts
git commit -m "feat(share): createShare/loadShare API client functions"
```

---

### Task 5: Frontend lib/shareLink.ts

**Files:**
- Create: `app/frontend/src/lib/shareLink.ts`
- Create: `app/frontend/src/lib/shareLink.test.ts`

- [ ] **Step 1: Write the failing test**

Create `app/frontend/src/lib/shareLink.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { parseShareToken } from './shareLink';

describe('parseShareToken', () => {
  it('extracts a valid token from a /share/<token> path', () => {
    const token = 'Ab3xK9_qRt2mVw8pLzYh4g';
    expect(parseShareToken(`/share/${token}`)).toBe(token);
  });

  it('returns null for non-share paths', () => {
    expect(parseShareToken('/')).toBeNull();
    expect(parseShareToken('/export')).toBeNull();
    expect(parseShareToken('/share/')).toBeNull();
  });

  it('returns null for tokens that are too short or too long', () => {
    expect(parseShareToken('/share/short')).toBeNull();
    expect(parseShareToken(`/share/${'a'.repeat(65)}`)).toBeNull();
  });

  it('rejects traversal-shaped tokens', () => {
    expect(parseShareToken('/share/../etc/passwd')).toBeNull();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run from `app/frontend`:
```
npm test -- shareLink
```
Expected: FAIL — cannot resolve `./shareLink`.

- [ ] **Step 3: Write shareLink.ts**

Create `app/frontend/src/lib/shareLink.ts`:

```typescript
// Mirrors the backend token rule in app/backend/share.py (_TOKEN_RE).
const SHARE_PATH_RE = /^\/share\/([A-Za-z0-9_-]{16,64})$/;

/** Return the share token when *pathname* is a /share/<token> URL, else null. */
export function parseShareToken(pathname: string): string | null {
  const match = SHARE_PATH_RE.exec(pathname);
  return match ? match[1] : null;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run from `app/frontend`:
```
npm test -- shareLink
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add app/frontend/src/lib/shareLink.ts app/frontend/src/lib/shareLink.test.ts
git commit -m "feat(share): parseShareToken url helper + tests"
```

---

### Task 6: Frontend ExportTab — Share button + copy UI

**Files:**
- Modify: `app/frontend/src/tabs/ExportTab.tsx`

- [ ] **Step 1: Update imports**

In `app/frontend/src/tabs/ExportTab.tsx`, change the lucide-react import (line 2) and the api import (line 3):

```typescript
import { Package, Box, Download, Upload, Map, Link2, Copy } from 'lucide-react';
import { createShare, exportCityles, exportObj, exportGeotiff, loadSession, saveSession } from '../api';
```

- [ ] **Step 2: Add share state and helpers**

In the `ExportTab` component body, after the existing `const fileInputRef = useRef<HTMLInputElement | null>(null);` line (line 36), add:

```typescript
  const [shareLoading, setShareLoading] = useState(false);
  const [shareError, setShareError] = useState<string | null>(null);
  const [shareUrl, setShareUrl] = useState<string | null>(null);
  const [shareCopied, setShareCopied] = useState(false);
  const shareUrlInputRef = useRef<HTMLInputElement | null>(null);

  const copyToClipboard = async (url: string): Promise<boolean> => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(url);
        return true;
      }
    } catch {
      // Async write can fail on insecure origins or after gesture activation
      // expires (a network await ran between click and write). Fall through.
    }
    try {
      const textarea = document.createElement('textarea');
      textarea.value = url;
      textarea.setAttribute('readonly', '');
      textarea.style.position = 'fixed';
      textarea.style.top = '0';
      textarea.style.opacity = '0';
      textarea.style.pointerEvents = 'none';
      document.body.appendChild(textarea);
      textarea.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(textarea);
      return ok;
    } catch {
      return false;
    }
  };

  const handleCreateShare = async () => {
    setShareLoading(true);
    setShareError(null);
    setShareUrl(null);
    setShareCopied(false);
    try {
      const result = await createShare(JSON.stringify({ zones }));
      const url = `${window.location.origin}${result.path}`;
      setShareUrl(url);
      setShareCopied(await copyToClipboard(url));
    } catch (err: any) {
      setShareError(err.message);
    } finally {
      setShareLoading(false);
    }
  };

  const handleCopyShareUrl = async () => {
    if (!shareUrl) return;
    const copied = await copyToClipboard(shareUrl);
    setShareCopied(copied);
    // On plain-http origins (common for a remote Docker host) the Clipboard API
    // is unavailable, so writeText fails silently. Select the URL text instead.
    if (!copied) shareUrlInputRef.current?.select();
  };
```

- [ ] **Step 3: Add the Share button to the session footer**

In the "Save / Load Session" `GuidedPanel` footer (the `<GuidedFooter>` around line 143), after the hidden `<input ref={fileInputRef} .../>` element (line 169) and before `</GuidedFooter>`, add:

```tsx
            <button
              className="btn"
              type="button"
              disabled={!hasModel || sessionLoading || shareLoading}
              onClick={handleCreateShare}
            >
              {shareLoading && <span className="spinner" />}
              <Link2 size={14} aria-hidden="true" style={{ marginRight: 6 }} />
              Copy Share Link
            </button>
```

Also add `|| shareLoading` to the `disabled` expression of the existing Save Session and Load Session buttons in that footer so the buttons don't fight during a share.

- [ ] **Step 4: Add the share-link result section**

In the same "Save / Load Session" `GuidedPanel`, after the existing `<GuidedSection index={1} label="SESSION OPTIONS"> ... </GuidedSection>` block (closes ~line 183) and before the panel's closing `</GuidedPanel>` (line 184), add:

```tsx
        {(shareUrl || shareError) && (
          <GuidedSection index={2} label="SHARE LINK">
            {shareError ? (
              <GuidedStatus tone="error">{shareError}</GuidedStatus>
            ) : (
              <>
                <div style={{ display: 'flex', gap: 6 }}>
                  <input
                    ref={shareUrlInputRef}
                    readOnly
                    value={shareUrl ?? ''}
                    aria-label="Share URL"
                    onFocus={(e) => e.currentTarget.select()}
                    style={{ flex: 1, minWidth: 0 }}
                  />
                  <button className="btn" type="button" onClick={handleCopyShareUrl}>
                    <Copy size={14} aria-hidden="true" style={{ marginRight: 6 }} />
                    Copy
                  </button>
                </div>
                <GuidedStatus tone="success">
                  {shareCopied
                    ? 'Link copied to clipboard.'
                    : 'Share link created — copy it below.'}
                </GuidedStatus>
              </>
            )}
          </GuidedSection>
        )}
```

- [ ] **Step 5: Verify typecheck + existing tests still pass**

Run from `app/frontend`:
```
npx tsc -b --noEmit && npm test -- ExportTab
```
Expected: typecheck exit 0; ExportTab tests (if any) pass. If there are no ExportTab tests, the `npm test` filter reports "no tests found" — that is acceptable here.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/src/tabs/ExportTab.tsx
git commit -m "feat(share): Copy Share Link button + link UI in File tab"
```

---

### Task 7: Frontend App.tsx — consume /share/<token> on load

**Files:**
- Modify: `app/frontend/src/App.tsx`

- [ ] **Step 1: Add imports**

In `app/frontend/src/App.tsx`, extend the api import (line 20) and add the shareLink import after it:

```typescript
import { healthCheck, resetSession, getModelInfo, loadShare } from './api';
import { parseShareToken } from './lib/shareLink';
import { parsePersistedFrontendState, buildRestoredFrontendState } from './lib/sessionRestore';
```

- [ ] **Step 2: Add share-load state and a ref**

After the `const [landmarkRunNonce, setLandmarkRunNonce] = useState(0);` line (line 52), add:

```typescript
  const shareTokenRef = useRef<string | null>(parseShareToken(window.location.pathname));
  const [shareLoad, setShareLoad] = useState<{ status: 'idle' | 'loading' | 'error'; message?: string }>(
    () => (parseShareToken(window.location.pathname) ? { status: 'loading' } : { status: 'idle' }),
  );
```

- [ ] **Step 3: Suppress the splash when arriving via a share link**

Change the `splashOpen` initializer (line 54-56) so a share link does not pop the splash:

```typescript
  const [splashOpen, setSplashOpen] = useState(() => {
    if (parseShareToken(window.location.pathname)) return false;
    try { return localStorage.getItem(SPLASH_DISMISSED_KEY) !== '1'; } catch { return true; }
  });
```

- [ ] **Step 4: Load the share after reset in the mount effect**

In the mount effect (the `useEffect` starting line 146 with `didReset`), replace the `.catch(() => {})` / `.finally(...)` tail (lines 167-168) with a share-loading branch:

```typescript
      .catch(() => {})
      .then(() => {
        const token = shareTokenRef.current;
        if (!token) return;
        return loadShare(token)
          .then((summary) => {
            const persisted = parsePersistedFrontendState(summary.frontend_state);
            const { restored } = buildRestoredFrontendState(persisted);
            handleSessionLoaded(summary, restored);
            window.history.replaceState(null, '', '/');
            setShareLoad({ status: 'idle' });
          })
          .catch((err: any) => {
            setShareLoad({ status: 'error', message: err?.message ?? String(err) });
          });
      })
      .finally(() => setInitialResetPending(false));
```

- [ ] **Step 5: Add the loading/error overlay**

In the returned JSX, immediately inside `<div className="app-container">` (after line 172) and before `<StartSplash ... />`, add:

```tsx
      {shareLoad.status !== 'idle' && (
        <div
          style={{
            position: 'fixed', inset: 0, zIndex: 2000,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(255, 255, 255, 0.96)',
          }}
        >
          {shareLoad.status === 'loading' ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="spinner" />
              Loading shared session…
            </div>
          ) : (
            <div style={{ textAlign: 'center', maxWidth: 420 }}>
              <p style={{ fontWeight: 600 }}>Could not load shared session</p>
              <p style={{ margin: '8px 0 16px' }}>{shareLoad.message}</p>
              <button
                className="btn btn-primary"
                type="button"
                onClick={() => { window.history.replaceState(null, '', '/'); setShareLoad({ status: 'idle' }); }}
              >
                Start fresh
              </button>
            </div>
          )}
        </div>
      )}
```

- [ ] **Step 6: Verify typecheck**

Run from `app/frontend`:
```
npx tsc -b --noEmit
```
Expected: exit 0.

- [ ] **Step 7: Commit**

```bash
git add app/frontend/src/App.tsx
git commit -m "feat(share): load /share/<token> links on app start"
```

---

### Task 8: Full verification + manual smoke

- [ ] **Step 1: Run the full backend + frontend test suites for touched areas**

Backend, from repo root:
```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_share.py app/backend/test_session_save_load.py app/backend/test_session_io.py -v
```
Expected: all PASS.

Frontend, from `app/frontend`:
```
npm test -- shareLink && npx tsc -b --noEmit
```
Expected: PASS + typecheck exit 0.

- [ ] **Step 2: Manual smoke (requires a running app + a generated model)**

Follow `app/command.txt` (or the project run instructions) to start backend + frontend. Then:
1. Generate a small model (Target → Generate).
2. Go to the **File** tab → click **Copy Share Link**. Confirm a URL appears and "Link copied to clipboard." shows.
3. Open the copied `/share/<token>` URL in a fresh browser tab. Confirm the "Loading shared session…" overlay appears, then the model loads and the URL resets to `/`.
4. Open a bad link `/share/aaaaaaaaaaaaaaaa` → confirm the "Could not load shared session" error overlay with a working "Start fresh" button.

Record the result of each step in the commit message or PR description.

- [ ] **Step 3: Final commit (if any smoke fixes were needed)**

```bash
git add -A
git commit -m "test(share): verify share create/load end-to-end"
```

---

### Self-Review Notes (verified against the spec)

- **Spec "Feature 2 — Copy Share Link / Backend":** covered by Tasks 1–3 (SHARE_DIR, share.py dropping `include_optimization`, two endpoints wired to `app_state`, token regex + atomic write + traversal guard retained verbatim from optree).
- **Spec "Feature 2 / Frontend":** covered by Tasks 4–7 (`createShare`/`loadShare`, ExportTab share section with clipboard→execCommand fallback, `parseShareToken`, App mount consumption + `history.replaceState('/')`, full-screen overlay). No routing library — pathname parsing only.
- **Spec "session/state compatibility" risk:** Task 8 Step 1 re-runs the existing session round-trip tests to confirm nothing regressed.
- **i18n:** intentionally deferred — all share strings are plain English here and will be wrapped by plan 4 of 4.
- **Type consistency:** `ShareCreateResult` (`token`, `path`) matches the backend `{token, path}` return; `loadShare` returns `SessionLoadSummary` (already defined in `api.ts`); `parseShareToken` regex matches the backend `_TOKEN_RE`.

---

## Plan 2 of 4 — DXF auxiliary-line import (+ Import-tab layout polish)

Port optree's DXF auxiliary-line import into VoxCity. This adds a second import mode alongside the existing OBJ import: DXF files (roads / site boundaries / reference geometry) are parsed into per-layer polylines, placed against the model with the same anchor/rotate/move/units controls as OBJ, then **baked to absolute lon/lat and stored as `auxiliary_lines`** — a non-voxelized overlay that renders as 3D reference lines and never mutates the voxel grid. The Import tab is reworked to optree's OBJ/DXF mode-toggle layout while preserving VoxCity's own specifics (`previewDisabled`/`previewGridShape`, `gizmoMode`, `plan-panel-header`, plain-English strings, `btn-secondary`).

Every optree endpoint uses `Depends(get_session)`; VoxCity has **no** `sessions.py` — every endpoint here is adapted to the global `app_state` (`from .state import app_state`) exactly like VoxCity's existing `import_obj` endpoints. The teal accent (`#009999`) is re-skinned to VoxCity indigo `#6666FF`; DXF UI accents use VoxCity's existing indigo `btn-primary`.

**Files**

Backend (new): `app/backend/dxf_import.py`, `app/backend/test_dxf_import.py`, `app/backend/test_dxf_endpoints.py`, `app/backend/test_auxiliary_lines_state.py`
Backend (edit): `app/backend/models.py`, `app/backend/state.py`, `app/backend/main.py`, `app/backend/session_io.py`, `app/backend/test_session_io.py`, `app/backend/requirements.txt`, `pyproject.toml`
Frontend (new): `app/frontend/src/lib/auxiliaryLines.ts`, `app/frontend/src/lib/auxiliaryLines.test.ts`, `app/frontend/src/components/DxfPlacementMap.tsx`, `app/frontend/src/components/AuxiliaryLinesControl.tsx`, `app/frontend/src/three/AuxiliaryLineLayer.tsx`
Frontend (edit): `app/frontend/src/api.ts`, `app/frontend/src/three/SceneViewer.tsx`, `app/frontend/src/tabs/ImportTab.tsx`

Conventions: backend tests run from repo root with `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest ...` and `PYTHONPATH=app` (imports are top-level `backend.*` / `from tests.importer.conftest import make_flat_voxcity`). Frontend: from `app/frontend`, `npm test -- <name>` and `npx tsc -b --noEmit`. Commit after each task.

> **NOTE (assembler):** Task-level ported code for Plan 2 is reproduced in full in the standalone draft. This combined document preserves the complete task list, file targets, commands, and self-review; where a code block is long it is retained verbatim from the optree source cited in each step. If any block was elided during assembly, recover it from the optree paths named in that step before implementing.

---

### Task 1 — Backend DXF parser (pure, ezdxf) + unit tests (TDD)

**Files:** `app/backend/dxf_import.py` (new), `app/backend/test_dxf_import.py` (new)

- [ ] **Step 1** Create `app/backend/dxf_import.py` with the verbatim optree parser (self-contained, no FastAPI/state deps):

```python
"""Pure DXF parsing for auxiliary reference lines (no FastAPI/state deps).

Extracts LINE / LWPOLYLINE / POLYLINE geometry, flattened to 2D and grouped by
DXF layer, for use as non-voxelized overlay polylines.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional

import ezdxf
import numpy as np
from ezdxf import colors as ezcolors
from ezdxf.recover import read as recover_read


class DxfParseError(ValueError):
    """Raised when the input cannot be parsed as a DXF document."""


# $INSUNITS code -> our units enum (only the ones the placement form supports).
_INSUNITS_TO_ENUM = {6: "m", 5: "cm", 4: "mm", 2: "ft", 1: "in"}
_DEFAULT_COLOR = "#888888"


@dataclass
class ParsedDxfLayer:
    name: str
    color: str                              # "#rrggbb"
    polylines: List[List[List[float]]]      # [ [ [x, y], ... ], ... ]


@dataclass
class ParsedDxf:
    layers: List[ParsedDxfLayer]
    bounds: List[List[float]]               # [[xmin, ymin], [xmax, ymax]]
    center: List[float]                     # [cx, cy]
    detected_units: Optional[str]
    insert_count: int = 0
    file_name: Optional[str] = None         # set by the upload endpoint


def _rgb_to_hex(rgb) -> Optional[str]:
    if not rgb:
        return None
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _resolve_color(entity, doc) -> str:
    # Entity true-color / ACI first, then fall back to the layer's color.
    try:
        if entity.dxf.hasattr("true_color"):
            hexc = _rgb_to_hex(entity.rgb)
            if hexc:
                return hexc
        aci = entity.dxf.color
        if aci and 0 < aci < 256:
            return _rgb_to_hex(ezcolors.aci2rgb(aci)) or _DEFAULT_COLOR
    except Exception:
        pass
    try:
        layer = doc.layers.get(entity.dxf.layer)
        if layer.dxf.hasattr("true_color"):
            hexc = _rgb_to_hex(layer.rgb)
            if hexc:
                return hexc
        aci = layer.dxf.color
        if aci and 0 < aci < 256:
            return _rgb_to_hex(ezcolors.aci2rgb(aci)) or _DEFAULT_COLOR
    except Exception:
        pass
    return _DEFAULT_COLOR


def _entity_polyline(entity) -> Optional[List[List[float]]]:
    t = entity.dxftype()
    if t == "LINE":
        s, e = entity.dxf.start, entity.dxf.end
        return [[float(s.x), float(s.y)], [float(e.x), float(e.y)]]
    if t == "LWPOLYLINE":
        pts = [[float(p[0]), float(p[1])] for p in entity.get_points()]
        if entity.closed and pts:
            pts.append([pts[0][0], pts[0][1]])
        return pts or None
    if t == "POLYLINE":
        # Old-style 2D polyline backed by VERTEX entities.
        pts = [[float(v.dxf.location.x), float(v.dxf.location.y)] for v in entity.vertices]
        if entity.is_closed and pts:
            pts.append([pts[0][0], pts[0][1]])
        return pts or None
    return None


def parse_dxf(data: bytes) -> ParsedDxf:
    try:
        doc, _auditor = recover_read(io.BytesIO(data))
        msp = doc.modelspace()  # touch modelspace here so a broken doc fails now
    except Exception as exc:  # ezdxf raises several types on bad input
        raise DxfParseError(f"Could not parse DXF: {exc}") from exc

    units_code = 0
    try:
        units_code = int(doc.header.get("$INSUNITS", 0))
    except Exception:
        units_code = 0
    detected_units = _INSUNITS_TO_ENUM.get(units_code)

    insert_count = 0
    order: List[str] = []
    by_layer: dict[str, ParsedDxfLayer] = {}
    xs: List[float] = []
    ys: List[float] = []

    for entity in msp:
        t = entity.dxftype()
        if t == "INSERT":
            insert_count += 1
            continue
        pts = _entity_polyline(entity)
        if not pts:
            continue
        name = entity.dxf.layer
        if name not in by_layer:
            by_layer[name] = ParsedDxfLayer(name=name, color=_resolve_color(entity, doc), polylines=[])
            order.append(name)
        by_layer[name].polylines.append(pts)
        for x, y in pts:
            xs.append(x)
            ys.append(y)

    layers = [by_layer[n] for n in order]
    if xs and ys:
        bounds = [[min(xs), min(ys)], [max(xs), max(ys)]]
        center = [(bounds[0][0] + bounds[1][0]) / 2.0, (bounds[0][1] + bounds[1][1]) / 2.0]
    else:
        bounds = [[0.0, 0.0], [0.0, 0.0]]
        center = [0.0, 0.0]

    return ParsedDxf(
        layers=layers,
        bounds=bounds,
        center=center,
        detected_units=detected_units,
        insert_count=insert_count,
    )


def bake_polylines_to_lonlat(polylines, mat, grid_geom) -> List[List[List[float]]]:
    """Map model-XY polylines to lon/lat via a model->voxel-index matrix + grid.

    `mat` is the 4x4 from voxcity.importer.transform.build_placement_transform;
    `grid_geom` is the dict from compute_grid_geometry (origin/u_vec/v_vec/adj_mesh).
    """
    origin = np.asarray(grid_geom["origin"], dtype=float)
    u_vec = np.asarray(grid_geom["u_vec"], dtype=float)
    v_vec = np.asarray(grid_geom["v_vec"], dtype=float)
    dx = float(grid_geom["adj_mesh"][0])
    dy = float(grid_geom["adj_mesh"][1])
    out: List[List[List[float]]] = []
    for ring in polylines:
        baked: List[List[float]] = []
        for x, y in ring:
            ijk = mat @ np.array([float(x), float(y), 0.0, 1.0])
            i_f, j_f = float(ijk[0]), float(ijk[1])
            lon = origin[0] + (i_f * dx) * u_vec[0] + (j_f * dy) * v_vec[0]
            lat = origin[1] + (i_f * dx) * u_vec[1] + (j_f * dy) * v_vec[1]
            baked.append([float(lon), float(lat)])
        out.append(baked)
    return out
```

- [ ] **Step 2** Create `app/backend/test_dxf_import.py` (ported from optree; the ONLY change is the import path `app.backend.dxf_import` -> `backend.dxf_import`):

```python
import io

import ezdxf
import pytest

from backend.dxf_import import parse_dxf, ParsedDxf


def _to_bytes(doc) -> bytes:
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


def _doc(units: int | None = None):
    doc = ezdxf.new()
    if units is not None:
        doc.header["$INSUNITS"] = units
    return doc


def test_parses_line_as_two_point_polyline():
    doc = _doc()
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 5), dxfattribs={"layer": "walls"})
    parsed = parse_dxf(_to_bytes(doc))
    assert isinstance(parsed, ParsedDxf)
    layer = next(l for l in parsed.layers if l.name == "walls")
    assert layer.polylines == [[[0.0, 0.0], [10.0, 5.0]]]


def test_parses_open_and_closed_lwpolyline():
    doc = _doc()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (1, 0), (1, 1)], close=False, dxfattribs={"layer": "a"})
    msp.add_lwpolyline([(0, 0), (2, 0), (2, 2)], close=True, dxfattribs={"layer": "a"})
    parsed = parse_dxf(_to_bytes(doc))
    a = next(l for l in parsed.layers if l.name == "a")
    assert a.polylines[0] == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    assert a.polylines[1] == [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 0.0]]


def test_parses_old_style_polyline_with_vertices():
    doc = _doc()
    msp = doc.modelspace()
    pl = msp.add_polyline2d([(0, 0), (3, 0), (3, 3)], dxfattribs={"layer": "window"})
    pl.close(True)
    parsed = parse_dxf(_to_bytes(doc))
    w = next(l for l in parsed.layers if l.name == "window")
    assert w.polylines[0] == [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 0.0]]


def test_groups_by_layer_in_first_seen_order():
    doc = _doc()
    msp = doc.modelspace()
    msp.add_line((0, 0), (1, 0), dxfattribs={"layer": "second"})
    msp.add_line((0, 0), (1, 0), dxfattribs={"layer": "first"})
    msp.add_line((0, 0), (1, 0), dxfattribs={"layer": "second"})
    parsed = parse_dxf(_to_bytes(doc))
    names = [l.name for l in parsed.layers]
    assert names == ["second", "first"]


def test_bounds_and_center():
    doc = _doc()
    msp = doc.modelspace()
    msp.add_line((-2, -4), (6, 10), dxfattribs={"layer": "a"})
    parsed = parse_dxf(_to_bytes(doc))
    assert parsed.bounds == [[-2.0, -4.0], [6.0, 10.0]]
    assert parsed.center == [2.0, 3.0]


def test_color_is_hex_string():
    doc = _doc()
    msp = doc.modelspace()
    msp.add_line((0, 0), (1, 1), dxfattribs={"layer": "a", "true_color": 0xFF8800})
    parsed = parse_dxf(_to_bytes(doc))
    a = next(l for l in parsed.layers if l.name == "a")
    assert a.color.lower() == "#ff8800"


@pytest.mark.parametrize(
    "code,expected",
    [(6, "m"), (5, "cm"), (4, "mm"), (2, "ft"), (1, "in"), (0, None)],
)
def test_detects_insunits(code, expected):
    doc = _doc(units=code)
    doc.modelspace().add_line((0, 0), (1, 1), dxfattribs={"layer": "a"})
    parsed = parse_dxf(_to_bytes(doc))
    assert parsed.detected_units == expected


def test_missing_insunits_is_none():
    doc = _doc()
    doc.modelspace().add_line((0, 0), (1, 1), dxfattribs={"layer": "a"})
    lines = _to_bytes(doc).decode("utf-8").splitlines()
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "$INSUNITS" and cleaned and cleaned[-1].strip() == "9":
            cleaned.pop()
            i += 3
            continue
        cleaned.append(lines[i])
        i += 1
    data = ("\n".join(cleaned) + "\n").encode("utf-8")
    parsed = parse_dxf(data)
    assert parsed.detected_units is None


def test_empty_document_has_no_layers_and_zero_inserts():
    parsed = parse_dxf(_to_bytes(_doc()))
    assert parsed.layers == []
    assert parsed.insert_count == 0


def test_counts_inserts_when_geometry_only_in_block():
    doc = _doc()
    blk = doc.blocks.new(name="B")
    blk.add_line((0, 0), (1, 1))
    doc.modelspace().add_blockref("B", (0, 0))
    parsed = parse_dxf(_to_bytes(doc))
    assert parsed.layers == []
    assert parsed.insert_count == 1


def test_malformed_input_raises_dxf_parse_error():
    from backend.dxf_import import DxfParseError
    with pytest.raises(DxfParseError):
        parse_dxf(b"this is not a dxf file")


import numpy as np

from backend.dxf_import import bake_polylines_to_lonlat


def test_bake_polylines_identity_grid():
    mat = np.eye(4)
    grid_geom = {
        "origin": [100.0, 50.0],
        "u_vec": [1.0, 0.0],
        "v_vec": [0.0, 1.0],
        "adj_mesh": [1.0, 1.0],
    }
    out = bake_polylines_to_lonlat([[[2.0, 3.0], [4.0, 5.0]]], mat, grid_geom)
    assert out == [[[102.0, 53.0], [104.0, 55.0]]]
```

- [ ] **Step 3** Install ezdxf into the env (Task 6 also records it in requirements/pyproject; install now so tests run):

```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pip install "ezdxf>=1.1,<2"
```

Expected: `Successfully installed ezdxf-1.x.x` (or "already satisfied").

- [ ] **Step 4** Run the parser tests:

```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_dxf_import.py -v
```

Expected: all pass (`test_detects_insunits` runs 6 param cases). No import errors for `backend.dxf_import`.

- [ ] **Step 5** Commit: `git add -A && git commit -m "feat(dxf): pure DXF polyline parser + bake-to-lonlat with unit tests"`

---

### Task 2 — Pydantic models for DXF import + auxiliary lines

**Files:** `app/backend/models.py`

- [ ] **Step 1** In `app/backend/models.py`, immediately after the existing `ImportObjCommitResponse` class (find it with `grep -n "class ImportObjCommitResponse" app/backend/models.py`), insert the DXF models verbatim from optree (`Dict`, `List`, `Optional`, `Field`, `BaseModel` are already imported at the top of the module — confirm with `grep -n "from pydantic" app/backend/models.py`):

```python
class DxfLayerInfo(BaseModel):
    name: str
    color: str            # "#rrggbb"
    n_segments: int


class DxfPreviewLayer(BaseModel):
    name: str
    color: str
    polylines: List[List[List[float]]]   # model-XY, decimated if large


class ImportDxfPreview(BaseModel):
    layers: List[DxfPreviewLayer]


class ImportDxfUploadResponse(BaseModel):
    import_id: str
    layers: List[DxfLayerInfo]
    model_bounds: List[List[float]]       # [[xmin,ymin],[xmax,ymax]]
    model_center: List[float]             # [cx, cy] (default placement pivot)
    detected_units: Optional[str] = None  # from $INSUNITS, if present
    preview: ImportDxfPreview
    warning: Optional[str] = None


class DxfPlacement(BaseModel):
    anchor_lonlat: List[float] = Field(..., min_length=2, max_length=2)  # [lon, lat]
    anchor_model_point: List[float] = Field(default_factory=lambda: [0.0, 0.0], min_length=2, max_length=2)
    rotation: float = 0.0                 # degrees
    move: List[float] = Field(default_factory=lambda: [0.0, 0.0], min_length=2, max_length=2)  # [east, north] m
    units: str = "m"


class ImportDxfCommitRequest(BaseModel):
    import_id: str = Field(..., max_length=64)
    placement: DxfPlacement
    layer_visibility: Dict[str, bool] = Field(default_factory=dict)


class AuxiliaryLine(BaseModel):
    id: str
    file_name: str
    layer: str
    color: str
    points: List[List[float]]             # [[lon, lat], ...]


class ImportDxfCommitResponse(BaseModel):
    auxiliary_lines: List[AuxiliaryLine]
    warning: Optional[str] = None
```

- [ ] **Step 2** Confirm the module still imports cleanly:

```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -c "import sys; sys.path.insert(0,'app'); from backend import models; print(models.AuxiliaryLine.__fields__.keys())"
```

Expected: prints `dict_keys(['id', 'file_name', 'layer', 'color', 'points'])`.

- [ ] **Step 3** Commit: `git add -A && git commit -m "feat(dxf): pydantic models for DXF import + auxiliary lines"`

---

### Task 3 — `auxiliary_lines` state field + test

**Files:** `app/backend/state.py`, `app/backend/test_auxiliary_lines_state.py` (new)

- [ ] **Step 1** In `app/backend/state.py`, add the field to `AppState`. Insert it right after the `land_cover_source: str = "OpenStreetMap"` line (state.py:53):

```python
    land_cover_source: str = "OpenStreetMap"

    # Baked auxiliary reference lines imported from DXF, in absolute lon/lat,
    # for the 2D/3D overlay. Single source of truth for this geometry; never
    # voxelized. Each entry: {"id": str, "file_name": str, "layer": str,
    # "color": "#rrggbb", "points": [[lon, lat], ...]}.
    auxiliary_lines: List[Dict[str, Any]] = field(default_factory=list)
```

(`List`, `Dict`, `Any`, `field` are already imported at state.py:10-12.)

- [ ] **Step 2** In `AppState.store_generation_result` (state.py:148), clear stale aux lines whenever a brand-new model replaces the old one. Add this line at the end of the method body, after `self.raw_data = { ... }` is assigned (i.e. after state.py:176):

```python
        # A freshly generated model invalidates any previously imported DXF
        # overlay (its lon/lat geometry belongs to the old grid).
        self.auxiliary_lines = []
```

- [ ] **Step 3** Create `app/backend/test_auxiliary_lines_state.py` (ported, import path adapted to `backend.state`):

```python
from backend.state import AppState


def test_auxiliary_lines_defaults_empty():
    s = AppState()
    assert s.auxiliary_lines == []
```

- [ ] **Step 4** Run:

```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_auxiliary_lines_state.py -v
```

Expected: 1 passed.

- [ ] **Step 5** Commit: `git add -A && git commit -m "feat(dxf): auxiliary_lines state field, cleared on regeneration"`

---

### Task 4 — DXF endpoints (upload / commit / delete) adapted to global `app_state` + geo exposure + tests

**Files:** `app/backend/main.py`, `app/backend/test_dxf_endpoints.py` (new)

- [ ] **Step 1** In `app/backend/main.py`, extend the `from .models import (...)` block (main.py:39-68) to add the new symbols (keep the existing entries; add these names anywhere in the parenthesized list):

```python
    AuxiliaryLine,
    DxfLayerInfo,
    DxfPreviewLayer,
    ImportDxfPreview,
    ImportDxfUploadResponse,
    DxfPlacement,
    ImportDxfCommitRequest,
    ImportDxfCommitResponse,
```

- [ ] **Step 2** Add the parser import and the in-memory store. Directly below the existing `import_obj_store: Dict[str, str] = {}` (main.py:172), add:

```python
from .dxf_import import parse_dxf, bake_polylines_to_lonlat, DxfParseError, ParsedDxf

# import_id -> ParsedDxf (uploaded-but-not-yet-committed DXF documents).
import_dxf_store: Dict[str, "ParsedDxf"] = {}

_MAX_PREVIEW_SEGMENTS = 20000
```

- [ ] **Step 3** Expose committed aux lines through `GET /api/model/geo`. In the returned dict of `model_geo()` (main.py:2741-2759), add one key alongside `"land_cover_source"`:

```python
            "land_cover_source": lc_source,
            "auxiliary_lines": list(app_state.auxiliary_lines),
            "building_geojson": building_fc,
```

- [ ] **Step 4** Append the three endpoints. Insert them immediately after `import_obj_commit` finishes (after its `return ImportObjCommitResponse(...)` block near main.py ~3560; place before the next `@app.` decorator). These are the optree endpoints with **every** `state: AppState = Depends(get_session)` parameter removed and every `state.` replaced by `app_state.`, and `_require_model(state)` -> `_require_model()`:

```python
@app.post("/api/model/import_dxf/upload", response_model=ImportDxfUploadResponse)
async def import_dxf_upload(file: UploadFile = File(...)):
    """Parse an uploaded DXF into per-layer polylines + preview; register import_id."""
    _require_model()
    import uuid
    data = await file.read()
    try:
        parsed = parse_dxf(data)
    except DxfParseError as e:
        raise HTTPException(status_code=400, detail=str(e))

    import_id = uuid.uuid4().hex
    parsed.file_name = file.filename or "import.dxf"
    import_dxf_store[import_id] = parsed

    total = sum(len(pl) for layer in parsed.layers for pl in layer.polylines)
    stride = max(1, (total + _MAX_PREVIEW_SEGMENTS - 1) // _MAX_PREVIEW_SEGMENTS)
    layer_infos: List[DxfLayerInfo] = []
    preview_layers: List[DxfPreviewLayer] = []
    for layer in parsed.layers:
        n_seg = sum(max(0, len(pl) - 1) for pl in layer.polylines)
        layer_infos.append(DxfLayerInfo(name=layer.name, color=layer.color, n_segments=n_seg))
        preview_polys = layer.polylines if stride == 1 else [pl for i, pl in enumerate(layer.polylines) if i % stride == 0]
        preview_layers.append(DxfPreviewLayer(name=layer.name, color=layer.color, polylines=preview_polys))

    warning = None
    if not parsed.layers:
        if parsed.insert_count > 0:
            warning = ("Supported geometry was found only inside block/INSERT references, which "
                       "are not imported; no auxiliary lines were extracted.")
        else:
            warning = "No LINE/LWPOLYLINE/POLYLINE geometry found in the DXF."
    elif parsed.detected_units is None:
        warning = "DXF has no $INSUNITS; assuming meters — set the units before placing if that's wrong."

    return ImportDxfUploadResponse(
        import_id=import_id,
        layers=layer_infos,
        model_bounds=parsed.bounds,
        model_center=parsed.center,
        detected_units=parsed.detected_units,
        preview=ImportDxfPreview(layers=preview_layers),
        warning=warning,
    )


@app.post("/api/model/import_dxf/commit", response_model=ImportDxfCommitResponse)
async def import_dxf_commit(req: ImportDxfCommitRequest):
    """Bake the placed DXF polylines to lon/lat and store as auxiliary lines."""
    _require_model()
    import uuid
    from voxcity.importer.transform import build_placement_transform
    from voxcity.geoprocessor.draw._common import compute_grid_geometry

    parsed = import_dxf_store.get(req.import_id)
    if parsed is None:
        raise HTTPException(status_code=404, detail="Unknown or expired import_id; please re-upload.")

    p = req.placement
    for name, vec, n in (("anchor_lonlat", p.anchor_lonlat, 2), ("anchor_model_point", p.anchor_model_point, 2), ("move", p.move, 2)):
        if len(vec) != n or not all(math.isfinite(float(x)) for x in vec):
            raise HTTPException(status_code=400, detail=f"{name} must be {n} finite numbers")
    if not math.isfinite(float(p.rotation)):
        raise HTTPException(status_code=400, detail="rotation must be finite")

    rect = app_state.rectangle_vertices
    if rect is None and isinstance(app_state.voxcity.extras, dict):
        rect = app_state.voxcity.extras.get("rectangle_vertices")
    if rect is None:
        raise HTTPException(status_code=400, detail="Model has no rectangle_vertices")
    grid_geom = compute_grid_geometry(rect, float(app_state.meshsize))
    if grid_geom is None:
        raise HTTPException(status_code=500, detail="compute_grid_geometry returned None")

    mat = build_placement_transform(
        app_state.voxcity,
        anchor_lonlat=(float(p.anchor_lonlat[0]), float(p.anchor_lonlat[1])),
        anchor_elevation=0.0,   # DXF is 2D; height output ignored
        anchor_model_point=(float(p.anchor_model_point[0]), float(p.anchor_model_point[1]), 0.0),
        rotation=float(p.rotation),
        move=(float(p.move[0]), float(p.move[1]), 0.0),
        units=p.units,
    )

    new_lines: List[Dict[str, Any]] = []
    file_name = parsed.file_name or "import.dxf"
    for layer in parsed.layers:
        if req.layer_visibility and req.layer_visibility.get(layer.name) is False:
            continue
        baked = bake_polylines_to_lonlat(layer.polylines, mat, grid_geom)
        for ring in baked:
            new_lines.append({
                "id": uuid.uuid4().hex,
                "file_name": file_name,
                "layer": layer.name,
                "color": layer.color,
                "points": ring,
            })
    app_state.auxiliary_lines.extend(new_lines)
    import_dxf_store.pop(req.import_id, None)

    return ImportDxfCommitResponse(
        auxiliary_lines=[AuxiliaryLine(**ln) for ln in new_lines],
        warning=None if new_lines else "No visible layers were committed.",
    )


@app.delete("/api/model/auxiliary_lines")
async def delete_auxiliary_lines(file_name: Optional[str] = None, id: Optional[str] = None):
    """Clear auxiliary lines (all, or filtered by file_name/id). Idempotent."""
    if file_name is None and id is None:
        app_state.auxiliary_lines = []
    else:
        def _matches(ln: Dict[str, Any]) -> bool:
            if file_name is not None and ln.get("file_name") == file_name:
                return True
            if id is not None and ln.get("id") == id:
                return True
            return False
        app_state.auxiliary_lines = [ln for ln in app_state.auxiliary_lines if not _matches(ln)]
    return {"auxiliary_lines": app_state.auxiliary_lines}
```

Note: `math`, `HTTPException`, `UploadFile`, `File`, `Optional`, `Dict`, `Any`, `List` are all already imported in `main.py` (used by the OBJ endpoints); do not re-import them at module scope.

- [ ] **Step 5** Create `app/backend/test_dxf_endpoints.py` — ported from optree but rewritten to VoxCity's global-`app_state` fixture pattern (copied from `test_import_obj.py`: autouse fixture installs a flat model into the global `app_state`; no `dependency_overrides`, no `sessions`):

```python
"""Tests for the DXF import endpoints (upload / commit / delete)."""
from __future__ import annotations

import io

import ezdxf
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app, import_dxf_store
from backend.state import app_state
from tests.importer.conftest import make_flat_voxcity


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _model_loaded():
    app_state.voxcity = make_flat_voxcity(nx=30, ny=30, nz=12, meshsize=1.0)
    app_state.rectangle_vertices = app_state.voxcity.extras["rectangle_vertices"]
    app_state.land_cover_source = "OpenStreetMap"
    app_state.auxiliary_lines = []
    import_dxf_store.clear()
    yield
    app_state.voxcity = None
    app_state.rectangle_vertices = None
    app_state.auxiliary_lines = []
    import_dxf_store.clear()


def _dxf_bytes() -> bytes:
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (4, 0), (4, 3)], close=True, dxfattribs={"layer": "window"})
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


def test_dxf_upload_returns_layers_and_center(client):
    files = {"file": ("test.dxf", _dxf_bytes(), "application/dxf")}
    r = client.post("/api/model/import_dxf/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert [l["name"] for l in body["layers"]] == ["window"]
    assert body["model_center"] == [2.0, 1.5]
    assert body["import_id"]


def test_dxf_commit_populates_auxiliary_lines_and_geo(client):
    up = client.post(
        "/api/model/import_dxf/upload",
        files={"file": ("test.dxf", _dxf_bytes(), "application/dxf")},
    ).json()
    geo0 = client.get("/api/model/geo").json()
    anchor = geo0["center"][::-1]  # geo center is [lat, lon]; placement wants [lon, lat]
    commit = client.post("/api/model/import_dxf/commit", json={
        "import_id": up["import_id"],
        "placement": {"anchor_lonlat": anchor, "anchor_model_point": up["model_center"],
                      "rotation": 0, "move": [0, 0], "units": "m"},
        "layer_visibility": {"window": True},
    })
    assert commit.status_code == 200, commit.text
    lines = commit.json()["auxiliary_lines"]
    assert len(lines) == 1
    assert lines[0]["layer"] == "window"
    assert lines[0]["file_name"] == "test.dxf"
    assert len(lines[0]["points"]) == 4  # closed ring

    geo = client.get("/api/model/geo").json()
    assert len(geo["auxiliary_lines"]) == 1


def test_dxf_upload_warns_when_geometry_only_in_blocks(client):
    doc = ezdxf.new()
    blk = doc.blocks.new(name="B")
    blk.add_line((0, 0), (1, 1))
    doc.modelspace().add_blockref("B", (0, 0))
    buf = io.StringIO(); doc.write(buf)
    r = client.post(
        "/api/model/import_dxf/upload",
        files={"file": ("b.dxf", buf.getvalue().encode(), "application/dxf")},
    )
    assert r.status_code == 200
    assert "block" in (r.json().get("warning") or "").lower()


def test_dxf_upload_rejects_malformed(client):
    r = client.post(
        "/api/model/import_dxf/upload",
        files={"file": ("bad.dxf", b"not a dxf", "application/dxf")},
    )
    assert r.status_code == 400


def test_delete_auxiliary_lines(client):
    up = client.post("/api/model/import_dxf/upload",
                     files={"file": ("t.dxf", _dxf_bytes(), "application/dxf")}).json()
    anchor = client.get("/api/model/geo").json()["center"][::-1]
    client.post("/api/model/import_dxf/commit", json={
        "import_id": up["import_id"],
        "placement": {"anchor_lonlat": anchor, "anchor_model_point": up["model_center"],
                      "rotation": 0, "move": [0, 0], "units": "m"},
        "layer_visibility": {}})
    r = client.delete("/api/model/auxiliary_lines")
    assert r.status_code == 200
    assert client.get("/api/model/geo").json()["auxiliary_lines"] == []
    assert client.delete("/api/model/auxiliary_lines").status_code == 200


def test_upload_requires_model(client):
    app_state.voxcity = None
    r = client.post("/api/model/import_dxf/upload",
                    files={"file": ("t.dxf", _dxf_bytes(), "application/dxf")})
    assert 400 <= r.status_code < 500


def test_commit_requires_model(client):
    app_state.voxcity = None
    r = client.post("/api/model/import_dxf/commit", json={
        "import_id": "nope",
        "placement": {"anchor_lonlat": [139.7, 35.69], "anchor_model_point": [0, 0],
                      "rotation": 0, "move": [0, 0], "units": "m"},
        "layer_visibility": {}})
    assert 400 <= r.status_code < 500


def test_commit_layer_visibility_filters(client):
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (1, 0), (1, 1)], close=True, dxfattribs={"layer": "a"})
    msp.add_lwpolyline([(2, 2), (3, 2), (3, 3)], close=True, dxfattribs={"layer": "b"})
    buf = io.StringIO(); doc.write(buf)
    up = client.post("/api/model/import_dxf/upload",
                     files={"file": ("m.dxf", buf.getvalue().encode(), "application/dxf")}).json()
    anchor = client.get("/api/model/geo").json()["center"][::-1]
    commit = client.post("/api/model/import_dxf/commit", json={
        "import_id": up["import_id"],
        "placement": {"anchor_lonlat": anchor, "anchor_model_point": up["model_center"],
                      "rotation": 0, "move": [0, 0], "units": "m"},
        "layer_visibility": {"a": True, "b": False}})
    assert commit.status_code == 200, commit.text
    layers = {ln["layer"] for ln in commit.json()["auxiliary_lines"]}
    assert layers == {"a"}


def test_delete_auxiliary_lines_by_file_and_id(client):
    up1 = client.post("/api/model/import_dxf/upload",
                      files={"file": ("f1.dxf", _dxf_bytes(), "application/dxf")}).json()
    anchor = client.get("/api/model/geo").json()["center"][::-1]
    pl = {"anchor_lonlat": anchor, "anchor_model_point": up1["model_center"],
          "rotation": 0, "move": [0, 0], "units": "m"}
    client.post("/api/model/import_dxf/commit",
                json={"import_id": up1["import_id"], "placement": pl, "layer_visibility": {}})
    up2 = client.post("/api/model/import_dxf/upload",
                      files={"file": ("f2.dxf", _dxf_bytes(), "application/dxf")}).json()
    pl2 = {**pl, "anchor_model_point": up2["model_center"]}
    client.post("/api/model/import_dxf/commit",
                json={"import_id": up2["import_id"], "placement": pl2, "layer_visibility": {}})
    lines = client.get("/api/model/geo").json()["auxiliary_lines"]
    assert {ln["file_name"] for ln in lines} == {"f1.dxf", "f2.dxf"}
    assert client.delete("/api/model/auxiliary_lines?file_name=f1.dxf").status_code == 200
    lines = client.get("/api/model/geo").json()["auxiliary_lines"]
    assert {ln["file_name"] for ln in lines} == {"f2.dxf"}
    target_id = lines[0]["id"]
    assert client.delete(f"/api/model/auxiliary_lines?id={target_id}").status_code == 200
    remaining = client.get("/api/model/geo").json()["auxiliary_lines"]
    assert all(ln["id"] != target_id for ln in remaining)


def test_far_from_origin_lands_on_map(client):
    base = 1_000_000.0
    doc = ezdxf.new()
    doc.modelspace().add_lwpolyline(
        [(base, base), (base + 4, base), (base + 4, base + 3)],
        close=True, dxfattribs={"layer": "window"})
    buf = io.StringIO(); doc.write(buf)
    up = client.post("/api/model/import_dxf/upload",
                     files={"file": ("far.dxf", buf.getvalue().encode(), "application/dxf")}).json()
    anchor = client.get("/api/model/geo").json()["center"][::-1]
    commit = client.post("/api/model/import_dxf/commit", json={
        "import_id": up["import_id"],
        "placement": {"anchor_lonlat": anchor, "anchor_model_point": up["model_center"],
                      "rotation": 0, "move": [0, 0], "units": "m"},
        "layer_visibility": {}}).json()
    pts = commit["auxiliary_lines"][0]["points"]
    lons = [p[0] for p in pts]; lats = [p[1] for p in pts]
    assert 139.699 < min(lons) and max(lons) < 139.702
    assert 35.689 < min(lats) and max(lats) < 35.692


def test_commit_does_not_mutate_voxels(client):
    before = int(np.asarray(app_state.voxcity.voxels.classes).sum())
    up = client.post("/api/model/import_dxf/upload",
                     files={"file": ("t.dxf", _dxf_bytes(), "application/dxf")}).json()
    anchor = client.get("/api/model/geo").json()["center"][::-1]
    client.post("/api/model/import_dxf/commit", json={
        "import_id": up["import_id"],
        "placement": {"anchor_lonlat": anchor, "anchor_model_point": up["model_center"],
                      "rotation": 0, "move": [0, 0], "units": "m"},
        "layer_visibility": {}})
    after = int(np.asarray(app_state.voxcity.voxels.classes).sum())
    assert before == after
```

Note: the optree fixtures assumed `make_flat_voxcity` returns a rectangle spanning `_RECT` near Tokyo. VoxCity's `tests.importer.conftest.make_flat_voxcity(nx, ny, nz, meshsize)` is the one `test_import_obj.py` already uses; the `test_far_from_origin` / `test_dxf_upload_returns_layers_and_center` assertions about `model_center == [2.0, 1.5]` depend only on the DXF geometry, not the model rectangle. If `make_flat_voxcity`'s rectangle differs from `139.700/35.690`, adjust the two lon/lat bounds asserts in `test_far_from_origin_lands_on_map` to `geo["center"]` +/- a few 1e-3; verify against the actual center printed by the failing assertion.

- [ ] **Step 6** Run:

```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_dxf_endpoints.py -v
```

Expected: all pass. If `test_far_from_origin_lands_on_map` fails only on the numeric bounds, re-anchor them to the model center as noted; the structural assertions (counts, layers, no-voxel-mutation, filtering, delete) must pass unchanged.

- [ ] **Step 7** Commit: `git add -A && git commit -m "feat(dxf): import_dxf upload/commit + auxiliary_lines delete endpoints on app_state; expose in /model/geo"`

---

### Task 5 — Session save/load round-trip for auxiliary lines + test

**Files:** `app/backend/session_io.py`, `app/backend/test_session_io.py`

- [ ] **Step 1** In `app/backend/session_io.py`, add the archive member name next to the others (session_io.py:23-25):

```python
_VOXCITY_NAME = "voxcity.h5"
_META_NAME = "meta.json"
_FRONTEND_STATE_NAME = "frontend_state.json"
_AUXILIARY_LINES_NAME = "auxiliary_lines.json"
```

- [ ] **Step 2** Add the field to `ParsedSession` (session_io.py:32-39):

```python
@dataclass
class ParsedSession:
    """Output of parse_session_zip — fully validated, in-memory representation."""

    meta: Dict[str, Any]
    voxcity_h5_path: str
    frontend_state: Optional[str] = None
    sim_results: Optional[Dict[str, Any]] = None  # filled in by a later task
    auxiliary_lines: Optional[list] = None        # DXF overlay, [[...], ...]
```

- [ ] **Step 3** Write the file during save. In `save_session_to_zip`, right after the `frontend_state` block (after session_io.py:93, before `if include_sim_results:`), add:

```python
        auxiliary_lines = getattr(state, "auxiliary_lines", None)
        if auxiliary_lines:
            (tmp / _AUXILIARY_LINES_NAME).write_text(
                json.dumps(auxiliary_lines),
                encoding="utf-8",
            )
```

- [ ] **Step 4** Read it back during parse. In `parse_session_zip`, after `frontend_state` is resolved and before `sim_results = _parse_sim_results(...)` (session_io.py:157), add:

```python
        auxiliary_lines = None
        aux_path = tmp / _AUXILIARY_LINES_NAME
        if aux_path.is_file():
            try:
                loaded_aux = json.loads(aux_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SessionLoadError(
                    f"auxiliary_lines.json is not valid JSON: {exc}"
                ) from exc
            if isinstance(loaded_aux, list):
                auxiliary_lines = loaded_aux
```

Then include it in the `ParsedSession(...)` constructor at session_io.py:159-164:

```python
        return ParsedSession(
            meta=meta,
            voxcity_h5_path=str(voxcity_path),
            frontend_state=frontend_state,
            sim_results=sim_results,
            auxiliary_lines=auxiliary_lines,
        )
```

- [ ] **Step 5** Restore into state. In `apply_session_to_state`, after `state.reset_for_session_load()` (session_io.py:399), add:

```python
    state.auxiliary_lines = list(parsed.auxiliary_lines or [])
```

- [ ] **Step 6** Add a round-trip test to `app/backend/test_session_io.py` (uses the existing monkeypatched save/load fixture in that file, so no real H5 is needed):

```python
def test_auxiliary_lines_round_trip(tmp_path):
    from types import SimpleNamespace
    from backend.session_io import parse_session_zip, save_session_to_zip

    aux = [
        {"id": "a1", "file_name": "roads.dxf", "layer": "centerline",
         "color": "#6666ff", "points": [[139.70, 35.69], [139.701, 35.6905]]},
    ]
    state = SimpleNamespace(
        voxcity=SimpleNamespace(voxels=SimpleNamespace(meta=SimpleNamespace(meshsize=5.0))),
        rectangle_vertices=[[139.0, 35.0], [139.1, 35.0], [139.1, 35.1], [139.0, 35.1]],
        land_cover_source="OpenStreetMap",
        auxiliary_lines=aux,
    )
    buf = save_session_to_zip(state)
    parsed = parse_session_zip(buf)
    assert parsed.auxiliary_lines == aux


def test_auxiliary_lines_absent_when_empty(tmp_path):
    from types import SimpleNamespace
    from backend.session_io import parse_session_zip, save_session_to_zip

    state = SimpleNamespace(
        voxcity=SimpleNamespace(voxels=SimpleNamespace(meta=SimpleNamespace(meshsize=5.0))),
        rectangle_vertices=[[139.0, 35.0], [139.1, 35.0], [139.1, 35.1], [139.0, 35.1]],
        land_cover_source="OpenStreetMap",
        auxiliary_lines=[],
    )
    parsed = parse_session_zip(save_session_to_zip(state))
    assert parsed.auxiliary_lines is None  # no file written for empty list
```

- [ ] **Step 7** Run:

```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_session_io.py -v
```

Expected: existing tests still pass + the 2 new tests pass.

- [ ] **Step 8** Commit: `git add -A && git commit -m "feat(dxf): persist auxiliary_lines through session save/load round-trip"`

---

### Task 6 — Declare the `ezdxf` dependency

**Files:** `app/backend/requirements.txt`, `pyproject.toml`

- [ ] **Step 1** Append to `app/backend/requirements.txt` (currently 4 lines):

```
ezdxf>=1.1,<2
```

- [ ] **Step 2** `pyproject.toml` uses Poetry (`[tool.poetry.dependencies]`, `pkg = "spec"` form). Add `ezdxf` in that table, e.g. right after `trimesh = "*"` (pyproject.toml:52):

```toml
trimesh = "*"
ezdxf = ">=1.1,<2"
```

- [ ] **Step 3** Verify the file still parses as TOML:

```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('ok')"
```

Expected: `ok`.

- [ ] **Step 4** Commit: `git add -A && git commit -m "build(dxf): add ezdxf dependency (requirements.txt + pyproject)"`

---

### Task 7 — Frontend API surface for DXF

**Files:** `app/frontend/src/api.ts`

- [ ] **Step 1** Extend `ModelGeoResult` (api.ts:334-352). Add one field before the closing brace (after `land_cover_geojson: any;`):

```typescript
  land_cover_geojson: any;
  // Baked DXF auxiliary lines in absolute lon/lat (backend-authoritative).
  auxiliary_lines?: AuxiliaryLineDto[];
}
```

- [ ] **Step 2** Append the DXF DTOs + calls to `api.ts` (near the OBJ import functions, after `commitImportObj` at api.ts:695). Ported verbatim from optree (`BASE` and `request` helpers already exist in this file):

```typescript
export interface AuxiliaryLineDto {
  id: string;
  file_name: string;
  layer: string;
  color: string;
  points: [number, number][]; // [lon, lat]
}

export interface DxfLayerInfoDto { name: string; color: string; n_segments: number; }
export interface DxfPreviewLayerDto { name: string; color: string; polylines: [number, number][][]; }

export interface ImportDxfUploadResult {
  import_id: string;
  layers: DxfLayerInfoDto[];
  model_bounds: [number, number][];
  model_center: [number, number];
  detected_units: string | null;
  preview: { layers: DxfPreviewLayerDto[] };
  warning: string | null;
}

export interface DxfPlacementDto {
  anchor_lonlat: [number, number];
  anchor_model_point: [number, number];
  rotation: number;
  move: [number, number];
  units: string;
}

export interface ImportDxfCommitRequestDto {
  import_id: string;
  placement: DxfPlacementDto;
  layer_visibility: Record<string, boolean>;
}

export interface ImportDxfCommitResult {
  auxiliary_lines: AuxiliaryLineDto[];
  warning: string | null;
}

export async function uploadImportDxf(file: File): Promise<ImportDxfUploadResult> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/model/import_dxf/upload`, { method: 'POST', body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function commitImportDxf(req: ImportDxfCommitRequestDto) {
  return request<ImportDxfCommitResult>('/model/import_dxf/commit', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function clearAuxiliaryLines(params: { fileName?: string; id?: string } = {}) {
  const qs = new URLSearchParams();
  if (params.fileName) qs.set('file_name', params.fileName);
  if (params.id) qs.set('id', params.id);
  const suffix = qs.toString() ? `?${qs.toString()}` : '';
  return request<{ auxiliary_lines: AuxiliaryLineDto[] }>(`/model/auxiliary_lines${suffix}`, {
    method: 'DELETE',
  });
}
```

- [ ] **Step 2b** Confirm `AuxiliaryLineDto` is declared before its use in `ModelGeoResult` — TypeScript interface declarations are hoisted within a module, so ordering does not matter; no reordering needed.

- [ ] **Step 3** Typecheck (from `app/frontend`):

```
npx tsc -b --noEmit
```

Expected: no errors (nothing consumes the new symbols yet — that's fine).

- [ ] **Step 4** Commit: `git add -A && git commit -m "feat(dxf): frontend api types + upload/commit/clear calls"`

---

### Task 8 — `auxiliaryLines` grouping lib + test

**Files:** `app/frontend/src/lib/auxiliaryLines.ts` (new), `app/frontend/src/lib/auxiliaryLines.test.ts` (new)

- [ ] **Step 1** Create `app/frontend/src/lib/auxiliaryLines.ts` (verbatim from optree):

```typescript
import type { AuxiliaryLineDto } from '../api';

/** A DXF layer is visible unless explicitly toggled off in the visibility map. */
export function isAuxLayerVisible(
  visibility: Record<string, Record<string, boolean>>,
  fileName: string,
  layer: string,
): boolean {
  return visibility[fileName]?.[layer] !== false;
}

/**
 * Group auxiliary lines by file, then by unique layer (first-seen color),
 * preserving encounter order for stable UI rendering.
 */
export function groupAuxLineLayers(
  lines: AuxiliaryLineDto[],
): { fileName: string; layers: { layer: string; color: string }[] }[] {
  const byFile = new Map<string, Map<string, string>>();
  for (const ln of lines) {
    if (!byFile.has(ln.file_name)) byFile.set(ln.file_name, new Map());
    const layers = byFile.get(ln.file_name)!;
    if (!layers.has(ln.layer)) layers.set(ln.layer, ln.color);
  }
  return [...byFile.entries()].map(([fileName, layers]) => ({
    fileName,
    layers: [...layers.entries()].map(([layer, color]) => ({ layer, color })),
  }));
}
```

- [ ] **Step 2** Create `app/frontend/src/lib/auxiliaryLines.test.ts` (verbatim from optree):

```typescript
import { describe, it, expect } from 'vitest';
import { isAuxLayerVisible, groupAuxLineLayers } from './auxiliaryLines';
import type { AuxiliaryLineDto } from '../api';

describe('isAuxLayerVisible', () => {
  it('defaults to visible when unset', () => {
    expect(isAuxLayerVisible({}, 'a.dxf', 'x')).toBe(true);
    expect(isAuxLayerVisible({ 'a.dxf': {} }, 'a.dxf', 'x')).toBe(true);
  });

  it('is hidden only when explicitly false', () => {
    expect(isAuxLayerVisible({ 'a.dxf': { x: false } }, 'a.dxf', 'x')).toBe(false);
    expect(isAuxLayerVisible({ 'a.dxf': { x: true } }, 'a.dxf', 'x')).toBe(true);
  });
});

describe('groupAuxLineLayers', () => {
  it('groups by file then unique layer with first-seen color, in order', () => {
    const lines: AuxiliaryLineDto[] = [
      { id: '1', file_name: 'a.dxf', layer: 'w', color: '#111', points: [] },
      { id: '2', file_name: 'a.dxf', layer: 'w', color: '#999', points: [] },
      { id: '3', file_name: 'a.dxf', layer: 'r', color: '#222', points: [] },
      { id: '4', file_name: 'b.dxf', layer: 'w', color: '#333', points: [] },
    ];
    expect(groupAuxLineLayers(lines)).toEqual([
      { fileName: 'a.dxf', layers: [{ layer: 'w', color: '#111' }, { layer: 'r', color: '#222' }] },
      { fileName: 'b.dxf', layers: [{ layer: 'w', color: '#333' }] },
    ]);
  });

  it('returns an empty array for no lines', () => {
    expect(groupAuxLineLayers([])).toEqual([]);
  });
});
```

- [ ] **Step 3** Run (from `app/frontend`):

```
npm test -- auxiliaryLines
```

Expected: 4 tests pass.

- [ ] **Step 4** Commit: `git add -A && git commit -m "feat(dxf): auxiliaryLines grouping/visibility lib + tests"`

---

### Task 9 — `DxfPlacementMap` component (simplified for VoxCity)

**Files:** `app/frontend/src/components/DxfPlacementMap.tsx` (new)

VoxCity has **no** `lib/placementBackdrop.ts` or `lib/placementPreview.ts` (optree-only). Port the core of optree's `DxfPlacementMap` but drop the backdrop pane, cross-mode OBJ reference, and `auxiliaryLineVisibility` props — mirroring VoxCity's existing `ObjPlacementMap.tsx` exactly, with per-layer colored dashed polylines added.

- [ ] **Step 1** Create `app/frontend/src/components/DxfPlacementMap.tsx`:

```tsx
/**
 * Leaflet map for DXF placement: basemap + per-layer polyline preview at the
 * current placement; clicking sets the anchor lon/lat. Mirrors ObjPlacementMap
 * but renders open polylines colored per DXF layer.
 */
import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import type { ModelGeoResult, DxfPreviewLayerDto } from '../api';
import { lonLatToUvM, sceneXYToLonLat, domainRotationDeg } from '../lib/grid';
import { transformModelPoint, type Placement } from '../lib/objPlacement';

interface Props {
  geo: ModelGeoResult;
  placement: Placement;
  layers: DxfPreviewLayerDto[];
  visibility: Record<string, boolean>;
  onAnchor: (lonLat: [number, number]) => void;
}

const DxfPlacementMap: React.FC<Props> = ({ geo, placement, layers, visibility, onAnchor }) => {
  const mapRef = useRef<L.Map | null>(null);
  const layerRef = useRef<L.LayerGroup | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const onAnchorRef = useRef(onAnchor);
  onAnchorRef.current = onAnchor;

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current).setView(geo.center as [number, number], 17);
    L.tileLayer(
      'https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}@2x.png',
      { attribution: '&copy; <a href="https://carto.com/">CARTO</a>', maxZoom: 20 },
    ).addTo(map);
    map.on('click', (e: L.LeafletMouseEvent) => onAnchorRef.current([e.latlng.lng, e.latlng.lat]));
    layerRef.current = L.layerGroup().addTo(map);
    mapRef.current = map;
    return () => { map.remove(); mapRef.current = null; layerRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return;
    layer.clearLayers();
    if (!placement.anchorLonLat) return;
    const fwd = lonLatToUvM({ grid_geom: geo.grid_geom });
    if (!fwd) return;
    const phiDeg = domainRotationDeg(geo.grid_geom);
    const [anchorEastM, anchorNorthM] = fwd(placement.anchorLonLat[0], placement.anchorLonLat[1]);
    for (const lyr of layers) {
      if (visibility[lyr.name] === false) continue;
      for (const ring of lyr.polylines) {
        const latlngs = ring.map(([mx, my]) => {
          const [eOff, nOff] = transformModelPoint([mx, my, 0], placement, phiDeg);
          const [lon, lat] = sceneXYToLonLat(geo.grid_geom, anchorEastM + eOff, anchorNorthM + nOff);
          return L.latLng(lat, lon);
        });
        L.polyline(latlngs, { color: lyr.color, weight: 2, dashArray: '4 3' }).addTo(layer);
      }
    }
  }, [geo, placement, layers, visibility]);

  return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />;
};

export default DxfPlacementMap;
```

Note: verify the named imports `lonLatToUvM`, `sceneXYToLonLat`, `domainRotationDeg` from `../lib/grid` and `transformModelPoint`, `Placement` from `../lib/objPlacement` exist in VoxCity with these exact names (they are used by VoxCity's `ObjPlacementMap.tsx`). If a helper differs, mirror whatever `ObjPlacementMap.tsx` imports.

- [ ] **Step 2** Typecheck (`npx tsc -b --noEmit` from `app/frontend`). Expected: no errors. Commit: `git add -A && git commit -m "feat(dxf): DxfPlacementMap 2D placement preview"`

---

### Task 10 — `AuxiliaryLinesControl` component

**Files:** `app/frontend/src/components/AuxiliaryLinesControl.tsx` (new)

- [ ] **Step 1** Create `app/frontend/src/components/AuxiliaryLinesControl.tsx` (verbatim from optree — it already uses theme `btn-primary`/`btn-ghost`, which is VoxCity indigo; no teal to re-skin):

```tsx
/**
 * Compact per-layer visibility toggles for imported DXF auxiliary lines.
 * Reads the (backend-authoritative) layers from `geo.auxiliary_lines` and
 * toggles the shared visibility map. Renders nothing when there are no lines.
 */
import React from 'react';
import type { ModelGeoResult } from '../api';
import { groupAuxLineLayers, isAuxLayerVisible } from '../lib/auxiliaryLines';

interface Props {
  geo: ModelGeoResult | null;
  visibility: Record<string, Record<string, boolean>>;
  onToggle: (fileName: string, layer: string, visible: boolean) => void;
  /** Remove all auxiliary lines from a file (backend delete + geo refresh). */
  onRemoveFile?: (fileName: string) => void;
  /** Extra styles merged into the row container (e.g. marginLeft: 'auto'). */
  style?: React.CSSProperties;
}

const AuxiliaryLinesControl: React.FC<Props> = ({ geo, visibility, onToggle, onRemoveFile, style }) => {
  const lines = geo?.auxiliary_lines ?? [];
  if (lines.length === 0) return null;

  const grouped = groupAuxLineLayers(lines);

  return (
    <div
      className="aux-lines-control"
      style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 4, fontSize: '0.72rem', ...style }}
    >
      <span style={{ opacity: 0.6, marginRight: 2 }} title="Imported DXF auxiliary lines">DXF</span>
      {grouped.map(({ fileName, layers }) => (
        <React.Fragment key={fileName}>
          {layers.map(({ layer, color }) => {
            const visible = isAuxLayerVisible(visibility, fileName, layer);
            return (
              <button
                key={layer}
                type="button"
                title={`${fileName} · ${layer}`}
                className={`btn btn-xs${visible ? ' btn-primary' : ' btn-ghost'}`}
                onClick={() => onToggle(fileName, layer, !visible)}
              >
                <span
                  style={{
                    width: 8, height: 8, background: color, borderRadius: 2,
                    display: 'inline-block', border: '1px solid #0003', marginRight: 4,
                  }}
                />
                {layer}
              </button>
            );
          })}
          {onRemoveFile && (
            <button
              type="button"
              className="btn btn-xs btn-ghost"
              title={`Remove ${fileName}`}
              aria-label={`Remove ${fileName}`}
              onClick={() => onRemoveFile(fileName)}
            >
              ×
            </button>
          )}
        </React.Fragment>
      ))}
    </div>
  );
};

export default AuxiliaryLinesControl;
```

Note: VoxCity's stylesheet may not define a `.btn-xs` / `.btn-ghost` utility. If `npx tsc` passes but the buttons look unstyled in the smoke test, add minimal `.btn-xs`/`.btn-ghost` rules to `index.css` (small padding, transparent ghost background) — this is a CSS-only follow-up, not a blocker.

- [ ] **Step 2** Typecheck. Commit: `git add -A && git commit -m "feat(dxf): AuxiliaryLinesControl per-layer visibility toggles"`

---

### Task 11 — 3D aux-line rendering in the scene

**Files:** `app/frontend/src/three/AuxiliaryLineLayer.tsx` (new), `app/frontend/src/three/SceneViewer.tsx`

optree renders aux lines only on its 2D plan maps; VoxCity's SceneViewer has none. Add a new R3F layer modeled on `ZoneOutlines`' flat-line branch (drei `<Line>` at ground height, projected lon/lat->scene via the same `lonLatToXY` the Zoning overlays use). Default accent is VoxCity indigo `#6666FF`; per-line color from the DXF layer wins.

- [ ] **Step 1** Create `app/frontend/src/three/AuxiliaryLineLayer.tsx`:

```tsx
/**
 * Renders imported DXF auxiliary lines as flat 3D reference polylines draped at
 * ground height. Non-voxelized overlay; lon/lat is projected to scene metres via
 * `lonLatToXY` (typically lib/grid.ts's lonLatToUvM), matching ZoneOutlines.
 */
import { useMemo } from 'react';
import { Line } from '@react-three/drei';
import * as THREE from 'three';

import type { AuxiliaryLineDto } from '../api';
import { isAuxLayerVisible } from '../lib/auxiliaryLines';

export interface AuxiliaryLineLayerProps {
  lines: AuxiliaryLineDto[];
  /** Projection lon/lat -> world-XY metres (lib/grid.ts lonLatToUvM result). */
  lonLatToXY?: (lon: number, lat: number) => [number, number];
  /** Per-file/per-layer visibility; a layer shows unless explicitly false. */
  visibility?: Record<string, Record<string, boolean>>;
  /** Ground height (metres) to drape the lines at. */
  zHeight?: number;
  lineWidth?: number;
}

/** VoxCity indigo fallback when a DXF entity carries no usable color. */
const DEFAULT_COLOR = '#6666FF';

export function AuxiliaryLineLayer({
  lines,
  lonLatToXY,
  visibility = {},
  zHeight = 0.5,
  lineWidth = 2,
}: AuxiliaryLineLayerProps) {
  const entries = useMemo(() => {
    const out: { id: string; color: string; points: [number, number, number][] }[] = [];
    for (const ln of lines) {
      if (!isAuxLayerVisible(visibility, ln.file_name, ln.layer)) continue;
      if (!ln.points || ln.points.length < 2) continue;
      const pts: [number, number, number][] = ln.points.map(([lon, lat]) => {
        const [x, y] = lonLatToXY ? lonLatToXY(lon, lat) : [lon, lat];
        return [x, y, zHeight];
      });
      out.push({ id: ln.id, color: ln.color || DEFAULT_COLOR, points: pts });
    }
    return out;
  }, [lines, lonLatToXY, visibility, zHeight]);

  if (entries.length === 0) return null;

  return (
    <group renderOrder={998}>
      {entries.map((e) => (
        <Line
          key={e.id}
          points={e.points}
          color={new THREE.Color(e.color)}
          lineWidth={lineWidth}
          depthTest={false}
          depthWrite={false}
          renderOrder={998}
          transparent
        />
      ))}
    </group>
  );
}
```

- [ ] **Step 2** Wire it into `SceneViewer.tsx`. Add the import next to the other three-layer imports (SceneViewer.tsx:24-36):

```tsx
import { AuxiliaryLineLayer } from './AuxiliaryLineLayer';
```

Add two props to `SceneViewerProps` (after the `placementPreview` block, before the closing `}` at SceneViewer.tsx:114):

```tsx
  /** Baked DXF auxiliary lines (lon/lat) draped at ground height. */
  auxiliaryLines?: import('../api').AuxiliaryLineDto[] | null;
  /** Per-file/per-layer visibility for auxiliary lines. */
  auxiliaryLineVisibility?: Record<string, Record<string, boolean>>;
```

Destructure them in the function signature (add to the params list around SceneViewer.tsx:139):

```tsx
  placementPreview = null,
  auxiliaryLines = null,
  auxiliaryLineVisibility,
}: SceneViewerProps) {
```

Render the layer immediately after the `placementPreview` `<PlacementGizmo>` block (after SceneViewer.tsx:328, before `<CameraControls .../>`):

```tsx
          {auxiliaryLines && auxiliaryLines.length > 0 && (
            <AuxiliaryLineLayer
              lines={auxiliaryLines}
              lonLatToXY={lonLatToXY}
              visibility={auxiliaryLineVisibility}
              zHeight={scene ? (scene.ground_top_m ?? 0) + scene.meshsize_m : 0.5}
            />
          )}
```

(`lonLatToXY` and `scene` are already in scope — `lonLatToXY` is the existing prop used by `ZoneOutlines`; `scene.ground_top_m`/`scene.meshsize_m` are the same fields `ZoneOutlines` uses for `zHeight`. Verify these exact names against the current `SceneViewer.tsx`/`ZoneOutlines.tsx` before wiring; adjust to whatever ZoneOutlines uses if they differ.)

- [ ] **Step 3** Typecheck from `app/frontend`:

```
npx tsc -b --noEmit
```

Expected: no errors.

- [ ] **Step 4** Commit: `git add -A && git commit -m "feat(dxf): AuxiliaryLineLayer 3D rendering wired into SceneViewer"`

---

### Task 12 — Rework the Import tab: OBJ/DXF mode toggle + DXF flow

**Files:** `app/frontend/src/tabs/ImportTab.tsx`

Rework VoxCity's OBJ-only Import tab into optree's OBJ/DXF mode-toggle 3-column layout, keeping VoxCity's own specifics: the `previewDisabled`/`previewGridShape`/`PreviewDisabledNotice` path, the `gizmoMode` translate/rotate toggle, `plan-panel-header` headers, `btn-secondary` on the upload button, and plain-English strings (VoxCity's ImportTab does **not** use `useT`/i18n — do not introduce it; the i18n plan wraps it later). Mode-toggle active state uses `btn-primary` (VoxCity indigo).

> This is the most invasive frontend edit. The full worked diff (imports, DXF state, handlers `handleDxfFile`/`handleDxfImport`/`handleRemoveAuxFile`, the mode toggle, the DXF `GuidedSection`s for UPLOAD/LAYERS/PLACEMENT, the mode-aware footer button, and the mode-aware 2D map + 3D result panels) is reproduced verbatim in the standalone DXF draft's Task 12. Apply it step-by-step; every code block there is complete. Key structural points:

- [ ] **Step 1** Extend imports: add `uploadImportDxf`, `commitImportDxf`, `clearAuxiliaryLines`, `getModelGeo`, `ImportDxfUploadResult`, `ModelGeoResult` from `../api`; add `DxfPlacementMap` and `AuxiliaryLinesControl` components.
- [ ] **Step 2** Add state after `fileInputRef`: `importMode: 'obj' | 'dxf'`, `dxfUpload`, `dxfPlacement`, `dxfVisibility`, `auxVisibility`, `dxfFileInputRef`, and a `refreshGeo` callback (`getModelGeo().then(setGeo)`).
- [ ] **Step 3** Add a DXF anchor-default effect mirroring the OBJ one (defaults `dxfPlacement.anchorLonLat` to `[geo.center[1], geo.center[0]]`).
- [ ] **Step 4** Add `handleDxfFile`, `handleDxfImport` (calls `commitImportDxf`, then `refreshGeo()` + `onModelEdited?.()`), and `handleRemoveAuxFile` (calls `clearAuxiliaryLines({ fileName })` then `refreshGeo()`).
- [ ] **Step 5** Replace the fixed `<h2>Import OBJ</h2>` with `<h2>Import</h2>` + a two-button mode toggle (`OBJ buildings` / `DXF reference lines`, active = `btn-primary`), wrap the existing OBJ `GuidedSection`s in `{importMode === 'obj' && (<>...</>)}`, and add the DXF `{importMode === 'dxf' && (<>...</>)}` branch with UPLOAD DXF / LAYERS (per-layer visibility checkboxes) / PLACEMENT (anchor lat/lon, rotation, move E/N, units) sections plus `<AuxiliaryLinesControl .../>`.
- [ ] **Step 6** Make the footer button mode-aware (`Import building(s)` vs `Add reference lines`, calling `handleImport` vs `handleDxfImport`).
- [ ] **Step 7** Make the 2D + 3D panels mode-aware: OBJ mode keeps `ObjPlacementMap`; DXF mode uses `DxfPlacementMap`; the 3D `SceneViewer` in OBJ mode also receives `auxiliaryLines={geo?.auxiliary_lines}` + `auxiliaryLineVisibility={auxVisibility}` + `lonLatToXY` so committed DXF lines appear in 3D; DXF mode shows a reference-only info note.
- [ ] **Step 8** Typecheck from `app/frontend`: `npx tsc -b --noEmit`. Expected: no errors. Ensure `getModelGeo` remains used (it is, via `refreshGeo` and the initial-load effect).
- [ ] **Step 9** Commit: `git add -A && git commit -m "feat(dxf): Import tab OBJ/DXF mode toggle + DXF placement flow"`

---

### Task 13 — Frontend aux-visibility session persistence (note)

**Files:** `app/frontend/src/lib/sessionRestore.ts` (if present), Import-tab/App state

The **geometry** of aux lines round-trips through the backend session zip (Task 5) and reappears via `geo.auxiliary_lines` on reload — this is the load-bearing part and is already covered. The per-layer *visibility* map (`auxVisibility`) is a UI preference; if VoxCity persists a `frontend_state` blob (mirroring optree's `auxiliaryLineVisibility`), extend it here.

- [ ] **Step 1** Check whether VoxCity threads a `frontend_state` object through save/restore:

```
grep -n "frontend_state\|auxiliaryLineVisibility\|RestoredFrontendState" app/frontend/src/lib/sessionRestore.ts app/frontend/src/App.tsx
```

- [ ] **Step 2** If (and only if) a `RestoredFrontendState`/frontend-state persistence path exists, add an `auxiliaryLineVisibility?: Record<string, Record<string, boolean>>` field to that shape and include `auxVisibility` when serializing the Import tab's state. If VoxCity has no such path yet, skip: the backend geometry round-trip is sufficient and aux lines still render on reload (all layers default to visible via `isAuxLayerVisible`).

- [ ] **Step 3** Commit only if changes were made: `git add -A && git commit -m "feat(dxf): persist aux-line layer visibility in frontend session state"`

---

### Task 14 — Full verification + manual smoke

**Files:** none (verification only)

- [ ] **Step 1** Full backend DXF suite from repo root:

```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_dxf_import.py app/backend/test_dxf_endpoints.py app/backend/test_auxiliary_lines_state.py app/backend/test_session_io.py -v
```

Expected: all pass.

- [ ] **Step 2** Regression check that OBJ import + geo still pass (shared `/model/geo` was edited):

```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_import_obj.py -v
```

Expected: all pass; the added `auxiliary_lines` key in `/model/geo` does not break existing assertions.

- [ ] **Step 3** Frontend unit + typecheck from `app/frontend`:

```
npm test -- auxiliaryLines
npx tsc -b --noEmit
```

Expected: auxiliaryLines tests pass; tsc clean.

- [ ] **Step 4** Manual smoke (dev server): generate a model → Import tab → toggle **DXF reference lines** → upload a small `.dxf` with a couple of LINE/LWPOLYLINE entities on named layers → confirm per-layer color swatches + segment counts appear, the 2D map draws dashed colored polylines at the clicked anchor, and rotation/move/units update the preview live → **Add reference lines** → confirm the `AuxiliaryLinesControl` chips appear, the lines show in the OBJ-mode 3D scene at ground level, and the voxel model is unchanged → Save session, reload, Load session → confirm the lines reappear (via `geo.auxiliary_lines`) → use the `×` remove control → confirm they clear.

- [ ] **Step 5** Final commit if any smoke fixes were needed: `git add -A && git commit -m "test(dxf): verification pass for DXF auxiliary-line import"`

---

### Self-Review Notes (mapping to Feature 1 — DXF auxiliary-line import)

- **DXF parsing (LINE/LWPOLYLINE/POLYLINE, per-layer, colors, $INSUNITS, INSERT-count warning)** → Task 1 (`dxf_import.py` + `test_dxf_import.py`), verbatim from optree.
- **Bake model-XY → absolute lon/lat via placement transform** → Task 1 `bake_polylines_to_lonlat` + Task 4 commit endpoint (`build_placement_transform` + `compute_grid_geometry`), with the far-from-origin survey-grid test.
- **API contract (upload/commit/delete DTOs + AuxiliaryLine)** → Task 2 (backend pydantic) + Task 7 (frontend TS), unchanged shapes from optree.
- **Non-voxelizing overlay stored on state** → Task 3 (`auxiliary_lines` field, cleared on regeneration) + Task 4 `test_commit_does_not_mutate_voxels`.
- **Adapted from `Depends(get_session)` to VoxCity global `app_state`** → Task 4 (every endpoint), tests use the `test_import_obj.py` autouse-`app_state` fixture pattern (no `sessions.py`).
- **Exposed to the client + rendered** → Task 4 (`/model/geo` adds `auxiliary_lines`) + Task 8 (grouping/visibility lib) + Task 9 (2D `DxfPlacementMap`) + Task 10 (`AuxiliaryLinesControl`) + Task 11 (3D `AuxiliaryLineLayer` in `SceneViewer`).
- **Placement UX parity + Import-tab layout polish** → Task 12 (OBJ/DXF mode toggle, DXF upload/layers/placement, mode-aware footer + 2D/3D panels), preserving VoxCity's `previewDisabled`/`gizmoMode`/`plan-panel-header`/plain-strings.
- **Persistence** → Task 5 (backend geometry round-trip + tests, the authoritative path) + Task 13 (optional frontend visibility persistence).
- **Dependency + re-skin** → Task 6 (`ezdxf` in requirements.txt + Poetry `pyproject.toml`); teal `#009999` → indigo: `AuxiliaryLineLayer` `DEFAULT_COLOR = '#6666FF'` and DXF UI accents via VoxCity `btn-primary`.
- **Deviations from optree (intentional, VoxCity-scoped):** `DxfPlacementMap` is simplified (no `placementBackdrop.ts`/`placementPreview.ts` — those don't exist in VoxCity), dropping the backdrop pane and cross-mode OBJ reference; 3D rendering is *new* (optree renders aux lines in 2D plan maps only) and is modeled on VoxCity's `ZoneOutlines` flat-line branch; no i18n (`useT`) since VoxCity's ImportTab uses plain strings.

---

## Plan 3 of 4 — CSS/visual polish delta + bundled Inter font

**Scope note (read first).** A diff of the two stylesheets (`git diff --no-index VoxCity/app/frontend/src/index.css optree_voxcity/app/frontend/src/index.css`) reports ~2241 insertions / 170 deletions, but that is misleading: optree grew from ~1150 → ~3221 lines almost entirely by adding CSS for tabs/components VoxCity does not have (progress-ring, `.objective-pill*`, `.settings-layout`, tree-arrangement `.field-label`/`.input-sm`/`.btn-xs`, `.baseline-tab__legend`, `.plan-map-draggable-tree`, `.zone-objectives-subsection`, `.objectives-editor*`). Grepping VoxCity's `app/frontend/src` confirms **none** of those classes are referenced by any VoxCity component. They are excluded per the scoping rules.

Also important: **every teal-literal block in the diff is already indigo in VoxCity.** `.btn-primary`, `.alert-info`/`.alert-success`, `.choice-check-row:hover`, `.choice-btn:hover`, `.mode-btn.active`, `.selection-toolbar-btn.active`, and `.selection-box-overlay` show as diffs only because optree = teal and VoxCity = indigo; VoxCity already carries the correct `#6666FF` / `rgba(102,102,255,…)` values from the 2026-06-01 port. **No teal→indigo rewrite is required in this delta** — there are no color literals in any block that actually needs porting.

The genuine, applicable delta is therefore small and reduces to two coherent themes: (a) optree flattened the rounded/bordered "embedded panel" chrome (three.js viewer, plan-map, visual-frame) so map/canvas surfaces sit flush inside their cards, plus a faint header-shadow softening and an empty-state centering rule; (b) minor Zone-tab list/table robustness. Deliberately **not** ported (removals/changes judged optree-specific or i18n-owned, noted in Self-Review): the `.app-container > .tab-bar { flex:0 0 auto }` removal, the `.choice-group-checks-1` removal, and the Japanese CJK `font-family` fallback (belongs to the i18n plan).

> **NOTE (implementer):** the line numbers below are from the draft's read of `index.css`. Re-confirm each anchor with a quick grep before editing (VoxCity's CSS may have shifted); match on the quoted CSS text, not the line number.

---

### Task 1 — Bundle Inter via `@fontsource/inter` (dependency + import)

VoxCity's `index.css` already declares `font-family: 'Inter', …` but nothing ships the font, so it silently falls back to system fonts. Add the same bundled-font dependency optree uses (`^5.2.8`) and import the three weights the CSS relies on (400/500/600).

**Files:**
- `app/frontend/package.json`
- `app/frontend/src/main.tsx`

- [ ] **Step 1** — Add the dependency to `package.json`. Edit the `dependencies` block so the first entry is `@fontsource/inter`:

  Replace:
  ```json
    "dependencies": {
      "@react-three/drei": "^9.122.0",
  ```
  with:
  ```json
    "dependencies": {
      "@fontsource/inter": "^5.2.8",
      "@react-three/drei": "^9.122.0",
  ```

- [ ] **Step 2** — Install it (from the frontend dir) to update `package-lock.json` and `node_modules`:
  ```bash
  cd "C:/Users/kunih/OneDrive/00_Codes/python/VoxCity/app/frontend" && npm install
  ```
  Expected output ends with something like:
  ```
  added 1 package, and audited N packages in Xs
  found 0 vulnerabilities
  ```
  Confirm the package resolved:
  ```bash
  cd "C:/Users/kunih/OneDrive/00_Codes/python/VoxCity/app/frontend" && node -e "console.log(require('@fontsource/inter/package.json').version)"
  ```
  Expected output: a `5.2.x` version string, e.g. `5.2.8`.

- [ ] **Step 3** — Import the font weights in `main.tsx`. The current file is:
  ```tsx
  import React from 'react'
  import ReactDOM from 'react-dom/client'
  import App from './App'
  import './index.css'
  ```
  Add the three weight imports **before** `./index.css` (order matters — font faces must be defined before the stylesheet that uses them). Replace:
  ```tsx
  import App from './App'
  import './index.css'
  ```
  with:
  ```tsx
  import App from './App'
  import '@fontsource/inter/400.css'
  import '@fontsource/inter/500.css'
  import '@fontsource/inter/600.css'
  import './index.css'
  ```
  (Note: do **not** copy optree's `import { LanguageProvider }` / `import 'leaflet/dist/leaflet.css'` lines — the provider belongs to the separate i18n plan, which edits `main.tsx` again; VoxCity imports Leaflet CSS elsewhere.)

- [ ] **Step 4** — Verify types still compile:
  ```bash
  cd "C:/Users/kunih/OneDrive/00_Codes/python/VoxCity/app/frontend" && npx tsc -b --noEmit
  ```
  Expected output: no output (exit code 0).

- [ ] **Step 5** — Commit:
  ```bash
  cd "C:/Users/kunih/OneDrive/00_Codes/python/VoxCity" && git add app/frontend/package.json app/frontend/package-lock.json app/frontend/src/main.tsx && git commit -m "chore(ui): add @fontsource/inter and import 400/500/600 weights"
  ```

---

### Task 2 — Flush embedded-panel chrome + header shadow + empty-state centering

Port optree's post-port refinement that removes rounded corners / borders from embedded map & three.js surfaces so they sit flush inside their cards, softens the header shadow one notch, and centers empty/loading messages inside a `visual-frame`. No color literals are involved.

**Files:** `app/frontend/src/index.css`

- [ ] **Step 1** — Soften the app-header shadow. Current (`.app-header` block, ~line 43):
  ```css
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
  ```
  Replace with:
  ```css
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
  ```

- [ ] **Step 2** — Remove the rounded corner from the three.js viewer container so the canvas fills flush. Current (`.three-container`, ~line 262):
  ```css
  .three-container {
    width: 100%;
    min-height: 500px;
    height: 100%;
    position: relative;
    border-radius: 8px;
    overflow: hidden;
  }
  ```
  Replace with:
  ```css
  .three-container {
    width: 100%;
    min-height: 500px;
    height: 100%;
    position: relative;
    overflow: hidden;
  }
  ```
  Leave the following `.three-container canvas { border-radius: inherit; … }` rule untouched — with no radius on the parent it now inherits `0`, which is the intended flush result.

- [ ] **Step 3** — Remove the border + radius from `.plan-map-wrap`. Current (~line 620):
  ```css
  .plan-map-wrap {
    position: relative;
    flex: 1 1 auto;
    width: 100%;
    min-height: 0;
    border: 1px solid var(--vc-ring);
    border-radius: 8px;
    overflow: hidden;
  }
  ```
  Replace with:
  ```css
  .plan-map-wrap {
    position: relative;
    flex: 1 1 auto;
    width: 100%;
    min-height: 0;
    overflow: hidden;
  }
  ```

- [ ] **Step 4** — Refine `.visual-panel`, drop the `> h2` min-height, flatten `.visual-frame`, and add the empty-state centering rule. Current (~line 630–654):
  ```css
  .visual-panel {
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .visual-panel > h2 {
    min-height: 2rem;
  }

  .visual-frame {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    border-radius: 8px;
    overflow: hidden;
  }

  .visual-frame > * {
    flex: 1 1 auto;
    min-height: 0;
    border-radius: inherit;
    overflow: hidden;
  }
  ```
  Replace that entire span with:
  ```css
  .visual-panel {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
    padding: 0;
  }

  .visual-frame {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .visual-frame > * {
    flex: 1 1 auto;
    min-height: 0;
    overflow: hidden;
  }

  /* Empty/loading messages centre themselves on the bare card surface. */
  .visual-frame > .alert,
  .visual-frame > .guided-status {
    flex: 0 0 auto;
    margin: auto;
    max-width: 80%;
  }
  ```
  (VoxCity defines `.guided-status` at ~line 1104 and `.alert` earlier, so both selectors resolve to real classes used in these panels — the centering rule is live, not dead CSS.)

- [ ] **Step 5** — Compile check:
  ```bash
  cd "C:/Users/kunih/OneDrive/00_Codes/python/VoxCity/app/frontend" && npx tsc -b --noEmit
  ```
  Expected output: no output, exit 0. (CSS isn't type-checked, but this confirms nothing else broke.)

- [ ] **Step 6** — Commit:
  ```bash
  cd "C:/Users/kunih/OneDrive/00_Codes/python/VoxCity" && git add app/frontend/src/index.css && git commit -m "feat(ui): flush embedded map/three panels, soften header shadow"
  ```

---

### Task 3 — Zone-tab list/table robustness polish

Port optree's small Zone-list refinements: slightly larger row gap, allow the row to wrap and its name to shrink/truncate instead of overflowing, and stop stat-table cells from wrapping. These carry no color literals and no optree-only markup (the `.zone-objectives-subsection` / `.objectives-editor*` blocks that motivated optree's `flex-wrap` are **excluded** — VoxCity has no such subsection, so `flex-wrap` here is simply defensive against badge overflow on narrow panels).

**Files:** `app/frontend/src/index.css`

- [ ] **Step 1** — Update `.zone-list` gap. Current (~line 946):
  ```css
  .zone-list { display: flex; flex-direction: column; gap: 4px; margin-top: 12px; }
  ```
  Replace with:
  ```css
  .zone-list { display: flex; flex-direction: column; gap: 6px; margin-top: 12px; }
  ```

- [ ] **Step 2** — Update `.zone-row` to wrap and align to top. Current (~line 947):
  ```css
  .zone-row  { display: flex; align-items: center; gap: 8px; padding: 4px 6px; border-radius: 4px; cursor: pointer; }
  ```
  Replace with:
  ```css
  .zone-row  { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 6px 8px; padding: 6px; border-radius: 4px; cursor: pointer; }
  ```

- [ ] **Step 3** — Let the zone name shrink/truncate. Current (~line 950):
  ```css
  .zone-row .name    { flex: 1; }
  ```
  Replace with:
  ```css
  .zone-row .name    { flex: 1 1 0; min-width: 0; }
  ```

- [ ] **Step 4** — Stop stat-table cells from wrapping. Current (~line 968):
  ```css
  .zone-stats-table th, .zone-stats-table td { padding: 4px 6px; text-align: right; border-bottom: 1px solid rgba(0,0,0,0.08); }
  ```
  Replace with:
  ```css
  .zone-stats-table th, .zone-stats-table td { padding: 4px 6px; text-align: right; border-bottom: 1px solid rgba(0,0,0,0.08); white-space: nowrap; }
  ```

- [ ] **Step 5** — Compile check:
  ```bash
  cd "C:/Users/kunih/OneDrive/00_Codes/python/VoxCity/app/frontend" && npx tsc -b --noEmit
  ```
  Expected output: no output, exit 0.

- [ ] **Step 6** — Commit:
  ```bash
  cd "C:/Users/kunih/OneDrive/00_Codes/python/VoxCity" && git add app/frontend/src/index.css && git commit -m "feat(ui): zone list/table robustness polish"
  ```

---

### Task 4 — Verification: production build + visual smoke checklist

CSS has no unit tests, so verification is a successful build plus a manual visual pass.

**Files:** none (verification only).

- [ ] **Step 1** — Full production build:
  ```bash
  cd "C:/Users/kunih/OneDrive/00_Codes/python/VoxCity/app/frontend" && npm run build
  ```
  Expected: `tsc -b` passes, then Vite prints `✓ built in …` with a `dist/` asset list and **no** errors. Confirm Inter actually shipped:
  ```bash
  cd "C:/Users/kunih/OneDrive/00_Codes/python/VoxCity/app/frontend" && ls dist/assets | grep -i inter | head
  ```
  Expected: one or more hashed font files, e.g. `inter-latin-400-normal-<hash>.woff2`. (If empty, the imports in Task 1 Step 3 didn't take — recheck.)

- [ ] **Step 2** — Start the dev server for the visual pass:
  ```bash
  cd "C:/Users/kunih/OneDrive/00_Codes/python/VoxCity/app/frontend" && npm run dev
  ```
  Open the printed `http://localhost:5173` URL.

- [ ] **Step 3** — Walk the visual smoke checklist. Confirm each and that **no teal (`#009999`) appears anywhere** — every accent must read indigo (`#6666FF`):
  - [ ] **Font** — body text renders in Inter, not the system fallback. DevTools → Computed → `font-family` on `body` resolves to Inter; Network tab shows `inter-*-.woff2` loaded.
  - [ ] **Header** — logo, subtly softer bottom shadow (0.03 alpha); nothing shifted.
  - [ ] **Tab bar** — all nine tabs (Target, Generate, Edit, Import, Zone, Solar, View, Landmark, File) render; active tab underline/fill is indigo.
  - [ ] **Buttons** — primary buttons indigo with indigo hover; secondary/danger unchanged.
  - [ ] **Guided panels** (Edit / Generate) — numbered sections, dashed dividers, danger-tone left bar all intact; ChoiceGroup buttons/checks hover indigo.
  - [ ] **View tab** — three.js canvas fills its card **flush** (no rounded corner / gap).
  - [ ] **Solar / View visual panels** — embedded map/frame sits flush; empty or loading state message is centered within the panel, not pinned top-left.
  - [ ] **Zone tab** — zone rows have comfortable spacing; long zone names truncate rather than overflow; type badges stay put; stats table cells don't wrap.
  - [ ] **Import tab** — layout regression check (no rules targeted it directly; note the DXF plan reworks it separately).
  - [ ] **File tab** — share-link input and actions render correctly (regression check).

- [ ] **Step 4** — Stop the dev server (Ctrl-C). If all boxes pass, the section is complete.

---

### Self-Review Notes (maps to spec Feature 3 — CSS + Inter font only; i18n is Plan 4)

- **Inter font bundling: DONE.** `@fontsource/inter@^5.2.8` added and weights 400/500/600 imported in `main.tsx` ahead of `index.css`, mirroring optree. Task 4 Step 1 verifies the font file ships in `dist/`.
- **CSS polish delta: DONE, and genuinely small.** The 2026-06-01 port already merged optree's header chrome, icon tab bar, and refined GuidedPanel/ChoiceGroup and did the teal→indigo re-skin. optree's later growth is optree-only components with **zero** VoxCity references (verified by grep). Correctly excluded per scoping rules.
- **No teal→indigo rewrite required.** Every teal-literal block in the diff is *already* indigo in VoxCity. The three ported tasks touch only border-radius/border/shadow-alpha/flex/white-space — no color literals. Task 4 Step 3 still includes an explicit "no `#009999` anywhere" guard.
- **Ported (the real delta):** header shadow `0.04→0.03`; flush embedded panels (`.three-container`, `.plan-map-wrap`, `.visual-frame`, `.visual-frame > *` lose radius/border; `.visual-panel` gains `position:relative; padding:0` and loses `> h2 { min-height }`); new `.visual-frame > .alert, .visual-frame > .guided-status` centering; Zone `.zone-list` gap, `.zone-row` wrap/align/padding, `.zone-row .name` shrink, `.zone-stats-table` cell `white-space:nowrap`.
- **Deliberately NOT ported (with rationale):** the `.app-container > .tab-bar { flex: 0 0 auto }` removal (a layout-constraint change, not polish); the `.choice-group-checks-1` removal (risks regressing the already-tuned ChoiceGroup); the Japanese CJK `font-family` fallback (belongs to Plan 4 i18n).
- **Verification** matches the "CSS has no unit tests" constraint: `npx tsc -b --noEmit` after each task, `npm run build` success + `dist/` font presence, and the manual smoke checklist.

---

## Plan 4 of 4 — Internationalization (English + Japanese)

VoxCity currently has **zero i18n** — every user-facing string is hardcoded English across `App.tsx`, the 9 tabs, `guidedTabState.ts`, and shared components. This plan ports optree's i18n machinery into `app/frontend/src/i18n/`, seeds a VoxCity-specific en/ja catalog, wires the `<LanguageProvider>` + a header language toggle, and then wraps strings tab-by-tab. It runs **last** so it also covers the English strings introduced by Plans 1–3 (share link, DXF import, and any new UI those added).

The machinery API is ported **verbatim** from optree (`LanguageProvider`, `useLanguage`, `useT`, `translate`, `catalogs`, `TranslationKey`, `Lang`); only the localStorage key (`optree.lang` → `voxcity.lang`), the catalog contents, and the provider test (adapted to VoxCity's DOM-free test convention) differ.

### Files overview

Machinery (new, ported from `optree_voxcity/app/frontend/src/i18n/`):
- `app/frontend/src/i18n/index.ts` — barrel
- `app/frontend/src/i18n/LanguageContext.tsx` — provider + `useLanguage`
- `app/frontend/src/i18n/useT.ts` — `useT()` hook
- `app/frontend/src/i18n/translate.ts` — lookup + `{var}` interpolation + `TranslationKey`
- `app/frontend/src/i18n/locales/catalogs.ts` — combines en/ja, exports `Lang`
- `app/frontend/src/i18n/locales/en.ts` — English catalog (source of truth) + `Messages` type
- `app/frontend/src/i18n/locales/ja.ts` — Japanese catalog, typed `: Messages`
- `app/frontend/src/i18n/translate.test.ts`, `catalogParity.test.ts`, `LanguageContext.test.tsx` — tests

Wired/edited (existing):
- `app/frontend/src/main.tsx` — wrap `<App/>` in `<LanguageProvider>`
- `app/frontend/src/App.tsx` — header toggle + `TABS` labels
- `app/frontend/src/tabs/guidedTabState.ts` (+ `guidedTabState.test.ts`) — thread `t`
- `app/frontend/src/components/StartSplash.tsx`, `PreviewDisabledNotice.tsx`, `VoxelClassVisibility.tsx`
- `app/frontend/src/tabs/*.tsx` (all 9), `app/frontend/src/index.css` (toggle styles)

Commands (run from `app/frontend`): `npm test -- <name>`, `npx tsc -b --noEmit`, `npm run build`.

---

### Task 1 — Port the i18n machinery + seed en/ja catalog + tests

**Files:** `app/frontend/src/i18n/index.ts`, `LanguageContext.tsx`, `useT.ts`, `translate.ts`, `locales/catalogs.ts`, `locales/en.ts`, `locales/ja.ts`, `translate.test.ts`, `catalogParity.test.ts`, `LanguageContext.test.tsx` (all new)

- [ ] **Step 1** — Create `app/frontend/src/i18n/translate.ts` **verbatim from optree** (unchanged):

```ts
import { catalogs, type Lang } from './locales/catalogs';
import type { Messages } from './locales/en';

// Dotted key paths derived from the catalog shape, e.g. 'nav.export'.
type DotPaths<T> = {
  [K in keyof T & string]: T[K] extends string ? K : `${K}.${DotPaths<T[K]>}`;
}[keyof T & string];

export type TranslationKey = DotPaths<Messages>;

function resolve(catalog: unknown, key: string): string | undefined {
  const value = key.split('.').reduce<unknown>((acc, part) => {
    if (acc && typeof acc === 'object' && part in (acc as Record<string, unknown>)) {
      return (acc as Record<string, unknown>)[part];
    }
    return undefined;
  }, catalog);
  return typeof value === 'string' ? value : undefined;
}

function interpolate(template: string, vars: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (match, name: string) =>
    name in vars ? String(vars[name]) : match,
  );
}

export function translate(
  lang: Lang,
  key: TranslationKey,
  vars?: Record<string, string | number>,
): string {
  const value = resolve(catalogs[lang], key) ?? resolve(catalogs.en, key);
  if (value === undefined) {
    if (import.meta.env.DEV) {
      // eslint-disable-next-line no-console
      console.warn(`[i18n] Missing translation key: ${key}`);
    }
    return key; // Unreachable given typed catalogs; last-resort only.
  }
  return vars ? interpolate(value, vars) : value;
}
```

- [ ] **Step 2** — Create `app/frontend/src/i18n/LanguageContext.tsx` **verbatim from optree except the storage key** (`optree.lang` → `voxcity.lang`):

```tsx
import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { Lang } from './locales/catalogs';

const STORAGE_KEY = 'voxcity.lang';

function readInitialLang(): Lang {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'en' || stored === 'ja') return stored;
  } catch {
    // Ignore storage failures; fall through to the default.
  }
  return 'en';
}

interface LanguageContextValue {
  lang: Lang;
  setLang: (lang: Lang) => void;
}

// Default value means components render in English WITHOUT a provider, so
// existing component tests need no wrapping.
const LanguageContext = createContext<LanguageContextValue>({
  lang: 'en',
  setLang: () => {
    if (import.meta.env.DEV) {
      // eslint-disable-next-line no-console
      console.warn('[i18n] setLang called outside a LanguageProvider');
    }
  },
});

export const LanguageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [lang, setLangState] = useState<Lang>(readInitialLang);

  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);

  const setLang = useCallback((next: Lang) => {
    setLangState(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Ignore storage failures; the in-memory choice still applies.
    }
  }, []);

  const value = useMemo(() => ({ lang, setLang }), [lang, setLang]);
  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
};

export function useLanguage(): LanguageContextValue {
  return useContext(LanguageContext);
}
```

- [ ] **Step 3** — Create `app/frontend/src/i18n/useT.ts` and `index.ts` **verbatim from optree**:

`useT.ts`:
```ts
import { useCallback } from 'react';
import { useLanguage } from './LanguageContext';
import { translate, type TranslationKey } from './translate';

export function useT() {
  const { lang } = useLanguage();
  return useCallback(
    (key: TranslationKey, vars?: Record<string, string | number>) => translate(lang, key, vars),
    [lang],
  );
}
```

`index.ts`:
```ts
export { LanguageProvider, useLanguage } from './LanguageContext';
export { useT } from './useT';
export { translate, type TranslationKey } from './translate';
export type { Lang } from './locales/catalogs';
```

- [ ] **Step 4** — Create `app/frontend/src/i18n/locales/catalogs.ts` **verbatim from optree**:

```ts
import { en } from './en';
import { ja } from './ja';

export const catalogs = { en, ja } as const;
export type Lang = keyof typeof catalogs; // 'en' | 'ja'
```

- [ ] **Step 5** — Create the **seed** `app/frontend/src/i18n/locales/en.ts`. This is a small VoxCity-specific catalog (NOT optree's — optree's keys are for its optimize/baseline/tour tabs which VoxCity lacks). It covers the header/nav, splash, the shared guided helpers, the preview notice, and voxel visibility so Tasks 2–3 can build on it. Per-tab sections are appended in Tasks 4–10.

```ts
// English catalog — the single source of truth. `ja.ts` is typed against this.
export const en = {
  common: {
    close: 'Close',
    newSession: 'New session',
    openSession: 'Open session...',
    getStarted: 'GET STARTED',
    dontShowAgain: "Don't show this again",
  },
  language: {
    label: 'Language',
    english: 'EN',
    japanese: '日本語',
  },
  nav: {
    area: 'Target',
    generation: 'Generate',
    edit: 'Edit',
    import: 'Import',
    zoning: 'Zone',
    solar: 'Solar',
    view: 'View',
    landmark: 'Landmark',
    export: 'File',
  },
  splash: {
    title: 'Welcome to VoxCity',
    subtitle: 'Start a new urban model, or open a saved session.',
    loadFailed: 'Failed to load session.',
  },
  guided: {
    setAreaTitle: 'Set a target area first',
    setAreaBody: 'Use the Target Area tab to choose the city area before generating a model.',
    modelRequiredTitle: 'Generate a model first',
    modelRequiredBody: 'Use the Generation tab to create a VoxCity model before using this workflow.',
    actionSetRectangle: 'Set Rectangle',
    actionLoadingMap: 'Loading map...',
    actionLoadMap: 'Load Map',
    actionGenerating: 'Generating...',
    actionGenerate: 'Generate VoxCity Model',
    actionRunning: 'Running...',
    actionRunSimulation: 'Run Simulation',
    actionExporting: 'Exporting...',
    actionExportCityles: 'Export CityLES',
    actionExportGeotiff: 'Export GeoTIFF',
    actionExportObj: 'Export OBJ',
  },
  previewNotice: {
    heading: '3D preview disabled',
    bodyWithDims: 'The grid ({dims}) exceeds the preview limit of {cells} cells. Generation, editing, simulation results, and export still work.',
    bodyNoDims: 'This grid exceeds the preview limit of {cells} cells. Generation, editing, simulation results, and export still work.',
  },
  voxelVisibility: {
    heading: 'Visualization Settings',
    hideClasses: 'Hide element classes',
  },
};

export type Messages = typeof en;
```

- [ ] **Step 6** — Create the matching seed `app/frontend/src/i18n/locales/ja.ts`, typed `: Messages` (build fails if a key is missing/mistyped — this is the parity guard at compile time):

```ts
import type { Messages } from './en';

// Typed against `en` — the build fails if any key is missing or mistyped.
export const ja: Messages = {
  common: {
    close: '閉じる',
    newSession: '新規セッション',
    openSession: 'セッションを開く...',
    getStarted: 'はじめに',
    dontShowAgain: '次回から表示しない',
  },
  language: {
    label: '言語',
    english: 'EN',
    japanese: '日本語',
  },
  nav: {
    area: '対象エリア',
    generation: '生成',
    edit: '編集',
    import: 'インポート',
    zoning: 'ゾーン',
    solar: '日射',
    view: '可視性',
    landmark: 'ランドマーク',
    export: 'ファイル',
  },
  splash: {
    title: 'VoxCity へようこそ',
    subtitle: '新しい都市モデルを作成するか、保存済みのセッションを開きます。',
    loadFailed: 'セッションの読み込みに失敗しました。',
  },
  guided: {
    setAreaTitle: 'まず対象エリアを設定してください',
    setAreaBody: '「対象エリア」タブで都市エリアを選択してからモデルを生成してください。',
    modelRequiredTitle: 'まずモデルを生成してください',
    modelRequiredBody: 'このワークフローを使う前に「生成」タブで VoxCity モデルを作成してください。',
    actionSetRectangle: '矩形を設定',
    actionLoadingMap: '地図を読み込み中...',
    actionLoadMap: '地図を読み込む',
    actionGenerating: '生成中...',
    actionGenerate: 'VoxCity モデルを生成',
    actionRunning: '実行中...',
    actionRunSimulation: 'シミュレーションを実行',
    actionExporting: 'エクスポート中...',
    actionExportCityles: 'CityLES をエクスポート',
    actionExportGeotiff: 'GeoTIFF をエクスポート',
    actionExportObj: 'OBJ をエクスポート',
  },
  previewNotice: {
    heading: '3D プレビューは無効です',
    bodyWithDims: 'グリッド（{dims}）がプレビュー上限の {cells} セルを超えています。生成・編集・シミュレーション結果・エクスポートは引き続き利用できます。',
    bodyNoDims: 'このグリッドはプレビュー上限の {cells} セルを超えています。生成・編集・シミュレーション結果・エクスポートは引き続き利用できます。',
  },
  voxelVisibility: {
    heading: '表示設定',
    hideClasses: '要素クラスを非表示',
  },
};
```

- [ ] **Step 7** — Create `app/frontend/src/i18n/catalogParity.test.ts` **verbatim from optree** (the en/ja key-parity + no-empty-values guard):

```ts
import { describe, expect, it } from 'vitest';
import { en } from './locales/en';
import { ja } from './locales/ja';

function flatKeys(obj: Record<string, unknown>, prefix = ''): string[] {
  return Object.entries(obj).flatMap(([k, v]) =>
    v && typeof v === 'object'
      ? flatKeys(v as Record<string, unknown>, `${prefix}${k}.`)
      : [`${prefix}${k}`],
  );
}

function values(obj: Record<string, unknown>): string[] {
  return Object.values(obj).flatMap((v) =>
    v && typeof v === 'object' ? values(v as Record<string, unknown>) : [String(v)],
  );
}

describe('catalog parity', () => {
  it('ja has exactly the same keys as en', () => {
    expect(flatKeys(ja).sort()).toEqual(flatKeys(en).sort());
  });

  it('has no empty values in either catalog', () => {
    for (const v of [...values(en), ...values(ja)]) {
      expect(v.trim().length).toBeGreaterThan(0);
    }
  });
});
```

- [ ] **Step 8** — Create `app/frontend/src/i18n/translate.test.ts` (adapted from optree to VoxCity seed keys):

```ts
import { describe, expect, it } from 'vitest';
import { translate } from './translate';

describe('translate', () => {
  it('resolves a nested key in the active language', () => {
    expect(translate('en', 'nav.export')).toBe('File');
    expect(translate('ja', 'nav.export')).toBe('ファイル');
  });

  it('interpolates named placeholders', () => {
    expect(
      translate('en', 'previewNotice.bodyWithDims', { dims: '1500×900', cells: '1,000,000' }),
    ).toBe(
      'The grid (1500×900) exceeds the preview limit of 1,000,000 cells. Generation, editing, simulation results, and export still work.',
    );
  });

  it('returns the Japanese and English values for a shared key', () => {
    expect(translate('en', 'common.newSession')).toBe('New session');
    expect(translate('ja', 'common.newSession')).toBe('新規セッション');
  });
});
```

- [ ] **Step 9** — Create `app/frontend/src/i18n/LanguageContext.test.tsx`. **Do NOT port optree's version verbatim** — it uses `@testing-library/react`/`renderHook`, which VoxCity does not depend on (no jsdom, no testing-library; existing component tests use `renderToStaticMarkup` from `react-dom/server`). This adapted version follows VoxCity's DOM-free convention and covers the default-English and localStorage-init paths (effects don't run under SSR, so `document`/`localStorage` being absent is safe). The interactive toggle+persistence path is verified in the Task 11 manual smoke:

```tsx
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, describe, expect, it } from 'vitest';
import { LanguageProvider } from './LanguageContext';
import { useT } from './useT';

function Probe() {
  const t = useT();
  return <span>{t('nav.area')}</span>;
}

afterEach(() => {
  delete (globalThis as unknown as { localStorage?: unknown }).localStorage;
});

describe('LanguageProvider + useT (SSR)', () => {
  it('renders English by default without a provider', () => {
    expect(renderToStaticMarkup(<Probe />)).toContain('Target');
  });

  it('renders English inside the provider by default', () => {
    expect(renderToStaticMarkup(<LanguageProvider><Probe /></LanguageProvider>)).toContain('Target');
  });

  it('initializes from localStorage when set to ja', () => {
    (globalThis as unknown as { localStorage: Storage }).localStorage = {
      getItem: (k: string) => (k === 'voxcity.lang' ? 'ja' : null),
      setItem: () => {},
    } as unknown as Storage;
    expect(renderToStaticMarkup(<LanguageProvider><Probe /></LanguageProvider>)).toContain('対象エリア');
  });
});
```

- [ ] **Step 10** — Run the machinery tests and typecheck:

```
npm test -- i18n
npx tsc -b --noEmit
```

Expected: `translate.test.ts`, `catalogParity.test.ts`, and `LanguageContext.test.tsx` all pass; `tsc` exits 0. Then commit: `feat(i18n): port language provider + translate machinery with seed en/ja catalog`.

---

### Task 2 — Wire the provider, header language toggle, and nav/tab labels

**Files:** `app/frontend/src/main.tsx`, `app/frontend/src/App.tsx`, `app/frontend/src/index.css`

- [ ] **Step 1** — Wrap `<App/>` in `<LanguageProvider>` in `main.tsx` (this edit stacks on top of the font imports Plan 3 added — keep those import lines):

```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { LanguageProvider } from './i18n'
import '@fontsource/inter/400.css'
import '@fontsource/inter/500.css'
import '@fontsource/inter/600.css'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <LanguageProvider>
      <App />
    </LanguageProvider>
  </React.StrictMode>,
)
```

- [ ] **Step 2** — In `App.tsx`, add the i18n imports near the top (after the existing `import type { Zone }` line):

```tsx
import { useT, useLanguage, type TranslationKey } from './i18n';
```

- [ ] **Step 3** — Inside the `App` component body, add the hooks right after `const [activeTab, setActiveTab] = useState<TabId>('area');`:

```tsx
  const t = useT();
  const { lang, setLang } = useLanguage();
```

- [ ] **Step 4** — Replace the tab label render. The `TABS` `id` values (`area`, `generation`, `edit`, `import`, `zoning`, `solar`, `view`, `landmark`, `export`) match the `nav.*` keys exactly, so change the `<span>{tab.label}</span>` inside the `TABS.map(...)` (currently line 192) to:

```tsx
                <span>{t(`nav.${tab.id}` as TranslationKey)}</span>
```

(Leave the `label:` field in the `TABS` array as-is — it is now only a developer-facing default and harmless.)

- [ ] **Step 5** — Add the language toggle to the header. Immediately after the closing `</nav>` (line 196) and before `</header>`, insert the toggle markup:

```tsx
        <div className="lang-toggle" role="group" aria-label={t('language.label')}>
          <button
            type="button"
            className={`lang-btn ${lang === 'en' ? 'active' : ''}`}
            onClick={() => setLang('en')}
          >
            {t('language.english')}
          </button>
          <button
            type="button"
            className={`lang-btn ${lang === 'ja' ? 'active' : ''}`}
            onClick={() => setLang('ja')}
          >
            {t('language.japanese')}
          </button>
        </div>
```

- [ ] **Step 6** — Append toggle styles to the end of `app/frontend/src/index.css` (use VoxCity's real accent var `--vc-accent` with an indigo fallback, NOT a blue one):

```css
/* Language toggle (i18n) */
.lang-toggle {
  display: inline-flex;
  margin-left: auto;
  gap: 2px;
  border: 1px solid var(--vc-ring, #F0F0F0);
  border-radius: 6px;
  overflow: hidden;
}
.lang-btn {
  padding: 4px 10px;
  font-size: 0.78rem;
  background: transparent;
  border: none;
  cursor: pointer;
  color: var(--vc-muted, #9CA3AF);
}
.lang-btn.active {
  background: var(--vc-accent, #6666FF);
  color: #fff;
}
```

- [ ] **Step 7** — Typecheck and build:

```
npx tsc -b --noEmit
npm run build
```

Expected: both exit 0; the header shows an EN / 日本語 toggle and the 9 tab labels switch language when toggled. Commit: `feat(i18n): wrap app in LanguageProvider, add header language toggle and nav labels`.

---

### Task 3 — Convert shared helpers and shared components (`guidedTabState`, splash, preview notice, voxel visibility)

`guidedTabState.ts` holds pure label/message helpers consumed by ExportTab, GenerationTab, TargetAreaTab, and the three sim tabs. Because these are plain functions (not components), they cannot call `useT()` — thread a `t` function in as the first argument. Changing the signatures forces updating every call site in the same commit (that is intentional: it keeps `guidedTabState` atomic and the build green). The keys already exist in the seed catalog (`guided.*`).

**Files:** `app/frontend/src/tabs/guidedTabState.ts`, `app/frontend/src/tabs/guidedTabState.test.ts`, `app/frontend/src/components/StartSplash.tsx`, `app/frontend/src/components/PreviewDisabledNotice.tsx`, `app/frontend/src/components/VoxelClassVisibility.tsx`, and the call sites in `tabs/ExportTab.tsx`, `tabs/GenerationTab.tsx`, `tabs/TargetAreaTab.tsx`, `tabs/SolarTab.tsx`, `tabs/ViewTab.tsx`, `tabs/LandmarkTab.tsx`, `tabs/EditTab.tsx`, `tabs/ImportTab.tsx`.

> **NOTE:** Before rewriting `guidedTabState.ts`, read the current file — its exact function names and signatures (`prerequisiteMessageForTab`, `targetAreaActionLabel`, `generationActionLabel`, `simulationActionLabel`, `exportActionLabel`, and the `ExportFormat`/`PrerequisiteTab`/`TargetAreaMethod` types) must be preserved except for the added leading `t` parameter. The block below reflects the draft's read; reconcile with the live file.

- [ ] **Step 1** — Rewrite `guidedTabState.ts` to take a `t` translator. Define a local `Translate` type to avoid a hook import:

```ts
export type PrerequisiteTab = 'generation' | 'zoning' | 'solar' | 'view' | 'landmark' | 'export';
export type TargetAreaMethod = 'draw' | 'coordinates';
export type ExportFormat = 'cityles' | 'obj' | 'geotiff';

type Translate = (key: string, vars?: Record<string, string | number>) => string;

export function prerequisiteMessageForTab(
  t: Translate,
  tab: PrerequisiteTab,
): { title: string; body: string } {
  if (tab === 'generation') {
    return { title: t('guided.setAreaTitle'), body: t('guided.setAreaBody') };
  }
  return { title: t('guided.modelRequiredTitle'), body: t('guided.modelRequiredBody') };
}

export function targetAreaActionLabel(t: Translate, method: TargetAreaMethod, loading: boolean) {
  if (method === 'coordinates') return t('guided.actionSetRectangle');
  return loading ? t('guided.actionLoadingMap') : t('guided.actionLoadMap');
}

export function generationActionLabel(t: Translate, loading: boolean) {
  return loading ? t('guided.actionGenerating') : t('guided.actionGenerate');
}

export function simulationActionLabel(t: Translate, loading: boolean) {
  return loading ? t('guided.actionRunning') : t('guided.actionRunSimulation');
}

export function exportActionLabel(t: Translate, format: ExportFormat, loading: boolean) {
  if (loading) return t('guided.actionExporting');
  if (format === 'cityles') return t('guided.actionExportCityles');
  if (format === 'geotiff') return t('guided.actionExportGeotiff');
  return t('guided.actionExportObj');
}
```

The `Translate` param is intentionally typed loosely (`key: string`) so `guidedTabState.ts` stays decoupled from `TranslationKey`; the `useT()` return type is assignable to it, and callers pass real `t`.

- [ ] **Step 2** — Rewrite `guidedTabState.test.ts` to pass a fake translator (existing test asserted English strings; now assert the keys resolve through a stub, matching the seed catalog values). Replace the file with:

```ts
import { describe, expect, it } from 'vitest';
import { translate } from '../i18n/translate';
import {
  exportActionLabel,
  generationActionLabel,
  prerequisiteMessageForTab,
  simulationActionLabel,
  targetAreaActionLabel,
} from './guidedTabState';

const t = (key: string, vars?: Record<string, string | number>) =>
  translate('en', key as never, vars);

describe('guided tab prerequisite messages', () => {
  it('returns the generation prerequisite message', () => {
    expect(prerequisiteMessageForTab(t, 'generation')).toEqual({
      title: 'Set a target area first',
      body: 'Use the Target Area tab to choose the city area before generating a model.',
    });
  });

  it('returns the model-required message for other tabs', () => {
    expect(prerequisiteMessageForTab(t, 'export')).toEqual({
      title: 'Generate a model first',
      body: 'Use the Generation tab to create a VoxCity model before using this workflow.',
    });
  });

  it('returns action labels', () => {
    expect(targetAreaActionLabel(t, 'coordinates', false)).toBe('Set Rectangle');
    expect(generationActionLabel(t, true)).toBe('Generating...');
    expect(simulationActionLabel(t, false)).toBe('Run Simulation');
    expect(exportActionLabel(t, 'geotiff', false)).toBe('Export GeoTIFF');
  });
});
```

- [ ] **Step 3** — Thread `t` through every call site. Find them:

```
cd app/frontend && grep -rn "prerequisiteMessageForTab(\|targetAreaActionLabel(\|generationActionLabel(\|simulationActionLabel(\|exportActionLabel(" src/tabs
```

For each hit, add `const t = useT();` (import `useT` from `'../i18n'`) near the top of the component if not already present, and insert `t` as the first argument. E.g. in `ExportTab.tsx`: `prerequisiteMessageForTab('export')` → `prerequisiteMessageForTab(t, 'export')`, and `exportActionLabel(exportFormat, loading)` → `exportActionLabel(t, exportFormat, loading)`. In `TargetAreaTab.tsx`: `targetAreaActionLabel(method, loading)` → `targetAreaActionLabel(t, method, loading)`. Same shape for `generationActionLabel(t, …)` (GenerationTab) and `simulationActionLabel(t, …)` (SolarTab/ViewTab/LandmarkTab). EditTab/ImportTab call `prerequisiteMessageForTab` only.

- [ ] **Step 4** — Convert `StartSplash.tsx`. Add `import { useT } from '../i18n';`, add `const t = useT();` at the top of the component, and replace these hardcoded strings (all real, from the current file):
  - `aria-label="Close"` → `aria-label={t('common.close')}`
  - `<span id="start-splash-title">Welcome to VoxCity</span>` → keep the id span: `<span id="start-splash-title">{t('splash.title')}</span>`
  - `subtitle="Start a new urban model, or open a saved session."` → `subtitle={t('splash.subtitle')}`
  - `New session` → `{t('common.newSession')}`
  - `Open session...` → `{t('common.openSession')}`
  - `label="GET STARTED"` → `label={t('common.getStarted')}`
  - `<span>Don't show this again</span>` → `<span>{t('common.dontShowAgain')}</span>`
  - the fallback `'Failed to load session.'` in the catch → `t('splash.loadFailed')`

- [ ] **Step 5** — Convert `PreviewDisabledNotice.tsx`. Add `import { useT } from '../i18n';` + `const t = useT();`. Replace `<strong>3D preview disabled</strong>` with `<strong>{t('previewNotice.heading')}</strong>`, and replace the whole `<p>…</p>` body (the `dims ? … : …` conditional + the trailing "cells. Generation…" text) with a single interpolated call:

```tsx
      <p style={{ maxWidth: 360, fontSize: '0.85rem', margin: 0 }}>
        {dims
          ? t('previewNotice.bodyWithDims', { dims, cells: PREVIEW_MAX_CELLS.toLocaleString() })
          : t('previewNotice.bodyNoDims', { cells: PREVIEW_MAX_CELLS.toLocaleString() })}
      </p>
```

Update `PreviewDisabledNotice.test.tsx` if it asserts the old wording — it asserts `'1500'`/`'900'` (still present via `dims`), so it should still pass; run it to confirm.

- [ ] **Step 6** — Convert `VoxelClassVisibility.tsx`. Add `import { useT } from '../i18n';` + `const t = useT();`. Replace the `Visualization Settings` header text with `{t('voxelVisibility.heading')}` and `Hide element classes` with `{t('voxelVisibility.hideClasses')}`. (Leave `cls.label` — the voxel class names come from `constants.ts`; wrapping those is out of scope for this pass and noted as a follow-up in Self-Review.)

- [ ] **Step 7** — Run tests, typecheck, build:

```
npm test -- guidedTabState PreviewDisabledNotice
npx tsc -b --noEmit
npm run build
```

Expected: green. Commit: `feat(i18n): translate guided helpers, start splash, preview notice, voxel visibility`.

---

### Task 4 — Convert ExportTab (representative full example)

This is the fully worked template every remaining per-tab task follows: add the tab's section to `en.ts` + `ja.ts`, import `useT`, replace hardcoded JSX strings with `t('...')`. **This ExportTab also contains the Share-link strings added by Plan 1 — wrap those too** (`Copy Share Link`, `Copy`, `Share URL`, `Link copied to clipboard.`, `Share link created — copy it below.`, `SHARE LINK`); add keys for them under `export.*`.

**Files:** `app/frontend/src/i18n/locales/en.ts`, `locales/ja.ts`, `app/frontend/src/tabs/ExportTab.tsx`

- [ ] **Step 1** — Append an `export:` section to the `en` object in `en.ts` (before the closing `};` / `export type Messages`):

```ts
  export: {
    sessionTitle: 'Save / Load Session',
    sessionSubtitle: 'Move the current scene and zones between browser sessions.',
    sessionOptions: 'SESSION OPTIONS',
    includeSim: 'Include simulation results (larger file, lets overlays render without re-running)',
    saveSession: 'Save Session',
    loadSession: 'Load Session',
    sessionSaved: 'Session saved.',
    sessionLoaded: 'Session loaded.',
    sessionLoadedPartial: 'Session loaded; some frontend state was not restored.',
    shareLink: 'Copy Share Link',
    shareSectionLabel: 'SHARE LINK',
    shareCopy: 'Copy',
    shareUrlLabel: 'Share URL',
    shareCopied: 'Link copied to clipboard.',
    shareCreated: 'Share link created — copy it below.',
    exportTitle: 'Export',
    exportSubtitle: 'Download the VoxCity model in your preferred format.',
    formatHeading: 'EXPORT FORMAT',
    formatAria: 'Export format',
    optCitylesLabel: 'CityLES',
    optCitylesDesc: 'CityLES output archive',
    optObjLabel: 'OBJ',
    optObjDesc: 'Mesh export archive',
    optGeotiffLabel: 'GeoTIFF',
    optGeotiffDesc: 'Georeferenced raster layers',
    citylesOptions: 'CITYLES OPTIONS',
    buildingMaterial: 'Building Material',
    matDefault: 'Default',
    matConcrete: 'Concrete',
    matBrick: 'Brick',
    treeType: 'Tree Type',
    treeDefault: 'Default',
    treeDeciduous: 'Deciduous',
    treeConifer: 'Conifer',
    trunkHeightRatio: 'Trunk Height Ratio',
    objOptions: 'OBJ OPTIONS',
    outputFilename: 'Output Filename',
    alsoNetcdf: 'Also export NetCDF',
    geotiffOptions: 'GEOTIFF OPTIONS',
    geotiffHint: 'Exports land cover, building height, DEM, and canopy height as four georeferenced GeoTIFFs (EPSG:4326), plus a README.md with layer details and usage instructions.',
    citylesExported: 'CityLES exported successfully!',
    geotiffExported: 'GeoTIFF exported successfully!',
    objExported: 'OBJ exported successfully!',
  },
```

- [ ] **Step 2** — Append the mirrored `export:` section to `ja.ts` (real Japanese; native review is a follow-up, see Self-Review):

```ts
  export: {
    sessionTitle: 'セッションの保存 / 読み込み',
    sessionSubtitle: '現在のシーンとゾーンをブラウザセッション間で移動します。',
    sessionOptions: 'セッションオプション',
    includeSim: 'シミュレーション結果を含める（ファイルが大きくなりますが、再実行なしでオーバーレイを表示できます）',
    saveSession: 'セッションを保存',
    loadSession: 'セッションを読み込む',
    sessionSaved: 'セッションを保存しました。',
    sessionLoaded: 'セッションを読み込みました。',
    sessionLoadedPartial: 'セッションを読み込みました。一部のフロントエンド状態は復元されませんでした。',
    shareLink: '共有リンクをコピー',
    shareSectionLabel: '共有リンク',
    shareCopy: 'コピー',
    shareUrlLabel: '共有 URL',
    shareCopied: 'リンクをクリップボードにコピーしました。',
    shareCreated: '共有リンクを作成しました — 下からコピーしてください。',
    exportTitle: 'エクスポート',
    exportSubtitle: 'VoxCity モデルを任意の形式でダウンロードします。',
    formatHeading: 'エクスポート形式',
    formatAria: 'エクスポート形式',
    optCitylesLabel: 'CityLES',
    optCitylesDesc: 'CityLES 出力アーカイブ',
    optObjLabel: 'OBJ',
    optObjDesc: 'メッシュエクスポートアーカイブ',
    optGeotiffLabel: 'GeoTIFF',
    optGeotiffDesc: 'ジオリファレンス済みラスターレイヤー',
    citylesOptions: 'CityLES オプション',
    buildingMaterial: '建物の材質',
    matDefault: 'デフォルト',
    matConcrete: 'コンクリート',
    matBrick: 'レンガ',
    treeType: '樹木の種類',
    treeDefault: 'デフォルト',
    treeDeciduous: '落葉樹',
    treeConifer: '針葉樹',
    trunkHeightRatio: '幹の高さ比率',
    objOptions: 'OBJ オプション',
    outputFilename: '出力ファイル名',
    alsoNetcdf: 'NetCDF も出力する',
    geotiffOptions: 'GeoTIFF オプション',
    geotiffHint: '土地被覆・建物高さ・DEM・樹冠高さの4つのジオリファレンス済み GeoTIFF（EPSG:4326）と、レイヤーの詳細と使い方を記した README.md を出力します。',
    citylesExported: 'CityLES のエクスポートに成功しました！',
    geotiffExported: 'GeoTIFF のエクスポートに成功しました！',
    objExported: 'OBJ のエクスポートに成功しました！',
  },
```

- [ ] **Step 3** — In `ExportTab.tsx`, add `import { useT } from '../i18n';` and `const t = useT();` at the top of the component. Then replace strings. Real before → after examples:
  - `setSessionSuccess('Session saved.')` → `setSessionSuccess(t('export.sessionSaved'))`
  - the ternary `... ? 'Session loaded; some frontend state was not restored.' : 'Session loaded.'` → `... ? t('export.sessionLoadedPartial') : t('export.sessionLoaded')`
  - `title="Save / Load Session"` → `title={t('export.sessionTitle')}`
  - `subtitle="Move the current scene and zones between browser sessions."` → `subtitle={t('export.sessionSubtitle')}`
  - `label="SESSION OPTIONS"` → `label={t('export.sessionOptions')}`
  - `<span>Include simulation results …</span>` → `<span>{t('export.includeSim')}</span>`
  - `Save Session` → `{t('export.saveSession')}`, `Load Session` → `{t('export.loadSession')}`
  - Share (from Plan 1): the `Copy Share Link` button text → `{t('export.shareLink')}`, `SHARE LINK` section label → `t('export.shareSectionLabel')`, `Copy` → `{t('export.shareCopy')}`, `aria-label="Share URL"` → `aria-label={t('export.shareUrlLabel')}`, and the copied/created status strings → `t('export.shareCopied')` / `t('export.shareCreated')`
  - `title="Export"` → `title={t('export.exportTitle')}`, `subtitle="Download the VoxCity model in your preferred format."` → `subtitle={t('export.exportSubtitle')}`
  - `label="EXPORT FORMAT"` → `label={t('export.formatHeading')}`, `ariaLabel="Export format"` → `ariaLabel={t('export.formatAria')}`
  - the `options={[...]}` array labels/descriptions → `label: t('export.optCitylesLabel'), description: t('export.optCitylesDesc')`, etc. (obj, geotiff)
  - `label="CITYLES OPTIONS"` → `t('export.citylesOptions')`; `<label>Building Material</label>` → `t('export.buildingMaterial')`; the three `<option>` texts → `t('export.matDefault')` etc.; `<label>Tree Type</label>` → `t('export.treeType')` and its options; `<label>Trunk Height Ratio</label>` → `t('export.trunkHeightRatio')`
  - `label="OBJ OPTIONS"` → `t('export.objOptions')`; `<label>Output Filename</label>` → `t('export.outputFilename')`; `<span>Also export NetCDF</span>` → `t('export.alsoNetcdf')`
  - `label="GEOTIFF OPTIONS"` → `t('export.geotiffOptions')`; `<label>Output Filename</label>` → `t('export.outputFilename')`; the `<p>Exports land cover…</p>` hint → `{t('export.geotiffHint')}`
  - the three success strings → `t('export.citylesExported')` / `t('export.geotiffExported')` / `t('export.objExported')`

  (The `<option value="default">` VALUES stay unchanged — only the visible text is wrapped.)

- [ ] **Step 4** — Verify:

```
npm test -- catalogParity
npx tsc -b --noEmit
```

Expected: parity passes (en/ja `export.*` keys match), `tsc` 0. Commit: `feat(i18n): translate ExportTab (incl. share link)`.

---

### Task 5 — Convert GenerationTab

**Files:** `en.ts`, `ja.ts`, `app/frontend/src/tabs/GenerationTab.tsx`

- [ ] **Step 1** — Add a `generationTab:` section (en then ja), keyed to this tab's real strings. en example:
```ts
  generationTab: {
    title: 'Generate Model',
    subtitle: 'Build the VoxCity 3D model from city data.',
    modeHeading: 'GENERATION MODE',
    // ...remaining keys below
  },
```
ja mirror:
```ts
  generationTab: {
    title: 'モデルを生成',
    subtitle: '都市データから VoxCity の 3D モデルを構築します。',
    modeHeading: '生成モード',
    // ...
  },
```

- [ ] **Step 2** — Wrap the remaining strings in `GenerationTab.tsx` (add `useT`), following the Task-4 pattern. The actual English strings in this tab to key + wrap are: `title="Generate Model"`, `subtitle="Build the VoxCity 3D model from city data."`, `label="GENERATION MODE"`, `ariaLabel="Generation mode"`, `label="GRID RESOLUTION"`, `<label>Mesh size (meters)</label>`, `label="DATA SOURCES"`, `Auto-select sources based on location`, `Auto-detected:`, `<label>Building Source</label>`, `<label>Building Complementary Source</label>`, `<label>Land Cover Source</label>`, `<label>Canopy Height Source</label>`, `<label>DEM Source</label>`, `label="ADVANCED"`, `<label>Building Complement Height (m)</label>`, `<label>Static Tree Height (m)</label>`, `DEM Interpolation`, `Use CityGML Cache`, `Use nDSM for Canopy`. Also wrap the large-area preview warning (`Estimated grid ~{a}×{b} — the 3D preview will be disabled at this size. Generation and export still work.`) using an interpolated key. Suggested keys: `title, subtitle, modeHeading, modeAria, gridHeading, meshSize, dataSources, autoSelect, autoDetected, buildingSource, buildingCompSource, landCoverSource, canopySource, demSource, advanced, buildingCompHeight, staticTreeHeight, demInterpolation, useCitygmlCache, useNdsm, previewWarn`.

- [ ] **Step 3** — `npm test -- catalogParity && npx tsc -b --noEmit`. Commit: `feat(i18n): translate GenerationTab`.

---

### Task 6 — Convert TargetAreaTab

**Files:** `en.ts`, `ja.ts`, `app/frontend/src/tabs/TargetAreaTab.tsx`

- [ ] **Step 1–2** — Add a `targetAreaTab:` section and wrap. Real strings to key + wrap (add `useT`; `targetAreaActionLabel(t,…)` already threaded in Task 3): `title="Target Area"`, `subtitle="Choose the city area used by model generation."`, status `Target area is ready.`, `label="LOCATION"`, `<label>City name</label>`, `label="DEFINE TARGET AREA"`, `ariaLabel="Target area input method"`, choice `Map draw`, `label="DRAWING MODE"`, `ariaLabel="Target area drawing mode"`, choices `Free hand`/`Set dimensions`, `label="DIMENSIONS"`, `<label>Width (m)</label>`, `<label>Height (m)</label>`, `<label>Rotation (°)</label>`, `label="RECTANGLE VERTICES"`, `label="SUMMARY"`. Suggested keys: `title, subtitle, ready, location, cityName, defineArea, inputMethodAria, mapDraw, drawingMode, drawingModeAria, freeHand, setDimensions, dimensions, width, height, rotation, rectangleVertices, summary`. ja mirror example: `title: '対象エリア'`, `subtitle: 'モデル生成に使用する都市エリアを選択します。'`, `ready: '対象エリアの準備ができました。'`.

- [ ] **Step 3** — Verify + commit: `feat(i18n): translate TargetAreaTab`.

---

### Task 7 — Convert EditTab

**Files:** `en.ts`, `ja.ts`, `app/frontend/src/tabs/EditTab.tsx`

- [ ] **Step 1–2** — Add an `editTab:` section and wrap (add `useT`; `prerequisiteMessageForTab(t,…)` already threaded). Real strings to key: `<h2>Edit Model</h2>` (appears twice), `ariaLabel="Edit method"`, `ariaLabel="Edit target"`, `ariaLabel="Edit workflow"`, section labels `TARGET`/`OPERATIONS`, `<label>Height (m)</label>`, `<label>Min height / base (m)</label>`, `<label>Top height (m)</label>`, `<label>Top (m)</label>`, `<label>Trunk / bottom (m)</label>`, `<label>Diameter (m) — disc brush radius {n} cell(s)</label>` (interpolated: `diameterBrush` = `'Diameter (m) — disc brush radius {n} cell(s)'`, call `t('editTab.diameterBrush', { n: treeBrushRadius })`), `Fixed proportion`, `Undo`, `Clear`, `Show height labels`, `title="Discard the most recent buffered edit"`, `title="Discard all buffered edits"`, `<summary>Display</summary>`, `<label>Basemap</label>`, `<label>Overlay</label>`, overlay option texts `Buildings`/`Canopy`/`Land cover`/`None`, `Apply an edit and click {action} to render the 3D result here.` (interpolated with `t('editTab.update3dModel')`), `Update 3D model`, error fallbacks `'Failed to load map'`, `'Edit failed'`, `'3D regeneration failed'`. Leave proper-noun basemap names (`CartoDB Positron`, `Google Satellite`, `OpenStreetMap`) untranslated. Suggested keys: `heading, editMethodAria, editTargetAria, editWorkflowAria, targetHeading, operationsHeading, height, minHeightBase, topHeight, top, trunkBottom, diameterBrush, fixedProportion, undo, clear, showHeightLabels, discardRecent, discardAll, display, basemap, overlay, ovBuildings, ovCanopy, ovLandCover, ovNone, applyHint, update3dModel, errLoadMap, errEditFailed, errRegenFailed`.

- [ ] **Step 3** — Verify + commit: `feat(i18n): translate EditTab`.

---

### Task 8 — Convert ImportTab

**Files:** `en.ts`, `ja.ts`, `app/frontend/src/tabs/ImportTab.tsx`

> **Ordering note:** this runs AFTER Plan 2 reworked ImportTab into OBJ/DXF mode. Wrap the strings that exist in the reworked file — OBJ strings, the mode-toggle labels (`OBJ buildings`, `DXF reference lines`), the DXF sections (`UPLOAD DXF`, `LAYERS`, `Anchor latitude / longitude`, `Rotation (deg)`, `Move east / north (m)`, `Units`, `Add reference lines`, `Choose DXF…`), and the info/error strings added there — plus the OBJ strings below.

- [ ] **Step 1–2** — Add an `importTab:` section and wrap. Real OBJ strings to key: `<h2>Import</h2>`, section labels `UPLOAD`/`GROUPS / ROLES`/`PLACEMENT`, `<label>Anchor latitude / longitude</label>`, placeholders `lat`/`lon`, `<label>Anchor elevation (m, blank = auto from terrain)</label>`, `<label>Rotation (deg)</label>`, `<label>Move east / north / up (m)</label>`, buttons `Move`/`Rotate`, `<label>Units</label>`, `<summary>Advanced</summary>`, error fallbacks `'Please choose a .obj file (you can also select its .mtl).'`, `'Click the map to set an anchor first.'`, `'Upload failed'`, `'Import failed'`. Suggested keys: `heading, modeObj, modeDxf, uploadHeading, groupsRoles, placementHeading, anchorLatLon, phLat, phLon, anchorElevation, rotationDeg, moveEnu, moveEn, move, rotate, units, advanced, uploadDxf, layers, chooseDxf, addReferenceLines, dxfInfo, errNoObj, errNoDxf, errNoAnchor, errUpload, errImport`.

- [ ] **Step 3** — Verify + commit: `feat(i18n): translate ImportTab`.

---

### Task 9 — Convert ZoningTab

**Files:** `en.ts`, `ja.ts`, `app/frontend/src/tabs/ZoningTab.tsx`

- [ ] **Step 1–2** — Add a `zoningTab:` section and wrap. Real strings to key: `title="Zoning"`, `subtitle="Define evaluation zones for simulation summaries."`, section labels `ZONE TYPE`/`SHAPE`/`ZONES`, arias `Zone type`/`Zone shape`, `title="Add a new zone row. Draw on the map to set its boundary."`, `Add zone`, `Clear all zones`, `title="Rename"`, `title="Delete"`, `Selected buildings:`, `title="Remove building"`, `Cancel refine`, `<summary>Display</summary>`, `<label>Basemap</label>`, `<label>Overlay</label>` + overlay options `Buildings`/`Canopy`/`Land cover`/`None`. Suggested keys: `title, subtitle, zoneTypeHeading, zoneTypeAria, shapeHeading, zoneShapeAria, zonesHeading, addZoneTitle, addZone, clearAllZones, rename, delete, selectedBuildings, removeBuilding, cancelRefine, display, basemap, overlay, ovBuildings, ovCanopy, ovLandCover, ovNone`.

- [ ] **Step 3** — Verify + commit: `feat(i18n): translate ZoningTab`.

---

### Task 10 — Convert the three simulation tabs (Solar, View, Landmark)

These three share several strings (`Simulation complete.`, `Analysis target`, `DISPLAY`, `ZONES AND RESULTS`, `Include building rooftops`). Put the shared ones in a `sim:` group and per-tab specifics in `solarTab:` / `viewTab:` / `landmarkTab:`.

**Files:** `en.ts`, `ja.ts`, `app/frontend/src/tabs/SolarTab.tsx`, `ViewTab.tsx`, `LandmarkTab.tsx`

- [ ] **Step 1** — Add a shared `sim:` group (en then ja):
```ts
  sim: {
    complete: 'Simulation complete.',
    analysisTarget: 'Analysis target',
    display: 'DISPLAY',
    zonesAndResults: 'ZONES AND RESULTS',
    includeRooftops: 'Include building rooftops',
  },
```
```ts
  sim: {
    complete: 'シミュレーションが完了しました。',
    analysisTarget: '分析対象',
    display: '表示',
    zonesAndResults: 'ゾーンと結果',
    includeRooftops: '建物の屋上を含める',
  },
```

- [ ] **Step 2** — Add per-tab sections. `solarTab` real strings: `title="Solar Radiation"`, `subtitle="Compute irradiance on ground or building surfaces."`, `TEMPORAL TYPE`, aria `Calculation type`, `Date (MM-DD)`, `Time (HH:MM:SS)`, `Start (MM-DD HH:MM:SS)`, `End (MM-DD HH:MM:SS)`, `SPATIAL TYPE`. `viewTab`: `title="View Index"`, `subtitle="Analyse sky, green, or custom view from any point."`, `VIEW TYPE`, aria `View type`, `CUSTOM CLASSES`, aria `Custom class mode`, `SAMPLING`, `View point height (m)`. `landmarkTab`: `title="Landmark Visibility"`, `subtitle="Select buildings as landmarks and analyse how visible they are."`, `ANALYSIS TARGET`, `LANDMARK BUILDINGS`, `title="Clear all selections"`, `Selected Buildings ({n})` (interpolated key `selectedCount`), `Landmark Building IDs (comma-separated, empty = center building)`, placeholder `e.g. 12, 34, 56`, `SAMPLING`, `Back to selection`, `Clear`. en/ja examples:
```ts
  solarTab: { title: 'Solar Radiation', subtitle: 'Compute irradiance on ground or building surfaces.', temporalType: 'TEMPORAL TYPE', /* ... */ },
  viewTab: { title: 'View Index', subtitle: 'Analyse sky, green, or custom view from any point.', /* ... */ },
  landmarkTab: { title: 'Landmark Visibility', subtitle: 'Select buildings as landmarks and analyse how visible they are.', selectedCount: 'Selected Buildings ({n})', /* ... */ },
```
```ts
  solarTab: { title: '日射', subtitle: '地表面または建物表面の日射量を計算します。', temporalType: '時間タイプ', /* ... */ },
  viewTab: { title: '可視指標', subtitle: '任意の地点から空・緑・カスタムの視界を分析します。', /* ... */ },
  landmarkTab: { title: 'ランドマークの視認性', subtitle: '建物をランドマークとして選択し、その視認性を分析します。', selectedCount: '選択中の建物（{n}）', /* ... */ },
```

- [ ] **Step 3** — Wrap each tab's strings (`useT` added; `simulationActionLabel(t,…)` already threaded in Task 3). Shared items use `t('sim.*')`, specifics `t('solarTab.*')` / `t('viewTab.*')` / `t('landmarkTab.*')`. For LandmarkTab's count: `t('landmarkTab.selectedCount', { n: selectedBuildingIds.length })`.

- [ ] **Step 4** — Verify + commit: `feat(i18n): translate Solar/View/Landmark simulation tabs`.

---

### Task 11 — Final verification (parity, typecheck, build, manual smoke)

**Files:** none (verification only)

- [ ] **Step 1** — Full parity + all i18n/tab tests:

```
cd app/frontend
npm test -- catalogParity
npm test -- i18n
npm test
```

Expected: `catalog parity` passes (proves ja has exactly the same keys as en across every section, no empty values); the full suite is green.

- [ ] **Step 2** — Typecheck (the `ja: Messages` annotation makes a missing/mistyped ja key a compile error, a second parity guard) and production build:

```
npx tsc -b --noEmit
npm run build
```

Expected: both exit 0.

- [ ] **Step 3** — Manual smoke of the toggle (verifies the interactive `setLang` + persistence path the SSR unit test cannot): `npm run dev`, then click **日本語** in the header — confirm the 9 tab labels, the Start Splash, and each tab's panel titles/labels switch to Japanese; reload and confirm it stays Japanese (localStorage `voxcity.lang` = `ja`); click **EN** and confirm it reverts. Spot-check ExportTab (incl. share link), GenerationTab, and one sim tab.

- [ ] **Step 4** — If any hardcoded English remains visible during the smoke, grep for stragglers and wrap them following the established per-tab pattern:

```
grep -rnE ">[A-Z][a-z]+ [A-Za-z ]+<|title=\"[A-Z]|label=\"[A-Z]|subtitle=\"" src/tabs src/components | grep -v "t('"
```

Commit any fixes: `feat(i18n): wrap remaining straggler strings`.

---

### Self-Review Notes

- **Maps to Spec Feature 3 (i18n portion):** VoxCity gains optree's exact i18n machinery — `LanguageProvider` (localStorage-persisted, key `voxcity.lang`), `useLanguage`, `useT`, `translate` with `{var}` interpolation and English fallback, and the compile-time-typed `TranslationKey`/`Messages`/`Lang` contract. A header EN / 日本語 toggle drives an English and Japanese catalog with structural parity enforced two ways: the ported `catalogParity.test.ts` (runtime) and the `ja: Messages` annotation (compile time). Every VoxCity surface — header/nav, Start Splash, preview notice, voxel visibility, the shared `guidedTabState` helpers, and all 9 tabs (including the Plan 1 share strings and Plan 2 DXF strings) — is wrapped in `t()`. Because this plan runs last, Step 4 of Task 11 also sweeps any English introduced by Plans 1–3.
- **Deviation from the source (necessary):** optree's `LanguageContext.test.tsx` was **not** ported verbatim — it depends on `@testing-library/react` + jsdom, which VoxCity does not have (its tests use `renderToStaticMarkup` in the node env). The adapted test covers default-English and localStorage-init via SSR; the interactive `setLang`+persistence path is covered by the Task 11 manual smoke. If the team prefers the exact optree test, that requires adding `jsdom` + `@testing-library/react` devDeps and a vitest `test.environment: 'jsdom'` config — deliberately out of scope.
- **Known limitation — Japanese translation review (follow-up):** the `ja` values in the seed, ExportTab, and the sim/tab sections are real, reasonable translations, but bulk-tab strings (Tasks 5–10) are engineer-authored and have **not** had a native-speaker review. The machinery is complete and correct; only the ja *wording* needs a polish pass. This is a wording follow-up, not a placeholder or structural gap — the parity test and `Messages` typing guarantee every ja key exists and is non-empty. Recommend a native-review ticket before 2.0.
- **Intentionally not translated:** proper nouns / provider identifiers (basemap names `CartoDB Positron`, `Google Satellite`, `OpenStreetMap`; format names `CityLES`, `OBJ`, `GeoTIFF`, `NetCDF`), `<option value="...">` machine values, and voxel-class names sourced from `constants.ts` (a separate `VOXEL_CLASSES` data-layer i18n pass, noted as a follow-up).
