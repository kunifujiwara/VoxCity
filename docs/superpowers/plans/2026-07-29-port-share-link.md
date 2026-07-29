# Copy Share Link — Implementation Plan (1 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Copy Share Link" feature to VoxCity: persist the current session server-side under an unguessable token and let anyone with the `/share/<token>` URL load that snapshot.

**Architecture:** Port optree's `share.py` (session-zip snapshot + token) adapted to VoxCity's single global `app_state` (dropping optree's optimization/per-session-cookie machinery). Two new endpoints (`POST /api/share`, `POST /api/share/{token}/load`) reuse the existing `save_session_to_zip` / `parse_session_zip` / `apply_session_to_state` pipeline. Frontend adds a Share section to the File tab and consumes `/share/<token>` URLs on app load via pathname parsing (no router).

**Tech Stack:** FastAPI + Pydantic (backend), React 18 + TypeScript + Vite + Vitest (frontend), pytest (backend tests).

**Part of:** `docs/superpowers/specs/2026-07-29-port-optree-features-design.md`. Strings added here are plain English; the later i18n plan (4 of 4) wraps them in `t()`.

---

## File Structure

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

## Task 1: Backend config — SHARE_DIR

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

## Task 2: Backend share.py module

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

## Task 3: Backend endpoints — /api/share and /api/share/{token}/load

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

## Task 4: Frontend api.ts — createShare / loadShare

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

## Task 5: Frontend lib/shareLink.ts

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

## Task 6: Frontend ExportTab — Share button + copy UI

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

## Task 7: Frontend App.tsx — consume /share/<token> on load

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

## Task 8: Full verification + manual smoke

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

## Self-Review Notes (verified against the spec)

- **Spec "Feature 2 — Copy Share Link / Backend":** covered by Tasks 1–3 (SHARE_DIR, share.py dropping `include_optimization`, two endpoints wired to `app_state`, token regex + atomic write + traversal guard retained verbatim from optree).
- **Spec "Feature 2 / Frontend":** covered by Tasks 4–7 (`createShare`/`loadShare`, ExportTab share section with clipboard→execCommand fallback, `parseShareToken`, App mount consumption + `history.replaceState('/')`, full-screen overlay). No routing library — pathname parsing only.
- **Spec "session/state compatibility" risk:** Task 8 Step 1 re-runs the existing session round-trip tests to confirm nothing regressed.
- **i18n:** intentionally deferred — all share strings are plain English here and will be wrapped by plan 4 of 4.
- **Type consistency:** `ShareCreateResult` (`token`, `path`) matches the backend `{token, path}` return; `loadShare` returns `SessionLoadSummary` (already defined in `api.ts`); `parseShareToken` regex matches the backend `_TOKEN_RE`.
