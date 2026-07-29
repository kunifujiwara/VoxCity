// Mirrors the backend token rule in app/backend/share.py (_TOKEN_RE).
const SHARE_PATH_RE = /^\/share\/([A-Za-z0-9_-]{16,64})$/;

/** Return the share token when *pathname* is a /share/<token> URL, else null. */
export function parseShareToken(pathname: string): string | null {
  const match = SHARE_PATH_RE.exec(pathname);
  return match ? match[1] : null;
}
