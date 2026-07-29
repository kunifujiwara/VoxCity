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
