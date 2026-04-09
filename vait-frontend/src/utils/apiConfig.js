export const BASE_URL =
  (import.meta.env.VITE_API_URL || 'http://localhost:8000').trim().replace(/\/+$/, '');

export function buildApiUrl(path) {
  const normalizedPath = path.startsWith('/') ? path : '/' + path;
  return BASE_URL + normalizedPath;
}
