const rawApiUrl = (import.meta.env.VITE_API_URL || '').trim();

const normalizedApiUrl = rawApiUrl.replace(/\/+$/, '');
const apiUrlContainsPrefix = /\/api\/vait$/i.test(normalizedApiUrl);

export function buildApiUrl(path) {
  const initialPath = path.startsWith('/') ? path : `/${path}`;

  if (!normalizedApiUrl) {
    return initialPath;
  }

  const normalizedPath =
    apiUrlContainsPrefix && initialPath.startsWith('/api/vait')
      ? initialPath.replace(/^\/api\/vait/i, '') || '/'
      : initialPath;

  return `${normalizedApiUrl}${normalizedPath}`;
}

export const API_BASE_URL = normalizedApiUrl;
