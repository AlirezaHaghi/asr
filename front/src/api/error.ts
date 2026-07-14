export async function apiError(response: Response): Promise<Error> {
  const fallback = `Request failed - ${response.status} ${response.statusText}`

  try {
    const body: unknown = await response.json()
    if (body && typeof body === 'object' && 'detail' in body) {
      const detail = (body as { detail: unknown }).detail
      if (typeof detail === 'string') return new Error(detail)
      if (detail != null) return new Error(JSON.stringify(detail))
    }
  } catch {
    // The fallback below handles non-JSON error responses.
  }

  return new Error(fallback)
}
