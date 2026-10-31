/**
 * SAFEOPS — central API configuration.
 *
 * All backend URLs live here so the whole app can be pointed at a real
 * backend by changing one value (or setting VITE_SAFEOPS_API_URL).
 */

export const API_CONFIG = {
  /** Override with VITE_SAFEOPS_API_URL when the real backend is deployed. */
  baseUrl: import.meta.env["VITE_SAFEOPS_API_URL"] ?? "http://127.0.0.1:8000/api/v1",
  timeoutMs: 20_000,
} as const;

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Thin fetch wrapper with timeout handling. Every call in detectionService.ts
 * goes through this function, so swapping in the real backend only requires
 * updating API_CONFIG and the individual endpoint paths.
 */
export async function apiRequest<TResponse>(path: string, init: RequestInit = {}): Promise<TResponse> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), API_CONFIG.timeoutMs);

  const isFormData = typeof FormData !== "undefined" && init.body instanceof FormData;
  const headers: Record<string, string> = {
    ...(isFormData ? {} : { "Content-Type": "application/json" }),
    ...(init.headers as Record<string, string>),
  };

  try {
    const response = await fetch(`${API_CONFIG.baseUrl}${path}`, {
      ...init,
      signal: controller.signal,
      headers,
    });

    if (!response.ok) {
      throw new ApiError(`Request failed (${response.status})`, response.status);
    }

    return (await response.json()) as TResponse;
  } finally {
    clearTimeout(timer);
  }
}
