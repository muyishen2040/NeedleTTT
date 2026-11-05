// These payloads follow docs/contracts.md (DEE-5).
// The bootstrap server uses an UNTRAINED teaching model and fictional items.

export type Variant = "base" | "vector" | "matrix";
export const VARIANTS: Variant[] = ["base", "vector", "matrix"];
export const VARIANT_LABELS: Record<Variant, string> = {
  base: "History baseline",
  vector: "Vector TTT",
  matrix: "Matrix TTT",
};

export interface CatalogItem {
  item_id: string;
  item_idx: number;
  title: string;
  categories: string[];
  image_url: string | null;
  source: string;
}

export interface Recommendation {
  item_id: string;
  title: string;
  categories: string[];
  image_url: string | null;
  score: number;
  rank: number;
}

export interface Rating {
  item_id: string;
  rating: number;
}

const BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(BASE + path, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
  });
  if (!response.ok) {
    let detail: string;
    try {
      detail = JSON.stringify(await response.json());
    } catch {
      detail = response.statusText;
    }
    throw new Error("API request failed: " + detail);
  }
  return (await response.json()) as T;
}

export async function listItems(): Promise<CatalogItem[]> {
  const data = await request<{ items: CatalogItem[] }>("/items");
  return data.items;
}

export async function startSession(): Promise<string> {
  const data = await request<{ session_id: string }>("/sessions", { method: "POST" });
  return data.session_id;
}

export async function getRecommendations(
  sessionId: string,
  variant: Variant,
): Promise<Recommendation[]> {
  const data = await request<{ recommendations: Recommendation[] }>(
    "/sessions/" + encodeURIComponent(sessionId) + "/recommendations?variant=" + variant,
  );
  return data.recommendations;
}

export async function addRating(sessionId: string, rating: Rating): Promise<void> {
  await request("/sessions/" + encodeURIComponent(sessionId) + "/ratings", {
    method: "POST",
    body: JSON.stringify(rating),
  });
}

export async function resetSession(sessionId: string): Promise<void> {
  await request("/sessions/" + encodeURIComponent(sessionId) + "/reset", {
    method: "POST",
  });
}
