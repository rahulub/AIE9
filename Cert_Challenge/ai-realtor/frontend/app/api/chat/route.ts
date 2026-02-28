/**
 * Proxies chat requests to the FastAPI backend and streams the response.
 * Ensures streaming works correctly (Next.js rewrites can sometimes buffer).
 */

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(req: Request) {
  const body = await req.json();
  const res = await fetch(`${BACKEND_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });

  if (!res.ok) {
    const err = await res.text();
    return new Response(err, { status: res.status });
  }

  const headers = new Headers();
  res.headers.forEach((value, key) => {
    if (key.toLowerCase() === "content-type" || key.toLowerCase() === "x-thread-id") {
      headers.set(key, value);
    }
  });

  return new Response(res.body, {
    status: res.status,
    headers,
  });
}
