/**
 * Proxies thread delete to the FastAPI backend.
 */

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ threadId: string }> }
) {
  const { threadId } = await params;
  const res = await fetch(`${BACKEND_URL}/api/chat/thread/${threadId}`, {
    method: "DELETE",
  });

  const data = await res.json().catch(() => ({}));
  return Response.json(data, { status: res.status });
}
