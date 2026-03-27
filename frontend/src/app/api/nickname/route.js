/**
 * Nickname API proxy routes.
 * POST /api/nickname — register or login
 */
export async function POST(req) {
  const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
  try {
    const body = await req.json();
    const res = await fetch(`${BACKEND_URL}/api/nickname`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    return new Response(JSON.stringify(data), {
      status: res.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    return new Response(
      JSON.stringify({ error: "Backend unavailable" }),
      { status: 502, headers: { "Content-Type": "application/json" } }
    );
  }
}
