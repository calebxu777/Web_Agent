/**
 * GET /api/nickname/[name]/check — Check if a nickname exists
 */
export async function GET(req, { params }) {
  const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
  const { name } = await params;
  try {
    const res = await fetch(
      `${BACKEND_URL}/api/nickname/${encodeURIComponent(name)}/check`
    );
    const data = await res.json();
    return new Response(JSON.stringify(data), {
      status: res.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    return new Response(
      JSON.stringify({ error: "Backend unavailable", exists: false }),
      { status: 502, headers: { "Content-Type": "application/json" } }
    );
  }
}
