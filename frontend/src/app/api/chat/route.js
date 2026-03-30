/**
 * Next.js API Route - Proxy to FastAPI backend
 * ============================================
 * Streams SSE responses from the Python backend to the browser and ensures
 * each browser session gets a stable session_id cookie. This lets the MVP
 * reuse Redis-backed short-horizon memory even when the UI does not pass an
 * explicit session_id on each turn.
 */

const SESSION_COOKIE = "commerce_session_id";
const SESSION_COOKIE_MAX_AGE = 60 * 60 * 24;

function parseCookies(cookieHeader) {
  const result = {};
  if (!cookieHeader) return result;

  for (const part of cookieHeader.split(";")) {
    const [rawKey, ...rawValueParts] = part.split("=");
    const key = rawKey?.trim();
    if (!key) continue;
    result[key] = decodeURIComponent(rawValueParts.join("=").trim());
  }
  return result;
}

function buildProxyHeaders(sessionId) {
  const headers = new Headers({
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });
  headers.append(
    "Set-Cookie",
    `${SESSION_COOKIE}=${encodeURIComponent(sessionId)}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${SESSION_COOKIE_MAX_AGE}`
  );
  return headers;
}

export async function POST(req) {
  const body = await req.json();
  const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
  const parsedCookies = parseCookies(req.headers.get("cookie") || "");
  const sessionId = String(body?.session_id || parsedCookies[SESSION_COOKIE] || crypto.randomUUID()).trim();
  const payload = {
    ...body,
    session_id: sessionId,
  };

  try {
    const backendResponse = await fetch(`${BACKEND_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!backendResponse.ok) {
      const errText = await backendResponse.text();
      return new Response(
        JSON.stringify({
          error: `Backend error: ${backendResponse.status}`,
          detail: errText,
        }),
        {
          status: backendResponse.status,
          headers: {
            "Content-Type": "application/json",
            "Set-Cookie": `${SESSION_COOKIE}=${encodeURIComponent(sessionId)}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${SESSION_COOKIE_MAX_AGE}`,
          },
        }
      );
    }

    return new Response(backendResponse.body, {
      status: 200,
      headers: buildProxyHeaders(sessionId),
    });
  } catch (err) {
    console.error("[API Proxy] Failed to reach backend:", err.message);

    const encoder = new TextEncoder();
    const errorStream = new ReadableStream({
      start(controller) {
        const errorEvent = JSON.stringify({
          type: "error",
          message: `Could not connect to backend at ${BACKEND_URL}. Is the FastAPI server running? (uvicorn src.api:app --port 8000)`,
        });
        controller.enqueue(encoder.encode(`data: ${errorEvent}\n\n`));
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      },
    });

    return new Response(errorStream, {
      status: 200,
      headers: buildProxyHeaders(sessionId),
    });
  }
}
