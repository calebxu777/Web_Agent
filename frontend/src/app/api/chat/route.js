/**
 * Next.js API Route — Proxy to FastAPI backend
 * ==============================================
 * Forwards chat requests to the Python backend at localhost:8000
 * and streams SSE responses back to the browser.
 *
 * This replaces the old mock-only route.js that simulated everything
 * in JavaScript. Now the real Python CommerceAgent handles it
 * (with its own mock_mode support).
 */

export async function POST(req) {
  const body = await req.json();

  const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

  try {
    const backendResponse = await fetch(`${BACKEND_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
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
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    // Stream the SSE response directly through to the browser
    return new Response(backendResponse.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (err) {
    console.error("[API Proxy] Failed to reach backend:", err.message);

    // Provide a helpful error that streams as SSE so the frontend can display it
    const encoder = new TextEncoder();
    const errorStream = new ReadableStream({
      start(controller) {
        const errorEvent = JSON.stringify({
          type: "error",
          message: `Could not connect to backend at ${BACKEND_URL}. Is the FastAPI server running? (uvicorn src.api:app --port 8000)`,
        });
        controller.enqueue(
          encoder.encode(`data: ${errorEvent}\n\n`)
        );
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      },
    });

    return new Response(errorStream, {
      status: 200, // Return 200 so the SSE parser works
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }
}
