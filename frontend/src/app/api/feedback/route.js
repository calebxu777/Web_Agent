/**
 * POST /api/feedback
 *
 * Receives user product feedback (upvote/downvote).
 * Upvoted products are forwarded to the Python backend's /api/ingest endpoint.
 */
export async function POST(req) {
  const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

  try {
    const body = await req.json();
    const { product, vote } = body;

    console.log(`[Feedback] ${vote === "up" ? "👍" : "👎"} ${product?.title}`);

    // Forward upvoted products to the backend for ingestion
    if (vote === "up") {
      try {
        const ingestRes = await fetch(`${BACKEND_URL}/api/ingest`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ product_data: product }),
        });
        const ingestData = await ingestRes.json();
        console.log(`[Feedback] Ingest result:`, ingestData);

        return new Response(
          JSON.stringify({
            success: true,
            message: "Product ingested into catalog",
            ingest: ingestData,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      } catch (err) {
        console.error("[Feedback] Backend ingest failed:", err.message);
        return new Response(
          JSON.stringify({
            success: false,
            message: "Feedback recorded but ingest failed — is the backend running?",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
    }

    // Downvote — just acknowledge
    return new Response(
      JSON.stringify({
        success: true,
        message: "Feedback recorded",
        vote,
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  } catch (err) {
    console.error("[Feedback] Error:", err);
    return new Response(
      JSON.stringify({ error: "Failed to process feedback" }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
