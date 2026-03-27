import { NextResponse } from "next/server";

/**
 * POST /api/feedback
 *
 * Receives user product feedback (upvote/downvote).
 * Upvoted web products are queued for async ingestion into the local catalog.
 *
 * Body:
 *   { product: { title, description, price, image, url, source }, vote: "up" | "down" }
 *
 * In production, this forwards to the Python agent server.
 * Currently mocks the response while logging the payload.
 */
export async function POST(req) {
  try {
    const body = await req.json();
    const { product, vote } = body;

    console.log(`[Feedback] ${vote === "up" ? "👍" : "👎"} ${product?.title}`);
    console.log(`[Feedback] Source: ${product?.source}, URL: ${product?.url}`);

    // In production: forward to Python backend for async ingestion
    // const backendUrl = process.env.AGENT_API_URL || "http://localhost:8000";
    // if (vote === "up" && product?.source === "web") {
    //   fetch(`${backendUrl}/api/ingest`, {
    //     method: "POST",
    //     headers: { "Content-Type": "application/json" },
    //     body: JSON.stringify({ product_data: product }),
    //   }).catch((err) => console.error("[Feedback] Backend ingest failed:", err));
    // }

    return NextResponse.json({
      success: true,
      message:
        vote === "up"
          ? "Product queued for catalog ingestion"
          : "Feedback recorded",
      product_id: product?.id,
      vote,
    });
  } catch (err) {
    console.error("[Feedback] Error:", err);
    return NextResponse.json(
      { error: "Failed to process feedback" },
      { status: 500 }
    );
  }
}
