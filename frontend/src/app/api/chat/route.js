import { NextResponse } from "next/server";

export async function POST(req) {
  try {
    const body = await req.json();
    const { message, hasImage, webSearch } = body;

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        const sendEvent = (data) => {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
        };
        const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

        // ---- Pipeline simulation ----
        sendEvent({ type: "status", stage: "cold_start", message: "Loading the model for cold start..." });
        await sleep(1200);

        sendEvent({ type: "status", stage: "intent_detection", message: "Understanding your request..." });
        await sleep(800);

        if (hasImage) {
          sendEvent({ type: "status", stage: "analyzing_image", message: "Analyzing your image with Florence-2..." });
          await sleep(1500);
        }

        sendEvent({ type: "status", stage: "decomposing_query", message: "Breaking down your query..." });
        await sleep(600);

        sendEvent({ type: "status", stage: "sourcing_matches", message: "Sourcing matches from local catalog..." });
        await sleep(1000);

        const isSearch = message && (
          message.toLowerCase().includes("recommend") ||
          message.toLowerCase().includes("find") ||
          message.toLowerCase().includes("looking") ||
          message.toLowerCase().includes("search") ||
          message.toLowerCase().includes("show") ||
          message.toLowerCase().includes("want") ||
          message.toLowerCase().includes("need") ||
          hasImage
        );

        if (webSearch && isSearch) {
          sendEvent({ type: "status", stage: "web_search", message: "Searching the web via Firecrawl..." });
          await sleep(2000);
          sendEvent({ type: "status", stage: "web_scraping", message: "Extracting product data from web results..." });
          await sleep(1200);
        }

        sendEvent({ type: "status", stage: "reranking", message: "Ranking products by relevance..." });
        await sleep(600);

        if (hasImage) {
          sendEvent({ type: "status", stage: "verifying", message: "Verifying visual matches..." });
          await sleep(800);
        }

        // ---- Products with images + descriptions ----
        if (isSearch) {
          const items = [
            {
              id: 1,
              title: "Premium Wool Overcoat",
              price: 199.99,
              source: "local",
              image: "https://images.unsplash.com/photo-1539533018447-63fcce2678e3?w=200&h=200&fit=crop",
              description: "Tailored double-breasted overcoat in Italian merino wool. Notch lapel, fully lined interior.",
            },
            {
              id: 2,
              title: "Minimalist Chelsea Boots",
              price: 150.0,
              source: "local",
              image: "https://images.unsplash.com/photo-1638247025967-b4e38f787b76?w=200&h=200&fit=crop",
              description: "Sleek suede Chelsea boots with elastic side panels. Comfortable cushioned insole.",
            },
          ];

          if (webSearch) {
            items.push(
              {
                id: 3,
                title: "Arc'teryx Therme Parka",
                price: 349.0,
                source: "web",
                url: "https://arcteryx.com",
                image: "https://images.unsplash.com/photo-1544923246-77307dd270b0?w=200&h=200&fit=crop",
                description: "GORE-TEX insulated parka with 750-fill down. Windproof and waterproof for extreme cold.",
              },
              {
                id: 4,
                title: "Everlane ReNew Long Puffer",
                price: 178.0,
                source: "web",
                url: "https://everlane.com",
                image: "https://images.unsplash.com/photo-1608063615781-e2ef8c73d0e4?w=200&h=200&fit=crop",
                description: "Made from 100% recycled polyester. Lightweight yet warm, perfect for daily wear.",
              }
            );
          } else {
            items.push({
              id: 3,
              title: "Classic Leather Tote Bag",
              price: 89.99,
              source: "local",
              image: "https://images.unsplash.com/photo-1590874103328-eac38a683ce7?w=200&h=200&fit=crop",
              description: "Full-grain vegetable-tanned leather tote with interior zip pocket. Ages beautifully.",
            });
          }

          sendEvent({ type: "products", items });
        }

        // ---- Generate ----
        sendEvent({ type: "status", stage: "generating", message: "Generating your personalized recommendations..." });
        await sleep(500);

        const sampleResponse = isSearch
          ? webSearch
            ? "I've searched both our catalog and the web to find you the best options. The Premium Wool Overcoat is our top local pick — Italian merino wool with a tailored silhouette that works for both office and weekend wear. From the web, the Arc'teryx Therme Parka is the premium choice if you need serious weather protection, while the Everlane ReNew Puffer offers excellent sustainability at a more accessible price. Want me to compare any of these in more detail?"
            : "Great picks from our catalog! The Premium Wool Overcoat stands out for its exceptional warmth and tailored fit — perfect for both professional and casual settings. The Minimalist Chelsea Boots pair beautifully with it for a complete look, and the Classic Leather Tote rounds things out as a versatile daily carry that develops gorgeous patina over time. Would you like me to narrow things down by budget or style?"
          : "Hey there! I'm your personal shopping assistant. I can help you discover products by describing what you're looking for, or you can upload a photo and I'll find similar items. Toggle on 'Local + Web' search above the input to also search the broader web. What are you shopping for today?";

        const words = sampleResponse.split(" ");
        for (const word of words) {
          sendEvent({ type: "token", content: word + " " });
          await sleep(35);
        }

        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (err) {
    return NextResponse.json({ error: "Failed to process chat" }, { status: 500 });
  }
}
