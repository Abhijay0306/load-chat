// Cloudflare RAG Worker
// =====================
// Handles:
//   /chat         → Gemini Embedding 2 (3072d) pipeline
//   /health       → Health check

export interface Env {
  SUPABASE_URL: string;
  SUPABASE_KEY: string;
  DEEPSEEK_API_KEY: string;
  GOOGLE_API_KEY: string;
}

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
  });
}

// ── Embed query via Gemini Embedding 2 API (3072-dim) ──
async function embedQueryGemini(apiKey: string, text: string): Promise<number[]> {
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:embedContent?key=${apiKey}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "models/gemini-embedding-2-preview",
        content: { parts: [{ text }] },
        taskType: "RETRIEVAL_QUERY",
      }),
    }
  );
  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Gemini Embed error: ${response.status} ${err}`);
  }
  const data = (await response.json()) as { embedding: { values: number[] } };
  return data.embedding.values;
}

// ── Retrieve relevant chunks from Supabase ──
async function retrieveChunks(
  supabaseUrl: string,
  supabaseKey: string,
  queryEmbedding: number[],
  rpcName: string,
  topK = 5
): Promise<{ content: string; source: string; similarity: number }[]> {
  const response = await fetch(`${supabaseUrl}/rest/v1/rpc/${rpcName}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      apikey: supabaseKey,
      Authorization: `Bearer ${supabaseKey}`,
    },
    body: JSON.stringify({
      query_embedding: queryEmbedding,
      match_threshold: 0.1,
      match_count: topK,
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Supabase retrieval failed: ${response.status} ${err}`);
  }

  const rows = (await response.json()) as {
    content: string;
    metadata: { source: string };
    similarity: number;
  }[];

  return rows.map((r) => ({
    content: r.content,
    source: r.metadata?.source ?? "unknown",
    similarity: r.similarity,
  }));
}

// ── Strip base64 images from text before sending to LLM ──
function stripBase64Images(text: string): string {
  return text.replace(/!\[([^\]]*)\]\(data:image\/[^;]+;base64,[A-Za-z0-9+/=]+\)/g,
    (_match, alt) => `\n[📷 DIAGRAM/IMAGE${alt ? ': ' + alt : ''}]\n`
  );
}

// ── Generate answer using DeepSeek ──
async function generateAnswer(
  apiKey: string,
  query: string,
  chunks: { content: string; source: string }[]
): Promise<{ answer: string; inputTokens: number; outputTokens: number }> {
  const context = chunks
    .map((c, i) => `[Source ${i + 1}: ${c.source}]\n${stripBase64Images(c.content)}`)
    .join("\n\n---\n\n");

  const systemPrompt = `You are Miss MoMo, a professional technical assistant for Load Controls Inc.
You have two modes of operation:
1. **General Greeting/Conversation**: If the user says "Hello", "How are you", or other non-technical pleasantries, respond warmly and professionally as Miss MoMo. You do NOT need to cite sources for small talk.
2. **Technical Product Support**: If the user asks about products, installation, specifications, or company technology:
   - Use ONLY the provided context below to answer.
   - Be clear, detailed, and cite the document names you used.
   - Where you see [📷 DIAGRAM/IMAGE] in the context, mention that a wiring diagram or figure is shown below the answer.
   - If the answer isn't in the context, politely say: "I couldn't find specific details for that in our current documentation, but I can help with other Load Controls products."
   - Do NOT invent specifications.`;

  const userMessage = `Context from documentation:\n\n${context}\n\n---\n\nUser Question: ${query}`;

  const response = await fetch("https://api.deepseek.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "deepseek-chat",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userMessage },
      ],
      max_tokens: 1024,
      temperature: 0.1,
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`DeepSeek API error: ${response.status} ${err}`);
  }

  const data = (await response.json()) as {
    choices: { message: { content: string } }[];
    usage: { prompt_tokens: number; completion_tokens: number };
  };

  return {
    answer: data.choices[0].message.content,
    inputTokens: data.usage?.prompt_tokens ?? 0,
    outputTokens: data.usage?.completion_tokens ?? 0,
  };
}

// ── Shared chat handler ──
async function handleChat(
  request: Request,
  env: Env
): Promise<Response> {
  const body = (await request.json()) as { query?: string };
  const query = body?.query?.trim();

  if (!query) {
    return jsonResponse({ error: "Missing 'query' field in request body" }, 400);
  }

  // 1. Embed the query with Gemini 2 (3072d)
  let queryEmbedding: number[];
  try {
    queryEmbedding = await embedQueryGemini(env.GOOGLE_API_KEY, query);
    console.log("[Step 1] Gemini embed OK, dims:", queryEmbedding.length);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("[Step 1 FAILED] Gemini embed error:", msg);
    throw new Error(`Gemini embed failed: ${msg}`);
  }

  // 2. Retrieve relevant chunks from documents_gemini table
  const rpcName = "match_documents_gemini";
  let chunks: { content: string; source: string; similarity: number }[];
  try {
    chunks = await retrieveChunks(
      env.SUPABASE_URL,
      env.SUPABASE_KEY,
      queryEmbedding,
      rpcName,
      5
    );
    console.log("[Step 2] Supabase retrieve OK, chunks:", chunks.length);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("[Step 2 FAILED] Supabase retrieve error:", msg);
    throw new Error(`Supabase retrieve failed: ${msg}`);
  }

  if (chunks.length === 0) {
    return jsonResponse({
      answer: "I couldn't find any relevant information to answer your question. Please make sure documents have been ingested into the Gemini-powered knowledge base.",
      sources: [],
      input_tokens: 0,
      output_tokens: 0,
      engine: "gemini-embedding-2",
    });
  }

  // 3. Generate answer with DeepSeek
  let answer: string, inputTokens: number, outputTokens: number;
  try {
    ({ answer, inputTokens, outputTokens } = await generateAnswer(
      env.DEEPSEEK_API_KEY,
      query,
      chunks
    ));
    console.log("[Step 3] DeepSeek answer OK, tokens:", inputTokens, outputTokens);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("[Step 3 FAILED] DeepSeek error:", msg);
    throw new Error(`DeepSeek failed: ${msg}`);
  }

  return jsonResponse({
    answer,
    sources: [...new Set(chunks.map((c) => c.source))],
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    engine: "gemini-embedding-2",
    rich_chunks: chunks.map((c) => c.content),
  });
}

// ── Main Worker Handler ──
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS_HEADERS });
    }

    if (url.pathname === "/health") {
      return jsonResponse({ status: "ok", timestamp: new Date().toISOString() });
    }

    if (url.pathname === "/chat" && request.method === "POST") {
      try {
        return await handleChat(request, env);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error("Chat error:", message);
        return jsonResponse({ error: `Server error: ${message}` }, 500);
      }
    }

    return jsonResponse({ error: "Not Found" }, 404);
  },
};
