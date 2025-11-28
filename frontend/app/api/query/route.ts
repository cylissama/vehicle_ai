import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { question, top_k = 6, use_openai = true } = body;

    if (!question) {
      return NextResponse.json(
        { error: "Missing 'question' field." },
        { status: 400 }
      );
    }

    // FIX: Use the Docker service name via env var
    // Docker: "http://backend:8000"
    // Local: "http://127.0.0.1:8000"
    const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8000";
    
    // Construct the full URL
    const apiURL = `${backendUrl}/query`;

    const res = await fetch(apiURL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k, use_llm: use_openai }),
    });

    if (!res.ok) {
      const errorText = await res.text();
      console.error("FastAPI error:", errorText);
      return NextResponse.json(
        { error: "FastAPI backend error", details: errorText },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (err: any) {
    console.error("API route failure:", err);
    return NextResponse.json(
      { error: "Internal server error", details: err.message },
      { status: 500 }
    );
  }
}