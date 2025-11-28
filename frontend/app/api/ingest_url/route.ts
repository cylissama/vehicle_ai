import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const { url } = await req.json();

    if (!url) {
      return NextResponse.json(
        { error: "Missing 'url' in request body." },
        { status: 400 }
      );
    }

    // FIX: Get the backend URL from the environment (defined in docker-compose)
    // In Docker: "http://backend:8000"
    // Local fallback: "http://127.0.0.1:8000"
    const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8000";

    // FIX: Use the absolute URL. 
    // We assume the FastAPI endpoint is "/ingest_url" (without the /api prefix).
    const fastapiRes = await fetch(`${backendUrl}/ingest_url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });

    const data = await fastapiRes.json();

    if (!fastapiRes.ok) {
      return NextResponse.json(
        { error: data.error || "Backend error" }, 
        { status: fastapiRes.status }
      );
    }

    return NextResponse.json(data, { status: 200 });

  } catch (err: any) {
    console.error("Ingest URL error:", err);
    return NextResponse.json(
      { error: "Server error or FastAPI unreachable." },
      { status: 500 }
    );
  }
}