import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const file = formData.get("file");

    if (!file || !(file instanceof Blob)) {
      return NextResponse.json(
        { error: "No PDF file uploaded." },
        { status: 400 }
      );
    }

    // FIX: Use the environment variable defined in docker-compose.yml
    // In Docker, this resolves to "http://backend:8000"
    // Locally, it falls back to localhost
    const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8000";

    // Note: Make sure your FastAPI endpoint is actually named "/ingest"
    const backendRes = await fetch(`${backendUrl}/ingest`, {
      method: "POST",
      body: formData, 
      // specific headers are usually not needed for FormData as fetch sets the boundary automatically
    });

    // Check if the response body is empty before parsing JSON
    const text = await backendRes.text();
    let data;
    try {
        data = text ? JSON.parse(text) : {};
    } catch (e) {
        console.error("Failed to parse backend response:", text);
        return NextResponse.json({ error: "Invalid response from backend" }, { status: 500 });
    }

    if (!backendRes.ok) {
      return NextResponse.json(
        { error: data.error || "Ingest failed" },
        { status: backendRes.status }
      );
    }

    return NextResponse.json(data);
  } catch (err: any) {
    console.error("Ingest API error:", err);
    return NextResponse.json(
      { error: "Internal Server Error", details: err.message },
      { status: 500 }
    );
  }
}