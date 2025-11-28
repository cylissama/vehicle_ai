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

    // Forward to FastAPI server
    const fastapiRes = await fetch("http://localhost:8000/ingest_url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });

    const data = await fastapiRes.json();

    if (!fastapiRes.ok) {
      return NextResponse.json({ error: data.error }, { status: fastapiRes.status });
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