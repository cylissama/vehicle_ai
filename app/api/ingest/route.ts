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

    const backendRes = await fetch("http://localhost:8000/ingest", {
      method: "POST",
      body: formData, // send multipart form-data straight through
    });

    const data = await backendRes.json();

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