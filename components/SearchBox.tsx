"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export default function SearchBox() {
  const router = useRouter();
  const [query, setQuery] = useState("");

  return (
    <div className="flex gap-3 w-full max-w-xl">
      <input
        className="flex-1 border px-3 py-2 rounded"
        placeholder="Search your manual…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button
        onClick={() => router.push(`/ask?query=${query}`)}
        className="bg-blue-600 text-white px-4 py-2 rounded"
      >
        Go
      </button>
    </div>
  );
}