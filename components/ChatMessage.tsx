export default function ChatMessage({ role, content }: any) {
  const isUser = role === "user";

  return (
    <div
      className={`p-3 rounded-lg max-w-[80%] ${
        isUser ? "bg-blue-100 ml-auto" : "bg-gray-100"
      }`}
    >
      {content}
    </div>
  );
}