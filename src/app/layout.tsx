import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "中古車価格予測AI",
  description: "中古車の価格をAIで予測します",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <body>
        {/* children を直接レンダリング */}
        {children}
      </body>
    </html>
  );
}
