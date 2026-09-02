import type { Metadata } from "next";
import { IBM_Plex_Mono, Syne } from "next/font/google";
import "./globals.css";

const display = Syne({ subsets: ["latin"], variable: "--font-display" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "Helix — AlphaZero loop on four games",
  description: "Self-play, MCTS, policy-value nets for Connect Four, Gomoku, Hex, and Othello. Not a chatbot.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${display.variable} ${mono.variable} font-display antialiased`}>{children}</body>
    </html>
  );
}
