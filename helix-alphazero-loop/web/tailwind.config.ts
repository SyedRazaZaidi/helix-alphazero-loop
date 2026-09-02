import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        void: "#07090c",
        signal: "#c8ff4a",
        copper: "#e8a87c",
        ash: "#9aa3b2",
        panel: "#10141c",
        line: "#1c2430",
        p1: "#c8ff4a",
        p2: "#5b8cff",
      },
      fontFamily: {
        display: ["var(--font-display)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
