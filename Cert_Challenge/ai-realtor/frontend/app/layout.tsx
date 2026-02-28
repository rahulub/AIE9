import type { Metadata } from "next";
import { Inter, DM_Serif_Display } from "next/font/google";
import "./globals.css";

const fontSans = Inter({ subsets: ["latin"], variable: "--font-sans" });
const fontSerif = DM_Serif_Display({ weight: "400", subsets: ["latin"], variable: "--font-serif" });

export const metadata: Metadata = {
  title: "AI Realtor - Property Analyzer",
  description: "AI-powered property inspection report analysis. Get expert insights on your home inspection findings and buyer priorities.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${fontSans.variable} ${fontSerif.variable}`}>
      <body
        className="antialiased"
      >
        {children}
      </body>
    </html>
  );
}
