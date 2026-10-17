import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GateKeeper AI — Command Center",
  description: "Edge person detection, restricted-zone alerts, and incident history.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
