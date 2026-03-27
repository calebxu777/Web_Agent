import "./globals.css";

export const metadata = {
  title: "Commerce Agent",
  description: "A high-performance Compound AI System for commerce",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
