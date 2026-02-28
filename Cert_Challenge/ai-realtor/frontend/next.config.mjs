/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  async rewrites() {
    return [
      { source: "/api/ingest", destination: "http://localhost:8000/api/ingest" },
    ]
  },
}

export default nextConfig
