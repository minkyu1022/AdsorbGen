/** @type {import('next').NextConfig} */

// When served behind a reverse-proxy subpath (e.g. code-server's
// /proxy/3000/), set NEXT_PUBLIC_ASSET_PREFIX to the full external prefix so
// /_next/* asset URLs resolve under the proxy instead of the host root.
// Empty (default) = served at root.
const assetPrefix = process.env.NEXT_PUBLIC_ASSET_PREFIX || "";

const nextConfig = {
  reactStrictMode: true,
  assetPrefix,
  async rewrites() {
    // Proxy /api/* to the FastAPI backend during dev.
    const backend = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};

export default nextConfig;
