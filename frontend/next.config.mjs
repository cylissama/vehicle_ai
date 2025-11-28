/** @type {import('next').NextConfig} */
const nextConfig = {
  // This is required for Docker to work properly
  output: "standalone", 
  
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        // Docker sees "http://backend:8000", Local sees "http://127.0.0.1:8000"
        destination: process.env.BACKEND_URL 
          ? `${process.env.BACKEND_URL}/:path*` 
          : 'http://127.0.0.1:8000/:path*',
      },
    ]
  },
};

export default nextConfig;