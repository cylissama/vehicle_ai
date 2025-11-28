// app/layout.tsx
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import './home.css';
import NavBar from '../components/NavBar';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Vehicle Maintenance RAG',
  description: 'AI-powered vehicle maintenance assistant using RAG technology',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen flex flex-col bg-gray-50">
          {/* Navigation Bar */}
          <NavBar />
          
          {/* Main Content */}
          <main className="flex-1">
            {children}
          </main>
          
          {/* Footer */}
          <footer className="bg-gray-800 text-gray-300 py-6 mt-auto">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
                <div className="text-sm">
                  © {new Date().getFullYear()} Vehicle Maintenance RAG. Powered by AI.
                </div>
                <div className="flex space-x-6 text-sm">
                  <a href="#" className="hover:text-white transition-colors">
                    Documentation
                  </a>
                  <a href="#" className="hover:text-white transition-colors">
                    GitHub
                  </a>
                  <a href="#" className="hover:text-white transition-colors">
                    API Status
                  </a>
                </div>
              </div>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}