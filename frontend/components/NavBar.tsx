// components/NavBar.tsx
'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import './NavBar.css';

export default function NavBar() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { name: 'Home', path: '/' },
    { name: 'Ask Question', path: '/ask' },
    { name: 'Ingest Documents', path: '/ingest' },
  ];

  const isActive = (path: string) => {
    return pathname === path;
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-content">
          {/* Logo and Desktop Navigation */}
          <div className="navbar-left">
            {/* Logo */}
            <Link href="/" className="navbar-logo">
              <span className="logo-text">Vehicle Maintenance RAG</span>
            </Link>

            {/* Desktop Navigation */}
            <div className="nav-desktop">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  href={item.path}
                  className={`nav-link ${isActive(item.path) ? 'active' : ''}`}
                >
                  {item.name}
                </Link>
              ))}
            </div>
          </div>

          {/* Status Indicator (Desktop) */}
          <div className="status-desktop">
            <div className="status-badge">
              <div className="status-dot"></div>
              <span className="status-text">API Connected</span>
            </div>
          </div>

          {/* Mobile menu button */}
          <div className="mobile-menu-button">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="menu-toggle"
              aria-label="Toggle menu"
            >
              {!mobileMenuOpen ? (
                <svg className="menu-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              ) : (
                <svg className="menu-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileMenuOpen && (
        <div className="mobile-menu">
          <div className="mobile-menu-items">
            {navItems.map((item) => (
              <Link
                key={item.path}
                href={item.path}
                onClick={() => setMobileMenuOpen(false)}
                className={`mobile-nav-link ${isActive(item.path) ? 'active' : ''}`}
              >
                {item.name}
              </Link>
            ))}
          </div>
          {/* Mobile Status */}
          <div className="mobile-status">
            <div className="status-dot"></div>
            <span className="status-text">API Connected</span>
          </div>
        </div>
      )}
    </nav>
  );
}