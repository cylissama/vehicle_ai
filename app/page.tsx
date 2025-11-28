// app/page.tsx
import Link from 'next/link';
import './home.css';

export default function Home() {
  return (
    <div className="home-container">
      {/* Hero Section */}
      <div className="hero-section">
        <div className="hero-content">
          <div className="hero-text">
            <h1 className="hero-title">
              Vehicle Maintenance Assistant
            </h1>
            <p className="hero-subtitle">
              Get instant answers from your vehicle manuals using AI-powered Retrieval Augmented Generation
            </p>
            <div className="hero-buttons">
              <Link href="/ask" className="btn btn-primary">
                Ask a Question
              </Link>
              <Link href="/ingest" className="btn btn-secondary">
                Upload Documents
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="features-section">
        <h2 className="section-title">How It Works</h2>
        <div className="features-grid">
          {/* Feature 1 */}
          <div className="feature-card">
            <h3 className="feature-title">1. Upload Documents</h3>
            <p className="feature-description">
              Upload your vehicle manuals, service guides, and maintenance documents in PDF format.
            </p>
          </div>

          {/* Feature 2 */}
          <div className="feature-card">
            <h3 className="feature-title">2. AI Processing</h3>
            <p className="feature-description">
              Our system extracts, chunks, and indexes your documents using advanced embedding models.
            </p>
          </div>

          {/* Feature 3 */}
          <div className="feature-card">
            <h3 className="feature-title">3. Ask Questions</h3>
            <p className="feature-description">
              Ask questions in natural language and get accurate answers with source citations.
            </p>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="stats-section">
        <div className="stats-container">
          <div className="stats-grid">
            <div className="stat-item">
              <div className="stat-value">AI-Powered</div>
              <div className="stat-label">Vector Search</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">OCR</div>
              <div className="stat-label">Scanned Documents</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">Instant</div>
              <div className="stat-label">Answers</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">Source</div>
              <div className="stat-label">Citations</div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="cta-section">
        <h2 className="cta-title">Ready to Get Started?</h2>
        <p className="cta-subtitle">
          Upload your first document or ask a question about your vehicle
        </p>
        <div className="cta-buttons">
          <Link href="/ask" className="btn btn-primary">
            Start Asking Questions
          </Link>
          <Link href="/ingest" className="btn btn-outline">
            Upload Your First Document
          </Link>
        </div>
      </div>
    </div>
  );
}