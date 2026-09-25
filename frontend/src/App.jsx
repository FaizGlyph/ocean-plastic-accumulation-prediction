import { useState } from 'react'
import './App.css'

function App() {
  const [activePage, setActivePage] = useState('map')

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">🌊</div>
          <div>
            <h2>OceanTrace</h2>
            <span>Plastic Prediction System</span>
          </div>
        </div>

        <nav>
          <button
            className={activePage === 'map' ? 'nav-item active' : 'nav-item'}
            onClick={() => setActivePage('map')}
          >
            <span>◉</span>
            Prediction Map
          </button>

          <button
            className={activePage === 'analysis' ? 'nav-item active' : 'nav-item'}
            onClick={() => setActivePage('analysis')}
          >
            <span>▥</span>
            Model Analysis
          </button>

          <button
            className={activePage === 'methodology' ? 'nav-item active' : 'nav-item'}
            onClick={() => setActivePage('methodology')}
          >
            <span>◇</span>
            Methodology
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="project-status">
            <span className="status-dot"></span>
            Model pipeline active
          </div>

          <div className="version">
            CSE7102 · Semester VII
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="main">
        <header className="topbar">
          <div>
            <p className="eyebrow">CSE7102 · MINI PROJECT</p>
            <h1>OceanTrace</h1>
            <p className="project-subtitle">
              Machine Learning-Based Ocean Plastic Accumulation Prediction & GIS
              Visualization
            </p>
          </div>

          <div className="header-badge">
            <span></span>
            Research Prototype
          </div>
        </header>

        {activePage === 'map' && <PredictionMap />}
        {activePage === 'analysis' && <ModelAnalysis />}
        {activePage === 'methodology' && <Methodology />}
      </main>
    </div>
  )
}


/* =========================
   PREDICTION MAP
========================= */

function PredictionMap() {
  return (
    <div className="page">
      <div className="page-intro">
        <div>
          <h2>Prediction Map</h2>
          <p>
            Explore model-predicted plastic concentration across the study
            region.
          </p>
        </div>

        <div className="map-note">
          Predictions are model estimates, not direct plastic detection.
        </div>
      </div>

      {/* Statistics */}
      <div className="stats-grid">
        <StatCard value="614" label="Historical observations" />
        <StatCard value="12" label="ML features" />
        <StatCard value="6,615" label="Prediction grid points" />
        <StatCard value="5-fold" label="Cross validation" />
      </div>

      {/* Map */}
      <section className="map-card">
        <div className="card-header">
          <div>
            <h3>Accumulation Potential</h3>
            <p>Pacific Ocean study region</p>
          </div>

          <div className="legend">
            <span><i className="low"></i> Low</span>
            <span><i className="medium"></i> Medium</span>
            <span><i className="high"></i> High</span>
          </div>
        </div>

        <div className="map-container">
          <iframe
            src="/map.html"
            title="Ocean Plastic Prediction Map"
          ></iframe>
        </div>
      </section>

      {/* Support explanation */}
      <section className="info-grid">
        <div className="info-card">
          <div className="info-icon">◎</div>
          <div>
            <h3>Prediction Support</h3>
            <p>
              The project checks how close each prediction is to historical
              observations and how familiar its environmental conditions are
              to the model.
            </p>
          </div>
        </div>

        <div className="info-card">
          <div className="info-icon warning">!</div>
          <div>
            <h3>Extrapolation Warning</h3>
            <p>
              Areas outside the training-data range are flagged instead of
              being presented as equally reliable predictions.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}


/* =========================
   MODEL ANALYSIS
========================= */

function ModelAnalysis() {
  const features = [
    ['Wind speed', '26.9%'],
    ['Longitude', '14.3%'],
    ['Wave direction', '12.2%'],
    ['Wave height', '8.6%'],
    ['Current speed', '7.4%'],
    ['Month', '6.6%'],
  ]

  return (
    <div className="page">
      <div className="page-intro">
        <div>
          <h2>Model Analysis</h2>
          <p>
            Performance and feature information from the Random Forest model.
          </p>
        </div>
      </div>

      <div className="analysis-grid">
        <div className="metric-card">
          <span>Random CV MAE</span>
          <strong>1.9246</strong>
          <small>5-fold cross validation</small>
        </div>

        <div className="metric-card">
          <span>Random CV R²</span>
          <strong>0.5264</strong>
          <small>Mean across five folds</small>
        </div>

        <div className="metric-card caution-card">
          <span>Spatial CV MAE</span>
          <strong>2.7195</strong>
          <small>2° × 2° spatial blocks</small>
        </div>

        <div className="metric-card caution-card">
          <span>Spatial CV R²</span>
          <strong>-0.0939</strong>
          <small>Spatial generalization diagnostic</small>
        </div>
      </div>

      <section className="content-card">
        <div className="card-header">
          <div>
            <h3>Feature Importance</h3>
            <p>Random Forest feature importance</p>
          </div>
        </div>

        <div className="feature-list">
          {features.map(([name, value]) => (
            <div className="feature-row" key={name}>
              <div className="feature-name">
                <span>{name}</span>
                <strong>{value}</strong>
              </div>

              <div className="bar-background">
                <div
                  className="bar"
                  style={{ width: value }}
                ></div>
              </div>
            </div>
          ))}
        </div>

        <div className="analysis-note">
          <strong>Important:</strong> Feature importance indicates how the
          model used the variables. It does not prove that a feature causes
          higher plastic concentration.
        </div>
      </section>

      <section className="content-card">
        <div className="card-header">
          <div>
            <h3>Validation Interpretation</h3>
            <p>Why two validation approaches are shown</p>
          </div>
        </div>

        <div className="comparison">
          <div>
            <span className="comparison-label">Random CV</span>
            <p>
              Randomly separates observations into training and testing
              groups. This measures general predictive performance within the
              sampled data distribution.
            </p>
          </div>

          <div>
            <span className="comparison-label spatial">Spatial CV</span>
            <p>
              Separates observations using geographic blocks. The weaker
              spatial result indicates that predictions become more difficult
              when moving into different geographic areas.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}


/* =========================
   METHODOLOGY
========================= */

function Methodology() {
  const steps = [
    {
      number: '01',
      title: 'Historical plastic observations',
      text: '614 marine plastic observations provide the target concentration values.',
    },
    {
      number: '02',
      title: 'Environmental data',
      text: 'ERA5 wind and wave variables are combined with Copernicus GLOBCURRENT surface current data.',
    },
    {
      number: '03',
      title: 'Feature engineering',
      text: 'Wind speed, current speed and month are derived from the environmental data.',
    },
    {
      number: '04',
      title: 'Machine learning',
      text: 'A Random Forest regression model learns the relationship between environmental conditions and observed plastic concentration.',
    },
    {
      number: '05',
      title: 'Spatial prediction',
      text: 'The trained model generates predictions over a 6,615-point spatial grid.',
    },
    {
      number: '06',
      title: 'GIS visualization',
      text: 'Predictions are displayed as an interactive map with accumulation-potential categories and prediction-support information.',
    },
  ]

  return (
    <div className="page">
      <div className="page-intro">
        <div>
          <h2>Methodology</h2>
          <p>
            From historical observations to machine-learning-based spatial
            prediction.
          </p>
        </div>
      </div>

      <section className="content-card workflow-card">
        <div className="workflow">
          {steps.map((step) => (
            <div className="workflow-step" key={step.number}>
              <div className="step-number">{step.number}</div>

              <div>
                <h3>{step.title}</h3>
                <p>{step.text}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <div className="data-source-grid">
        <div className="source-card">
          <span className="source-label">PLASTIC DATA</span>
          <h3>614 observations</h3>
          <p>
            Historical microplastic measurements covering the Pacific Ocean
            study region.
          </p>
        </div>

        <div className="source-card">
          <span className="source-label">ERA5</span>
          <h3>Wind + Waves</h3>
          <p>
            10 m wind components, wave direction, wave period and significant
            wave height.
          </p>
        </div>

        <div className="source-card">
          <span className="source-label">GLOBCURRENT</span>
          <h3>Ocean Currents</h3>
          <p>
            Daily surface current velocity components used to derive current
            speed.
          </p>
        </div>
      </div>
    </div>
  )
}


/* =========================
   COMPONENTS
========================= */

function StatCard({ value, label }) {
  return (
    <div className="stat-card">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  )
}

export default App