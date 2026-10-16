
import { useState } from 'react'
import './App.css'

function App() {
  const [activePage, setActivePage] = useState('map')

  return (
    <div className="app">
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
          <button
            className={
              activePage === 'comparison'
                ? 'nav-item active'
                : 'nav-item'
            }
            onClick={() => setActivePage('comparison')}
          >
            <span>▤</span>
            Model Comparison
          </button>          
        </nav>

        <div className="sidebar-bottom">
          <div className="project-status">
            <span className="status-dot"></span>
            SVR prediction pipeline
          </div>
          <div className="version">CSE7102 · Semester VII</div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <p className="eyebrow">CSE7102 · MINI PROJECT</p>
            <h1>OceanTrace</h1>
            <p className="project-subtitle">
              Machine Learning-Based Ocean Plastic Accumulation Prediction
              &amp; GIS Visualization
            </p>
          </div>

          <div className="header-badge">
            <span></span>
            Research Prototype
          </div>
        </header>

        {activePage === 'map' && <PredictionMap />}
        {activePage === 'analysis' && <ModelAnalysis />}
        {activePage === 'comparison' && <ModelComparison />}
        {activePage === 'methodology' && <Methodology />}
      </main>
    </div>
  )
}

function PredictionMap() {
  return (
    <div className="page">
      <div className="page-intro">
        <div>
          <h2>Prediction Map</h2>
          <p>
            Explore SVR-predicted microplastic concentration across the
            Pacific Ocean study region.
          </p>
        </div>

        <div className="map-note">
          Model estimates only — not direct plastic detection.
        </div>
      </div>

      <div className="stats-grid">
        <StatCard value="614" label="Historical observations" />
        <StatCard value="12" label="Model input features" />
        <StatCard value="6,615" label="Prediction grid points" />
        <StatCard value="5-fold" label="Spatial cross-validation" />
      </div>

      <section className="map-card">
        <div className="card-header">
          <div>
            <h3>SVR-Predicted Concentration</h3>
            <p>Pacific Ocean · Predicted concentration in pieces/m³</p>
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
            title="SVR Ocean Plastic Prediction and Support Map"
            style={{ width: '100%', height: '100%', border: 0 }}
          />
        </div>
      </section>

      <section className="info-grid">
        <div className="info-card">
          <div className="info-icon">◎</div>
          <div>
            <h3>Prediction Support</h3>
            <p>
              Geographic distance and environmental similarity are used
              to identify locations that are better represented by the
              available training observations.
            </p>
          </div>
        </div>

        <div className="info-card">
          <div className="info-icon warning">!</div>
          <div>
            <h3>Extrapolation Warning</h3>
            <p>
              Support categories are diagnostic indicators, not statistical
              confidence estimates. Validation has not established a
              consistent increase in SVR error across all support categories.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}

function ModelAnalysis() {
  return (
    <div className="page">
      <div className="page-intro">
        <div>
          <h2>Model Analysis</h2>
          <p>
            Performance of the Support Vector Regression model using
            spatially separated validation folds.
          </p>
        </div>
      </div>

      <div className="analysis-grid">
        <div className="metric-card">
          <span>Out-of-fold MAE</span>
          <strong>2.5495</strong>
          <small>pieces/m³ · All held-out predictions</small>
        </div>

        <div className="metric-card">
          <span>Out-of-fold R²</span>
          <strong>0.2348</strong>
          <small>Pooled spatial cross-validation predictions</small>
        </div>

        <div className="metric-card caution-card">
          <span>SVR kernel</span>
          <strong>RBF</strong>
          <small>Nonlinear regression</small>
        </div>

        <div className="metric-card caution-card">
          <span>Validation</span>
          <strong>5-fold</strong>
          <small>Geographic blocks of 2° × 2°</small>
        </div>
      </div>

      <section className="content-card">
        <div className="card-header">
          <div>
            <h3>Model Configuration</h3>
            <p>Settings used by the current SVR pipeline</p>
          </div>
        </div>

        <div className="feature-list">
          {[
            ['Algorithm', 'Support Vector Regression'],
            ['Kernel', 'Radial Basis Function (RBF)'],
            ['Feature scaling', 'StandardScaler'],
            ['Regularization (C)', '10.0'],
            ['Epsilon', '0.1'],
            ['Input features', '12'],
            ['Training observations', '614'],
          ].map(([name, value]) => (
            <div className="feature-row" key={name}>
              <div className="feature-name">
                <span>{name}</span>
                <strong>{value}</strong>
              </div>
            </div>
          ))}
        </div>

        <div className="analysis-note">
          <strong>Metric interpretation:</strong> MAE measures the average
          absolute prediction error in pieces/m³. R² measures predictive
          performance relative to a mean-value baseline; it is not accuracy
          expressed as a percentage.
        </div>
      </section>

      <section className="content-card">
        <div className="card-header">
          <div>
            <h3>Spatial Validation</h3>
            <p>Testing generalization across geographic regions</p>
          </div>
        </div>

        <div className="comparison">
          <div>
            <span className="comparison-label">Out-of-fold MAE</span>
            <p>
              The pooled mean absolute error across all 614 held-out
              predictions is 2.5495 pieces/m³. Each prediction is produced
              by a model that did not train on that observation.
            </p>
          </div>

          <div>
            <span className="comparison-label spatial">Limitations</span>
            <p>
              The dataset contains a limited number of observations in a
              geographically restricted region. Predictions away from
              observed locations should be interpreted cautiously.
            </p>
          </div>
        </div>
      </section>

      <section className="content-card">
        <div className="card-header">
          <div>
            <h3>Support Error Validation</h3>
            <p>Exploratory comparison using nearest-grid support labels</p>
          </div>
        </div>

        <div className="feature-list">
          {[
            ['Supported · 555 samples', 'MAE 2.5251'],
            ['Caution · 35 samples', 'MAE 2.8742'],
            ['Extrapolation · 24 samples', 'MAE 2.6401'],
          ].map(([name, value]) => (
            <div className="feature-row" key={name}>
              <div className="feature-name">
                <span>{name}</span>
                <strong>{value}</strong>
              </div>
            </div>
          ))}
        </div>

        <div className="analysis-note">
          Support labels do not show a consistent increase in error from
          Supported to Caution to Extrapolation. The category sample sizes
          are imbalanced, so these results should be treated as exploratory.
        </div>
      </section>
    </div>
  )
}

function Methodology() {
  const steps = [
    {
      number: '01',
      title: 'Historical plastic observations',
      text: '614 marine plastic observations provide the target microplastic concentration values in pieces/m³.',
    },
    {
      number: '02',
      title: 'Environmental data',
      text: 'ERA5 wind and wave variables are combined with Copernicus GLOBCURRENT surface-current data.',
    },
    {
      number: '03',
      title: 'Feature engineering',
      text: 'Wind speed and current speed are derived from velocity components. Month, location and environmental variables form the model inputs.',
    },
    {
      number: '04',
      title: 'SVR model training',
      text: 'Support Vector Regression with an RBF kernel and standardized features learns relationships between the input variables and observed concentration.',
    },
    {
      number: '05',
      title: 'Spatial prediction',
      text: 'The trained SVR model generates predictions over a 6,615-point spatial grid using environmental conditions from the selected reference date.',
    },
    {
      number: '06',
      title: 'GIS visualization',
      text: 'Folium displays the predicted concentration heatmap alongside geographic and environmental support diagnostics.',
    },
  ]

  return (
    <div className="page">
      <div className="page-intro">
        <div>
          <h2>Methodology</h2>
          <p>
            From historical observations to machine-learning-based spatial
            prediction and GIS visualization.
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
            Historical microplastic measurements from the Pacific Ocean
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
            Daily surface-current velocity components used to derive
            current speed.
          </p>
        </div>

        <div className="source-card">
          <span className="source-label">MACHINE LEARNING</span>
          <h3>SVR with RBF kernel</h3>
          <p>
            Standardized environmental and geographic features are used
            to predict microplastic concentration.
          </p>
        </div>
      </div>
    </div>
  )
}

function StatCard({ value, label }) {
  return (
    <div className="stat-card">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  )
}

function ModelComparison() {
  const models = [
    {
      name: 'Random Forest',
      mae: 2.7195,
      r2: -0.0939,
      description: 'Ensemble of decision trees.',
    },
    {
      name: 'Gradient Boosting',
      mae: 2.8361,
      r2: -0.0603,
      description: 'Sequential boosting of decision trees.',
    },
    {
      name: 'Extra Trees',
      mae: 2.6049,
      r2: 0.1224,
      description: 'Ensemble of highly randomized trees.',
    },
    {
      name: 'Support Vector Regression (SVR)',
      mae: 2.5516,
      r2: 0.1558,
      description: 'RBF kernel with standardized input features.',
    },
  ]

  return (
    <div className="page">
      <div className="page-intro">
        <div>
          <h2>Model Comparison</h2>
          <p>
            Comparing four regression algorithms using recorded
            spatial cross-validation results.
          </p>
        </div>
      </div>

      <section className="content-card">
        <div className="card-header">
          <div>
            <h3>Spatial Validation Results</h3>
            <p>Lower MAE and higher R² are preferred.</p>
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            textAlign: 'left',
          }}>
            <thead>
              <tr>
                {['Model', 'Mean MAE ↓', 'Mean R² ↑'].map((heading) => (
                  <th key={heading} style={{
                    padding: '14px 10px',
                    borderBottom: '1px solid #29414a',
                  }}>
                    {heading}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map((model) => (
                <tr key={model.name}>
                  <td style={{
                    padding: '14px 10px',
                    borderBottom: '1px solid #203840',
                  }}>
                    <strong>{model.name}</strong>
                    {model.name.startsWith('Support Vector') && (
                      <span style={{
                        display: 'block',
                        color: '#43d9b4',
                        fontSize: '12px',
                        marginTop: '4px',
                      }}>
                        Current selected model
                      </span>
                    )}
                  </td>
                  <td style={{
                    padding: '14px 10px',
                    borderBottom: '1px solid #203840',
                  }}>
                    {model.mae.toFixed(4)}
                  </td>
                  <td style={{
                    padding: '14px 10px',
                    borderBottom: '1px solid #203840',
                  }}>
                    {model.r2.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="analysis-note">
          <strong>Interpretation:</strong> Based on the currently recorded
          results, SVR has the lowest mean spatial MAE and highest mean
          spatial R² among these four experiments.
        </div>
      </section>

      <section className="content-card">
        <div className="card-header">
          <div>
            <h3>Algorithm Overview</h3>
            <p>How the four models make predictions</p>
          </div>
        </div>

        <div className="feature-list">
          {models.map((model) => (
            <div className="feature-row" key={model.name}>
              <div className="feature-name">
                <span>{model.name}</span>
              </div>
              <p style={{ marginTop: '8px' }}>
                {model.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      <section className="content-card">
        <div className="card-header">
          <div>
            <h3>Validation Notes</h3>
            <p>Important limitations of this comparison</p>
          </div>
        </div>

        <ul>
          <li>
            The results are recorded fold averages from spatial
            cross-validation experiments.
          </li>
          <li>
            Identical held-out fold assignments must be verified before
            treating the ranking as a definitive comparison.
          </li>
          <li>
            R² is not percentage accuracy. Negative R² indicates performance
            below the mean-target baseline for the evaluated folds.
          </li>
          <li>
            The results describe this dataset and validation setup;
            they do not establish that one algorithm is universally best.
          </li>
        </ul>
      </section>
    </div>
  )
}

export default App
