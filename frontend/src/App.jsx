
import { useEffect, useState } from "react";

import {
  Activity,
  Brain,
  LayoutDashboard,
  Microscope,
  FlaskConical,
  Settings,
  Bell,
  Search,
  Users,
  Dna,
  Target,
  Menu,
  ChevronRight,
} from "lucide-react";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

import "./App.css";

// --------------------------------------------------
// API CONFIGURATION
// --------------------------------------------------

const API_BASE = "http://127.0.0.1:8001";

// --------------------------------------------------
// INITIAL PREDICTION FORM VALUES
// --------------------------------------------------

const initialForm = {
  "Female Age": 34,
  AMH: 2.5,
  "Male Age": 36,
  "Sperm count": 35,
  "Sperm motility": 45,
  "Sperm morphology": 0.05,
  "Oocyte Maturity MII": 8,
  "Oocyte Maturity MI": 2,
  "Oocyte Maturity others": 1,
  "Number of fertilized eggs": 6,
  "Embryo Grade 1": 3,
  "Embryo Grade 2": 2,
  "Embryo Grade 3": 0,
  "Embryo Grade 4": 0,
  "Number of embryos": 2,
};

// --------------------------------------------------
// NAVIGATION
// --------------------------------------------------

const navigation = [
  { name: "Overview", icon: LayoutDashboard },
  { name: "Prediction", icon: Brain },
  { name: "Model Analytics", icon: Activity },
  { name: "Explainability", icon: Microscope },
  { name: "Research", icon: FlaskConical },
];

// --------------------------------------------------
// DATA NORMALIZATION
// --------------------------------------------------


function normalizeModelMetrics(raw) {
  let entries = [];

  if (Array.isArray(raw)) {
    entries = raw;
  } else if (raw && typeof raw === "object") {
    entries = Object.entries(raw).map(([name, metrics]) => ({
      name,
      ...(typeof metrics === "object" && metrics !== null
        ? metrics
        : { roc_auc: metrics }),
    }));
  }

  return entries
    .map((model) => {
      const name =
        model.name ||
        model.model ||
        model.model_name ||
        model.algorithm ||
        "Unknown model";

      const aucKey = Object.keys(model).find(
        (key) =>
          key.toLowerCase().replace(/[^a-z]/g, "") === "rocauc"
      );

      return {
        name,
        auc: aucKey ? Number(model[aucKey]) : NaN,
      };
    })
    .filter(
      (item) =>
        item.name !== "Unknown model" &&
        Number.isFinite(item.auc) &&
        item.auc >= 0 &&
        item.auc <= 1
    );
}

function normalizeShapFeatures(raw) {
  if (!Array.isArray(raw)) return [];

  return raw
    .map((item) => {
      const keys = Object.keys(item);

      const featureKey = keys.find((key) =>
        [
          "feature",
          "feature_name",
          "features",
          "name",
        ].includes(key.toLowerCase())
      );

      const valueKey = keys.find((key) =>
        [
          "mean_abs_shap",
          "mean_absolute_shap",
          "mean_abs_shap_value",
          "mean_absolute_shap_value",
          "shap_importance",
          "importance",
          "value",
        ].includes(
          key.toLowerCase().replace(/[\s-]/g, "_")
        )
      );

      return {
        name: featureKey ? String(item[featureKey]) : "",
        value: valueKey ? Number(item[valueKey]) : NaN,
      };
    })
    .filter(
      (item) =>
        item.name &&
        Number.isFinite(item.value) &&
        item.value >= 0
    )
    .sort((a, b) => b.value - a.value);
}

// --------------------------------------------------
// MAIN APPLICATION
// --------------------------------------------------

function App() {
  const [page, setPage] = useState("Overview");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [modelData, setModelData] = useState([]);
  const [features, setFeatures] = useState([]);

  const [analyticsLoading, setAnalyticsLoading] = useState(true);
  const [analyticsError, setAnalyticsError] = useState("");

  // Fetch live analytics when the dashboard loads.
  useEffect(() => {
    async function loadAnalytics() {
      try {
        setAnalyticsLoading(true);
        setAnalyticsError("");

        const [modelResponse, shapResponse] = await Promise.all([
          fetch(`${API_BASE}/api/v1/analytics/model-comparison`),
          fetch(`${API_BASE}/api/v1/analytics/shap`),
        ]);

        if (!modelResponse.ok) {
          throw new Error("Could not load model comparison data.");
        }

        if (!shapResponse.ok) {
          throw new Error("Could not load SHAP feature importance.");
        }

        const modelJson = await modelResponse.json();
        const shapJson = await shapResponse.json();

        const normalizedModels = normalizeModelMetrics(
          modelJson.models
        );

        const normalizedFeatures = normalizeShapFeatures(
          shapJson.features
        );

        if (normalizedModels.length === 0) {
          throw new Error(
            "Model metrics were returned, but no valid ROC-AUC values were found."
          );
        }

        if (normalizedFeatures.length === 0) {
          throw new Error(
            "SHAP data was returned, but its feature/value columns could not be recognized."
          );
        }

        setModelData(normalizedModels);
        setFeatures(normalizedFeatures);
      } catch (error) {
        setAnalyticsError(
          error.message || "Unable to connect to analytics API."
        );
      } finally {
        setAnalyticsLoading(false);
      }
    }

    loadAnalytics();
  }, []);

  return (
    <div className="app-shell">
      {/* SIDEBAR */}

      {sidebarOpen && (
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-logo">
              <Dna size={24} />
            </div>

            <div>
              <h2>FertilityIQ</h2>
              <span>RESEARCH PLATFORM</span>
            </div>
          </div>

          <p className="nav-heading">WORKSPACE</p>

          <nav className="navigation">
            {navigation.map(({ name, icon: Icon }) => (
              <button
                key={name}
                className={`nav-link ${
                  page === name ? "selected" : ""
                }`}
                onClick={() => setPage(name)}
              >
                <Icon size={18} />
                <span>{name}</span>

                {page === name && <ChevronRight size={16} />}
              </button>
            ))}
          </nav>

          <div className="sidebar-footer">
            <div className="online-indicator">
              <span className="green-dot" />
              Research environment active
            </div>

            <button className="nav-link">
              <Settings size={18} />
              Settings
            </button>

            <div className="workspace-profile">
              <div className="profile-avatar">FI</div>

              <div>
                <strong>Research Workspace</strong>
                <small>Local environment</small>
              </div>
            </div>
          </div>
        </aside>
      )}

      {/* MAIN AREA */}

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="header-icon"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <Menu size={20} />
            </button>

            <span className="breadcrumb">
              Workspace / <strong>{page}</strong>
            </span>
          </div>

          <div className="topbar-right">
            <span className="environment-label">
              <span className="green-dot" />
              LOCAL ENVIRONMENT
            </span>

            <button className="header-icon">
              <Search size={18} />
            </button>

            <button className="header-icon">
              <Bell size={18} />
            </button>
          </div>
        </header>

        <div className="page-content">
          {analyticsLoading && (
            <div className="muted-note">
              Loading live research analytics...
            </div>
          )}

          {analyticsError && (
            <div className="error-message">
              Analytics: {analyticsError}
            </div>
          )}

          {page === "Overview" && (
            <Overview
              modelData={modelData}
              features={features}
            />
          )}

          {page === "Prediction" && <Prediction />}

          {page === "Model Analytics" && <ModelAnalytics />}

          {page === "Explainability" && (
            <Explainability features={features} />
          )}

          {page === "Research" && <Research />}
        </div>
      </main>
    </div>
  );
}

// --------------------------------------------------
// REUSABLE COMPONENTS
// --------------------------------------------------

function PageHeading({ eyebrow, title, description }) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, note }) {
  return (
    <div className="stat-card">
      <div className="stat-card-top">
        <span>{label}</span>

        <div className="stat-icon">
          <Icon size={19} />
        </div>
      </div>

      <strong className="stat-value">{value}</strong>
      <small>{note}</small>
    </div>
  );
}

function Panel({ title, subtitle, children, tag }) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h3>{title}</h3>
          {subtitle && <p>{subtitle}</p>}
        </div>

        {tag && <span className="panel-tag">{tag}</span>}
      </div>

      {children}
    </section>
  );
}

function StatusRow({ label, value }) {
  return (
    <div className="status-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

// --------------------------------------------------
// FEATURE IMPORTANCE COMPONENT
// --------------------------------------------------

function FeatureBars({ features }) {
  if (!features || features.length === 0) {
    return (
      <p className="muted-note">
        SHAP feature importance is not available.
      </p>
    );
  }

  const maxValue = Math.max(
    ...features.map((item) => item.value)
  );

  return (
    <div className="feature-list">
      {features.slice(0, 10).map(({ name, value }) => (
        <div className="feature-item" key={name}>
          <div className="feature-label">
            <span>{name}</span>
            <strong>{value.toFixed(3)}</strong>
          </div>

          <div className="bar-background">
            <div
              className="bar-value"
              style={{
                width: `${(value / maxValue) * 100}%`,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// --------------------------------------------------
// OVERVIEW PAGE
// --------------------------------------------------

function Overview({ modelData, features }) {
  const pipeline = [
    "Dataset validation",
    "Exploratory analysis",
    "Feature engineering",
    "Model training",
    "Test evaluation",
    "SHAP explainability",
    "FastAPI integration",
  ];

  return (
    <>
      <PageHeading
        eyebrow="RESEARCH OVERVIEW / 2026"
        title="Research Dashboard"
        description="Monitor IVF dataset analysis, model performance and research experiments."
      />

      <div className="stats-grid">
        <StatCard
          icon={Users}
          label="Dataset records"
          value="5,000"
          note="Valid outcome records"
        />

        <StatCard
          icon={Brain}
          label="Test ROC-AUC"
          value="0.8601"
          note="XGBoost model"
        />

        <StatCard
          icon={Target}
          label="Test accuracy"
          value="77.20%"
          note="750 test records"
        />

        <StatCard
          icon={FlaskConical}
          label="Models evaluated"
          value="03"
          note="LR · XGBoost · LightGBM"
        />
      </div>

      <div className="two-column">
        <Panel
          title="Model performance"
          subtitle="ROC-AUC comparison on the validation dataset"
          tag="VALIDATION"
        >
          {modelData.length > 0 ? (
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={270}>
                <BarChart
                  data={modelData}
                  margin={{
                    top: 15,
                    right: 10,
                    left: 0,
                    bottom: 10,
                  }}
                >
                  <CartesianGrid
                    stroke="#26344a"
                    strokeDasharray="3 3"
                    vertical={false}
                  />

                  <XAxis
                    dataKey="name"
                    stroke="#9aa9bd"
                    tick={{ fontSize: 10 }}
                  />

                  <YAxis
                    domain={[0, 1]}
                    stroke="#9aa9bd"
                    tick={{ fontSize: 10 }}
                  />

                  <Tooltip
                    formatter={(value) => [
                      Number(value).toFixed(4),
                      "Validation ROC-AUC",
                    ]}
                    contentStyle={{
                      background: "#172337",
                      border: "1px solid #35445b",
                      color: "#fff",
                    }}
                  />

                  <Bar
                    dataKey="auc"
                    name="Validation ROC-AUC"
                    fill="#7896ff"
                    radius={[6, 6, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="muted-note">
              No validation model metrics available.
            </p>
          )}

          <p className="muted-note">
            All models are compared using validation ROC-AUC.
            Final XGBoost test ROC-AUC: 0.8601.
          </p>
        </Panel>

        <Panel
          title="Research pipeline"
          subtitle="Completed project milestones"
        >
          {pipeline.map((item) => (
            <div className="pipeline-row" key={item}>
              <span className="complete-check">✓</span>
              <span>{item}</span>
              <small>Complete</small>
            </div>
          ))}

          <div className="pipeline-row">
            <span className="pending-check">•</span>
            <span>Dynamic dashboard integration</span>
            <small className="pending-text">In progress</small>
          </div>
        </Panel>
      </div>

      <div className="two-column lower-section">
        <Panel
          title="Explainability insights"
          subtitle="Mean absolute SHAP values"
          tag="SHAP"
        >
          <FeatureBars features={features} />
        </Panel>

        <Panel
          title="Research integrity"
          subtitle="Current verification status"
        >
          <StatusRow
            label="Target class mapping"
            value="Unverified"
          />

          <StatusRow
            label="External validation"
            value="Not performed"
          />

          <StatusRow
            label="Patient-level split"
            value="Not established"
          />

          <p className="muted-note">
            This is an experimental research prototype,
            not a clinical decision system.
          </p>
        </Panel>
      </div>
    </>
  );
}

// --------------------------------------------------
// PREDICTION PAGE
// --------------------------------------------------

function Prediction() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const updateField = (name, value) => {
    setForm((current) => ({
      ...current,
      [name]: value,
    }));
  };

  async function submitPrediction(event) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const payload = Object.fromEntries(
        Object.entries(form).map(([key, value]) => [
          key,
          Number(value),
        ])
      );

      const response = await fetch(
        `${API_BASE}/api/v1/predict`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Prediction request failed."
        );
      }

      setResult(data.prediction);
    } catch (err) {
      setError(
        err.message || "Could not connect to the prediction API."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <PageHeading
        eyebrow="MODEL INFERENCE"
        title="IVF Research Prediction"
        description="Enter experimental feature values to query the trained model."
      />

      <Panel
        title="Feature input"
        subtitle="Enter all 15 model features"
        tag="RESEARCH ONLY"
      >
        <form onSubmit={submitPrediction}>
          <div className="form-grid">
            {Object.entries(form).map(([name, value]) => (
              <label className="form-field" key={name}>
                <span>{name}</span>

                <input
                  type="number"
                  min="0"
                  step="any"
                  required
                  value={value}
                  onChange={(event) =>
                    updateField(name, event.target.value)
                  }
                />
              </label>
            ))}
          </div>

          <button
            className="primary-button"
            type="submit"
            disabled={loading}
          >
            {loading
              ? "Running model..."
              : "Generate research prediction"}
          </button>
        </form>

        {error && (
          <div className="error-message">{error}</div>
        )}

        {result && (
          <div className="prediction-result">
            <h3>
              Predicted class: {result.predicted_class}
            </h3>

            <div className="probability-grid">
              <div>
                <span>Class 0 probability</span>

                <strong>
                  {(
                    result.class_0_probability * 100
                  ).toFixed(2)}
                  %
                </strong>
              </div>

              <div>
                <span>Class 1 probability</span>

                <strong>
                  {(
                    result.class_1_probability * 100
                  ).toFixed(2)}
                  %
                </strong>
              </div>
            </div>

            <p>
              Class mapping is unverified. This output is
              not a clinical diagnosis or treatment recommendation.
            </p>
          </div>
        )}
      </Panel>
    </>
  );
}

// --------------------------------------------------
// MODEL ANALYTICS PAGE
// --------------------------------------------------

function ModelAnalytics() {
  const metrics = [
    ["Accuracy", "77.20%"],
    ["Precision", "72.31%"],
    ["Recall", "87.97%"],
    ["F1-score", "79.37%"],
    ["ROC-AUC", "0.8601"],
    ["PR-AUC", "0.8391"],
    ["Brier score", "0.1474"],
  ];

  return (
    <>
      <PageHeading
        eyebrow="MODEL EVALUATION"
        title="Model Analytics"
        description="Final test results from the held-out dataset."
      />

      <Panel
        title="XGBoost test performance"
        subtitle="750 test records"
        tag="FINAL TEST"
      >
        <div className="metric-grid">
          {metrics.map(([name, value]) => (
            <div className="metric-card" key={name}>
              <span>{name}</span>
              <strong>{value}</strong>
            </div>
          ))}
        </div>

        <p className="muted-note">
          These results are from the held-out test split.
          They are separate from the validation model comparison.
        </p>
      </Panel>
    </>
  );
}

// --------------------------------------------------
// EXPLAINABILITY PAGE
// --------------------------------------------------

function Explainability({ features }) {
  return (
    <>
      <PageHeading
        eyebrow="MODEL INTERPRETABILITY"
        title="Explainability"
        description="Global feature importance calculated using SHAP."
      />

      <Panel
        title="SHAP feature importance"
        subtitle="Mean absolute SHAP values"
        tag="SHAP"
      >
        <FeatureBars features={features} />

        <p className="muted-note">
          SHAP importance describes model behaviour, not causation
          or clinical importance.
        </p>
      </Panel>
    </>
  );
}

// --------------------------------------------------
// RESEARCH PAGE
// --------------------------------------------------

function Research() {
  return (
    <>
      <PageHeading
        eyebrow="PROJECT DOCUMENTATION"
        title="Research methodology"
        description="Dataset profile, experiment design and current limitations."
      />

      <div className="two-column">
        <Panel title="Dataset profile">
          <StatusRow
            label="Original records"
            value="5,002"
          />

          <StatusRow
            label="Valid target records"
            value="5,000"
          />

          <StatusRow
            label="Model features"
            value="15"
          />

          <StatusRow
            label="Training records"
            value="3,500"
          />

          <StatusRow
            label="Validation records"
            value="750"
          />

          <StatusRow
            label="Test records"
            value="750"
          />
        </Panel>

        <Panel title="Research limitations">
          <ul className="limitations">
            <li>Target class mapping has not been verified.</li>
            <li>Feature units and definitions need confirmation.</li>
            <li>Patient-level independence is not established.</li>
            <li>
              Embryo features may introduce prediction-time leakage.
            </li>
            <li>External clinical validation has not been performed.</li>
            <li>Model output must not guide patient care.</li>
          </ul>
        </Panel>
      </div>
    </>
  );
}

export default App;
