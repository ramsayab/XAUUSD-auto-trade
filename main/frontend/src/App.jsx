import { useCallback, useEffect, useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value)))
    return "--";
  return Number(value).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
};

const formatDate = (value) =>
  value
    ? new Date(value).toLocaleString("en-US", {
        day: "2-digit",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "--";

function App() {
  const [position, setPosition] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);

  const loadDashboard = useCallback(async () => {
    setError("");
    try {
      const [positionResponse, historyResponse] = await Promise.all([
        fetch(`${API_URL}/current_pos`),
        fetch(`${API_URL}/history_pos`),
      ]);
      if (!positionResponse.ok || !historyResponse.ok)
        throw new Error("API error");
      setPosition(await positionResponse.json());
      setHistory(await historyResponse.json());
      setLastUpdated(new Date());
    } catch {
      setError(
        "Backend belum tersambung. Pastikan FastAPI berjalan di port 8000.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDashboard();
    const interval = window.setInterval(loadDashboard, 1000);
    return () => window.clearInterval(interval);
  }, [loadDashboard]);

  const profit = position?.profit ?? 0;
  const hasPosition = Boolean(position?.order_type);
  const isLong = position?.order_type === "Long";

  return (
    <main className="shell">
      <section className="intro">
        <div>
          <p className="eyebrow">XAUUSD / H1</p>
          <h1>Trading overview</h1>
          <p className="muted">
            Monitor the model's current exposure and recent closes.
          </p>
        </div>
        <div className="updated">
          Last sync{" "}
          <strong>
            {lastUpdated ? formatDate(lastUpdated) : "waiting..."}
          </strong>
        </div>
      </section>
      {error && <div className="alert">{error}</div>}
      <section className="stats-grid">
        <article className="stat-card accent-card">
          <span>ACCOUNT BALANCE</span>
          <strong>
            {position?.balance !== null && position?.balance !== undefined
              ? `$${formatNumber(position.balance)}`
              : "--"}
          </strong>
          <small>MT5 account</small>
        </article>
        <article className="stat-card">
          <span>OPEN P&amp;L</span>
          <strong className={profit >= 0 ? "positive" : "negative"}>
            {hasPosition
              ? `${profit >= 0 ? "+" : ""}$${formatNumber(profit)}`
              : "--"}
          </strong>
          <small>{hasPosition ? "Current position" : "No open position"}</small>
        </article>
        <article className="stat-card">
          <span>ACTIVE SIDE</span>
          <strong>{position?.order_type || "FLAT"}</strong>
          <small>
            {hasPosition
              ? `${formatNumber(position.lot)} lot`
              : "Waiting for signal"}
          </small>
        </article>
      </section>
      <section className="content-grid">
        <article className="panel position-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">CURRENT POSITION</p>
              <h2>{hasPosition ? "XAUUSD position" : "No active position"}</h2>
            </div>
            {hasPosition && (
              <span className={`side-pill ${isLong ? "long" : "short"}`}>
                {isLong ? "LONG" : "SHORT"}
              </span>
            )}
          </div>
          {loading ? (
            <div className="empty-state">Loading market data...</div>
          ) : hasPosition ? (
            <div className="position-details">
              <div className="price-block">
                <span>UNREALIZED PROFIT</span>
                <strong className={profit >= 0 ? "positive" : "negative"}>
                  {profit >= 0 ? "+" : ""}${formatNumber(profit)}
                </strong>
              </div>
              <dl>
                <div>
                  <dt>Entry price</dt>
                  <dd>{formatNumber(position.entry_price)}</dd>
                </div>
                <div>
                  <dt>Current price</dt>
                  <dd>{formatNumber(position.current_price)}</dd>
                </div>
                <div>
                  <dt>Volume</dt>
                  <dd>{formatNumber(position.lot)} lot</dd>
                </div>
              </dl>
            </div>
          ) : (
            <div className="empty-state">
              <span className="empty-icon">—</span>
              <strong>Portfolio is flat</strong>
              <span>The bot is waiting for its next signal.</span>
            </div>
          )}
        </article>
        <article className="panel signal-panel">
          <p className="eyebrow">SYSTEM STATUS</p>
          <h2>Model execution</h2>
          <div className="status-row">
            <span>
              <i className="status-dot" /> Strategy runner
            </span>
            <strong>{error ? "Paused" : "Running"}</strong>
          </div>
          <div className="status-row">
            <span>
              <i className="status-dot" /> Instrument
            </span>
            <strong>XAUUSD</strong>
          </div>
          <div className="status-row">
            <span>
              <i className="status-dot" /> Refresh rate
            </span>
            <strong>1 sec</strong>
          </div>
        </article>
      </section>
      <section className="panel history-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">EXECUTION LOG</p>
            <h2>Recent history</h2>
          </div>
          <span className="count-label">{history.length} records</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>TYPE</th>
                <th>RESULT</th>
                <th>OPENED</th>
                <th>CLOSED</th>
              </tr>
            </thead>
            <tbody>
              {history.length ? (
                history.map((item, index) => (
                  <tr key={`${item.end_time}-${index}`}>
                    <td>
                      <span
                        className={`trade-type ${item.order_type === "Long" ? "long-text" : "short-text"}`}
                      >
                        <i />
                        {item.order_type}
                      </span>
                    </td>
                    <td className={item.profit >= 0 ? "positive" : "negative"}>
                      {item.profit >= 0 ? "+" : ""}${formatNumber(item.profit)}
                    </td>
                    <td>{formatDate(item.start_time)}</td>
                    <td>{formatDate(item.end_time)}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="4" className="table-empty">
                    No closed trades yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
      <footer>
        <span>Powered by PPO model</span>
        <span>MT5 CONNECTED</span>
      </footer>
    </main>
  );
}

export default App;
