"use client";

import { useCallback, useEffect, useState } from "react";

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
// Keep false on any publicly accessible deployment. The API reset endpoint is not authenticated.
const DEMO_RESET_ENABLED = process.env.NEXT_PUBLIC_ENABLE_DEMO_RESET === "true";

type ApiStatus = "checking" | "online" | "partial" | "offline";

interface Incident {
  id: number;
  timestamp: string;
  track_id: number;
  zone_name: string;
  threat_level: string;
  camera_mode: string;
  dwell_time_sec: number;
  snapshot_base64?: string | null;
}

interface Stats {
  total_incidents: number;
  critical_breaches: number;
  warnings: number;
  avg_dwell_time_sec: number;
}

const EMPTY_STATS: Stats = {
  total_incidents: 0,
  critical_breaches: 0,
  warnings: 0,
  avg_dwell_time_sec: 0,
};

async function getJson<T>(url: string): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 4500);
  try {
    const response = await fetch(url, {
      cache: "no-store",
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error(`Request failed (${response.status})`);
    return (await response.json()) as T;
  } finally {
    window.clearTimeout(timeout);
  }
}

export default function SecurityCommandCenter() {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [stats, setStats] = useState<Stats>(EMPTY_STATS);
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [resetBusy, setResetBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    const stored = window.localStorage.getItem("gatekeeper-theme");
    if (stored === "light" || stored === "dark") setTheme(stored);
  }, []);

  useEffect(() => {
    window.localStorage.setItem("gatekeeper-theme", theme);
  }, [theme]);

  const refreshData = useCallback(async () => {
    setIsRefreshing(true);
    const [incidentResult, statsResult] = await Promise.allSettled([
      getJson<Incident[]>(`${API_BASE_URL}/api/incidents?limit=25`),
      getJson<Stats>(`${API_BASE_URL}/api/stats`),
    ]);

    const incidentsOk = incidentResult.status === "fulfilled";
    const statsOk = statsResult.status === "fulfilled";

    if (incidentsOk) setIncidents(incidentResult.value);
    if (statsOk) setStats(statsResult.value);
    setApiStatus(incidentsOk && statsOk ? "online" : incidentsOk || statsOk ? "partial" : "offline");
    setLastUpdated(new Date().toLocaleTimeString());
    setIsRefreshing(false);
  }, []);

  useEffect(() => {
    let stopped = false;
    let timer: number | undefined;

    const refreshThenSchedule = async () => {
      await refreshData();
      if (!stopped) timer = window.setTimeout(refreshThenSchedule, 5000);
    };

    void refreshThenSchedule();
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [refreshData]);

  const clearLogs = async () => {
    if (!DEMO_RESET_ENABLED) return;
    const confirmed = window.confirm("Clear every incident in the demo database? This cannot be undone.");
    if (!confirmed) return;
    setResetBusy(true);
    setNotice(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/incidents/clear`, { method: "DELETE" });
      if (!response.ok) throw new Error(`Reset failed (${response.status})`);
      setNotice("Demo incident logs cleared.");
      await refreshData();
    } catch {
      setNotice("Could not reset the log. Check API availability and private-demo configuration.");
    } finally {
      setResetBusy(false);
    }
  };

  const statusLabel = {
    checking: "Checking API",
    online: "API connected",
    partial: "API partially responding",
    offline: "API unavailable",
  }[apiStatus];
  const statusClass = {
    checking: "bg-amber-400",
    online: "bg-emerald-400",
    partial: "bg-amber-400",
    offline: "bg-red-500",
  }[apiStatus];

  return (
    <main data-theme={theme} className="min-h-screen bg-slate-950 p-4 font-sans text-slate-100 sm:p-6">
      <div className="mx-auto max-w-7xl">
        <header className="mb-8 flex flex-col items-start justify-between gap-4 border-b border-slate-800 pb-5 md:flex-row md:items-center">
          <div>
            <div className="flex flex-wrap items-center gap-3">
              <span className={`h-3 w-3 rounded-full ${statusClass}`} aria-hidden="true" />
              <h1 className="text-2xl font-bold tracking-tight text-white md:text-3xl">
                GateKeeper AI — Command Center
              </h1>
            </div>
            <p className="mt-2 text-sm text-slate-400">
              Local person detection · Restricted-zone alerts · Cloud incident history
            </p>
            <p className="mt-2 text-xs text-slate-500" role="status">
              {statusLabel}{lastUpdated ? ` · Last checked ${lastUpdated}` : ""}
              {isRefreshing ? " · Refreshing…" : ""}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              className="theme-toggle rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-200 transition"
              onClick={() => setTheme((current) => (current === "light" ? "dark" : "light"))}
              aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
            >
              {theme === "light" ? "🌙 Dark mode" : "☀️ Light mode"}
            </button>
            <button
              type="button"
              onClick={() => void refreshData()}
              disabled={isRefreshing}
              className="rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-200 transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Refresh now
            </button>
            {DEMO_RESET_ENABLED && (
              <button
                type="button"
                onClick={() => void clearLogs()}
                disabled={resetBusy}
                className="rounded-lg border border-red-900 bg-red-950/60 px-4 py-2 text-sm text-red-200 transition hover:bg-red-900 disabled:opacity-60"
              >
                {resetBusy ? "Resetting…" : "Reset demo logs"}
              </button>
            )}
          </div>
        </header>

        {notice && (
          <div className="mb-5 rounded-lg border border-slate-700 bg-slate-900 p-3 text-sm text-slate-200" role="status">
            {notice}
          </div>
        )}
        {apiStatus === "offline" && (
          <div className="mb-5 rounded-lg border border-red-900/70 bg-red-950/40 p-4 text-sm text-red-200">
            The dashboard cannot reach the API at <code className="break-all">{API_BASE_URL}</code>. Check the backend service, deployment URL, and CORS configuration.
          </div>
        )}
        {apiStatus === "partial" && (
          <div className="mb-5 rounded-lg border border-amber-900/70 bg-amber-950/30 p-4 text-sm text-amber-200">
            The API responded to only part of the dashboard request. Some figures may be stale; check the backend logs.
          </div>
        )}

        <section aria-label="Incident summary" className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Kpi title="Incidents logged" value={stats.total_incidents} />
          <Kpi title="Critical breaches" value={stats.critical_breaches} tone="red" />
          <Kpi title="Warnings" value={stats.warnings} tone="amber" />
          <Kpi title="Average dwell time" value={`${stats.avg_dwell_time_sec}s`} tone="amber" />
        </section>

        <section>
          <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
            <div>
              <h2 className="text-xl font-semibold text-white">Incident evidence feed</h2>
              <p className="mt-1 text-sm text-slate-400">Refreshes approximately every five seconds; this is event polling, not a live CCTV stream.</p>
            </div>
            <span className="text-xs text-slate-500">Showing up to 25 recent incidents</span>
          </div>

          {incidents.length === 0 ? (
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-10 text-center text-slate-400">
              {apiStatus === "offline" ? "Waiting for an API connection." : "No incidents returned yet. Start the local edge app and process a test clip."}
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
              {incidents.map((incident) => (
                <article key={incident.id} className="flex min-w-0 flex-col overflow-hidden rounded-xl border border-slate-800 bg-slate-900 shadow-lg">
                  {incident.snapshot_base64 ? (
                    // The edge API validates and limits snapshots before they are stored.
                    <img
                      src={`data:image/jpeg;base64,${incident.snapshot_base64}`}
                      alt={`Evidence crop for incident ${incident.id}`}
                      loading="lazy"
                      className="h-56 w-full bg-black object-contain"
                    />
                  ) : (
                    <div className="flex h-56 items-center justify-center bg-slate-800 text-sm text-slate-500">No evidence snapshot</div>
                  )}
                  <div className="flex flex-1 flex-col justify-between p-4">
                    <div>
                      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                        <span className={`rounded-full border px-2.5 py-1 text-xs font-bold ${incident.threat_level.includes("CRITICAL") ? "border-red-500/30 bg-red-500/15 text-red-300" : "border-amber-500/30 bg-amber-500/15 text-amber-300"}`}>
                          {incident.threat_level}
                        </span>
                        <span className="text-xs text-slate-400">Track #{incident.track_id}</span>
                      </div>
                      <h3 className="break-words text-lg font-bold text-white">{incident.zone_name}</h3>
                      <p className="mt-2 text-sm text-slate-400">
                        Camera: <span className="text-slate-200">{incident.camera_mode}</span>
                      </p>
                      <p className="mt-1 text-sm text-slate-400">
                        Dwell: <span className="font-semibold text-amber-300">{incident.dwell_time_sec}s</span>
                      </p>
                    </div>
                    <p className="mt-4 border-t border-slate-800 pt-3 text-xs text-slate-500">
                      {Number.isNaN(Date.parse(incident.timestamp)) ? "Timestamp unavailable" : `Logged: ${new Date(incident.timestamp).toLocaleString()}`}
                    </p>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
        <footer className="mt-10 border-t border-slate-800 pt-4 text-xs text-slate-500">
          Hackathon prototype, not a certified security or biometric-identification system. Use approved test footage and protect incident evidence.
        </footer>
      </div>
    </main>
  );
}

function Kpi({ title, value, tone = "default" }: { title: string; value: string | number; tone?: "default" | "red" | "amber" }) {
  const valueClass = tone === "red" ? "text-red-400" : tone === "amber" ? "text-amber-300" : "text-white";
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
      <p className="text-xs uppercase tracking-wider text-slate-400">{title}</p>
      <p className={`mt-2 text-3xl font-bold ${valueClass}`}>{value}</p>
    </div>
  );
}
