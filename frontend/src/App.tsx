import { AnimatePresence, motion } from "framer-motion";
import { Activity, Microscope, ShieldAlert, Stethoscope, Zap } from "lucide-react";
import { Suspense, lazy, useEffect, useState } from "react";
import { CreatorsCard } from "./components/CreatorsCard";
import { TabNav, type TabDef } from "./components/TabNav";
import { LoadingBlock } from "./components/ui";
import { COLORS } from "./lib/format";
import { tabVariants } from "./lib/motion";

// Cada pestaña es un trozo (chunk) independiente: la primera carga solo descarga
// el panel clínico. Las otras dos se precargan en segundo plano tras el arranque.
const loadClinical = () => import("./tabs/ClinicalTab");
const loadTelemetry = () => import("./tabs/TelemetryTab");
const loadAudit = () => import("./tabs/AuditTab");
const ClinicalTab = lazy(() => loadClinical().then((m) => ({ default: m.ClinicalTab })));
const TelemetryTab = lazy(() => loadTelemetry().then((m) => ({ default: m.TelemetryTab })));
const AuditTab = lazy(() => loadAudit().then((m) => ({ default: m.AuditTab })));

type TabKey = "clinical" | "telemetry" | "audit";

const TABS: TabDef<TabKey>[] = [
  { key: "clinical", label: "Panel Clínico & Simulación", short: "Clínico", icon: <Stethoscope className="h-4 w-4" />, color: COLORS.benign },
  { key: "telemetry", label: "Telemetría del Motor Numérico", short: "Telemetría", icon: <Zap className="h-4 w-4" />, color: COLORS.engine },
  { key: "audit", label: "Auditoría Matemática", short: "Auditoría", icon: <Microscope className="h-4 w-4" />, color: COLORS.accent },
];

export default function App() {
  const [tab, setTab] = useState<TabKey>("clinical");
  // Se incrementa cuando la pestaña de telemetría promueve un modelo nuevo al panel clínico.
  const [modelVersion, setModelVersion] = useState(0);

  useEffect(() => {
    const t = window.setTimeout(() => {
      void loadTelemetry();
      void loadAudit();
    }, 1200);
    return () => window.clearTimeout(t);
  }, []);

  return (
    <div className="relative z-10 mx-auto max-w-[1440px] px-4 pb-10 pt-5 md:px-8">
      <header className="relative z-40 mb-5 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="relative rounded-lg border border-edge bg-white p-2.5">
                <Activity className="h-5 w-5 text-engine" />
              </div>
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-slate-900 md:text-2xl">
                Plataforma de Optimización y Diagnóstico Oncológico{" "}
                <span className="text-engine">WDBC</span>
              </h1>
              <p className="mt-0.5 text-xs text-slate-500 md:text-sm">
                Regresión logística como problema convexo sin restricciones{" "}
                <span className="font-serif italic text-slate-700">
                  min<sub className="not-italic">w∈ℝ<sup>d</sup>, b∈ℝ</sub> J(w, b)
                </span>{" "}
                · NumPy puro
              </p>
            </div>
          </div>
        </motion.div>
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.05, duration: 0.3 }}>
          <CreatorsCard />
        </motion.div>
      </header>

      <TabNav tabs={TABS} active={tab} onChange={setTab} />

      <main className="mt-5">
        <AnimatePresence mode="wait">
          <motion.div key={tab} variants={tabVariants} initial="initial" animate="animate" exit="exit" style={{ willChange: "transform, opacity" }}>
            <Suspense fallback={<LoadingBlock label="Cargando módulo…" height={420} />}>
              {tab === "clinical" && <ClinicalTab modelVersion={modelVersion} />}
              {tab === "telemetry" && <TelemetryTab onModelPromoted={() => setModelVersion((v) => v + 1)} />}
              {tab === "audit" && <AuditTab />}
            </Suspense>
          </motion.div>
        </AnimatePresence>
      </main>

      <footer className="mt-8 flex flex-col items-start gap-2 border-t border-edge pt-4 text-[11px] text-slate-500 md:flex-row md:items-center md:justify-between">
        <span className="flex items-center gap-1.5">
          <ShieldAlert className="h-3.5 w-3.5 text-amber-700" />
          Herramienta académica de apoyo a la decisión. No es un dispositivo médico ni sustituye el criterio clínico ni el estudio histopatológico.
        </span>
        <span className="font-mono">Wisconsin Diagnostic Breast Cancer · 569 muestras · 30 características</span>
      </footer>
    </div>
  );
}
