import { motion } from "framer-motion";
import type { ReactNode } from "react";

export interface TabDef<K extends string> {
  key: K;
  label: string;
  short: string;
  icon: ReactNode;
  color: string;
}

export function TabNav<K extends string>({ tabs, active, onChange }: { tabs: TabDef<K>[]; active: K; onChange: (k: K) => void }) {
  return (
    <nav className="glass flex gap-1 p-1" role="tablist">
      {tabs.map((t, i) => {
        const selected = t.key === active;
        return (
          <button
            key={t.key}
            role="tab"
            aria-selected={selected}
            onClick={() => onChange(t.key)}
            className={`relative flex flex-1 items-center justify-center gap-2.5 rounded-lg px-4 py-2.5 text-sm font-medium transition-colors ${
              selected ? "text-slate-900" : "text-slate-500 hover:text-slate-800"
            }`}
          >
            {selected && (
              <motion.span
                layoutId="tab-pill"
                className="absolute inset-0 rounded-lg border-b-2"
                style={{ borderColor: t.color, background: `${t.color}0d` }}
                transition={{ duration: 0.2, ease: "easeOut" }}
              />
            )}
            <span className="relative z-10" style={{ color: selected ? t.color : undefined }}>
              {t.icon}
            </span>
            <span className="relative z-10 hidden md:inline">
              <span className="mr-1.5 font-mono text-[11px] text-slate-500">0{i + 1}</span>
              {t.label}
            </span>
            <span className="relative z-10 md:hidden">{t.short}</span>
          </button>
        );
      })}
    </nav>
  );
}
