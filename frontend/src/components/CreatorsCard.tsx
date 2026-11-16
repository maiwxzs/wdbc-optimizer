import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, GraduationCap, Sigma, Users } from "lucide-react";
import { useState } from "react";
import { PROJECT, TEAM } from "../config/team";

/** Tarjeta permanente de firma de creadores (borde con degradado rotatorio). */
export function CreatorsCard() {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <motion.button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="spin-border block text-left"
        aria-expanded={open}
      >
        <div className="flex items-center gap-3 px-4 py-2.5">
          <div className="rounded-md border border-edge bg-night p-1.5">
            <Sigma className="h-4 w-4 text-engine" />
          </div>
          <div className="leading-tight">
            <div className="text-[12.5px] font-semibold text-slate-900">
              {PROJECT.title} <span className="text-slate-500">|</span>{" "}
              <span className="text-engine">{PROJECT.subtitle}</span>
            </div>
            <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-slate-500">
              <Users className="h-3 w-3" />
              {TEAM.map((m) => m.name).join(" · ")}
            </div>
          </div>
          <motion.span animate={{ rotate: open ? 180 : 0 }} className="ml-1 text-slate-500">
            <ChevronDown className="h-4 w-4" />
          </motion.span>
        </div>
      </motion.button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
            className="glass absolute right-0 z-50 mt-2 w-[380px] p-4 shadow-lg"
          >
            <div className="mb-3 flex items-center gap-2 text-xs text-slate-500">
              <GraduationCap className="h-4 w-4 text-accent" />
              <span>{PROJECT.institution}</span>
            </div>
            <motion.ul initial="hidden" animate="show" variants={{ show: { transition: { staggerChildren: 0.06 } } }} className="space-y-2.5">
              {TEAM.map((m) => (
                <motion.li
                  key={m.name}
                  variants={{ hidden: { opacity: 0, x: 12 }, show: { opacity: 1, x: 0 } }}
                  className="rounded-lg border border-edge bg-night p-3"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-slate-900">{m.name}</span>
                    <span className="rounded-full border border-engine/40 bg-engine/10 px-2 py-0.5 text-[10px] font-semibold text-engine">
                      {m.role}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-slate-500">{m.focus}</p>
                  {m.credential && <p className="mt-1 font-mono text-[10.5px] text-slate-500">{m.credential}</p>}
                </motion.li>
              ))}
            </motion.ul>
            <div className="mt-3 flex justify-between border-t border-edge pt-2.5 text-[11px] text-slate-500">
              <span>{PROJECT.course}</span>
              <span>{PROJECT.advisor}</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
