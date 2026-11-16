import katex from "katex";
import "katex/dist/katex.min.css";
import { useMemo } from "react";

/** Fórmula LaTeX renderizada con KaTeX. Solo la importan las pestañas (carga diferida). */
export function TeX({ children, block = false }: { children: string; block?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(children, { displayMode: block, throwOnError: false, strict: "ignore" }),
    [children, block],
  );
  return block ? (
    <div className="overflow-x-auto py-1 text-slate-800" dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span dangerouslySetInnerHTML={{ __html: html }} />
  );
}
