import { useCallback, useEffect, useRef, useState } from "react";

export interface AsyncState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  run: () => Promise<T | null>;
  setData: (d: T | null) => void;
}

/** Ejecuta una función asíncrona, descartando respuestas obsoletas (carreras). */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[], immediate = true): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(immediate);
  const callId = useRef(0);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const run = useCallback(async () => {
    const id = ++callId.current;
    setLoading(true);
    setError(null);
    try {
      const out = await fn();
      if (id === callId.current) setData(out);
      return out;
    } catch (e) {
      if (id === callId.current) setError(e instanceof Error ? e.message : String(e));
      return null;
    } finally {
      if (id === callId.current) setLoading(false);
    }
  }, deps);

  useEffect(() => {
    if (immediate) void run();
  }, [run, immediate]);

  return { data, error, loading, run, setData };
}
