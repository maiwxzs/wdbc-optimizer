/** @type {import('tailwindcss').Config} */
// Tema claro y sobrio: fondo papel, tinta gris oscura y una paleta de acentos
// desaturada. Los nombres de color se conservan para no tocar los componentes.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        night: "#F3F4F6", // superficies hundidas (campos, iconos)
        panel: "#FFFFFF",
        edge: "#DDE1E6", // bordes
        benign: "#2E7D5B",
        malignant: "#B23A2E",
        engine: "#1F5A8A",
        accent: "#5E4B8B",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
      boxShadow: {
        // Antes eran halos de neón; ahora un contorno fino del color del estado.
        "neon-cyan": "0 0 0 1px rgba(31,90,138,.35), 0 1px 3px rgba(16,24,40,.06)",
        "neon-violet": "0 0 0 1px rgba(94,75,139,.30), 0 1px 3px rgba(16,24,40,.06)",
        "neon-red": "0 0 0 1px rgba(178,58,46,.45), 0 1px 3px rgba(16,24,40,.06)",
        "neon-green": "0 0 0 1px rgba(46,125,91,.45), 0 1px 3px rgba(16,24,40,.06)",
      },
    },
  },
  plugins: [],
};
