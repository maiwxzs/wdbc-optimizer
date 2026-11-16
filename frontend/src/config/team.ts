// ✏️ Edita este archivo con los datos reales del equipo.
export interface TeamMember {
  name: string;
  role: string;
  focus: string;
  credential?: string;
}

export const PROJECT = {
  title: "Motor de Optimización Numérica & CDSS Oncológico",
  subtitle: "Proyecto Métodos Numéricos",
  institution: "Proyecto académico · Métodos Numéricos",
  course: "Métodos Numéricos",
  advisor: "WDBC · Regresión logística en NumPy",
};

export const TEAM: TeamMember[] = [
  {
    name: "Maicol Beltran Aguilar",
    role: "Lead Numerical Engineer",
    focus: "Motor numérico: gradiente analítico, estabilidad de σ(z), diferencias finitas y condiciones de parada",
  },
  {
    name: "Brayan Fernandez",
    role: "Frontend Architect",
    focus: "Plataforma web: React, animaciones con Framer Motion y visualización científica con ECharts",
  },
];
