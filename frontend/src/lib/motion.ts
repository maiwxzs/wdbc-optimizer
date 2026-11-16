import type { Variants } from "framer-motion";

/** Transición entre pestañas: un fundido corto, sin desenfoque ni rebote. */
export const tabVariants: Variants = {
  initial: { opacity: 0, y: 6 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.22, ease: "easeOut" } },
  exit: { opacity: 0, transition: { duration: 0.12, ease: "easeIn" } },
};

/** Contenedor que escalona, de forma breve, la entrada de sus hijos. */
export const staggerContainer: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.03 } },
};

/** Tarjeta que aparece con un fundido y un desplazamiento mínimo. */
export const cardItem: Variants = {
  hidden: { opacity: 0, y: 6 },
  show: { opacity: 1, y: 0, transition: { duration: 0.25, ease: "easeOut" } },
};
