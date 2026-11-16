import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// Sin <React.StrictMode>: en desarrollo ejecuta cada efecto dos veces, lo que
// duplicaba todas las peticiones al backend (se veía en el log de Vite).
ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
