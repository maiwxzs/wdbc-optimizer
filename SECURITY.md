# Política de seguridad

## Alcance

WDBC Optimizer es un **proyecto académico** pensado para ejecutarse en local (`localhost`). No está diseñado ni endurecido para exponerse a Internet:

- la API **no tiene autenticación** ni límites de uso;
- cualquier cliente puede reemplazar el modelo activo con `POST /api/train` y `set_active=true`;
- peticiones con `max_iter` alto (hasta 100 000) consumen CPU durante segundos, lo que permitiría una denegación de servicio trivial.

Si la despliega en un servidor accesible desde la red, añada como mínimo autenticación, límites de uso, `CORS_ORIGINS` restringido y un *proxy* inverso con HTTPS.

## Versiones soportadas

| Versión | Soporte |
|---|---|
| 1.0.x | sí |

## Cómo informar de una vulnerabilidad

**No abra un *issue* público.** Use la función *Report a vulnerability* de la pestaña [Security](https://github.com/maiwxzs/wdbc-optimizer/security) del repositorio (*Private vulnerability reporting*) o contacte directamente con el mantenedor, [maiwxzs](https://github.com/maiwxzs). Incluya:

1. una descripción del problema y de su impacto;
2. los pasos o la petición HTTP que lo reproducen;
3. la versión o el *commit* afectado.

Los mantenedores intentarán responder en un plazo razonable para un proyecto académico, sin compromisos de tiempo formales.

## Dependencias

Las dependencias se declaran con versión mínima en `backend/requirements.txt` y se fijan en `frontend/package-lock.json`. Se recomienda activar Dependabot en GitHub para recibir alertas.
