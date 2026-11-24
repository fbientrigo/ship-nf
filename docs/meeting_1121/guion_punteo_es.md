# Guion en punteo — SHiP CM 39 (computing), 15 min

**Mensaje central:** Generar muones con un Normalizing Flow es rápido; el problema es elegir cuáles vale la pena simular con FairShip. Queremos usar los resultados de FairShip para orientar esa elección.

**Cómo usarlo:** **Ancla** = idea que debes transmitir. **Evidencia** = uno o dos datos. **Puente** = lo que dices mirando la slide actual, justo antes del clic. No es necesario pronunciar los números de sección. Los tiempos son objetivos de término de cada slide; señala las figuras en vez de hablar de corrido.

**Atención:** el benchmark de cola (slides 7–9) usa **afterMS**; el loop con FairShip usa **P2**. Una buena traza reconstruida **no** equivale al background final.

**Contexto:** justo antes habla Eduard sobre normalizing flows para el flujo de neutrinos. No vuelvas a explicar qué es un flow, y cada limitación se atribuye a *nuestro* flow pequeño y a la cola que *nosotros* dejamos fuera.

## 1. Título · 0:50

- **Ancla:** pregunta del título: *Which muons should FairShip simulate next?*
- Preséntate: doctorando de Chile; NF + FairShip para muones post-shield.
- Idea oral: "Generating muons with a flow is fast. The challenge is choosing which ones are worth simulating."
- Estado: el intercambio ya corre en P2; aún falta medir candidatos útiles por CPU-hora.
- **Puente:** *"So, how does this work?"* → clic.

## 2. Mapa · 1:45

- **Ancla:** NF propone, FairShip simula, U aprende de los resultados.
- Precisión: hoy U apunta a **≥1 good track**, no a toda la selección.
- Menciona que habrá un estudio aparte: hasta dónde extrapola nuestro flow hacia la cola.
- **Puente:** *"But first, why is muon DIS a problem?"* → clic.

## 3. Muon DIS · 2:45

- **Ancla:** algunos eventos DIS de muones pueden imitar la firma de dos trazas.
- Izquierda: señal con dos trazas apuntando al target. Derecha: muón post-shield interactúa en material cercano y produce una firma similar.
- Combinatorial es distinto y no entra en esta charla.
- **Puente:** *"The problem is that very few of these events actually survive the selection."* → clic.

## 4. El embudo · 4:00

- **Ancla:** un evento no es el problema; la gran cantidad necesaria para encontrar candidatos útiles sí lo es.
- No afirmar que *muon DIS* sea raro por sí mismo. El subconjunto que sobrevive a selección/veto es el relevante.
- Evidencia histórica de Martina, 32nd CM: **4.5 M** simulados ≈ **0.02%** del total esperado; extrapolación de coste ≈ **8 años** de CERN farm bajo el supuesto citado.
- Evitar "we've simulated" si suena a que tú generaste esa muestra.
- **Puente:** *"So, instead of simulating everything, can we be smarter about which muons we choose?"* → clic.

## 5. El loop · 5:15

- **Ancla:** FairShip simula; U aprende de las etiquetas que devuelve.
- Un estado en P2 → varias realizaciones DIS → contar éxitos de **≥1 good track**. Repetir un mismo estado no equivale a nuevos muones independientes de P0.
- U estima esa probabilidad condicional. No aprende "la física" de la densidad del NF.
- **Puente:** *"Now, how do we actually connect this to FairShip?"* → clic.

## 6. FairShip · 6:30

- **Ancla:** FairShip se mantiene sin modificar.
- Scripts existentes; entrada `MCTrack[0]` en P2; resultados en `cbmsim` y `ship_reco_sim`.
- Versión fija: cambios de interfaz pueden detener la ejecución; cambios de definición física pueden cambiar resultados sin fallar.
- **Puente:** *"So the interface works. Now, how far can our flow go beyond its training data?"* → clic.

## 7. Cola de nuestro flow · 7:45

- **Ancla:** entrenado sin la cola, nuestro flow casi no llega a ella. La dejamos fuera a propósito.
- Primero aclarar: *"This test uses the afterMS sample, not P2."*
- Enlazar con Eduard: *"Eduard already explained how flows work; ours is a small RealNVP on five muon coordinates."*
- Train: `|zpx| ≤ 2`; tail: `|zpx| > 2.5`. Sólo **41/50 000** muestras del flow llegan allí: ≈ **1% de la masa de la cola real**.
- **Puente:** *"So how fast does the error grow as we move away from the training data?"* → clic.

## 8. Error versus distancia · 9:00

- **Ancla:** el error del flow frente a muones reales que no vio crece con la distancia al training.
- El error es una distancia entre distribuciones (sliced Wasserstein) sobre las cinco coordenadas. Línea gris = dos muestras reales entre sí, el mismo piso de ruido que usó Eduard.
- Evidencia: **3×** sobre el piso cerca del training, **12×** en la banda más lejana.
- Las líneas verticales marcan dónde está el 95% de los muones de training y de validación.
- **Puente:** *"What does that mean in momentum?"* → clic.

## 9. Error en momento · 10:00

- **Ancla:** el mismo error en GeV/c.
- Evidencia: px desplazado **0.3 GeV/c** cerca (estadística sola: 0.08) y **5.7 GeV/c** en la banda más lejana.
- Cerrar con el backup: el mismo flow entrenado **con** muones de cola llega a ≈ **90%**. Es nuestro flow extrapolando, no un límite de los flows.
- **Puente:** *"So our flow is reliable near its data. That's why we want FairShip feedback to decide where to propose."* → clic.

## 10. Inclinar con U · 11:15

- **Ancla:** aprender U a partir de FairShip, favorecer estados prometedores, entrenar NF y volver a evaluar.
- **Lo demostrado:** round trip de integración. **Lo pendiente:** mejor rendimiento por CPU-hora.
- Piloto P2: 19 estados ponderados → **Kish ESS = 3.6**. Remuestrear no aporta información física independiente.
- Pregunta de benchmark: *"Does U-guided sampling find more useful candidates than unguided sampling for the same CPU budget?"*
- **Puente:** *"But we have a problem: we don't have any examples passing the final selection yet."* → clic.

## 11. Una etapa a la vez · 12:15

- **Ancla:** sin positivos en el endpoint final, empezar por una etapa anterior con positivos.
- Propuesta: aprender U en un endpoint intermedio; usarlo para orientar la búsqueda de la siguiente etapa. No asumir que mejor etapa anterior implica mejor etapa final.
- Pregunta abierta: cuántos positivos / qué evidencia hacen falta antes de avanzar.
- **Puente:** *"So instead of targeting the final selection directly, we can work through it one stage at a time."* → clic.

## 12. Etapas de selección · 13:30

- **Ancla:** siete grupos visuales resumen los cortes específicos; no son una sustitución de sus definiciones individuales.
- Estado actual: **≥1 good reconstructed track**; todavía no hay evidencia de superar toda la selección.
- Métrica final del piloto: **candidatos útiles por CPU-hora**, guiado vs. no guiado, mismo presupuesto.
- **Puente:** *"So, where are we now?"* → clic.

## 13. Resumen y agradecimientos · 14:45

- **Ancla:** round trip NF–FairShip en P2 operativo; en afterMS nuestro flow sin cola llega a ≈1% de su masa y a ≈90% con muones de cola en el training; falta comparar rendimiento guiado vs. baseline a mismo coste.
- Evita repetir literalmente la frase larga de la apertura.
- Agradecimiento natural:
  > "I'd also like to thank Orlando for supervising my work, Oliver for helping me with FairShip, and Martina for sharing her work and the muon samples. I really appreciate their support."
- **Puente:** termina el agradecimiento y pasa a la slide final.

## 14. Questions · 15:00

- Di una sola invitación, sin duplicar la pregunta:
  > "Thank you for your attention. I'd really appreciate your feedback, especially on which selection stage we should target next."
- Mira a la sala y deja la diapositiva proyectada mientras contestas.

## Si vas atrasado

- **Slide 3:** sólo la diferencia señal / DIS.
- **Slides 8–9:** di 3× y 12× en la 8 y sólo los 5.7 GeV/c de la 9.
- **Slides 11–12:** combínalas verbalmente; no expliques dos veces el escalonamiento.
- **Slides 4 y 10:** conserva la motivación de coste y el benchmark pendiente.

## Backups

- **16 (A vs B):** mismo flow, mismos 400 pasos: 0.6% de la masa de cola sin muones de cola, 89% con todos, 90% con el mismo tamaño que A. Lo que importa es ver muones de cola, no tener más filas.
- **17 (cuántos):** respecto a su propio training, la masa de cola sale bien desde ~100 muones de cola; la forma mejora hasta ~450 y queda en 1.6× el piso.
- **18 (distancia, A vs B):** cerca del training ambos coinciden con los reales; lejos sólo se despega el flow sin cola. En la última banda el completo también sube: 0.46 contra 1.31.
- **19–20:** NLL vs. distancia y "más datos del centro" (slides viejas 8 y 9).
- **22:** qué es nuestro flow y el split. **23:** inclinar el latente hacia la cola.

## Preguntas probables (respuestas cortas)

- **"¿Entonces los NF no sirven para colas?"** No es eso. Dejamos la cola fuera a propósito y es un RealNVP pequeño; con muones de cola en el training llega a ≈90% (backup 16). Eduard mostró flows entrenados en todo el flujo que reproducen sus colas.
- **"¿Esto es error o likelihood?"** Distancia entre muestras del flow y muones reales held-out, más razones de masa. La NLL está en backup 19.
- **"¿Qué es la línea gris?"** Lo que dan dos muestras reales del mismo tamaño entre sí; bajo eso no se puede medir error.
- **"¿Por qué 95% y no el máximo?"** El último muón de training y de validación está en 14.5 y 9.9, más lejos que cualquier muón de test, por unas pocas filas aisladas. El 95% describe dónde está casi todo el training.
- **"¿Cuántos muones de cola hacen falta?"** ~100 para la masa, unos cientos para la forma (backup 17).
- **"¿Y los pesos de afterMS?"** Todo sin pesos, training y comparación. Pendiente cuando se confirme qué significan.
- **"¿0.02% de qué?"** Cifra histórica de Martina (32nd CM) respecto a interacciones DIS esperadas; no la conviertas a spills sin base.
- **"¿Por qué no muestrear la cola de la gaussiana base?"** Las muestras pueden llegar a la cola, pero se concentran cerca de su borde (mediana `px`: 4.5 vs. 5.6 GeV/c en la cola real; backup 23).
- **"¿Y si el flow se entrena con sus propios datos?"** En el test de cinco generaciones de autoconsumo, la cobertura de cola reportada bajó de 0.99 a 0.81–0.88. No extrapoles sin indicar el experimento.
- **"¿Qué es NLL?"** Negative log-likelihood: `-ln qθ(x)` en nats para la representación declarada. Más alto = menor densidad asignada; no prueba error absoluto de densidad.
- **"¿Por qué algunos muones lejanos tienen NLL baja?"** En la muestra reportada son 36, casi todos con `px` negativo; la tendencia describe el grueso de la cola, no cada punto.
- **"¿Y el veto?"** El endpoint demostrado es good track; veto y selección final aún no están evaluados como endpoint en el loop.
- **"¿Qué pasa si FairShip cambia?"** Se fija `8a1cc53`; registrar el commit también en outputs es trabajo pendiente.
- **"¿Cuánto cuesta?"** Pruebas locales: ~0.4 a 1.7 s por DIS, según la región/material; no universalizar.
- **"No sé la respuesta."** "I haven't measured that yet. I'd need to check it in FairShip."
