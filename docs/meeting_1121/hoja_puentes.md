# Hoja de puentes — SHiP CM 39 (15 min)

## Regla rápida

1. Termina la idea de la slide actual.
2. Di una frase breve que abra la pregunta siguiente y **haz clic al terminarla**.
3. En la slide nueva empieza por la respuesta, no por anunciar "section 02" o "section 03".

Las frases son opciones para hablar, no un libreto obligatorio. Si olvidas una, utiliza la pregunta visible en el título de la próxima diapositiva.

## Los 13 puentes

| De → a | Conexión | Puente natural (antes del clic) | Aterrizaje (después del clic) |
|---|---|---|---|
| **1 → 2** | Pregunta general → método | *"So, how does this work?"* | "The flow proposes, FairShip simulates, and U learns from the results." |
| **2 → 3** | Método → motivación física | *"But first, why is muon DIS a problem?"* | "Some muon DIS events can look like the signal." |
| **3 → 4** | Firma peligrosa → coste | *"The problem is that very few of these events actually survive the selection."* | "Simulating one event isn't the problem. It's how many we need." |
| **4 → 5** | Demasiados eventos → selección inteligente | *"So, instead of simulating everything, can we be smarter about which muons we choose?"* | "Here's our approach: the flow proposes a muon at P2." |
| **5 → 6** | Idea del loop → integración real | *"Now, how do we actually connect this to FairShip?"* | "We don't modify FairShip; we call its existing scripts." |
| **6 → 7** | Interfaz operativa → extrapolación | *"So the interface works. Now, how far can our flow go beyond its training data?"* | "This is an afterMS benchmark, not the P2 loop. Eduard already explained flows; ours is a small RealNVP." |
| **7 → 8** | Poca masa en cola → error vs. distancia | *"So how fast does the error grow as we move away from the training data?"* | "Here I compare the flow with real muons it never saw, by distance to the training data." |
| **8 → 9** | Error en 5D → error en momento | *"What does that mean in momentum?"* | "Same bands, only px, in GeV/c." |
| **9 → 10** | Fiable cerca de sus datos → feedback | *"So our flow is reliable near its data. That's why we want FairShip feedback to decide where to propose."* | "We learn U from FairShip results and use it to favour promising states." |
| **10 → 11** | Loop funciona → faltan positivos finales | *"But we have a problem: we don't have any examples passing the final selection yet."* | "So we start with an earlier stage where we have positives." |
| **11 → 12** | Aprender abajo → avanzar por etapas | *"So instead of targeting the final selection directly, we can work through it one stage at a time."* | "These seven groups summarise the selection cuts." |
| **12 → 13** | Estrategia → balance honesto | *"So, where are we now?"* | "The NF–FairShip connection works at P2. The CPU-hour benchmark is next." |
| **13 → 14** | Agradecimientos → invitación | *"I really appreciate their support."* | "Thank you for your attention. I'd really appreciate your feedback, especially on which selection stage we should target next." |

## Cierre literal recomendado

**Slide 13 (resumen):**

> "To wrap up: we have the Normalizing Flow–FairShip connection working at P2. Separately, on afterMS, our flow trained without the tail gets about 1% of its mass, and about 90% once tail muons are in training. The next step is to test whether FairShip feedback helps us find more useful candidates at the same computing cost."

**Slide 13 (agradecimiento):**

> "I'd also like to thank Orlando for supervising my work, Oliver for helping me with FairShip, and Martina for sharing her work and the muon samples. I really appreciate their support."

**Slide 14 (preguntas):**

> "Thank you for your attention. I'd really appreciate your feedback, especially on which selection stage we should target next."

Deja la slide 14 visible. **No repitas la pregunta** como segunda invitación.

## Frases a evitar (por la charla de Eduard)

- "Normalizing flows fail in the tails" → *"Our flow, trained without the tail, doesn't reach it."*
- "More data doesn't help" → *"More centre data doesn't help the tail; tail examples do (backup 16)."*
- "The NF is bad at extrapolating" → *"Our small RealNVP doesn't extrapolate into a region it never saw."*

## Números para recordar

| Slide | Número clave | Qué significa |
|---|---|---|
| 4 | 4.5 M ≈ 0.02%; ~8 años | Estimación histórica de Martina, no medida de este piloto |
| 7 | 41/50 000; ≈1% | Muestreo de cola respecto a la masa real en **afterMS**, cola fuera del training a propósito |
| 8 | 3× → 12× el piso | Error en 5D (sliced W1) frente a muones reales held-out, de la banda más cercana a la más lejana |
| 9 | 0.3 → 5.7 GeV/c | Desplazamiento de px; la estadística sola da 0.08 → 0.31 |
| 10 | 19 estados; Kish ESS 3.6 | Piloto de ponderación en **P2**; no prueba de mejora de eficiencia |
| 12 | ≥1 good track | Endpoint intermedio alcanzado; selección final pendiente |
| backup 16 | 0.6% → 89% (90% mismo tamaño) | Mismo flow sin y con muones de cola en el training |
| backup 17 | ~100 masa; ~450 forma | Muones de cola necesarios en ese experimento |
