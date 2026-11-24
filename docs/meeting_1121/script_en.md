# Talk script — SHiP CM 39 (computing session), 15 min

> **Core message:** Generating muons with a Normalizing Flow is fast. The challenge is choosing which ones are worth simulating in FairShip. We want to use FairShip's feedback to make that choice better.

**How to use this script:** Times are the *target end time* for each slide, not an instruction to speak continuously. Pause to point at the plots. **Bold** marks emphasis; *italics* mark the transition, said just before changing slides. These are speaking prompts, not lines to recite word for word.

**Context:** Eduard's talk on normalizing flows for the neutrino flux comes right before this one. Don't re-explain how flows work, and keep every limitation tied to *our* small flow and *our* held-out tail.

## 1. Title · 0:50

Hello everyone. I'm Fabián Trigo, a PhD student from Chile. I work on machine learning for simulation in SHiP, combining a **Normalizing Flow** with FairShip to study muons after the shield.

So, which muons should FairShip simulate next? Generating muons with a flow is fast. The challenge is choosing **which ones** are worth simulating. Our interface already works at P2, but we still need to measure whether it helps us get more useful candidates per CPU-hour.

*So, how does this work?*

## 2. Map · 1:45

The main idea is simple: the flow **proposes** a muon, FairShip simulates what happens next, and a utility score, U, learns from the results. For now, U is meant to estimate the chance of getting **at least one good track**, not passing the full selection.

I'll also show a separate test of how far our flow can extrapolate into the tail of the muon distribution.

*But first, why is muon DIS a problem?*

## 3. Muon DIS · 2:45

On the left, we have the signal: two charged tracks pointing back to the target. On the right, a muon that survives the shield interacts in nearby material, and the particles it produces can **look like the signal**.

There's also combinatorial background, but today I'll focus on muon DIS.

*The problem is that very few of these events actually survive the selection.*

## 4. The funnel · 4:00

Simulating one event isn't the problem. The problem is **how many** we need to simulate before finding a useful background candidate. Most events are rejected somewhere along this funnel.

As Martina showed at the 32nd meeting, about **4.5 million** muon DIS events had been simulated. That's only **0.02%** of the expected total. At the same reported computing cost, simulating the full sample would take roughly eight years on the CERN computing farm.

*So, instead of simulating everything, can we be smarter about which muons we choose?*

## 5. The loop · 5:15

Here's our approach. The flow proposes a muon state at **P2**. FairShip generates the DIS interaction, transports the particles, and reconstructs the event.

We can simulate several DIS interactions for the same muon and count how often we get **at least one good track**. That success fraction gives us a label to learn from. U tries to estimate this probability from the muon's state. It learns from **FairShip results**, not from the flow's own density.

This is still an intermediate endpoint, not the final background selection.

*Now, how do we actually connect this to FairShip?*

## 6. FairShip · 6:30

**We don't modify FairShip.** We call three of its scripts. The muon enters at P2 as `MCTrack[0]`, and we read the results from `cbmsim` and `ship_reco_sim`.

We pin a specific FairShip commit so we can reproduce the setup. If an interface changes, the run may stop. If a selection definition changes, it could quietly change the results. That's why version tracking matters.

*So the interface works. Now, how far can our flow go beyond its training data?*

## 7. The tail · 7:45

This test uses the **afterMS** sample, not P2. Eduard already explained how flows work; ours is a small RealNVP on five muon coordinates.

We train it on the centre, where `|zpx| ≤ 2`, and **hold the tail out on purpose**, beyond `2.5`, to see how far it extrapolates. Out of 50,000 generated muons, only **41** land in that tail. That's about **1% of the tail mass** in the real sample.

*So how fast does the error grow as we move away from the training data?*

## 8. Error versus distance · 9:00

Here I compare the flow with real muons it never saw, grouped by how far each one is from the training data. The error is a distance between the two distributions over all five coordinates. The grey line is what two real samples of the same size give each other, the same noise-floor idea Eduard used. An error only counts above it.

Close to the training data the flow is about **3 times** above the floor. In the farthest band it's **12 times** above. The vertical lines mark where 95% of the training and validation muons sit.

*What does that mean in momentum?*

## 9. Error in momentum · 10:00

Same bands, only px, in GeV/c. Near the training data the flow's px distribution is off by about **0.3 GeV/c**, where statistics alone give 0.08. In the farthest band it's off by **5.7 GeV/c**.

In backup, the same flow trained **with** tail muons gets about **90%** of the tail mass. So what we see here is our flow extrapolating into a region it never saw.

*So our flow is reliable near its data. That's why we want FairShip feedback to decide where to propose.*

## 10. Tilt by U · 11:15

Here's the idea. We learn U from FairShip results, use it to **favour promising muons**, and train the flow with those weights. Then we send new proposals back to FairShip.

The round trip runs, but we **haven't shown an efficiency gain** yet. In this small P2 test, weighting 19 states gives a Kish ESS of 3.6. Resampling those states doesn't create new physics information.

The real test is still ahead: **does U-guided sampling find more useful candidates than unguided sampling for the same CPU budget?**

*But we have a problem: we don't have any examples passing the final selection yet.*

## 11. One stage at a time · 12:15

Right now, we don't have any examples that pass the **full selection**. So we can't directly learn a useful U for that final endpoint from positive examples.

Instead, we can start at an earlier stage, where we do have positives, and use what we learn to propose states for the next stage. We still need to decide **how much evidence** is enough before moving on.

*So instead of targeting the final selection directly, we can work through it one stage at a time.*

## 12. Selection stages · 13:30

These seven groups summarise the selection cuts. The idea is to use the information from one stage to guide proposals for the next.

So far, we have reached **at least one good reconstructed track**. We don't yet have a muon passing the full selection. And improving an early stage doesn't automatically mean we improve the final one.

What we ultimately want to compare is **useful candidates per CPU-hour**, guided versus unguided, with the same simulation budget.

*So, where are we now?*

## 13. Summary and acknowledgements · 14:45

To wrap up: we have the **Normalizing Flow–FairShip connection** working at P2. Separately, on afterMS, our flow trained without the tail gets about 1% of its mass, and about 90% once tail muons are in training. The next step is to test whether FairShip feedback helps us find **more useful candidates at the same computing cost**.

I'd also like to thank **Orlando** for supervising my work, **Oliver** for helping me with FairShip, and **Martina** for sharing her work and the muon samples. I really appreciate their support.

*[Change to the final slide.]*

## 14. Questions · 15:00

Thank you for your attention. I'd really appreciate your **feedback**, especially on which **selection stage** we should target next.

[Pause and invite questions. Do not repeat the question in another form.]

---

## If you are running late

- **Slide 3:** Give the signal-versus-DIS contrast and skip the extra background explanation.
- **Slides 8–9:** Say the two numbers on slide 8 (3× near, 12× far) and give only the 5.7 GeV/c from slide 9.
- **Slides 11–12:** Explain the staged approach while showing the seven groups; don't repeat it twice.
- **Slides 4 and 10:** Keep the cost motivation and the fair-budget benchmark question.

## Backup slides

| slide | content | use it when asked |
|---|---|---|
| 16 | Same flow with and without tail muons in training: 0.6% vs 89% of the tail mass, 90% at equal training size | "Is it the model or the data?" |
| 17 | How many tail muons: amount right from ~100, shape levels off around 450 at 1.6× the floor | "How many tail samples do you need?" |
| 18 | Error vs distance for both flows: only the centre-only flow leaves the floor | "Does the full flow also degrade far out?" |
| 19–20 | Old NLL vs distance; more centre data | "What about likelihood?" / "More data?" |
| 22 | What our flow is and the data split | "Which flow, which split?" |
| 23 | Steering the latent into the tail | "Why not sample the tail of the base Gaussian?" |

Oral lines for the new backups:

- **16:** "Same flow, same 400 steps. Trained on the centre only, it gives 0.6% of the real tail mass. Trained on everything, 89%. If I cut the full set down to the centre-only size, still 90%, so what matters is seeing tail muons, not having more rows."
- **17:** "Measured against its own training data, the flow gets the tail mass right from about 100 tail muons. The tail shape keeps improving up to about 450, and then stays at 1.6 times the floor."
- **18:** "Same distance axis for both flows. Near the training data both match the real muons. Further out only the centre-only flow falls behind. In the last band the full flow rises too, to 0.46 against 1.31."

## Scientific guardrails for questions

- **afterMS ≠ P2:** The tail diagnostics are on afterMS; the FairShip round-trip uses P2. Do not merge the empirical claims.
- **Our flow ≠ normalizing flows in general:** We held the tail out on purpose, and the model is a small RealNVP. Trained with tail muons it recovers about 90%. Never say flows can't do tails.
- **Distribution distance ≠ likelihood:** Slides 8–9 compare flow samples with held-out real muons. The NLL plot is in backup, and higher NLL only means lower fitted density.
- **Unweighted:** Training and comparison ignore the afterMS weights.
- **Good track ≠ final background:** Preselection and veto are not part of the current demonstrated endpoint.
- **Technical failures ≠ rejected physics events:** A failed FairShip run is not evidence that a muon failed a cut.
- **No efficiency claim yet:** The CPU-hour comparison against a matching unguided baseline is pending.
