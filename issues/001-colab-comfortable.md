# 001: make Colab a comfortable compute target

Status: OPEN, deferred by the user on 2026-11-24 ("dejalo en una carpeta de issues para hacerlo despues").

Why: the server has 8 cores and 7.8 GB RAM, shared with FairShip production at night. When flows get bigger or
experiments need more seeds, Colab (100 credits available) should take the load.

What exists: `scripts/colab.sh MODULE LOCAL_OUT [args]` (commit `1fd1010`) ships `git archive HEAD` plus the afterMS
sample, runs the module, downloads the JSON and always stops the VM. Never run end to end.

Blockers and gaps:
- One-time login by the user: `colab sessions` asks for an OAuth code in a browser.
- The output JSON records `commit: ""` on Colab (no `.git` in the archive); pass the commit in an env var.
- Data upload: the afterMS sample goes to a Google VM. Confirm that this is acceptable for Martina's sample.
- Measure: wall time and credits for one planted run on CPU, T4 and L4, to know when Colab wins.

When to switch (rule written 2026-11-23): a bigger flow, more seeds than 8 local cores finish in an hour, or a night
when the server is busy with FairShip.
