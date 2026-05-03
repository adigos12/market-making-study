# Market Making Under Asymmetric Information

Synthetic-data implementation of the Glosten–Milgrom (1985) model of
market making with adverse selection. Companion code to the LaTeX
write-up of the same name.

## Components

- `generate.py` — synthetic trade-data generator under known parameters
- `mle.py` — panel maximum likelihood estimator for (μ, θ)
- `em.py` — latent-class logistic regression via Expectation–Maximisation
- `impact.py` — price-impact regression with bootstrapped standard errors

## Status

In development. See LaTeX write-up for full mathematical context.

---

Aditya Vasudev — University of Edinburgh, BSc Mathematics
