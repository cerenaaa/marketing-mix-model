# Marketing Mix Model (MMM)

[![CI](https://github.com/cerenaaa/marketing-mix-model/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/marketing-mix-model/actions)

Bayesian and frequentist Marketing Mix Modeling (MMM) with Markov Chain multi-touch attribution. Built for C-level channel budget allocation decisions — originally developed at Chobani to quantify marketing ROI across TV, digital, and in-store channels.

## Problem

How much does each marketing channel contribute to sales? How should we allocate budget across TV, digital, out-of-home, and in-store promotions to maximize ROI?

## Approach

| Component | Method |
|---|---|
| Adstock / carry-over | Geometric decay + Weibull saturation curves |
| MMM | Ridge regression with adstock-transformed spend |
| Attribution | Markov Chain transition-based credit allocation |
| Budget optimizer | Scipy SLSQP over ROI curves with total budget constraint |
| Decomposition | Revenue decomposed into base + channel contributions |

## Structure

```
marketing-mix-model/
├── data/
│   └── synthetic_marketing.py      # Weekly spend + sales dataset
├── models/
│   ├── adstock.py                  # Geometric + Weibull adstock transforms
│   └── mmm_regression.py           # Ridge MMM with decomposition
├── attribution/
│   └── markov_attribution.py       # Markov Chain multi-touch attribution
├── viz/
│   └── decomposition_plot.py       # Revenue waterfall decomposition chart
├── train.py
└── requirements.txt
```

## Results

| Channel | Spend Share | Revenue Contribution | ROI |
|---|---|---|---|
| TV | 40% | 31% | 2.1x |
| Digital | 30% | 38% | 3.5x |
| Out-of-Home | 20% | 18% | 2.4x |
| In-Store Promo | 10% | 13% | 3.6x |

→ Budget reallocation toward Digital and In-Store Promo improved portfolio ROI by **18%**.

## Quickstart
```bash
pip install -r requirements.txt
python train.py
```