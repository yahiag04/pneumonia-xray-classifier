# Model improvement results

Data: 2026-06-28

## Obiettivo

Questa fase misura miglioramenti su dataset interno e terzo dataset tramite:

- soglia decisionale scelta su validation;
- fine-tuning controllato con backbone congelato o ultimo blocco sbloccato.

NIH resta una valutazione separata di domain shift e non viene usato come target
di ottimizzazione in questa fase.

## Esperimenti pianificati

### Threshold tuning

Modelli:

- `efficientnet_b0`
- `resnet18`

Output:

```text
outputs/threshold_sweeps/<model>_internal_validation.json
outputs/threshold_sweeps/<model>_internal_validation.csv
outputs/threshold_sweeps/<model>_third_validation.json
outputs/threshold_sweeps/<model>_third_validation.csv
```

### Fine-tuning controllato

Modalita:

- `head`
- `last_block`

Output:

```text
outputs/runs_improved/<model>_<mode>/best.pt
outputs/evaluations_improved/<model>_<mode>_third_test.json
```

## Tabella risultati

| Esperimento | Modello | Dataset test | Soglia | TN | FP | FN | TP | Accuracy | Sensitivity | Specificity | Balanced acc. | F1 Pneumonia | ROC-AUC | PR-AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | efficientnet_b0 | Interno | 0.50 | 174 | 60 | 1 | 389 | 0.9022 | 0.9974 | 0.7436 | 0.8705 | 0.9273 | 0.9859 | n/a |
| Baseline | resnet18 | Terzo balanced | 0.50 | 143 | 91 | 1 | 233 | 0.8034 | 0.9957 | 0.6111 | 0.8034 | 0.8351 | 0.9620 | 0.9565 |

## Comandi

Comandi da compilare dopo l'implementazione:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/select_thresholds.py --help
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/finetune_model.py --help
```
