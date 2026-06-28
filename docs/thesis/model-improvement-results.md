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
| Threshold tuning | resnet18 | Terzo balanced | 0.55 | 146 | 88 | 2 | 232 | 0.8077 | 0.9915 | 0.6239 | 0.8077 | 0.8375 | 0.9620 | 0.9565 |
| Head-only fine-tuning | efficientnet_b0 | Terzo balanced | 0.50 | 46 | 188 | 1 | 233 | 0.5962 | 0.9957 | 0.1966 | 0.5962 | 0.7115 | 0.9091 | 0.9155 |
| Last-block fine-tuning | resnet18 | Terzo balanced | 0.50 | 120 | 114 | 2 | 232 | 0.7521 | 0.9915 | 0.5128 | 0.7521 | 0.8000 | 0.9543 | 0.9504 |

## Lettura preliminare

Il threshold tuning sul checkpoint `resnet18` gia fine-tuned migliora
leggermente il test bilanciato del terzo dataset: la soglia scelta su validation
e `0.55`, con balanced accuracy test `0.8077` contro `0.8034` della baseline a
soglia `0.50`.

I due fine-tuning controllati eseguiti non migliorano la baseline:

- `efficientnet_b0` head-only si ferma a best epoch `2` e crolla in specificity
  sul test (`0.1966`);
- `resnet18` last-block raggiunge una validation molto alta, ma sul test resta a
  balanced accuracy `0.7521`, sotto la baseline `0.8034`.

Questi risultati indicano che, per il terzo dataset, la modifica piu utile tra
quelle provate e la calibrazione della soglia sul checkpoint gia fine-tuned,
mentre i nuovi fine-tuning conservativi non sono candidati migliori.

## Note operative

Il comando di threshold tuning interno per `efficientnet_b0` non e stato
eseguito perche il dataset interno non era disponibile nel path storico:

```text
/Users/yahiaghallale/Downloads/chest_xray
```

I manifest del terzo dataset contengono path relativi, quindi i comandi sono
stati eseguiti dalla root del checkout principale usando gli script del worktree.

## Comandi

Comandi verificati:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/select_thresholds.py --help
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/finetune_model.py --help
```
