# NIH Threshold Sweep Design

## Obiettivo

Valutare tutti i cinque checkpoint fine-tuned sul terzo dataset sul manifest NIH
completo usando soglie decisionali `0.50`, `0.60`, `0.65` e `0.70`.
L'esperimento deve mostrare se una soglia piu alta riduce i falsi positivi
`No Finding -> Pneumonia` e migliora la balanced accuracy.

## Modelli

- `pneumonia_net`
- `resnet18`
- `mobilenet_v3_large`
- `efficientnet_b0`
- `densenet121`

I checkpoint sono letti da:

```text
outputs/runs_third_finetune/<model>/best.pt
```

Il dataset esterno e:

```text
outputs/nih/nih_224_binary_manifest.csv
```

## Architettura

Ogni modello esegue una sola inferenza sull'intero manifest NIH. Durante
l'inferenza vengono raccolte etichette reali e probabilita predette, che sono
poi riutilizzate per calcolare le metriche a tutte le soglie richieste.

La logica di calcolo delle metriche resta centralizzata in
`thesis.metrics.compute_binary_metrics`. Lo script dedicato coordina i modelli,
salva i risultati e non modifica i checkpoint.

## Output

Lo script produce:

- un JSON contenente metadati e metriche complete per modello e soglia;
- un CSV tabellare con una riga per ogni coppia modello-soglia;
- un riepilogo in console con balanced accuracy, sensitivity e specificity.

Per ogni soglia vengono salvati:

- `tn`, `fp`, `fn`, `tp`;
- accuracy;
- sensitivity;
- specificity;
- balanced accuracy;
- precision e F1 per entrambe le classi;
- ROC-AUC;
- PR-AUC;
- supporto delle classi.

ROC-AUC e PR-AUC non cambiano al variare della soglia, ma restano nel file per
rendere ogni riga autosufficiente.

## Interpretazione

La soglia `0.50` resta la baseline ufficiale per il confronto zero-shot gia
riportato. Le soglie superiori costituiscono un'analisi esplorativa della
distribuzione delle probabilita.

La soglia con balanced accuracy piu alta puo essere evidenziata per ciascun
modello, ma non deve essere presentata come prestazione finale su test esterno:
la sua selezione sullo stesso manifest NIH introduce data leakage. Un futuro
risultato calibrato richiedera uno split NIH validation/test separato,
preferibilmente a livello di paziente.

## Validazione

I test automatici devono verificare che:

- l'inferenza restituisca etichette e probabilita riutilizzabili;
- ogni soglia produca una riga distinta;
- al crescere della soglia i positivi predetti non aumentino;
- JSON e CSV contengano tutti i cinque modelli e tutte le quattro soglie.

L'esecuzione reale deve completarsi per tutti i modelli, inclusa DenseNet121,
senza ripetere l'inferenza per ogni soglia.
