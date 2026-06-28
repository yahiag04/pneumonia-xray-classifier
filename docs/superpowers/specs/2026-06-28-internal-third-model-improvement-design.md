# Internal and Third Dataset Model Improvement Design

## Obiettivo

Migliorare le prestazioni dei modelli sul dataset interno e sul terzo dataset,
senza usare NIH come target principale di ottimizzazione. Il miglioramento deve
essere misurato soprattutto con balanced accuracy e specificity, mantenendo
sensitivity alta sulla classe `pneumonia`.

La baseline di confronto e costituita dai checkpoint fair gia disponibili:

```text
outputs/runs_fair/<model>/best.pt
```

e dai checkpoint fine-tuned sul terzo dataset:

```text
outputs/runs_third_finetune/<model>/best.pt
```

## Modelli

Gli esperimenti principali partono da:

- `efficientnet_b0`, perche e il migliore sul test interno fair;
- `resnet18`, perche e il migliore sul terzo dataset post fine-tuning per
  balanced accuracy;
- `mobilenet_v3_large`, opzionale, se serve includere un modello piu leggero.

`DenseNet121` resta utile come riferimento sperimentale, ma non e il primo
candidato per iterazioni rapide per via del costo computazionale superiore.

## Approccio

Il lavoro procede in due fasi.

### 1. Threshold tuning

Per ogni modello candidato si calcola la soglia decisionale migliore sulla
validation disponibile, invece di usare sempre `0.5`.

Le soglie vengono selezionate con criteri espliciti:

- massimizzazione della balanced accuracy;
- massimizzazione di F1 sulla classe `pneumonia`;
- eventuale vincolo minimo di sensitivity, per esempio `sensitivity >= 0.95`.

La soglia scelta sulla validation viene poi applicata al test interno e al test
bilanciato del terzo dataset. La soglia non deve essere scelta sul test.

### 2. Fine-tuning controllato

Si confrontano varianti piu conservative del fine-tuning:

- head-only: backbone congelato, si allena solo il classificatore finale;
- last-block: si sbloccano solo gli ultimi blocchi della rete;
- learning rate basso, ad esempio `1e-6` o `5e-6`;
- early stopping basato su balanced accuracy o su una metrica composita, non
  solo su validation loss.

L'obiettivo e ridurre i falsi positivi `normal -> pneumonia` mantenendo pochi
falsi negativi.

## Dati

Dataset interno:

```text
/Users/yahiaghallale/Downloads/chest_xray
```

Terzo dataset:

```text
/Users/yahiaghallale/pneumonia-xray-classifier/third dataset/third
```

Manifest bilanciati gia disponibili:

```text
outputs/third_dataset/third_train_balanced.csv
outputs/third_dataset/third_val_balanced.csv
outputs/third_dataset/third_test_balanced.csv
```

Il test interno e il test del terzo dataset devono restare set finali: non
vengono usati per scegliere soglie, checkpoint o iperparametri.

## Metriche

Ogni run deve riportare:

- `tn`, `fp`, `fn`, `tp`;
- accuracy;
- sensitivity;
- specificity;
- balanced accuracy;
- precision e F1 per la classe `pneumonia`;
- ROC-AUC;
- PR-AUC;
- tempo medio per immagine.

Il successo non viene definito dalla sola accuracy. Un esperimento e utile se
aumenta balanced accuracy e specificity senza una perdita eccessiva di
sensitivity.

## Output

Gli output sperimentali devono essere separati dalle baseline esistenti:

```text
outputs/threshold_sweeps/
outputs/runs_improved/
outputs/evaluations_improved/
```

I risultati riassuntivi vanno documentati in:

```text
docs/thesis/model-improvement-results.md
```

La bozza LaTeX puo essere aggiornata solo dopo aver scelto gli esperimenti
finali da includere.

## Interpretazione per la tesi

Questa fase deve essere presentata come miglioramento controllato del protocollo
sperimentale, non come ricerca indiscriminata della metrica piu alta.

La storia attesa e:

1. baseline con soglia `0.5`;
2. miglioramento tramite soglia calibrata su validation;
3. miglioramento o confronto tramite fine-tuning conservativo;
4. verifica sul test interno e sul terzo dataset;
5. discussione separata del fatto che questi miglioramenti non garantiscono
   automaticamente trasferimento su NIH.

## Validazione

Prima di accettare i risultati finali bisogna verificare che:

- le soglie siano state scelte solo su validation;
- i test set non siano usati per tuning;
- ogni tabella indichi checkpoint, soglia e dataset;
- gli script siano riproducibili da comandi documentati;
- i test automatici esistenti continuino a passare.
