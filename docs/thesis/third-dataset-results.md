# Third dataset - valutazione e fine-tuning

Data: 2026-06-04

## Dataset

Path locale:

```text
/Users/yahiaghallale/pneumonia-xray-classifier/third dataset/third
```

Struttura:

```text
train/COVID19
train/NORMAL
train/PNEUMONIA
train/TURBERCULOSIS
val/COVID19
val/NORMAL
val/PNEUMONIA
val/TURBERCULOSIS
test/COVID19
test/NORMAL
test/PNEUMONIA
test/TURBERCULOSIS
```

Nota: la cartella si chiama `TURBERCULOSIS`, con refuso rispetto a `TUBERCULOSIS`. Nel codice viene comunque trattata come classe patologica.

## Mappatura binaria

Per mantenere il task coerente con gli altri esperimenti:

- `NORMAL` -> `normal`;
- `COVID19`, `PNEUMONIA`, `TURBERCULOSIS` -> `pneumonia`.

Questa scelta va spiegata in tesi: il terzo dataset non e' solo Normal/Pneumonia classico, ma contiene piu' condizioni patologiche toraciche. In questa fase vengono unificate nella classe positiva per valutare un task binario normal vs anomalia/polmonite-like.

## Conteggi originali

| Split | COVID19 | Normal | Pneumonia | Turberculosis | Normal binario | Pneumonia binaria |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 460 | 1341 | 3875 | 650 | 1341 | 4985 |
| Val | 10 | 8 | 8 | 12 | 8 | 30 |
| Test | 106 | 234 | 390 | 41 | 234 | 537 |

## Bilanciamento usato

Dato che lo split `val` originale contiene solo `8` immagini normal, per il fine-tuning sono stati combinati `train+val`, bilanciati per classe e poi divisi in train/val stratificati.

Manifest generati:

```text
outputs/third_dataset/third_train_balanced.csv
outputs/third_dataset/third_val_balanced.csv
outputs/third_dataset/third_test_balanced.csv
outputs/third_dataset/third_trainval_balanced.csv
outputs/third_dataset/third_dataset_counts.csv
```

Conteggi:

| Manifest | Normal | Pneumonia | Totale |
| --- | ---: | ---: | ---: |
| Raw train+val | 1349 | 5015 | 6364 |
| Raw test | 234 | 537 | 771 |
| Balanced train | 1214 | 1214 | 2428 |
| Balanced val | 135 | 135 | 270 |
| Balanced test | 234 | 234 | 468 |

## Inference prima del fine-tuning

Protocollo:

- checkpoint: `outputs/runs_fair/<model>/best.pt`;
- manifest: `outputs/third_dataset/third_test_balanced.csv`;
- nessun training sul terzo dataset;
- soglia: `0.5`.

| Modello | TN | FP | FN | TP | Accuracy | Sensitivity | Specificity | Balanced acc. | F1 Pneumonia | ROC-AUC | PR-AUC | Sec/img |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PneumoniaNet | 99 | 135 | 3 | 231 | 0.7051 | 0.9872 | 0.4231 | 0.7051 | 0.7700 | 0.9122 | 0.8930 | 0.0099 |
| ResNet18 | 106 | 128 | 6 | 228 | 0.7137 | 0.9744 | 0.4530 | 0.7137 | 0.7729 | 0.9103 | 0.9102 | 0.0222 |
| MobileNetV3-Large | 114 | 120 | 2 | 232 | 0.7393 | 0.9915 | 0.4872 | 0.7393 | 0.7918 | 0.9384 | 0.9365 | 0.0310 |
| EfficientNet-B0 | 123 | 111 | 7 | 227 | 0.7479 | 0.9701 | 0.5256 | 0.7479 | 0.7937 | 0.9227 | 0.9285 | 0.0456 |
| DenseNet121 | 113 | 121 | 7 | 227 | 0.7265 | 0.9701 | 0.4829 | 0.7265 | 0.7801 | 0.9300 | 0.9294 | 0.0668 |

## Lettura pre-fine-tuning

Sul terzo dataset bilanciato i modelli generalizzano meglio che su NIH, ma mostrano ancora una tendenza chiara:

- sensitivity molto alta sulla classe positiva;
- specificity bassa o moderata sulla classe normal;
- molti falsi positivi normal -> pneumonia.

`EfficientNet-B0` e' il migliore a soglia `0.5` per balanced accuracy prima del fine-tuning. `MobileNetV3-Large` ha la ROC-AUC piu' alta tra questi risultati pre-fine-tuning.

## Fine-tuning

Protocollo previsto:

- checkpoint iniziale: `outputs/runs_fair/<model>/best.pt`;
- train: `outputs/third_dataset/third_train_balanced.csv`;
- val: `outputs/third_dataset/third_val_balanced.csv`;
- test: `outputs/third_dataset/third_test_balanced.csv`;
- output: `outputs/runs_third_finetune/<model>/best.pt`;
- learning rate: `1e-5`;
- massimo epoche: `10`;
- early stopping: `patience=3`;
- soglia: `0.5`.

## Risultati post fine-tuning sul terzo dataset

| Modello | Best epoch | TN | FP | FN | TP | Accuracy | Sensitivity | Specificity | Balanced acc. | F1 Pneumonia | ROC-AUC | PR-AUC | Sec/img |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PneumoniaNet | 10 | 140 | 94 | 3 | 231 | 0.7927 | 0.9872 | 0.5983 | 0.7927 | 0.8265 | 0.9497 | 0.9340 | 0.0103 |
| ResNet18 | 10 | 143 | 91 | 1 | 233 | 0.8034 | 0.9957 | 0.6111 | 0.8034 | 0.8351 | 0.9620 | 0.9565 | 0.0281 |
| MobileNetV3-Large | 6 | 142 | 92 | 2 | 232 | 0.7991 | 0.9915 | 0.6068 | 0.7991 | 0.8315 | 0.9618 | 0.9540 | 0.0382 |
| EfficientNet-B0 | 10 | 141 | 93 | 5 | 229 | 0.7906 | 0.9786 | 0.6026 | 0.7906 | 0.8237 | 0.9585 | 0.9607 | 0.0584 |
| DenseNet121 | 7 | 137 | 97 | 1 | 233 | 0.7906 | 0.9957 | 0.5855 | 0.7906 | 0.8262 | 0.9654 | 0.9616 | 0.0848 |

Nota: `DenseNet121` e' stata fine-tunata e valutata sul terzo dataset, ma viene esclusa dalla comparazione NIH before/after completa per costo computazionale. La valutazione NIH post fine-tuning e' stata completata, ma la valutazione NIH baseline e' stata interrotta per evitare tempi eccessivi.

## Lettura post fine-tuning

Il fine-tuning sul terzo dataset bilanciato migliora tutti i modelli rispetto alla inference pre-fine-tuning. Il miglioramento principale riguarda la specificity: i modelli riducono i falsi positivi `normal -> pneumonia`, pur mantenendo sensitivity molto alta.

Sul test bilanciato del terzo dataset, `ResNet18` ha la balanced accuracy migliore (`0.8034`). Il margine rispetto a `PneumoniaNet`, `MobileNetV3-Large`, `EfficientNet-B0` e `DenseNet121` e' pero' contenuto. `DenseNet121` ha la ROC-AUC piu' alta sul terzo dataset post fine-tuning (`0.9654`), ma e' anche il modello piu' lento.

## Test NIH post fine-tuning

Protocollo:

- checkpoint: `outputs/runs_third_finetune/<model>/best.pt`;
- manifest: `outputs/nih/nih_224_binary_manifest.csv`;
- campioni: `61.765`;
- soglia: `0.5`;
- output: `outputs/evaluations/<model>_nih_224_after_third_ft.json`.

| Modello | TN | FP | FN | TP | Accuracy | Sensitivity | Specificity | Balanced acc. | F1 Pneumonia | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PneumoniaNet | 10854 | 49558 | 106 | 1247 | 0.1959 | 0.9217 | 0.1797 | 0.5507 | 0.0478 | 0.6098 | 0.0291 |
| ResNet18 | 7765 | 52647 | 60 | 1293 | 0.1467 | 0.9557 | 0.1285 | 0.5421 | 0.0468 | 0.6692 | 0.0383 |
| MobileNetV3-Large | 13606 | 46806 | 107 | 1246 | 0.2405 | 0.9209 | 0.2252 | 0.5731 | 0.0504 | 0.6646 | 0.0390 |
| EfficientNet-B0 | 7681 | 52731 | 55 | 1298 | 0.1454 | 0.9593 | 0.1271 | 0.5432 | 0.0469 | 0.6422 | 0.0352 |

`DenseNet121` post fine-tuning su NIH ha prodotto: balanced accuracy `0.5614`, sensitivity `0.9387`, specificity `0.1842`, ROC-AUC `0.6745`. Non viene inserita nella tabella comparativa principale perche' manca la valutazione baseline NIH completa, interrotta per tempi eccessivi.

## Confronto NIH prima/dopo fine-tuning sul terzo dataset

| Modello | Bal. acc. prima | Bal. acc. dopo | Delta | Sens. prima | Sens. dopo | Delta | Spec. prima | Spec. dopo | Delta | ROC-AUC prima | ROC-AUC dopo | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PneumoniaNet | 0.5783 | 0.5507 | -0.0276 | 0.7672 | 0.9217 | +0.1545 | 0.3894 | 0.1797 | -0.2097 | 0.6054 | 0.6098 | +0.0045 |
| ResNet18 | 0.5940 | 0.5421 | -0.0520 | 0.8692 | 0.9557 | +0.0865 | 0.3189 | 0.1285 | -0.1904 | 0.6908 | 0.6692 | -0.0216 |
| MobileNetV3-Large | 0.5924 | 0.5731 | -0.0193 | 0.8596 | 0.9209 | +0.0613 | 0.3252 | 0.2252 | -0.1000 | 0.6805 | 0.6646 | -0.0159 |
| EfficientNet-B0 | 0.5909 | 0.5432 | -0.0477 | 0.8404 | 0.9593 | +0.1190 | 0.3415 | 0.1271 | -0.2143 | 0.6696 | 0.6422 | -0.0274 |

## Lettura NIH post fine-tuning

Il fine-tuning sul terzo dataset migliora le prestazioni sul test del terzo dataset, ma non migliora la generalizzazione su NIH a soglia `0.5`.

Il pattern e' netto:

- la sensitivity su NIH aumenta per tutti i modelli confrontabili;
- la specificity crolla, quindi aumentano molto i falsi positivi `No Finding -> Pneumonia`;
- la balanced accuracy peggiora in tutti e quattro i modelli confrontabili;
- la ROC-AUC resta simile solo per `PneumoniaNet`, mentre cala per ResNet18, MobileNetV3-Large ed EfficientNet-B0.

Interpretazione per la tesi: il terzo dataset bilanciato aiuta l'adattamento al proprio dominio, ma non basta a risolvere il domain shift verso NIH. Anzi, a soglia `0.5` spinge ulteriormente i modelli verso una decisione positiva, aumentando la detection della classe patologica ma peggiorando la capacita' di riconoscere i casi `No Finding`.

## Analisi esplorativa delle soglie su NIH

Per verificare se la soglia decisionale `0.5` fosse troppo bassa dopo il
fine-tuning, ogni modello e' stato eseguito una sola volta sul manifest NIH. Le
probabilita' ottenute sono state poi riutilizzate per calcolare le metriche alle
soglie `0.50`, `0.60`, `0.65` e `0.70`.

Output:

```text
outputs/evaluations/nih_threshold_sweep_after_third_ft.json
outputs/evaluations/nih_threshold_sweep_after_third_ft.csv
```

| Modello | Soglia | TN | FP | FN | TP | Sensitivity | Specificity | Balanced acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PneumoniaNet | 0.50 | 10854 | 49558 | 106 | 1247 | 0.9217 | 0.1797 | 0.5507 |
| PneumoniaNet | 0.60 | 12309 | 48103 | 125 | 1228 | 0.9076 | 0.2038 | 0.5557 |
| PneumoniaNet | 0.65 | 13052 | 47360 | 133 | 1220 | 0.9017 | 0.2160 | 0.5589 |
| PneumoniaNet | 0.70 | 13922 | 46490 | 139 | 1214 | 0.8973 | 0.2305 | 0.5639 |
| ResNet18 | 0.50 | 7765 | 52647 | 60 | 1293 | 0.9557 | 0.1285 | 0.5421 |
| ResNet18 | 0.60 | 8996 | 51416 | 67 | 1286 | 0.9505 | 0.1489 | 0.5497 |
| ResNet18 | 0.65 | 9707 | 50705 | 68 | 1285 | 0.9497 | 0.1607 | 0.5552 |
| ResNet18 | 0.70 | 10486 | 49926 | 73 | 1280 | 0.9460 | 0.1736 | 0.5598 |
| MobileNetV3-Large | 0.50 | 13606 | 46806 | 107 | 1246 | 0.9209 | 0.2252 | 0.5731 |
| MobileNetV3-Large | 0.60 | 15622 | 44790 | 131 | 1222 | 0.9032 | 0.2586 | 0.5809 |
| MobileNetV3-Large | 0.65 | 16609 | 43803 | 143 | 1210 | 0.8943 | 0.2749 | 0.5846 |
| MobileNetV3-Large | 0.70 | 17634 | 42778 | 154 | 1199 | 0.8862 | 0.2919 | 0.5890 |
| EfficientNet-B0 | 0.50 | 7681 | 52731 | 55 | 1298 | 0.9593 | 0.1271 | 0.5432 |
| EfficientNet-B0 | 0.60 | 8870 | 51542 | 66 | 1287 | 0.9512 | 0.1468 | 0.5490 |
| EfficientNet-B0 | 0.65 | 9460 | 50952 | 75 | 1278 | 0.9446 | 0.1566 | 0.5506 |
| EfficientNet-B0 | 0.70 | 10122 | 50290 | 84 | 1269 | 0.9379 | 0.1675 | 0.5527 |
| DenseNet121 | 0.50 | 11128 | 49284 | 83 | 1270 | 0.9387 | 0.1842 | 0.5614 |
| DenseNet121 | 0.60 | 12126 | 48286 | 93 | 1260 | 0.9313 | 0.2007 | 0.5660 |
| DenseNet121 | 0.65 | 12711 | 47701 | 99 | 1254 | 0.9268 | 0.2104 | 0.5686 |
| DenseNet121 | 0.70 | 13315 | 47097 | 108 | 1245 | 0.9202 | 0.2204 | 0.5703 |

Nel range esaminato, la balanced accuracy cresce in modo monotono per tutti i
modelli e raggiunge il valore piu' alto a `0.70`. Questo risultato conferma che
la soglia `0.5` e' troppo permissiva sul dominio NIH. L'aumento della soglia
riduce i falsi positivi e migliora la specificity, mentre la sensitivity resta
alta:

| Modello | Bal. acc. 0.50 | Bal. acc. 0.70 | Delta | Sens. 0.70 | Spec. 0.70 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PneumoniaNet | 0.5507 | 0.5639 | +0.0132 | 0.8973 | 0.2305 |
| ResNet18 | 0.5421 | 0.5598 | +0.0177 | 0.9460 | 0.1736 |
| MobileNetV3-Large | 0.5731 | 0.5890 | +0.0160 | 0.8862 | 0.2919 |
| EfficientNet-B0 | 0.5432 | 0.5527 | +0.0095 | 0.9379 | 0.1675 |
| DenseNet121 | 0.5614 | 0.5703 | +0.0089 | 0.9202 | 0.2204 |

`MobileNetV3-Large` ottiene la balanced accuracy migliore dell'analisi
(`0.5890` a soglia `0.70`) e la specificity piu' alta (`0.2919`). Tuttavia, la
specificity resta bassa per tutti i modelli e il domain shift non viene risolto
dal solo cambiamento di soglia.

Poiche' tutti i modelli migliorano ancora al limite superiore del range
analizzato, `0.70` va descritta come la migliore soglia tra quelle provate, non
come soglia ottima. Inoltre, questa analisi usa lo stesso manifest NIH per
confrontare le soglie: e' quindi esplorativa e non costituisce una stima finale
priva di data leakage. Una calibrazione metodologicamente corretta richiede uno
split NIH validation/test separato, preferibilmente a livello di paziente.
