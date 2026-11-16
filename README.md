# Manutenzione predittiva

Pipeline dimostrativa basata sulle misure sperimentali di vibrazione dei
cuscinetti NASA FEMTO/PRONOSTIA. Scarica ed estrae il dataset, ricava feature
temporali dai canali orizzontale e verticale, addestra una regressione logistica
e salva modello e preprocessing.

I dati provengono da prove accelerate di laboratorio, non da macchine in
produzione. Le metriche sono una verifica tecnica della pipeline e non
dimostrano prestazioni industriali.

## Requisiti

`machineLearning` supporta Python 3.11 (`>=3.11,<3.12`). Crea l'ambiente,
installa NumPy e installa la libreria locale `ml_core` (le cartelle devono
essere sorelle):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

cd ../machineLearning
python -m pip install -e .
cd ../predictive_maintenance
```

## Download dei dati

Avvia esplicitamente il download NASA:

```bash
python scripts/download_femto.py
```

Se stai lavorando in una VM e hai configurato una cartella condivisa con il
sistema host, puoi salvarci i dati passando il percorso **visibile dalla VM**:

```bash
python scripts/download_femto.py --data-root /mnt/shared/predictive-maintenance
```

Il dataset sarà in
`/mnt/shared/predictive-maintenance/raw/femto/`. Il percorso va adattato al
punto di mount effettivo della cartella condivisa (ad esempio quello configurato
in VirtualBox, VMware, Hyper-V o WSL); il comando non può accedere direttamente
a un percorso host che la VM non ha montato.

L'archivio ufficiale è circa 1,16 GB e contiene un archivio ZIP annidato.
Servono quindi almeno alcuni GB liberi: lo script controlla lo spazio prima
del download e di ogni estrazione, riprende un download interrotto e rifiuta
percorsi pericolosi negli archivi. Al termine rimuove gli ZIP temporanei; usa
`--keep-archive` per conservarli.

I dati vengono estratti in `data/raw/femto/`. Dataset scaricati, dati grezzi e
feature processate sono esclusi da Git.

## Addestramento e valutazione

```bash
python scripts/train_predictive_maintenance.py
```

Per usare i dati sulla cartella condivisa, indica la stessa posizione e salva
anche le feature elaborate lì:

```bash
python scripts/train_predictive_maintenance.py \
  --data-dir /mnt/shared/predictive-maintenance/raw/femto \
  --processed-dir /mnt/shared/predictive-maintenance/processed
```

Modello e parametri di preprocessing restano in `models/` nel progetto; sono
molto più piccoli del dataset e non richiedono la cartella condivisa.

Durante il training, il terminale mostra il numero e la percentuale dei
campioni elaborati, la fase corrente della pipeline e l'avanzamento
dell'ottimizzazione del modello.

Il training usa le sequenze run-to-failure (Learning/Training set, quando
presenti). Di default tiene fuori un intero cuscinetto per il test; per
sceglierne uno esplicitamente:

```bash
python scripts/train_predictive_maintenance.py --test-bearing Bearing1_1
```

Si possono tenere fuori più cuscinetti ripetendo l'opzione. Il target è
positivo per le ultime `--horizon` acquisizioni della sequenza run-to-failure
(default: 20, inclusa l'ultima misura). Il conteggio considera i record
ordinati, anche se gli ID delle acquisizioni hanno salti. Questa è una proxy
riproducibile del guasto imminente, non una previsione in giorni o secondi. Non
si assume che le acquisizioni avvengano a intervalli temporali noti.

Il preprocessing calcola RMS, deviazione standard, picco-picco, massimo
assoluto, crest factor e kurtosis per ciascun canale. Standardizzazione e
modello sono stimati solo sui cuscinetti di training. Le feature estratte
vengono salvate in `processed/femto_features.npz` (oppure nel percorso indicato
con `--processed-dir`) con il relativo
`femto_features.json`; modello e parametri sono salvati in `models/`.
Il modello salvato è addestrato con il target corrente a 20 acquisizioni
totali; `models/preprocessing.json` riporta l'orizzonte e la partizione di
test usati.

Il dataset include poche sequenze indipendenti: i risultati dipendono molto dal
cuscinetto scelto come test e non sono una stima robusta della generalizzazione.
Per un lavoro sperimentale serio usare validazione raggruppata ripetuta,
metriche a livello di evento e definire l'orizzonte in base alla cadenza reale
delle acquisizioni.

### Validazione annidata delle politiche di allarme

Per eseguire la selezione delle politiche di allarme con una validazione
annidata leave-one-bearing-out:

```bash
python scripts/validate_nested_policies.py \
  --data-dir /mnt/shared/predictive-maintenance/raw/femto \
  --output-dir /mnt/shared/predictive-maintenance/processed
```

### Validazione Leave-One-Condition-Out (LOCO)

Il dataset FEMTO/PRONOSTIA raggruppa i cuscinetti per condizione operativa
secondo il prefisso del bearing (`Bearing1_*`, `Bearing2_*`, `Bearing3_*`). La
baseline LOCO lascia fuori tutti i cuscinetti di una condizione e seleziona la
politica di allarme solo sulle condizioni di training rimanenti, senza usare i
risultati della condizione esterna per scegliere parametri:

```bash
python scripts/validate_loco_policies.py \
  --data-dir /mnt/shared/predictive-maintenance/raw/femto \
  --output-dir /mnt/shared/predictive-maintenance/processed
```

L'opzione legacy `--weighted-loss` attiva oversampling della classe
minoritaria nel training, non una loss pesata. Non usarla per il confronto
standard; il JSON registra la modalità di campionamento usata.

Il file CSV generato riporta probabilità e notifica per campione; il JSON
riassume metriche per condizione e per obiettivo, in modo da confrontare la
baseline LOCO con la validazione leave-one-bearing-out esistente.

Per confrontare la regressione logistica con una baseline a soglia sulla sola
RMS orizzontale, genera prima (o conserva) i risultati standard LOCO e poi
riutilizzali per il confronto:

```bash
python scripts/validate_loco_policies.py \
  --data-dir /mnt/shared/predictive-maintenance/raw/femto \
  --output-dir /mnt/shared/predictive-maintenance/processed/loco_rms_comparison

python scripts/compare_loco_rms_baseline.py \
  --data-dir /mnt/shared/predictive-maintenance/raw/femto \
  --output-dir /mnt/shared/predictive-maintenance/processed/loco_rms_comparison
```

Il punteggio RMS è standardizzato usando solo il training di ciascuna piega.
Le soglie candidate sono ricavate dai punteggi out-of-fold interni; soglia,
conferma consecutiva e cooldown vengono selezionati con l'obiettivo evento
h20 (`balanced_f1`) senza usare la condizione esterna. Le predizioni e le
policy della regressione logistica vengono riutilizzate dal relativo LOCO
archiviato, evitando variazioni dovute a una nuova esecuzione del modello. Il
report include una riclassificazione esplorativa delle notifiche per finestre
di preavviso; tale analisi secondaria non partecipa alla selezione della
policy. Gli artefatti sono `femto_loco_rms_baseline_h20_predictions.csv` e
`femto_loco_rms_baseline_h20_results.json`.

Il valore predefinito dell'orizzonte è 20 acquisizioni totali. Le soglie,
conferme consecutive e cooldown sono selezionati solo sulle pieghe interne;
ogni cuscinetto della piega esterna resta escluso sia dall'addestramento sia
dalla selezione. Lo script salva predizioni e metriche nei file
`femto_nested_h20_oof_alarm_predictions.csv` e
`femto_nested_h20_validation_results.json`, senza modificare il modello
operativo o i suoi parametri. Per ogni cuscinetto vengono addestrati i modelli
delle pieghe interne e quello della piega esterna, quindi l'esecuzione può
richiedere alcuni minuti.

Per creare una timeline campione per campione e confrontare gli episodi della
politica F1 selezionata:

```bash
python scripts/analyze_alarm_timeline.py \
  --processed-dir /mnt/shared/predictive-maintenance/processed
```

Il CSV `femto_alarm_timeline_h20_balanced_f1.csv` contiene distanza
dall'ultima acquisizione, probabilità, notifica e feature orizzontali per ogni
campione. `femto_alarm_episodes_h20_balanced_f1.csv` riassume gli episodi sopra
soglia, inclusi quelli non confermati; il JSON associato confronta i
cuscinetti in fasce di distanza dal termine. `--objective` permette di
selezionare un'altra politica annidata.

### Analisi spettrale esplorativa dei segnali grezzi

Per confrontare i segnali originali di un cuscinetto tra riferimento lontano
dalla fine, episodi confermati prima di h20, tratto pre-finestra di un allarme
persistente e ultime 20 acquisizioni:

```bash
python scripts/analyze_raw_spectra.py \
  --data-dir /mnt/shared/predictive-maintenance/raw/femto \
  --processed-dir /mnt/shared/predictive-maintenance/processed
```

Il grafico e il CSV delle potenze per banda vengono salvati in
`processed/` (o nella directory indicata con `--output-dir`). L'analisi usa i
due canali grezzi, frequenza di campionamento di 25,6 kHz e confronta mediane
descrittive degli spettri e coerenza tra canali. Non individua automaticamente
descrittive degli spettri. Un secondo grafico aggiunge riferimenti di frequenza
per uno scenario NSK 6307DU: il modello è nominato in uno studio applicativo
PRONOSTIA, mentre numero e diametro degli elementi volventi (8, 13,49 mm),
diametro primitivo (57,5 mm) e angolo di contatto (0°) sono ipotesi geometriche
da verificare, non misure documentate nei file FEMTO. Con 1800 rpm nominali lo
scenario produce FTF ≈ 11,48 Hz, BSF ≈ 60,42 Hz, BPFO ≈ 91,85 Hz e BPFI ≈
148,15 Hz. Il grafico mostra fondamentale e seconda/terza armonica; il CSV
riporta la potenza nel bin FFT più vicino, il massimo nei bin adiacenti e il
rapporto rispetto al gruppo di riferimento per ciascuna armonica fino a 400 Hz.
Ogni acquisizione dura 0,1 s e la risoluzione FFT è 10 Hz,
quindi la vicinanza dei picchi è solo un controllo esplorativo, non una
diagnosi. La velocità effettiva può differire da quella nominale e le
acquisizioni sono brevi e temporalmente correlate.

## Predizione su una nuova acquisizione

Passa il CSV FEMTO originale a sei colonne (gli ultimi due campi sono i canali
di accelerazione orizzontale e verticale). Sono supportati anche CSV già ridotti
a due colonne numeriche senza intestazione:

```bash
python scripts/predict_machine.py --signal-csv data/raw/femto/Learning_set/Bearing1_1/acc_00001.csv
```

Puoi sovrascrivere la soglia usata per l'alert con `--threshold 0.4`.

## Cruscotto di replay FEMTO

Per visualizzare lo stato dei sei cuscinetti del benchmark, l'andamento RMS e
le notifiche della validazione LOCO:

```bash
streamlit run scripts/dashboard.py
```

Il cruscotto usa `processed/femto_features.npz` e
`processed/loco_rms_comparison/femto_loco_h20_alarm_predictions.csv` nella
cartella condivisa `/mnt/shared/predictive-maintenance/`. Se i file si trovano
altrove, cambia i percorsi nella barra laterale dell'app. La barra laterale
indica anche il dataset FEMTO/PRONOSTIA, l'accelerazione vibratoria misurata sui
due assi e l'RMS orizzontale mostrato nel grafico; l'unità non è specificata
negli artefatti utilizzati.

L'interfaccia ha le schede **Quadro d'insieme** e **Dettaglio**. Selezionando
una scheda cuscinetto nel quadro d'insieme si apre il relativo dettaglio con
lo stato, lo score e il grafico.

Il cursore **Posizione nella sequenza storica (%)** confronta i cuscinetti alla
stessa percentuale della rispettiva sequenza; non rappresenta tempo reale.
Verde indica score sotto la soglia sperimentale,
giallo un superamento non ancora confermato, rosso una notifica emessa dalla
policy LOCO e da sottoporre a verifica del manutentore; lo sfondo dei pulsanti
nel quadro d'insieme riprende il colore dello stato. Il rosso resta attivo
nel replay perché non sono disponibili dati di presa in carico o risoluzione.
È un prototipo dimostrativo basato su sei prove accelerate: non è collegato a
sensori live, lo score non è calibrato e i colori non sono validati per
decisioni operative o sostituzioni.

## Analisi esplorativa dei dataset pubblici

Oltre a FEMTO, il progetto include analisi indipendenti di NASA C-MAPSS e
XJTU-SY. Con i dati nella cartella condivisa prevista, esegui:

```bash
python scripts/analyze_public_datasets.py
```

Il comando si aspetta `raw/cmapss/` e `raw/xjtu_sy/` sotto
`/mnt/shared/predictive-maintenance/`. È possibile indicare un'altra radice e
cartella risultati:

```bash
python scripts/analyze_public_datasets.py \
  --data-root /path/to/predictive-maintenance/raw \
  --output-dir /path/to/predictive-maintenance/processed/public_datasets
```

L'analisi C-MAPSS stima la RUL in cicli con una regressione ridge e seleziona
la regolarizzazione tramite cross-validation a cinque fold raggruppata per
motore; riporta anche un baseline basato solo sul ciclo e valuta i modelli
sulle etichette RUL ufficiali del test. XJTU-SY confronta, sugli stessi
split leave-one-bearing-out, regressori ridge con solo tempo trascorso, sole
feature vibrazionali, tempo più vibrazioni e contesto operativo più vibrazioni.
Il baseline stima la vita tipica della condizione usando gli altri cuscinetti
della stessa condizione. La RUL è il numero di intervalli di campionamento di
un minuto fino all'ultima acquisizione registrata nel run-to-failure.
Nell'ablazione XJTU-SY, il modello tempo più vibrazioni ha ridotto la macro
MAE per bearing da 855,9 a 661,6 minuti rispetto al solo tempo (migliora in
11/15 bearing). Il modello completo ha ottenuto 425,0 minuti contro 465,8 del
baseline per condizione, ma il vantaggio varia tra le condizioni. È un
risultato retrospettivo ed esplorativo, non una validazione per decisioni
industriali.

I risultati sono salvati in JSON e CSV sotto `processed/public_datasets/`.
Le feature XJTU-SY estratte sono memorizzate in
`xjtu_sy_features_cache.npz`; il manifest dei percorsi, dimensioni e date
modifica evita di riusare la cache se i file cambiano. Per forzare l'estrazione
si può aggiungere `--refresh-xjtu-cache`. Le metriche dei dataset non vanno
aggregate o interpretate come prova diretta di prestazioni industriali o di
una decisione di sostituzione.
Fonti: [NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
e [XJTU-SY Bearing Datasets](https://biaowang.tech/xjtu-sy-bearing-datasets/).

## Test

I test unitari non scaricano il dataset:

```bash
python -m unittest discover -s tests
```

## Fonte e citazione

Dataset: [NASA Prognostics Data Repository — FEMTO Bearing](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
Il repository NASA richiede di citare dataset e donatori e dichiara che l'uso
dei dati è a rischio dell'utente.

Riferimento della piattaforma sperimentale:

> P. Nectoux, R. Gouriveau, K. Medjaher, E. Ramasso, B. Morello, N. Zerhouni,
> “PRONOSTIA: An Experimental Platform for Bearings Accelerated Life Test,”
> IEEE International Conference on Prognostics and Health Management, 2012.

## Licenza

Questo progetto è distribuito con licenza duale: GNU AGPL v3 o successiva per
l'uso open source oppure licenza commerciale per integrazioni proprietarie e
closed source. Per i termini della licenza commerciale, contattare
`mauriziopenna@gmail.com`. Il testo AGPL completo e l'avviso di dual licensing
sono nel file [LICENSE](./LICENSE).
