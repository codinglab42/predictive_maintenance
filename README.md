# Manutenzione predittiva

Pipeline sperimentale basata principalmente sul dataset NASA IMS Bearing.
Estrae feature vibrazionali dai tre test IMS, addestra un MLP implementato in
C++ e accessibile tramite pybind11 e usa il `2nd_test` come verifica tenuta
separata. La regressione logistica è mantenuta come benchmark. FEMTO/PRONOSTIA
resta disponibile come pipeline secondaria.

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

Per estrarre gli archivi IMS servono inoltre `py7zr`, `libarchive-c` e una
installazione di sistema di libarchive; i due pacchetti Python sono inclusi in
`requirements.txt`.

## Dataset principale: NASA IMS

Scarica l'archivio ufficiale del dataset IMS (contiene un 7z annidato, che a sua
volta contiene gli archivi RAR delle tre serie) ed estrailo nella cartella
condivisa:

```bash
mkdir -p /mnt/shared/predictive-maintenance/downloads
curl -L --fail \
  'https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip' \
  -o /mnt/shared/predictive-maintenance/downloads/ims_bearings.zip
python scripts/extract_ims.py
```

Il pacchetto ufficiale è grande circa 1,08 GB. `extract_ims.py` verifica i
percorsi interni e produce `raw/ims/1st_test`, `raw/ims/2nd_test` e
`raw/ims/3rd_test`; l'archivio della terza serie contiene internamente la
cartella `4th_test/txt`, che viene normalizzata a `3rd_test/txt`. Dati grezzi e
archivi sono esclusi da Git. Per riprendere l'estrazione dai RAR già preparati,
si può passare la loro cartella con `--archive`.

Per tenere i file grezzi nella cartella locale esclusa da Git invece che nella
cartella condivisa:

```bash
python scripts/extract_ims.py --output-dir data/raw/ims
```

Prepara le feature, addestra la rete neurale C++ tramite pybind11 e il
benchmark logistico, e valuta entrambi sul `2nd_test`:

```bash
python scripts/train_predictive_maintenance.py
```

Se i dati sono stati estratti localmente:

```bash
python scripts/train_predictive_maintenance.py --data-dir data/raw/ims
```

Il training usa solo `1st_test` e `3rd_test`. Per ogni bearing e serie, le sei
feature dei primi 30 file cronologici vengono mediate e sottratte dall'intera
sequenza; per il `2nd_test` la stessa trasformazione usa soltanto i suoi primi
30 file, senza etichette. Lo `StandardScaler` è poi fit-tato sul training
normalizzato e applicato al test senza rifit. Il modello è un MLP C++ `6-12-1`
con ReLU, output Sigmoid, learning rate `0.001`, 60 epoche e regolarizzazione
L2 (`0.01`); la pipeline usa `predict_proba` per ottenere i punteggi Sigmoide,
non le classi di `predict`. Le sei feature sono RMS, deviazione standard,
curtosi, asimmetria, picco-picco e fattore di cresta. Nel `1st_test` le due
colonne di ciascun cuscinetto sono elaborate separatamente e le sei feature
risultanti vengono mediate; nel `2nd_test` e `3rd_test` ogni cuscinetto ha una
sola colonna.
L'ordinamento e i grafici usano i timestamp dei file senza imporre intervalli
fissi: nel `1st_test`, per esempio, i primi due timestamp distano 2 minuti e
49 secondi.

I bearing sani hanno target 0 in tutti i campioni. Per i bearing indicati come
guasti nel dataset, il 75% iniziale della sequenza ha target 0 e l'ultimo 25%
target 1: bearing 3 e 4 in `1st_test`, bearing 1 in `2nd_test`, bearing 3 in
`3rd_test`. Per i campioni il cui numero non è divisibile per quattro, l'inizio
della finestra positiva è l'indice `floor(0.75 * N)`.

Il training valuta il MLP alle soglie `0.50`, `0.65`, `0.75`, `0.80`, `0.85`,
`0.90`, `0.95`, `0.97`, `0.99`, `0.995` e `0.999`, selezionando il massimo F1
e, a parità, meno allarmi sui bearing sani e poi la soglia maggiore. **Questa
selezione usa le etichette del `2nd_test`**: i risultati MLP così ottenuti sono
esplorativi e non costituiscono una verifica indipendente. La regressione
logistica resta riportata alla soglia baseline `probabilità > 0.7`. Il MLP è salvato in
`models/ims_neural_network.bin`, la regressione logistica di benchmark in
`models/ims_logistic_regression.bin`; parametri scaler e checksum sono in
`models/ims_preprocessing.json`. Le feature e le predizioni cronologiche del
test e le metriche comparative vengono salvate in
`/mnt/shared/predictive-maintenance/processed/`.
Nella verifica corrente la soglia selezionata è `0.999` (Precision `0.7524`,
Recall `0.6423`, F1 `0.6930`); la regressione logistica ha F1 `0.6059`.
`predict_proba` produce migliaia di valori distinti. I quattro bearing sono
verdi al primo file; il primo allarme del Bearing 1 è al file 703, prima
dell'inizio della coda target al file 739. Bearing 2 e 4 hanno alcuni falsi
allarmi più tardi (primi allarmi ai file 891 e 972); Bearing 3 non supera la
soglia selezionata.
Il training stampa e salva nel JSON anche i tempi di fit e verifica separati
per MLP e regressione logistica. Per ripetere soltanto fit e verifica usando le
feature già estratte:

```bash
python scripts/train_predictive_maintenance.py \
  --data-dir data/raw/ims \
  --processed-dir /mnt/shared/predictive-maintenance/processed \
  --reuse-features
```

Avvia il cruscotto collegato alle quattro sequenze del `2nd_test`:

```bash
streamlit run app.py
```

Il cruscotto mostra le probabilità cronologiche dei quattro cuscinetti del
`2nd_test`, con replay Play/Pausa, soglia configurabile, Health Index e stati
di allarme. Le probabilità sono state prodotte dalla rete neurale C++ tramite
pybind11; l'app usa i file elaborati e non riesegue l'inferenza al cambio di
acquisizione.

## Dataset secondario: FEMTO/PRONOSTIA

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

## Pipeline FEMTO/PRONOSTIA

```bash
python scripts/train_femto_predictive_maintenance.py
```

Per usare i dati sulla cartella condivisa, indica la stessa posizione e salva
anche le feature elaborate lì:

```bash
python scripts/train_femto_predictive_maintenance.py \
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
python scripts/train_femto_predictive_maintenance.py --test-bearing Bearing1_1
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
L'output riporta i percorsi effettivi del segnale e del modello e il checksum
SHA-256 del modello, così la previsione è riconducibile all'artefatto usato.
I nuovi modelli salvano il checksum anche in `models/preprocessing.json`;
metadati precedenti senza checksum restano utilizzabili, ma viene mostrato un
avviso perché l'abbinamento tra modello e preprocessing non è verificabile.
Se il checksum non corrisponde, la previsione viene rifiutata.

## Cruscotto industriale StoneCut PdM

Avvia il replay dimostrativo NASA IMS dalla radice del progetto:

```bash
streamlit run app.py
```

Il cruscotto usa `processed/ims_test_features.npz`,
`processed/ims_test_predictions.csv` e `processed/ims_test_metrics.json` nella
cartella condivisa `/mnt/shared/predictive-maintenance/`. Il modulo
`backend.py` carica e mette in cache i dati, mentre `app.py` gestisce la
visualizzazione. `scripts/dashboard.py` resta disponibile come entry point
compatibile.

La barra laterale permette di scegliere Italiano, English o 中文, scorrere i
984 file del `2nd_test`, avviare o mettere in pausa il replay e regolare la
soglia di allarme (preselezionata alla soglia scelta nel tuning MLP e salvata
nei risultati, con precisione al millesimo). La vista principale mostra lo stato della
macchina, l'indice di salute globale, i quattro componenti della macchina da
taglio e le probabilità di guasto fino all'acquisizione selezionata. Il
pannello tecnico confronta Precision/Recall dei due modelli, mostra RMS e
curtosi e consente di scaricare un report CSV o TXT per i cuscinetti critici.

È un replay storico del benchmark, non un collegamento a sensori live. La
coda positiva del 25% è un'etichetta proxy e non l'istante osservato del
guasto; la soglia MLP selezionata sul `2nd_test` e i relativi punteggi non sono
una misura indipendente né sono calibrati per uso industriale. Il report
raccomanda una verifica umana e non autorizza una sostituzione automatica.

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

Dataset principale: [NASA Prognostics Data Repository — IMS Bearings](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
Il dataset è fornito dal Center for Intelligent Maintenance Systems (IMS),
University of Cincinnati. Il repository NASA richiede di citare dataset e
donatori e dichiara che l'uso dei dati è a rischio dell'utente.

Dataset secondario: [NASA Prognostics Data Repository — FEMTO Bearing](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

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
