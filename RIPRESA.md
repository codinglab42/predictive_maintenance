# Ripresa: manutenzione predittiva NASA IMS

## Stato corrente (23 novembre 2026)

Il dataset principale del progetto è ora NASA IMS Bearing. L'archivio ufficiale
è stato scaricato in
`/mnt/shared/predictive-maintenance/downloads/ims_bearings.zip`; il pacchetto
contiene un 7z annidato, che a sua volta contiene gli archivi RAR di `1st_test`,
`2nd_test` e `3rd_test`. L'estrazione è completa nella cartella locale esclusa
da Git `predictive_maintenance/data/raw/ims/`; le feature sono state elaborate,
il modello è stato addestrato e la verifica su `2nd_test` è stata eseguita.
Le cache delle feature e i risultati sono in
`/mnt/shared/predictive-maintenance/processed/`.

Preparazione e protocollo richiesti:

- Ogni file viene ordinato per il timestamp nel nome. Per ciascun bearing si
  calcolano RMS, deviazione standard, curtosi, asimmetria, picco-picco e
  fattore di cresta. Il cruscotto usa i timestamp effettivi senza assumere
  cadenza fissa: nei primi due file di `1st_test` la differenza è 2 min 49 s.
- In `1st_test` ci sono otto colonne, due per bearing: le sei statistiche
  vengono calcolate separatamente sui due canali e poi mediate per ottenere un
  vettore di sei feature. `2nd_test` e `3rd_test` hanno una colonna per bearing.
- Nell'archivio NASA la terza serie è contenuta sotto `4th_test/txt`; lo script
  di estrazione normalizza questa directory in `3rd_test/txt`.
- Tutti i campioni dei bearing sani hanno target 0. Per i bearing che
  terminano in guasto, il primo 75% della sequenza è etichettato 0 e l'ultimo
  25% 1: bearing 3 e 4 di `1st_test`, bearing 1 di `2nd_test`, bearing 3 di
  `3rd_test`.
- `1st_test` e `3rd_test` formano il training; `2nd_test` resta separato per
  l'inferenza. Prima dello scaler, ogni bearing viene centrato sottraendo a
  tutte le sue feature la media delle prime 30 acquisizioni cronologiche. Per
  il test si usano solo le feature non etichettate dei suoi primi 30 file; lo
  scaler viene comunque fit-tato solo sul training.
- Il modello principale è l'MLP C++ esposto via pybind11 con architettura
  `6-12-1`, ReLU e Sigmoid, learning rate `0.001`, 60 epoche e regolarizzazione
  L2 (`0.01`). La pipeline usa `predict_proba` per il punteggio Sigmoide; il
  metodo `predict` del binding può restituire classi 0/1. La soglia MLP viene
  selezionata esplorativamente sul `2nd_test`; le metriche così ottenute non
  sono una valutazione indipendente.
- L'entry point principale è `scripts/train_predictive_maintenance.py`;
  la pipeline storica FEMTO resta disponibile in
  `scripts/train_femto_predictive_maintenance.py`.
- Il cruscotto riproduce in ordine cronologico le quattro sequenze di
  `2nd_test`; preimposta la soglia MLP selezionata e permette di regolarla.
  Sopra soglia mostra l'avviso **“Consigliato controllo/sostituzione”**.

La soglia e le etichette sono una convenzione sperimentale richiesta per questo
benchmark, non una regola operativa validata. Il punteggio non è calibrato e
l'avviso richiede verifica umana; non autorizza né prescrive una sostituzione.
Il lavoro precedente su FEMTO/PRONOSTIA resta documentato nelle sezioni
storiche seguenti e nella pipeline secondaria.

### Risultati IMS riproducibili

La rete C++ usa il seed `42`. Con le nuove impostazioni, `predict_proba`
restituisce 3.825 valori distinti sui 3.936 campioni del test. Il CSV delle
predizioni a soglia 0,999 ha SHA-256
`d075eed74353471f55a12fff6d04ffb8bc8baefed31f6b64b81403562db68333`.

| Modello sul `2nd_test` | Soglia | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Rete neurale C++ (`6-12-1`, tuning esplorativo) | 0,999 | 0,7524 | 0,6423 | 0,6930 |
| Regressione logistica (baseline) | 0,70 | 0,7688 | 0,5000 | 0,6059 |

Tutti e quattro i bearing sono verdi al file 1. Bearing 1 supera la soglia per
la prima volta al file **703**, prima dell'inizio della coda target proxy al
file **739**. A soglia 0,999 i conteggi sono Bearing 1 **194/984**, Bearing 2
**9/984**, Bearing 3 **0/984**, Bearing 4 **7/984**; i primi allarmi dei
bearing 2 e 4 sono rispettivamente ai file 891 e 972. La normalizzazione
elimina quindi l'allarme iniziale del Bearing 3 e, fino al file 739, il solo
allarme è quello anticipato sul Bearing 1; restano falsi allarmi tardivi sui
bearing sani 2 e 4.

| Soglia | Precision | Recall | F1-Score | Allarmi sani (B2+B3+B4) |
|---:|---:|---:|---:|---:|
| 0,50 | 0,5152 | 0,6911 | 0,5903 | 102 |
| 0,65 | 0,5611 | 0,6911 | 0,6193 | 81 |
| 0,75 | 0,5914 | 0,6707 | 0,6286 | 66 |
| 0,80 | 0,6052 | 0,6667 | 0,6344 | 60 |
| 0,85 | 0,6236 | 0,6667 | 0,6444 | 53 |
| 0,90 | 0,6560 | 0,6667 | 0,6613 | 40 |
| 0,95 | 0,6849 | 0,6626 | 0,6736 | 32 |
| 0,97 | 0,7026 | 0,6626 | 0,6820 | 27 |
| 0,99 | 0,7207 | 0,6504 | 0,6838 | 24 |
| 0,995 | 0,7407 | 0,6504 | 0,6926 | 19 |
| 0,999 (selezionata) | 0,7524 | 0,6423 | 0,6930 | 16 |

Il training contiene 33.920 campioni (2.659 positivi); `2nd_test` contiene
3.936 campioni (246 positivi). La soglia è `probabilità > 0,999`.

| Run | Fit rete neurale | Fit logistica | Verifica rete neurale | Verifica logistica | Totale fit + verifica |
|---|---:|---:|---:|---:|---:|
| Fit/verifica su feature cache | 1,417 s | 2,354 s | 0,002 s | < 0,001 s | 3,774 s |

I tempi comprendono fit, predizione sul test e calcolo delle metriche; escludono
estrazione feature, baseline normalization, scaling e scrittura degli
artefatti. La selezione della soglia usa le etichette del test ed è
esplorativa, non una stima indipendente della generalizzazione.

### Indicazioni tecniche reperite per definire i requisiti

Le linee guida pubbliche non forniscono un unico preavviso minimo o rapporto
di costo valido per ogni cuscinetto e impianto. ISO 17359 dà indicazioni
generali per impostare un programma di condition monitoring; ISO 13373-1
riguarda le procedure generali di monitoraggio delle vibrazioni. Sono
riferimenti per progettare il processo e le misure, non una soglia universale
di sostituzione applicabile direttamente ai punteggi di questo esperimento.
La documentazione del NASA Prognostics Data Repository specifica inoltre che
i dati sono raccolti per sviluppare algoritmi prognostici e che l'uso dei dati
e dei sistemi derivati è a rischio dell'utente: i benchmark run-to-failure non
validano da soli un allarme per un impianto.

Per questo progetto le indicazioni pratiche sono:

- **Preavviso:** definirlo a partire dal tempo necessario per verificare
  l'allarme, pianificare l'intervento e ottenere ricambi/fermo macchina, con un
  margine concordato con la manutenzione. Calcolarlo in tempo reale dai
  timestamp o dalla cadenza documentata; non convertire h20 in minuti finché
  la cadenza FEMTO non è verificata per gli artefatti usati.
- **Costo degli errori:** non adottare un rapporto generico trovato in un
  benchmark. Per un allarme d'ispezione, conteggiare almeno il carico/costo
  delle ispezioni non necessarie; per un mancato allarme, stimare guasto,
  fermo e possibili danni conseguenti. I pesi vanno concordati con chi gestisce
  l'impianto. Riportare anche recall per evento, falsi allarmi per unità di
  tempo e distribuzione del preavviso, non solo F1.
- **Output:** emettere un allarme di ispezione con evidenza e contesto; la
  sostituzione resta una decisione umana dopo verifica. Non trasformare il
  punteggio del classificatore in un comando automatico di sostituzione.

Fonti consultate il 12 novembre 2026:

- [ISO 17359:2018 — Condition monitoring and diagnostics of machines —
  General guidelines](https://www.iso.org/standard/71194.html).
- [ISO catalogue search — ISO 13373-1, vibration condition monitoring,
  general procedures](https://www.iso.org/search.html?q=ISO%2013373-1).
- [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/),
  descrizione e avvertenza sull'uso dei dataset.

## Stato storico FEMTO (12 novembre 2026)

Il lavoro corrente cerca di capire se le feature di vibrazione consentono di
anticipare il termine delle sequenze e di sostenere la decisione di
sostituzione, e quanto punteggi e politiche di notifica si trasferiscano tra
cuscinetti e condizioni operative. È una valutazione sperimentale, non una
regola industriale già validata né l'identificazione di una causa fisica.

- Dataset FEMTO/PRONOSTIA: 7.534 acquisizioni per 6 cuscinetti, due per
  ciascuna condizione; target h20 = ultime 20 acquisizioni positive per
  bearing (120 positivi, circa 1,59%).
- Le feature sono 12: RMS, deviazione standard, picco-picco, massimo
  assoluto, crest factor e kurtosis per ciascun canale orizzontale e verticale.
- Modello: regressione logistica; scaler stimato solo sui dati di training.
  Il modello operativo e il target h20 non sono stati modificati dagli
  esperimenti LOCO.
- I dati e gli artefatti elaborati si trovano in
  `/mnt/shared/predictive-maintenance/`.

### Risultato LOCO h20

La validazione Leave-One-Condition-Out tiene fuori entrambi i cuscinetti di
una condizione; la soglia e la politica sono selezionate usando solo gli altri
cuscinetti. Il confronto appaiato più recente con RMS orizzontale usa
l'obiettivo `balanced_f1`; la politica è soglia / conferma consecutiva /
cooldown, con durate espresse in acquisizioni. Questi risultati provengono
dalla nuova esecuzione standard salvata in
`/mnt/shared/predictive-maintenance/processed/loco_rms_comparison/`.
La baseline è lo z-score della sola RMS orizzontale, standardizzato sul
training di ogni piega. Le soglie candidate sono quantili dei punteggi
out-of-fold interni; soglia, conferma e cooldown sono selezionati solo sui
bearing interni con la metrica evento h20.

| Condizione test | Politica | Bearing | Rilevato in h20 | Episodi falsi | Precoci persistenti | Anticipo notifica |
|---|---|---|---:|---:|---:|---:|
| 1 | 0,50 / 3 / 0 | Bearing1_1 | No | 13 | 1 | — |
| 1 | 0,50 / 3 / 0 | Bearing1_2 | No | 0 | 1 | — |
| 2 | 0,10 / 3 / 50 | Bearing2_1 | No | 1 | 1 | — |
| 2 | 0,10 / 3 / 50 | Bearing2_2 | No | 2 | 1 | — |
| 3 | 0,30 / 5 / 50 | Bearing3_1 | No | 0 | 0 | — |
| 3 | 0,30 / 5 / 50 | Bearing3_2 | Sì | 0 | 0 | 4 |

Nel confronto appaiato, la regressione logistica rileva 1/6 bearing nella
finestra h20, con 16 episodi falsi, precisione evento 5,9%, recall 16,7% e F1
8,7%. Per condizione, il F1 evento è 0% (condizione 1), 0% (condizione 2) e
66,7% (condizione 3). La tabella per-bearing indica il conteggio complessivo
di falsi e precoci persistenti per fold nel riepilogo seguente:

| Modello | Rilevati | Episodi falsi | Precoci persistenti | Precisione evento | Recall evento | F1 evento |
|---|---:|---:|---:|---:|---:|---:|
| Regressione logistica | 1/6 | 16 | 4 | 5,9% | 16,7% | 8,7% |
| Baseline RMS orizzontale | 2/6 | 11 | 2 | 15,4% | 33,3% | 21,1% |

La baseline RMS rileva 1/2 bearing nella condizione 1, 0/2 nella condizione 2
e 1/2 nella condizione 3; i rispettivi F1 evento sono 16,7%, 0% e 66,7%.
Questa singola esecuzione favorisce la baseline sui valori aggregati, ma non
dimostra una superiorità generale con appena sei bearing.

**Nota di riproducibilità (verificata):** la differenza tra il riepilogo
storico e il confronto appaiato non era una variazione casuale del modello,
ma una diversa modalità di training. La ricostruzione dell'artefatto storico
con `--weighted-loss` ha prodotto un CSV delle predizioni identico byte per
byte a quello storico: quel flag attiva oversampling della classe minoritaria.
La riesecuzione standard senza oversampling ha invece prodotto un CSV
identico byte per byte a quello usato nel confronto appaiato.

È stato verificato anche il percorso della libreria C++ caricata: per default
il loader trova la copia installata nell'ambiente Python, mentre è presente
anche una build nella cartella sorella `machineLearning`. Pur avendo hash
diversi, entrambe le build hanno prodotto lo stesso CSV nella riesecuzione
standard controllata; la scelta del binario non spiega quindi la discrepanza
osservata in questo esperimento. L'artefatto storico non registrava i metadati
`weighted_loss` e `training_sampling`; quelli sono ora salvati nel JSON. I due
risultati descrivono regimi di training distinti e non vanno mescolati. Per il
confronto appaiato con RMS si usa la regressione logistica standard, senza
oversampling.

I risultati non indicano semplicemente che il modello assegni probabilità
basse in h20. Nel CSV appaiato, la logistica ha tutti i 20 campioni h20 sopra
soglia in quattro bearing, nessuno in Bearing3_1 e 9/20 in Bearing3_2; per
RMS i conteggi sono rispettivamente 20/20, 20/20, 20/20, 19/20, 0/20 e
13/20. Le notifiche h20 restano però una sola per la logistica e due per RMS:
le notifiche fuori finestra o persistenti prima di h20 non soddisfano la
tempistica richiesta. Il punteggio, il superamento della soglia e la
notifica operativa sono quindi risultati diversi.

### Separazione del punteggio dalla decisione di notifica

Come controllo descrittivo, l'AUC ROC è stata calcolata separatamente per
bearing sui punteggi LOCO esterni e sulle etichette h20. Non è stata usata per
selezionare le politiche; misura solo l'ordinamento dei campioni del bearing
esaminato e non dimostra che una soglia trasferisca a un nuovo bearing.

| Bearing | AUC logistica / RMS | Campioni h20 sopra soglia (logistica / RMS) | Notifiche in h20 (logistica / RMS) | Lead più vicino al termine, acquisizioni (logistica / RMS) |
|---|---:|---:|---:|---:|
| Bearing1_1 | 1,000 / 1,000 | 20/20 / 20/20 | 0 / 0 | 52 / 68 |
| Bearing1_2 | 1,000 / 1,000 | 20/20 / 20/20 | 0 / 1 | 23 / 16 |
| Bearing2_1 | 1,000 / 1,000 | 20/20 / 20/20 | 0 / 0 | 31 / 26 |
| Bearing2_2 | 0,926 / 0,992 | 20/20 / 19/20 | 0 / 0 | 42 / 24 |
| Bearing3_1 | 0,998 / 0,997 | 0/20 / 0/20 | 0 / 0 | — / — |
| Bearing3_2 | 1,000 / 1,000 | 9/20 / 13/20 | 1 / 1 | 3 / 7 |

Le AUC entro bearing sono alte, ma sono una misura retrospettiva e non
valutano una decisione calibrata tra bearing. In particolare, in
Bearing1_1, Bearing2_1 e Bearing2_2 le notifiche più vicine al termine sono
ancora fuori da h20; Bearing1_2 riceve una notifica RMS a 16 acquisizioni dal
termine ma nessuna notifica logistica nella finestra. In Bearing3_1 nessuna
delle due soglie produce una notifica; in Bearing3_2 entrambe notificano in
h20. Il lead è contato in acquisizioni, non convertito in tempo reale. Questi
casi puntano a un problema di trasferimento della soglia e di tempistica
della politica, non a una prova che sia già possibile decidere quando
sostituire il componente. Le notifiche h20 per bearing sono coerenti con i
totali appaiati: 1/6 per la logistica e 2/6 per RMS.

### Analisi descrittiva delle feature e preavviso

Confrontando le ultime 20 acquisizioni con il resto della sequenza
separatamente per bearing, le feature di ampiezza sono generalmente più alte
in h20. L'AUC univariata mediana di RMS orizzontale è 1,000, 0,996 e 0,999
nelle condizioni 1, 2 e 3; per RMS verticale è 1,000, 0,971 e 0,999. Crest
factor e kurtosis hanno andamenti meno uniformi. Questa è una comparazione
descrittiva entro bearing: non dimostra significatività statistica, causalità
né trasferibilità tra bearing.

Nel confronto appaiato, le notifiche sono state riclassificate con diverse
finestre di preavviso, senza riaddestrare né modificare la politica:

| Modello | Finestra accettata (acquisizioni prima del termine) | Bearing rilevati | Notifiche accettate | Notifiche fuori finestra |
|---|---|---:|---:|---:|
| Logistica | h20: 0–19 | 1/6 | 1 | 20 |
| Logistica | Ultimi 50: 0–49 | 4/6 | 4 | 17 |
| Logistica | Preavviso 10–50 | 3/6 | 3 | 18 |
| RMS | h20: 0–19 | 2/6 | 2 | 13 |
| RMS | Ultimi 50: 0–49 | 4/6 | 4 | 11 |
| RMS | Preavviso 10–50 | 3/6 | 3 | 12 |

Le finestre 10–50 sono state esaminate dopo aver osservato i lead time:
restano un'analisi esplorativa, non una metrica predefinita né un criterio per
selezionare il modello. Notifiche fuori finestra significa non conformi al
criterio temporale scelto, non necessariamente falsi in senso fisico.
Allargare la finestra modifica il compromesso tra copertura e preavviso; non
dimostra da solo una migliore capacità predittiva.

Un ricalcolo esplorativo senza cooldown fisso, mantenendo soglie e conferme,
non aumenta il rilevamento h20 (resta 2/6) né la copertura 10–50 (resta 4/6)
e aumenta le notifiche fuori finestra. Questa verifica appartiene alla
precedente serie di predizioni; non è stata ripetuta nel nuovo confronto
appaiato e non giustifica da sola la sostituzione del cooldown.

### Limitazioni e decisioni aperte

- Con sole tre condizioni e due bearing per condizione, LOCO è diagnostica
  esplorativa, non una stima precisa della generalizzazione industriale.
- L'aumento delle feature verso h20 non spiega perché la politica trasferisca
  diversamente tra condizioni. Carico, velocità e differenze individuali non
  sono stati isolati causalmente.
- L'opzione `--weighted-loss` della validazione LOCO duplica campioni della
  classe minoritaria; è ricampionamento, non pesatura effettiva della loss.
  La run storica con oversampling è stata riprodotta esattamente e il JSON
  ora registra la modalità; i suoi risultati vanno comunque tenuti separati
  dal confronto standard senza oversampling.
- La geometria impiegata nell'analisi spettrale è provvisoria; le frequenze
  caratteristiche non sono certificate per il cuscinetto installato.

#### Stabilità della selezione delle politiche LOCO

Per ciascuna piega LOCO sono state valutate tutte le 48 combinazioni della
griglia interna (4 soglie, 3 conferme, 4 cooldown) con `balanced_f1`. Nel
training standard, più politiche raggiungono lo stesso F1 massimo; dopo i
criteri di spareggio interni, la politica migliore è unica nelle condizioni
2 e 3, mentre nella condizione 1 quattro politiche sono esattamente a pari
merito:

| Condizione test | Politica selezionata (soglia / conferma / cooldown) | Politiche con F1 massimo | Pari merito esatto dopo gli spareggi |
|---|---:|---:|---:|
| 1 | 0,50 / 3 / 0 | 8/48 | 4/48 |
| 2 | 0,10 / 3 / 50 | 3/48 | 1/48 |
| 3 | 0,30 / 5 / 50 | 2/48 | 1/48 |

È stata poi perturbata la selezione standard lasciando fuori, a turno, uno dei
quattro bearing interni di ciascuna piega. Per ognuno dei quattro casi si sono
ricalcolate le predizioni leave-one-bearing-out sui tre bearing rimanenti,
addestrando ciascun modello interno sugli altri due, e riselezionate le 48
politiche. La politica completa originaria non è stata riselezionata in
nessuna delle 12 perturbazioni (0/4 per condizione); le combinazioni scelte
cambiano al variare dei bearing inclusi. Questo indica instabilità della
selezione, non che una particolare alternativa sia migliore.

Il risultato è diagnostico: ogni stima interna usa solo tre o quattro
bearing, perciò la griglia non giustifica di cambiare soglie o cooldown sulla
base di questi sei esemplari. Anche i risultati delle politiche con
oversampling hanno una propria superficie di selezione e non sono stati
inclusi nella perturbazione leave-one-out aggiuntiva. Dettagli di tutte le
combinazioni, metriche e perturbazioni:
`/mnt/shared/predictive-maintenance/processed/loco_policy_stability_20261103.json`.

La discrepanza tra gli artefatti LOCO è stata spiegata e riprodotta: la run
storica usa oversampling, quella del confronto RMS no; entrambe le modalità
sono registrate nei rispettivi JSON e i CSV sono stati confrontati
byte per byte.

Prossimo passo utile per l'allarme di ispezione: concordare la finestra minima
di preavviso in unità di tempo e una funzione di costo che renda espliciti
guasti mancati e ispezioni non necessarie. Prima di modificare soglie o
cooldown, aggiungere una valutazione temporale/costo-beneficio con selezione
annidata solo sui dati interni; mantenere h20 come benchmark storico, non come
regola di sostituzione, e tenere separati i risultati esplorativi. Finché
cadenza e costi non sono definiti, i lead time in acquisizioni e l'F1 evento
non bastano a produrre una decisione operativa affidabile.

### Aggiornamento 16 novembre 2026: cruscotto dimostrativo

È stato aggiunto `scripts/dashboard.py`, una dashboard Streamlit per riprodurre
le sei sequenze FEMTO con due schede: quadro d'insieme e dettaglio. Selezionando
un cuscinetto nel quadro si apre la scheda di dettaglio, che mostra stato,
score h20 e grafico di RMS orizzontale. Nel dettaglio si può cambiare
cuscinetto con un selettore senza etichetta visibile, formattato come titolo.
I pulsanti del quadro hanno lo sfondo del colore corrispondente allo stato e
mostrano cuscinetto, stato e score sulla prima riga, descrizione sulla seconda.
La barra laterale dichiara il dataset NASA FEMTO/PRONOSTIA, l'accelerazione
vibratoria misurata sui due assi e l'RMS orizzontale visualizzato; l'unità non
è specificata nei file. Il cursore "Posizione nella sequenza storica (%)"
confronta la stessa posizione relativa nei run, non il tempo reale. Usa le
feature in
`processed/femto_features.npz` e le predizioni LOCO standard in
`processed/loco_rms_comparison/femto_loco_h20_alarm_predictions.csv`.

Nel replay verde significa score sotto la soglia sperimentale, giallo score
sopra soglia senza conferma, rosso notifica emessa dalla policy LOCO. Il rosso
resta latched perché non sono disponibili prese in carico o risoluzioni. La
dashboard non legge sensori live; lo score non è calibrato e i colori non sono
validati per uso industriale. Rosso indica di valutare un'ispezione umana, non
una sostituzione automatica.

Avvio: `streamlit run scripts/dashboard.py`. I percorsi di feature e predizioni
sono configurabili dalla barra laterale. Il caricamento dei dati reali e i test
del progetto sono stati verificati; 27 test unitari passano.

Attività futura, non ancora avviata: rendere il progetto riproducibile da zero
su un ambiente pulito. Serviranno istruzioni versionate per repository e
dipendenze, acquisizione dei dati, comandi verificati, artefatti attesi e uno
script versionato per l'analisi di stabilità delle politiche.

Artefatti del confronto appaiato in
`/mnt/shared/predictive-maintenance/processed/loco_rms_comparison/`:
`femto_loco_h20_validation_results.json`,
`femto_loco_h20_alarm_predictions.csv`,
`femto_loco_rms_baseline_h20_results.json` e
`femto_loco_rms_baseline_h20_predictions.csv`.

### Aggiornamento 17 novembre 2026: previsione dalla dashboard

La dashboard mantiene il replay storico e aggiunge la scheda **Nuova
acquisizione**, con caricamento CSV a 6 colonne FEMTO o 2 canali numerici. Il
percorso di previsione è condiviso con il CLI tramite `src/predictor.py`;
percorsi e checksum degli artefatti sono condivisi in `src/model_artifacts.py`.
Modello e preprocessing vengono controllati prima di abilitare la previsione.
La GUI blocca gli artefatti non validi o con checksum discordante, avvisa per
i metadati legacy senza checksum e mostra score, soglia e orizzonte solo dopo
una previsione riuscita. Un CSV non valido mostra l'errore e non può essere
analizzato; se cambiano segnale o soglia, la previsione precedente viene
invalidata. Il replay usa ancora predizioni LOCO pre-calcolate: la nuova
scheda non è collegata a sensori live e il punteggio resta sperimentale, non
calibrato e non utilizzabile per comandare una sostituzione.

Il caricamento dei CSV e la gestione del checksum sono coperti da test mirati;
la suite completa passa con 36 test. È stata verificata anche una previsione
end-to-end dalla GUI con un CSV sintetico. Gli artefatti locali usati per
questa prova hanno metadati legacy, quindi la GUI mostra correttamente
l'avviso di checksum non verificabile. Da verificare dopo un nuovo training
il percorso con checksum registrato e quello con checksum non corrispondente.

## Sintesi del lavoro storicizzato

### Dataset e correzioni iniziali (29 ottobre 2026)

Il dataset è stato scaricato ed estratto nella cartella condivisa VirtualBox
`/mnt/shared/predictive-maintenance/raw/femto/`. L'archivio esterno aveva
1.157.035.288 byte; SHA-256:
`e21bb22bd8d54fd18ebe98b4b4e094c0c40469bda19811a2a642d5cc84ebd81f`.

Durante l'estrazione, `os.replace` non era supportato dal filesystem
VirtualBox: è stato aggiunto un fallback con copia. Il loader evita il doppio
conteggio del training set annidato, ordina le acquisizioni e usa gli ultimi
due canali dei CSV FEMTO a sei colonne. Queste correzioni restano parte del
codice corrente.

### Esperimenti con target h21 (storici e superati)

Le prime analisi usavano 21 positivi per bearing a causa della precedente
definizione dell'orizzonte. Le griglie di soglie, le tabelle di campioni in
allarme, le analisi di episodi/cooldown e la prima validazione annidata
appartengono a quel target e non vanno usate come risultati correnti. Il
target è stato poi definito come numero totale di acquisizioni finali,
includendo l'ultima; h20 significa esattamente 20 record.

### Validazione annidata leave-one-bearing-out h20

La validazione nested LOBO seleziona la politica solo sui bearing di training.
Con obiettivo `balanced_f1` ha rilevato 4/6 bearing, con 16 episodi falsi,
precisione evento 20%, recall 66,7% e F1 evento 30,8%. È un confronto utile
per descrivere la variabilità, ma non è direttamente confrontabile con LOCO:
LOBO misura il trasferimento a un bearing non visto mentre altre condizioni
possono essere presenti nel training; LOCO misura il trasferimento a una
condizione interamente esclusa. Entrambi restano incerti per il numero ridotto
di bearing.

Artefatti: `femto_nested_h20_oof_alarm_predictions.csv` e
`femto_nested_h20_validation_results.json` in
`/mnt/shared/predictive-maintenance/processed/`.

### Modello operativo, timeline e analisi spettrale

Il modello operativo h20 è stato addestrato sui primi cinque bearing e
valutato su Bearing3_2: accuracy 99,2%, precisione 100%, recall 35% (7/20),
F1 51,9%, TN=1617, FP=0, FN=13, TP=7. Un solo bearing di test non fornisce
una stima robusta.

È stata prodotta una timeline degli allarmi h20 con metriche per episodi e
feature per fasce temporali. L'analisi spettrale esplorativa di Bearing1_1 ha
riscontrato aumenti di potenza in alcuni gruppi pre-h20 e h20, ma usa
geometria provvisoria e risoluzione FFT di 10 Hz; non è una diagnosi di
guasto. Dettagli e grafici restano negli artefatti in
`/mnt/shared/predictive-maintenance/processed/`.

### Verifica della libreria `ml_core`

Il 29 ottobre la regressione logistica è stata confrontata su dati sintetici
con un risolutore Newton NumPy indipendente; coefficienti e probabilità
coincidevano entro `1e-5`, la loss entro `1e-9`, e il round-trip save/load
preservava le predizioni. Quattro test mirati erano passati. La verifica
supporta il corretto funzionamento di quel percorso, non certifica l'intera
libreria.

## Validazione aggiuntiva e percorso verso la decisione

La priorità immediata è definire requisiti decisionali verificabili
(preavviso minimo, cadenza delle misure e costo relativo di guasti mancati,
ispezioni e sostituzioni non necessarie), poi valutare soglie e politiche con
validazione annidata. I benchmark pubblici già analizzati restano verifiche
metodologiche separate: non si uniscono campioni né si confrontano direttamente
metriche ottenute con target e protocolli diversi. XJTU-SY è il controllo più
pertinente per i cuscinetti e ha cadenza dichiarata di un minuto; C-MAPSS
verifica la pipeline RUL su un componente diverso.

### C-MAPSS: benchmark metodologico su un altro componente

C-MAPSS contiene traiettorie multivariate di motori turbofan simulate, con più
unità, sensori e condizioni operative. I set di training arrivano al guasto;
per le traiettorie di test sono fornite le RUL reali. È quindi adatto a
verificare la pipeline di prognostica su più traiettorie e condizioni, ma non
è un dataset di macchine industriali misurate sul campo e non fornisce
evidenza diretta sui cuscinetti.

Fonte e descrizione: [NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

Attività:

1. Verificare fonte di download, termini d'uso, struttura e checksum dei file.
2. Implementare un loader e una pipeline C-MAPSS indipendenti dalla pipeline
   FEMTO; conservare i dati grezzi fuori da Git.
3. Rispettare la suddivisione ufficiale train/test; separare unità durante
   validazione e scelta del modello, evitando leakage tra cicli della stessa
   unità.
4. Valutare la stima RUL con metriche adatte alla prognostica e riportare i
   risultati per unità e per sottoinsieme, senza riutilizzare h20 come se fosse
   un'etichetta equivalente.

### XJTU-SY: benchmark aggiuntivo specifico per cuscinetti

La descrizione del dataset sul [sito dell'autore](https://biaowang.tech/xjtu-sy-bearing-datasets/)
riporta prove accelerate run-to-failure su 15 cuscinetti LDK UER204: cinque
per ciascuna di tre condizioni (2100 rpm/12 kN, 2250 rpm/11 kN e
2400 rpm/10 kN). Due accelerometri misurano vibrazioni orizzontali e
verticali a 25,6 kHz; ciascuna acquisizione contiene 32.768 campioni, pari a
1,28 s, ed è ripetuta ogni minuto. I file CSV hanno due colonne, una per asse.
La fonte descrive anche i guasti osservati e le vite dei singoli cuscinetti;
richiede di citare la pubblicazione associata quando i dati vengono usati.

Questo dataset è più pertinente al componente di interesse e permette di
esprimere l'orizzonte in tempo: venti intervalli di campionamento corrispondono
a circa 20 minuti (tra la prima e la ventesima acquisizione ci sono 19 minuti).
La definizione di fine vita e di qualsiasi target dovrà comunque essere
verificata dalla documentazione dei file prima di applicare un'etichetta come
h20. Le prove sono accelerate e da banco: più cuscinetti e una cadenza nota
migliorano la valutazione sperimentale, ma non dimostrano la trasferibilità a
un impianto reale.

Fonte descrittiva e collegamenti di download: [XJTU-SY Bearing Datasets](https://biaowang.tech/xjtu-sy-bearing-datasets/).

Attività:

1. Verificare accessibilità, condizioni di riuso, formato e metadati prima di
   scaricare i dati; registrare fonte e checksum.
2. Costruire un loader separato e documentare frequenza/cadenza delle misure,
   condizioni operative e criterio di fine vita.
3. Definire prima dell'analisi se il task è classificazione di allarme,
   rilevamento del degrado o stima RUL. Non assumere che h20 abbia lo stesso
   significato o la stessa durata temporale di FEMTO.
4. Valutare per cuscinetto, lasciando fuori esemplari interi; tenere separata
   la selezione di feature/politica dalla valutazione finale.

### Protocollo comune e limiti della conclusione

- Mantenere FEMTO come benchmark originale e conservare distinti risultati
  standard e risultati con oversampling.
- Per ogni benchmark pubblicare split, target, unità di valutazione, metriche,
  numero di unità e incertezze; non aggregare i punteggi tra dataset.
- Non fissare adesso pesi di uno score composito o soglie di accettazione
  industriale: devono derivare dai costi di mancato allarme, fermo macchina,
  ispezione e sostituzione, concordati con i responsabili della manutenzione.
- I benchmark pubblici possono sostenere una valutazione di ricerca, ma non
  bastano a decidere la sostituzione di un cuscinetto in impianto. Per quello
  serviranno dati rappresentativi dell'impianto, etichette da ispezioni e
  manutenzioni documentate, validazione indipendente e una fase in shadow mode
  con decisione umana.

### Stato e prossime azioni

I due dataset sono stati scaricati, estratti e spostati nell'area condivisa
`/mnt/shared/predictive-maintenance/`, seguendo la struttura usata per FEMTO:

- C-MAPSS: archivio `downloads/CMAPSSData.zip`; dati estratti in
  `raw/cmapss/`. L'integrità dello ZIP è stata verificata; sono presenti i
  quattro file di training, i quattro di test, i quattro file RUL e i due
  documenti di accompagnamento.
- XJTU-SY: i sei volumi RAR sono in `downloads/`; dati estratti in
  `raw/xjtu_sy/`. Sono presenti 15 directory bearing in tre condizioni e
  9.216 CSV, oltre al PDF introduttivo. I volumi RAR sono stati conservati;
  non è stata eseguita una verifica indipendente dell'integrità RAR.

È stato aggiunto lo script riproducibile
`scripts/analyze_public_datasets.py`, con loader separati per i due formati.
I test mirati e l'intera suite unit test sono passati (23 test).

#### Prima analisi C-MAPSS

È stata eseguita una regressione ridge su RUL in cicli. La regolarizzazione è
selezionata con 5-fold CV raggruppata per motore e MAE medio per motore; il
test ufficiale usa un solo campione finale per ciascuna unità e le etichette
RUL fornite. Il confronto baseline usa il solo ciclo operativo; l'altro modello
usa ciclo, condizioni operative e 21 sensori.

| Sottoinsieme | Motori train/test | MAE test: ciclo | MAE test: ciclo + condizioni + sensori | MAE OOF medio per motore |
|---|---:|---:|---:|---:|
| FD001 | 100 / 100 | 33,81 | 25,93 | 30,76 |
| FD002 | 260 / 259 | 35,32 | 26,36 | 31,17 |
| FD003 | 100 / 100 | 58,99 | 45,32 | 44,11 |
| FD004 | 249 / 248 | 54,46 | 38,14 | 41,67 |

Il modello con sensori migliora il baseline ciclo-only in tutti e quattro i
test ufficiali, ma l'errore resta ampio e aumenta nei sottoinsiemi FD003/FD004.
È un benchmark ridge semplice, non un confronto con metodi allo stato
dell'arte; le metriche C-MAPSS non sono confrontabili numericamente con quelle
dei cuscinetti. Predizioni e dettaglio completo:
`/mnt/shared/predictive-maintenance/processed/public_datasets/cmapss_rul_analysis.json`
e `cmapss_rul_test_predictions.csv`.

#### Analisi XJTU-SY

Nota metodologica: il confronto iniziale tra modello completo (tempo trascorso,
condizioni operative e feature vibrazionali) e baseline non isolava il
contributo delle vibrazioni. È stata quindi completata un'ablazione
leave-one-bearing-out sugli stessi split: tempo soltanto, vibrazioni soltanto,
tempo più vibrazioni e contesto operativo più vibrazioni. Le feature vengono
standardizzate sul training di ogni piega; i parametri e gli split non vengono
scelti sul bearing escluso.

Sono state estratte 10 feature temporali per acquisizione dai due canali
(RMS, deviazione standard, picco-picco, massimo assoluto e kurtosis). Il target
è il numero di intervalli di un minuto fino all'ultima acquisizione del run.
La valutazione lascia fuori un bearing intero; il ridge standardizzato usa
alpha=1 e viene confrontato con la mediana della vita residua degli altri
bearing della stessa condizione. Il target è una RUL osservata retrospettiva,
non una prognosi verificata prima del guasto.

| Condizione | Acquisizioni | MAE medio per bearing: ridge (min) | MAE medio per bearing: baseline (min) | Bearing in cui ridge migliora |
|---|---:|---:|---:|---:|
| 35 Hz / 12 kN | 616 | 60,9 | 37,8 | 1/5 |
| 37,5 Hz / 11 kN | 1.566 | 416,4 | 205,8 | 1/5 |
| 40 Hz / 10 kN | 7.034 | 797,9 | 1.153,7 | 4/5 |
| Macro media dei 15 bearing | 9.216 | 425,0 | 465,8 | 6/15 |

Il risultato aggregato favorisce di poco il ridge, ma non è coerente tra le
condizioni: il baseline è migliore nelle prime due, mentre il ridge è migliore
nella terza. Inoltre la condizione 40 Hz contiene il 76% delle acquisizioni,
quindi le metriche aggregate per campione sono dominate da quella condizione.
Con cinque bearing per condizione, questo è un risultato esplorativo e non
dimostra una regola trasferibile di sostituzione. Predizioni e metriche
dettagliate sono in
`/mnt/shared/predictive-maintenance/processed/public_datasets/xjtu_sy_rul_analysis.json`
e `xjtu_sy_rul_oof_predictions.csv`.

##### Ablazione del contributo delle vibrazioni

La tabella riporta la MAE media per bearing (minuti), con ogni bearing pesato
allo stesso modo:

| Modello | 35 Hz / 12 kN | 37,5 Hz / 11 kN | 40 Hz / 10 kN | Macro media dei 15 bearing |
|---|---:|---:|---:|---:|
| Solo tempo trascorso | 963,8 | 874,3 | 729,7 | 855,9 |
| Sole vibrazioni | 546,9 | 584,3 | 579,3 | 570,2 |
| Tempo + vibrazioni | 661,0 | 717,8 | 606,1 | 661,6 |
| Tempo + contesto operativo + vibrazioni | 60,9 | 416,4 | 797,9 | 425,0 |
| Baseline: vita mediana per condizione | 37,8 | 205,8 | 1.153,7 | 465,8 |

Rispetto al modello con solo tempo, aggiungere le vibrazioni riduce la macro
MAE da 855,9 a 661,6 minuti (22,7%) e migliora 11 bearing su 15: in questo
protocollo le vibrazioni contengono quindi un segnale predittivo incrementale,
ma non uniforme. Il modello completo con contesto operativo e vibrazioni ha
una macro MAE di 425,0 minuti contro 465,8 del baseline, ma batte il baseline
solo in 6 bearing su 15 (1/5 a 35 Hz, 1/5 a 37,5 Hz e 4/5 a 40 Hz). Il
vantaggio aggregato non si trasferisce in modo coerente tra condizioni.

Il target resta retrospettivo: per ogni acquisizione è il tempo fino all'ultima
registrazione del run-to-failure, senza una finestra di anticipo minimo
predefinita. Il leave-one-bearing-out testa un bearing nuovo solo nelle tre
condizioni già viste nel training; non verifica condizioni nuove, prestazioni
in shadow mode o decisioni industriali.

Prossimi passi:

1. Definire un anticipo utile per la manutenzione e ripetere la valutazione
   usando solo finestre disponibili a quell'anticipo, con metriche di allarme
   coerenti con il costo di falsi allarmi e guasti mancati.
2. Valutare la generalizzazione a condizioni operative non viste e, se
   possibile, a ulteriori bearing e dataset; mantenere separati C-MAPSS,
   XJTU-SY e FEMTO.
3. Solo dopo aver concordato requisiti e costi con i responsabili manutenzione,
   validare su dati di impianto etichettati da ispezioni/manutenzioni e in
   shadow mode. Non scegliere soglie industriali dai benchmark pubblici.

### Miglioramenti prioritari (18 novembre 2026)

Il progetto è già in grado di dimostrare replay storico, previsione su una
nuova acquisizione, controlli degli artefatti e confronti su benchmark
pubblici. Il miglioramento successivo deve renderlo più affidabile come
supporto decisionale, non limitarsi ad aumentare la complessità dei modelli.

1. **Definire il processo operativo.** Concordare con la manutenzione il
   preavviso minimo in unità di tempo reale, il costo di un'ispezione inutile,
   il costo di un guasto mancato e il costo di fermo/danno. Definire chi
   conferma un allarme, chi effettua l'ispezione e cosa accade dopo
   l'ispezione. h20 deve restare una proxy storica finché questi requisiti
   non sono formalizzati.
2. **Valutare decisioni, non solo punteggi.** Separare punteggio continuo,
   soglia, conferma consecutiva, cooldown, escalation e presa in carico.
   Selezionare soglia e politica con una funzione di costo, mantenendo h20 come
   benchmark storico. Le metriche principali devono includere recall per
   evento, falsi allarmi per ora o per unità, distribuzione del preavviso,
   rilevamento precoce/tardivo e costo atteso. Precisione, recall e curva
   precisione/recall devono essere riportati per condizione e per bearing.
3. **Usare unità temporali reali.** Verificare la cadenza effettiva di FEMTO
   e XJTU-SY; distinguere ore, minuti e acquisizioni senza convertirli
   automaticamente. Definire finestre coerenti con approvvigionamento,
   pianificazione e tempi di verifica. La finestra 10–50 acquisizioni resta
   esplorativa fino alla decisione operativa.
4. **Rafforzare la validazione.** Usare selezione nested per soglia, conferma
   e cooldown sui soli dati interni; evitare mescolamento temporale tra
   training e test; utilizzare split raggruppati per bearing, unità o motore;
   testare condizioni non viste e dati successivi nel tempo. Con soli sei
   bearing FEMTO, più dataset o più esemplari sono necessari prima di stimare
   la prestazione industriale.
5. **Calcolare intervalli di confidenza.** Integrare bootstrap raggruppato per
   bearing, unità o condizione. Esporre valori centrali e dispersione per
   F1, recall, falsi allarmi, lead time e costo atteso, senza presentare
   esiti su pochi esemplari come prestazioni certe.
6. **Migliorare feature e modelli con metodo incrementale.** Mantenere
   regressione logistica, ridge e baseline semplici come riferimento. In
   seguito confrontare gradient boosting, modelli temporali e reti ricorrenti
   solo se migliorano un protocollo preventivamente fissato. Estendere le
   feature a carico, velocità, temperatura, spettro, ore operative e contesto
   della macchina. Normalizzare differenze individuali e valutare
   l'incertezza predittiva.
7. **Tenere separati i risultati dei benchmark.** C-MAPSS, XJTU-SY e FEMTO
   hanno obiettivi e unità diverse e non devono essere messi insieme in una
   singola graduatoria. Conservare i metadati di training, oversampling,
   selezione della soglia, split e preprocessing accanto a ogni artefatto.
8. **Completare il workflow umano.** Integrazione, presa in carico, annotazione
   dell'esito, ispezione, intervento, risoluzione e audit trail devono essere
   stati espliciti della dashboard. Il rosso deve significare "verifica umana
   da valutare", mai sostituzione o arresto automatico.
9. **Tracciare eventi reali.** Dopo un pilota controllato, registrare
   timestamp, bearing/macchina, score, soglia, modello e checksum, evidenze,
   esito dell'ispezione, intervento e riscontro successivo. Questi dati
   alimenteranno una valutazione temporale, non soltanto nuovi esperimenti
   retrospettivi.
10. **Monitorare il sistema in esercizio.** Controllare deriva delle
    feature, deriva del sensore, distribuzione dei punteggi, falsi allarmi,
    prestazioni per macchina/condizione e compatibilità degli artefatti.
    Calibrare il modello solo su dati approvati e bloccare previsione o replay
    quando modello, preprocessing e checksum non sono coerenti.
11. **Completare la riproducibilità da zero.** Versionare dipendenze
    e environment lock, dati e istruzioni di acquisizione, comandi verificati,
    artefatti attesi e script di analisi. Estendere la CI oltre i test mirati
    con test di integrazione per training, validazione, caricamento GUI e
    pipeline end-to-end.

Ordine consigliato:

1. requisiti di preavviso e costi;
2. metrica costo-beneficio e finestre temporali;
3. selezione nested e intervalli di confidenza;
4. verifica della cadenza reale;
5. ampliamento del dataset;
6. presa in carico e annotazione umana;
7. pilota chiuso in shadow mode;
8. modelli più complessi solo dopo una baseline robusta.

Conclusione corrente: il progetto può dimostrare la pipeline e confrontare
metodi su benchmark pubblici; non è ancora in grado di sostenere una decisione
industriale di sostituzione basata solo sul modello.
