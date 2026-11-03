# Ripresa: manutenzione predittiva FEMTO/PRONOSTIA

## Scopo, dati e obiettivo

Il progetto studia se le vibrazioni misurate durante prove run-to-failure
contengano un segnale utile ad anticipare il termine della vita di un
cuscinetto. L'obiettivo è valutare, in modo sperimentale, quanto tale segnale
si trasferisca a cuscinetti e condizioni operative non usati per addestrare o
selezionare la politica di allarme. Non si sta certificando un sistema
industriale né formulando una diagnosi fisica del guasto.

Si usa il dataset NASA FEMTO/PRONOSTIA: 7.534 acquisizioni per sei cuscinetti
run-to-failure, due per ciascuna delle tre condizioni operative. Ogni
acquisizione fornisce due canali di vibrazione; da ciascun canale si estraggono
RMS, deviazione standard, picco-picco, massimo assoluto, crest factor e
kurtosis (12 feature complessive). Il target corrente h20 marca positive le
ultime 20 acquisizioni di ogni sequenza: 120 positivi in totale, circa 1,59%.

Obiettivo sperimentale immediato: confrontare la regressione logistica con
baseline semplici sulle stesse partizioni Leave-One-Condition-Out (LOCO),
selezionando ogni politica solo sui dati interni e misurando rilevamenti,
falsi allarmi e tempistica delle notifiche. La metrica primaria è il
rilevamento evento nella finestra h20; finestre di preavviso alternative sono
esplorative. I risultati, con soli sei cuscinetti, non sono una stima di
prestazioni industriali.

## Stato corrente (3 novembre 2026)

Il lavoro corrente cerca di capire se le feature di vibrazione anticipano il
termine delle sequenze e quanto il segnale e la politica di allarme si
trasferiscano tra cuscinetti e condizioni operative. Non si sta certificando
un sistema industriale né identificando una causa fisica.

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

**Nota di riproducibilità:** l'esecuzione appaiata recente non coincide con
il precedente riepilogo LOCO: quello riportava 2/2 bearing rilevati nella
condizione 3 e F1 100%, mentre la nuova esecuzione ne riporta 1/2 e F1
66,7%. Anche le politiche selezionate differiscono. I dati elaborati
ricostruiti dai CSV grezzi coincidono esattamente con `femto_features.npz`,
ma non è ancora stata identificata la causa della variazione dei risultati.
Perciò i numeri del confronto appaiato sono riportati separatamente e non
vanno mescolati con la precedente esecuzione; prima di trarre conclusioni è
necessario investigare e rendere riproducibile il training LOCO.

Questi risultati non dicono semplicemente che il modello assegna probabilità
basse in h20. Nel run appaiato, tutti i campioni h20 sono sopra la soglia in
quattro bearing, nessuno in Bearing3_1 e 9/20 in Bearing3_2. Nelle condizioni
1 e 2 le notifiche sono spesso molto anticipate o l'allarme persiste prima
della finestra, perciò la politica non soddisfa la metrica di notifica in
h20. È quindi necessario distinguere discriminazione del punteggio e
tempestività della notifica.

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
  Non è disponibile un riepilogo verificabile per fold della variante: non
  usare i risultati di rebalancing precedentemente riferiti come confronto
  confermato.
- La geometria impiegata nell'analisi spettrale è provvisoria; le frequenze
  caratteristiche non sono certificate per il cuscinetto installato.

Prossimo passo utile: scegliere e documentare una metrica temporale che
bilanci rilevamento, falsi allarmi e preavviso eccessivo; solo dopo valutare
una modifica della politica o del modello. Priorità immediata: spiegare la
variazione tra esecuzioni LOCO logistiche e assicurare che predizioni e
politiche siano riproducibili. Mantenere separati i risultati esplorativi
dalla valutazione sul target h20 originale.

Artefatti del confronto appaiato in
`/mnt/shared/predictive-maintenance/processed/loco_rms_comparison/`:
`femto_loco_h20_validation_results.json`,
`femto_loco_h20_alarm_predictions.csv`,
`femto_loco_rms_baseline_h20_results.json` e
`femto_loco_rms_baseline_h20_predictions.csv`.

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
