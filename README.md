# Oracolo – Guida rapida

Assistente vocale in italiano/inglese basato su OpenAI, con filtri linguistici, gestione dei documenti e controllo luci (Art-Net/sACN o WLED).

```
OcchioOnniveggente/
├── src/              # codice applicativo
├── scripts/          # utility (ingestione documenti, server realtime, dummy server)
├── data/             # filtri lingua, log, file temporanei
├── DataBase/         # archivio documenti indicizzati
└── tests/            # test automatici (pytest)
```

## Gestione delle chiavi

Le credenziali richieste (es. `OPENAI_API_KEY`) sono lette da variabili
d'ambiente. Puoi salvarle in un file `.env` caricato automaticamente con
`python-dotenv` oppure utilizzare un secret manager esterno. Il file `.env` è
escluso dal controllo versione per evitare di committare chiavi sensibili.

## Estensibilità e modularità

### Plugin backend

Registri di plugin consentono ora di estendere i provider STT/TTS/LLM tramite
entry point Python.  I backend predefiniti sono registrati automaticamente ma è
possibile aggiungerne altri definendo gli entry point `oracolo.stt`,
`oracolo.tts` e `oracolo.llm`.

### GUI modulare

Il front‑end è stato separato in componenti riusabili che espongono un'unica
API `run()`.  I wrapper `CLIFrontend` e `WebFrontend` permettono di integrare
facilmente l'interfaccia a riga di comando o quella web/REST.

### Configurazione

Lo schema completo delle impostazioni è generabile tramite lo script
`scripts/generate_settings_schema.py` che produce
`docs/settings.schema.json` con tutti i parametri validi e i relativi valori di
default.

## Domande fuori tema

Il file `data/domande_oracolo.json` contiene anche quesiti marcati con
`"type": "off_topic"`.  Ogni voce include una `categoria` (poetica, didattica,
evocativa o orientamento) che identifica il motivo del rifiuto.  Quando una
di queste domande viene selezionata, è possibile passare la categoria a
`oracle_answer(off_topic_category="poetica")` per ottenere una risposta di
cortese rifiuto adeguata al contesto.

## Sequenza delle categorie di domande

Il modulo `question_session` offre la classe `QuestionSession` per gestire la
scelta delle categorie quando non ne viene specificata una esplicitamente e per
memorizzare le risposte fornite dall'utente.  Di default le categorie vengono
proposte in rotazione **round‑robin**, evitando ripetizioni immediate.  È
possibile modificare le probabilità di estrazione definendo delle "pesature" in
`settings.yaml` oppure passando un dizionario `weights` al costruttore:

```python
from OcchioOnniveggente.src.question_session import QuestionSession

session = QuestionSession(weights={"poetica": 0.7, "didattica": 0.2, "orientamento": 0.1})
next_q = session.next_question()  # sceglie la categoria in base alle pesature
```

Esempio di configurazione nel file di impostazioni:

```yaml
question_weights:
  poetica: 0.5
  evocativa: 0.3
  didattica: 0.2
```

Quando le pesature non sono specificate, `QuestionSession` ruota le categorie
in ordine deterministico.

Le risposte e gli eventuali commenti dell'utente possono essere registrati
tramite `record_answer`:

```python
session.record_answer("risposta del sistema", reply="grazie")
print(session.answers, session.replies)
```

## Aggiornamenti backend

- **MetadataStore**: nuovo archivio dei metadati basato su SQLite FTS o PostgreSQL, con supporto opzionale al vector store FAISS.
- Le chiamate OpenAI utilizzano ora il client asincrono nativo (`openai.AsyncOpenAI`), eliminando il thread pool e semplificando l'integrazione.
- Funzioni TTS/STT locali con utilità di streaming a chunk in `audio/local_audio.py`.
- Backend LLM locale opzionale tramite `llm_backend=local` con fallback automatico a OpenAI.

## Monitoraggio risorse

Il modulo `src.metrics` registra periodicamente lo stato di GPU e CPU e
aggiorna tre metriche Prometheus:

- `gpu_memory_bytes` – memoria GPU allocata.
- `gpu_utilization_percent` – percentuale di utilizzo GPU.
- `cpu_usage_percent` – uso medio della CPU.

Il server realtime avvia automaticamente la raccolta e pubblica i valori
sull'endpoint `/metrics`, interrogabile con strumenti come Prometheus:

```bash
curl http://localhost:8000/metrics
```

Per limitare l'accesso è possibile impostare la variabile d'ambiente
`METRICS_TOKEN` e inviare il relativo bearer token nelle richieste
(`Authorization: Bearer <token>`). In alternativa proteggi l'endpoint tramite
un reverse proxy con autenticazione dedicata.

Nel log (livello `DEBUG`) sono visibili gli stessi dati. La funzione
`resolve_device` utilizza `gpu_utilization_percent` per dirottare le nuove
richieste sulla CPU quando l'uso della GPU supera il 90%.

Per ambienti **production** è possibile avviare un piccolo server di
esportazione con:

```python
from OcchioOnniveggente.src.metrics import metrics_loop, start_metrics_server

# espone le metriche su http://localhost:9000/metrics
start_metrics_server(9000)

# raccolta periodica CPU/GPU
asyncio.create_task(metrics_loop())
```

Le metriche possono quindi essere acquisite da Prometheus o da un collector
OpenTelemetry/OTLP e utilizzate come sorgente per sistemi di *autoscaling*.
La classe `Autoscaler` offre un semplice esempio di scalatore basato sull'uso
della CPU:

```python
from OcchioOnniveggente.src.metrics import Autoscaler

def scale_up():
    subprocess.run(["kubectl", "scale", "--replicas=3", "deploy/oracolo"])

def scale_down():
    subprocess.run(["kubectl", "scale", "--replicas=1", "deploy/oracolo"])

autoscaler = Autoscaler(scale_up, scale_down, high=80.0, low=20.0)
asyncio.create_task(autoscaler.run())
```

Su **Kubernetes** è possibile agganciare le metriche a un `HorizontalPodAutoscaler`
tramite il Prometheus Adapter. In **Docker Swarm** si può eseguire uno script
simile al precedente per regolare il numero di repliche in funzione dei valori
raccolti.

## LLM locale

Per usare un modello eseguito in locale è possibile impostare `llm_backend: local`
e indicare in `openai.llm_model` il percorso del modello. Il wrapper utilizza
`transformers` (o `llama-cpp`) e supporta diversi livelli di precisione per
riduire il consumo di memoria.

Requisiti indicativi:

- **fp32/fp16/bf16** – precisione piena o a 16 bit, richiede una GPU con almeno
  ~8‑16 GB di VRAM a seconda delle dimensioni del modello.
- **int4** – quantizzazione a 4 bit tramite `bitsandbytes`, consente di eseguire
  un modello 7B con ~4 GB di VRAM.

La precisione si seleziona in `settings.yaml` tramite `compute.llm.precision`:

```yaml
compute:
  llm:
    device: cuda         # auto | cpu | cuda
    precision: int4      # fp32 | fp16 | bf16 | int4
```

### Modelli ONNX

Per ridurre la latenza su CPU è possibile usare modelli convertiti in formato
ONNX tramite `onnxruntime`. Abilitare il supporto impostando
`compute.use_onnx: true` e indicando i percorsi dei file ONNX nei modelli:

```yaml
compute:
  use_onnx: true
  device: cpu
  stt:
    device: cpu
  llm:
    device: cpu
```

I modelli possono essere convertiti con `scripts/convert_to_onnx.py`:

```bash
# LLM
python scripts/convert_to_onnx.py --model gpt2 --output models/gpt2.onnx --type llm

# Whisper
python scripts/convert_to_onnx.py --model base --output models/whisper-base.onnx --type whisper
```

L'esecuzione tramite ONNX Runtime può offrire un miglioramento del 20‑30 % delle
prestazioni su CPU rispetto all'uso diretto di PyTorch.

In caso di errore il sistema effettua automaticamente il fallback al backend
OpenAI.

## Modalità streaming

È possibile ricevere la risposta dell'LLM in tempo reale utilizzando la nuova
funzione `stream_generate` in `src/oracle.py`. La funzione restituisce un
iteratore che emette piccoli chunk di testo man mano che vengono prodotti dal
modello, permettendo di aggiornare l'interfaccia senza attese.

Esempio minimo in modalità testuale:

```python
from OcchioOnniveggente.src.oracle import stream_generate
from OcchioOnniveggente.src.cli import stream_say
from openai import OpenAI

client = OpenAI()
tokens = stream_generate("Ciao?", "it", client, "gpt-4o", "")
stream_say(tokens)
```

Lo streaming può essere interrotto impostando un `threading.Event`, passando
un parametro `timeout` oppure premendo `CTRL+C` nella CLI.


## Protocollo di comunicazione con la UI

Il processo principale comunica con l'interfaccia inviando su **stdout** linee
JSON strutturate. I messaggi di chat hanno la forma:

```json
{"type": "chat", "role": "assistant", "text": "..."}
```

Ogni altra linea è interpretata come log e mostrata nel pannello di debug.
Script e test che leggono l'output devono quindi decodificare le linee JSON per
estrarre il testo della conversazione.


---

## 1. Installazione

```bash
pip install -r requirements.txt        # librerie principali
pip install pytest                     # necessario per eseguire i test
```

Servono inoltre:
- **Python 3.10+**
- Un'API key OpenAI (`OPENAI_API_KEY`)
- Motore TTS locale `pyttsx3` e VAD adattivo `webrtcvad` inclusi in `requirements.txt`.
- (Opzionale) GPU NVIDIA con ≥4 GB di VRAM per usare modelli Whisper locali; in assenza viene usata la CPU (più lenta).
  La selezione del device è gestita da `resolve_device` e può essere
  personalizzata impostando `compute.device` in `settings.yaml` o la
  variabile d'ambiente `ORACOLO_DEVICE` (`auto`, `cpu`, `cuda`). La soglia
  minima di VRAM (4 GB di default) è configurabile tramite
  `ORACOLO_MIN_CUDA_GB`.

---

## 2. Configurazione

Le impostazioni Pydantic si trovano in `settings.yaml` (parametri generali).
È possibile creare profili aggiuntivi copiando il file in
`settings.<nome>.yaml` e selezionandolo con la variabile d'ambiente
`ORACOLO_ENV=<nome>` (es. `ORACOLO_ENV=local`). I valori definiti nel profilo
scelto sovrascrivono quelli di base e qualsiasi parametro può essere
personalizzato (`debug`, `openai.api_key`, `audio.input_device`...).

Esempio minimale di `settings.yaml`:

```yaml
debug: true
stt_backend: openai  # openai | whisper
openai:
  stt_model: gpt-4o-mini-transcribe
```

`stt_backend` permette di usare l'API (`openai`) oppure una trascrizione
locale tramite `faster-whisper` (`whisper`).

Per avviare con dispositivi audio diversi:

```yaml
audio:
  input_device: 2   # indice input (da sounddevice.query_devices)
  output_device: 5
openai:
  tts_voice: alloy   # voce TTS locale (pyttsx3)

recording:
  use_webrtcvad: true
  vad_sensitivity: 2  # 0=più sensibile, 3=più severo
```

Per aggiungere un profilo dedicato (ad esempio `museo`):

```yaml
# settings.museo.yaml
debug: false
audio:
  input_device: "USB Microphone"
openai:
  api_key: "sk-..."
```

e attivalo esportando la variabile d'ambiente:

```bash
export ORACOLO_ENV=museo
```

Il file include inoltre sezioni per l'attivazione tramite **hotword** e per
definire più **profili** preconfigurati. Ad esempio:

```yaml
wake:
  enabled: true
  single_turn: false
  idle_timeout: 60
  it_phrases: ["ciao oracolo"]

profiles:
  Museo:
    oracle_system: "Sei l'oracolo del museo…"
    docstore_path: DataBase/museo.json
```

Queste opzioni permettono di risvegliare l'Oracolo con una frase chiave e di
passare rapidamente tra preset completi (prompt, dominio, archivio e memoria
della chat).

### Persona

In `settings.yaml` puoi anche configurare diverse **personalità** che definiscono tono e stile della risposta:

```yaml
persona:
  current: saggia
  profiles:
    saggia:
      tone: solenne
      style: poetico
    scherzosa:
      tone: allegro
      style: colloquiale
```

La personalità attiva viene inserita nel prompt di sistema. Durante la conversazione puoi dire (o digitare) `cambia personalità in scherzosa` per rendere l'Oracolo più leggero e informale.

---

## 3. Avvio dell'Oracolo

### Modalità standard (CLI)

Da `OcchioOnniveggente/`:

```bash
python -m src.main
```

Opzioni principali:
- `--autostart` avvia subito l'ascolto senza prompt
- `--quiet` nasconde i log dalla console (vista conversazione pulita)

Per separare conversazione e log in due "viewport" da terminale:

```bash
python -m src.main --quiet       # terminale principale, solo conversazione
tail -f data/logs/oracolo.log    # secondo terminale o pannello tmux per i log
```

Se si sviluppa un'interfaccia web, prevedere due componenti: una per la chat in tempo reale e una seconda, comprimibile, per mostrare i log solo all'occorrenza.

In Windows è disponibile uno script PS:

```powershell
.\run.ps1
```

### Interfaccia grafica (Tk)

```
python -m src.ui
```

Dal menu **Impostazioni** è possibile:
- Selezionare input/output audio
- Configurare modalità luce (sACN o WLED)
- Gestire i documenti indicizzati

La GUI offre inoltre:
- Scheda **Chat** con rendering Markdown e comandi slash (`/reset`,
  `/profile`, `/topic`, `/docs`, `/realtime`)
- Indicatori di livello audio in tempo reale e pulsanti per avviare/fermare
  il client WebSocket Realtime
- Dal menu **Server** è possibile avviare o fermare il server WebSocket
  realtime (`scripts/realtime_server.py`)
- Menu **Strumenti** per esportare la conversazione (TXT/MD/JSON), salvare le
  risposte in audio (WAV/MP3) e scaricare log o profili da condividere

### Front-end Qt/QML (sperimentale)

Per un'interfaccia moderna è incluso uno scheletro di client Qt/QML in
`src/frontend_qt`. Utilizza `QWebSocket` per la conversazione realtime e
riproduce l'audio PCM in streaming con `QAudioOutput`.

Compilazione e avvio:

```bash
cd OcchioOnniveggente/src/frontend_qt
mkdir build && cd build
cmake .. && cmake --build .
./oracolo_client
```

Il client espone le tab **Chat**, **Documenti** e **Impostazioni** oltre a un
menu a tendina per selezionare la modalità (Museo, Galleria, Conferenze,
Didattica).

---

## 4. Modalità Realtime (WebSocket)

### Server

- **Server completo**:
  ```bash
  python scripts/realtime_server.py
  ```
  (richiede chiave OpenAI, microfono e impostazioni valide in `settings.yaml`)
  Può essere avviato/fermato anche dal menu **Server** della GUI.

- **Server dummy di test**:
  ```bash
  python scripts/realtime_dummy_server.py
  ```

### Client

```
python -m src.realtime_oracolo
```

Variabili utili:
- `ORACOLO_WS_URL` – URL del server (default `ws://localhost:8765`)
- `--sr` – sample-rate, `--in-dev` e `--out-dev` per dispositivi audio.

### Modalità vocale

All'avvio il client carica le domande da `data/domande_oracolo.json`,
separandole tra **buone** e **off_topic**. Quando una nuova sessione inizia,
viene scelta una domanda buona casuale e letta tramite sintesi vocale locale.
Dopo ogni risposta valida l'Oracolo propone una micro‑domanda di follow‑up.
Esempio:

```
Domanda: "Chi sei?"
Risposta: "Sono un oracolo virtuale."
Follow-up: "Vuoi sapere come funziono?"
```
Se la trascrizione dell'utente corrisponde a una voce off-topic, il sistema
risponde con un cortese rifiuto generato dall'Oracolo e non propone follow‑up.

---

## 5. Gestione documenti (RAG)

Gli archivi consultati dall'Oracolo risiedono in `DataBase/`.
Per popolare l'indice predefinito `DataBase/index.json` inserisci i tuoi
documenti (ad esempio file di testo) nella cartella e genera l'indice con:

```bash
python scripts/ingest_docs.py --add DataBase
```

Il percorso dell'indice può essere personalizzato tramite `docstore_path` in
`settings.yaml`.

Script di ingestione/rimozione:

```bash
# Aggiunge singolo file o cartella
python scripts/ingest_docs.py --add path/to/file_or_dir

# Rimuove documenti dall'indice
python scripts/ingest_docs.py --remove path/to/file_or_dir

# Rigenera l'indice rileggendo i file già noti
python scripts/ingest_docs.py --reindex

# Svuota completamente l'indice
python scripts/ingest_docs.py --clear
```

Lo script `scripts/ingest_docs.py` ora rileva automaticamente un percorso con estensione `.db` o un DSN (`sqlite:///` o `postgresql://`) e utilizza il nuovo `MetadataStore` con indice FTS.
Prima di `--remove`, `--clear` o `--reindex` lo script crea un backup
`index.json.bak`. Per ripristinare l'indice basta copiare il file di backup
al nome originale:

```bash
cp index.json.bak index.json
```

È possibile saltare la creazione del backup aggiungendo il flag `--no-backup`.
Il recupero ora combina **BM25 + embedding** e, se configurato, applica un
mini **cross-encoder** per il reranking. Prima della ricerca la query può
essere riscritta automaticamente in 1–2 varianti (IT/EN) per migliorarne la
pertinenza.

Nell'interfaccia grafica sono disponibili le nuove finestre **Dominio…** e
**Conoscenza…**: la prima consente di definire parole chiave, prompt
oracolare e rigidità del filtro; la seconda permette di scegliere l'indice,
impostare il `top-k` e testare le query di recupero. Queste opzioni rendono
l'Oracolo adattabile a mostre, conferenze e installazioni, semplificando la
gestione dei documenti e migliorando la pertinenza delle risposte.

È possibile importare rapidamente nuovi file trascinandoli nella finestra
**Conoscenza…**, e durante le risposte viene mostrata un'anteprima dei
documenti/chunk utilizzati. Inoltre i profili di dominio possono essere
salvati e ricaricati come preset con un clic.

---

## 6. Chat, profili ed esportazione

La conversazione è multi-turno e può essere salvata in `data/logs/chat_sessions.jsonl`.
È possibile fissare messaggi, cambiare topic e passare da un profilo all'altro
anche da riga di comando (`/reset`, `/profile`, `/topic…`).

Dal menu **Strumenti** si possono esportare:
- la chat corrente in testo, Markdown o JSON;
- l'ultima risposta sintetizzata in audio (WAV/MP3);
- il log delle interazioni e le configurazioni di profilo, utili per la collaborazione.

## 7. Test

Esecuzione completa con pytest:

```bash
pytest
```

I test includono:
- validazione della configurazione (`tests/test_config.py`)
- funzioni del filtro linguistico (`tests/test_filters.py`)
- datastore di esempio (`tests/test_docstore.py`)

---

## 8. Moduli principali

| Modulo | Scopo |
| ------ | ----- |
| `audio.py` | Registrazione audio basata su VAD e riproduzione con pulsazione luci |
| `filters.py` | Filtro volgarità (IT/EN) con normalizzazione, wildcard e frasi multiple |
| `oracle.py` | Pipeline STT → validazione dominio → risposta LLM → TTS → log |
| `lights.py` / `wled_client.py` | Driver per sACN e WLED, con effetti pulsanti |
| `domain.py` | Controllo pertinenza domanda (keyword overlap + embeddings) |
| `retrieval.py` | BM25/fallback token overlap per recupero dei documenti |
| `realtime_oracolo.py` / `realtime_ws.py` | Client Realtime WS (microfono ↔ TTS) |
| `config.py` | Modelli Pydantic delle impostazioni e loader YAML con validazione |
| `ui.py` | Interfaccia grafica Tk interattiva (config, log, gestione documenti) |

---

## 9. Note rapide

- I file `data/filters/en_blacklist.txt` e `it_blacklist.txt` definiscono le parole/frasi bandite.
- `data/logs/dialoghi.csv` raccoglie cronologia delle conversazioni.
- `data/logs/chat_sessions.jsonl` salva le chat multi-turno.
- `tests/data/filters/*.txt` fornisce dizionari di esempio per i test.
- Alcuni script richiedono librerie opzionali (`pypdf`, `python-docx`, `rank-bm25`, `rapidfuzz`); installarle se necessarie per l'uso esteso.

## 10. Domande fuori tema

Il file `OcchioOnniveggente/data/domande_oracolo.json` include voci con
`"type": "off_topic"` e una `categoria` (poetica, didattica, evocativa,
orientamento). La funzione `load_questions` le carica e il modulo `oracle`
risponde con un rifiuto cortese specifico per categoria.

---

Con questo README hai una panoramica completa del progetto, delle sue funzionalità e dei comandi per avviare l'Oracolo, eseguire i test e gestire l'indice dei documenti. Buon divertimento!

