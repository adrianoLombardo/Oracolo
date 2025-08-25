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
## Aggiornamenti backend

- **MetadataStore**: nuovo archivio dei metadati basato su SQLite FTS o PostgreSQL, con supporto opzionale al vector store FAISS.
- Le chiamate OpenAI utilizzano ora il client asincrono nativo (`openai.AsyncOpenAI`), eliminando il thread pool e semplificando l'integrazione.
- Funzioni TTS/STT locali con utilità di streaming a chunk in `local_audio.py`.
- Backend LLM locale opzionale tramite `llm_backend=local` con fallback automatico a OpenAI.

### LLM locale

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


---

## 1. Installazione

```bash
pip install -r requirements.txt        # librerie principali
pip install pytest                     # necessario per eseguire i test
```

Servono inoltre:
- **Python 3.10+**
- Un'API key OpenAI (`OPENAI_API_KEY`)
- (Opzionale) GPU NVIDIA con ≥4 GB di VRAM per usare modelli Whisper locali; in assenza viene usata la CPU (più lenta).
  La selezione del device è gestita da `resolve_device` e può essere
  personalizzata impostando `compute.device` in `settings.yaml` o la
  variabile d'ambiente `ORACOLO_DEVICE` (`auto`, `cpu`, `cuda`). La soglia
  minima di VRAM (4 GB di default) è configurabile tramite
  `ORACOLO_MIN_CUDA_GB`.

---

## 2. Configurazione

Le impostazioni Pydantic si trovano in `settings.yaml` (parametri generali) e `settings.local.yaml` (override locali, es. dispositivi audio).
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
```

Il file include inoltre sezioni per l'attivazione tramite **hotword** e per
definire più **profili** preconfigurati. Ad esempio:

```yaml
wake:
  enabled: true
  single_turn: false
  idle_timeout: 50
  it_phrases: ["ciao oracolo"]

profiles:
  Museo:
    oracle_system: "Sei l'oracolo del museo…"
    docstore_path: DataBase/museo.json
```

Queste opzioni permettono di risvegliare l'Oracolo con una frase chiave e di
passare rapidamente tra preset completi (prompt, dominio, archivio e memoria
della chat).

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

---

## 5. Gestione documenti (RAG)

Gli archivi consultati dall'Oracolo risiedono in `DataBase/`.
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
Il percorso dell'indice è configurabile con `docstore_path` in `settings.yaml`.
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

---

Con questo README hai una panoramica completa del progetto, delle sue funzionalità e dei comandi per avviare l'Oracolo, eseguire i test e gestire l'indice dei documenti. Buon divertimento!

