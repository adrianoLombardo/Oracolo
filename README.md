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

---

## 1. Installazione

```bash
pip install -r requirements.txt        # librerie principali
pip install pytest                     # necessario per eseguire i test
```

Servono inoltre:
- **Python 3.10+**
- Un'API key OpenAI (`OPENAI_API_KEY`)

---

## 2. Configurazione

Le impostazioni Pydantic si trovano in `settings.yaml` (parametri generali) e `settings.local.yaml` (override locali, es. dispositivi audio).
Esempio minimale di `settings.yaml`:

```yaml
debug: true
openai:
  stt_model: gpt-4o-mini-transcribe
```

Per avviare con dispositivi audio diversi:

```yaml
audio:
  input_device: 2   # indice input (da sounddevice.query_devices)
  output_device: 5
```

---

## 3. Avvio dell'Oracolo

### Modalità standard (CLI)

Da `OcchioOnniveggente/`:

```bash
python -m src.main
```

Opzioni principali:
- `--autostart` avvia subito l'ascolto senza prompt
- `--quiet` riduce i log a schermo

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

---

## 4. Modalità Realtime (WebSocket)

### Server

- **Server completo**:
  ```bash
  python scripts/realtime_server.py
  ```
  (richiede chiave OpenAI, microfono e impostazioni valide in `settings.yaml`)

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
```

Il percorso dell'indice è configurabile con `docstore_path` in `settings.yaml`.

Nell'interfaccia grafica sono disponibili le nuove finestre **Dominio…** e
**Conoscenza…**: la prima consente di definire parole chiave, prompt
oracolare e rigidità del filtro; la seconda permette di scegliere l'indice,
impostare il `top-k` e testare le query di recupero. Queste opzioni rendono
l'Oracolo adattabile a mostre, conferenze e installazioni, semplificando la
gestione dei documenti e migliorando la pertinenza delle risposte.

---

## 6. Test

Esecuzione completa con pytest:

```bash
pytest
```

I test includono:
- validazione della configurazione (`tests/test_config.py`)
- funzioni del filtro linguistico (`tests/test_filters.py`)
- datastore di esempio (`tests/test_docstore.py`)

---

## 7. Moduli principali

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

## 8. Note rapide

- I file `data/filters/en_blacklist.txt` e `it_blacklist.txt` definiscono le parole/frasi bandite.
- `data/logs/dialoghi.csv` raccoglie cronologia delle conversazioni.
- `tests/data/filters/*.txt` fornisce dizionari di esempio per i test.
- Alcuni script richiedono librerie opzionali (`pypdf`, `python-docx`, `rank-bm25`, `rapidfuzz`); installarle se necessarie per l'uso esteso.

---

Con questo README hai una panoramica completa del progetto, delle sue funzionalità e dei comandi per avviare l'Oracolo, eseguire i test e gestire l'indice dei documenti. Buon divertimento!

