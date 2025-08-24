# Occhio Onniveggente

## Configurazione

Le impostazioni dell'applicazione sono definite in `settings.yaml` e vengono caricate tramite
modelli [Pydantic](https://docs.pydantic.dev/) presenti in `src/config.py`.
I valori mancanti o non validi vengono segnalati e sostituiti con default sensati.

Esempio di caricamento:

```python
from pathlib import Path
from pydantic import ValidationError
from src.config import Settings

try:
    settings = Settings.model_validate_yaml(Path("settings.yaml"))
except ValidationError as e:
    for err in e.errors():
        print(f"Errore in {err['loc']}: {err['msg']}")
    print("Uso impostazioni di default.")
    settings = Settings()
```

Un `settings.yaml` minimale potrebbe essere:

```yaml
debug: true
openai:
  stt_model: gpt-4o-mini-transcribe
```

Per limitare le risposte a un determinato contesto è possibile definire un profilo di dominio:

```yaml
domain:
  profile: "museo"  # profilo selezionato
  topic: ""        # contesto specifico opzionale
```

Se ad esempio `audio.sample_rate` contiene una stringa (`"ventiquattromila"`) invece di un
numero, Pydantic segnalerà `Input should be a valid integer` e userà `24000`.

## DataBase dei documenti

I testi consultati dall'Oracolo vanno inseriti nella cartella `DataBase/` alla radice del
progetto. Dopo aver aggiunto o rimosso file, esegui lo script di ingestione per
rigenerare l'indice:

```bash
python scripts/ingest_docs.py --add DataBase/nuovo_file.txt
python scripts/ingest_docs.py --remove DataBase/vecchio_file.txt
python scripts/ingest_docs.py --reindex
python scripts/ingest_docs.py --clear
```

Puoi anche indicizzare l'intera cartella in un'unica volta:

```bash
python scripts/ingest_docs.py --add DataBase
```

Rieseguendo lo script dopo ogni modifica l'indice viene aggiornato con i contenuti
presenti in `DataBase/`.

Prima di `--remove`, `--reindex` o `--clear` viene creato un backup
`index.json.bak`. Per ripristinare l'indice:

```bash
cp index.json.bak index.json
```

Usa `--no-backup` per saltare la creazione del file di backup. Argomenti
supplementari non riconosciuti (es. `--model-dir` o `--video_dir`) vengono
ignorati.

## Test

Per eseguire i test installa le dipendenze e lancia `pytest`:

```bash
pip install -r requirements.txt
pytest
```

## Interfaccia grafica

Un'interfaccia utente minimale con stile futuristico è disponibile nel modulo
`src.ui`. Per avviarla:

```bash
python -m src.ui
```
Dal menu **Impostazioni** è possibile selezionare i dispositivi audio e
configurare la luce via Art-Net/sACN o WLED.

