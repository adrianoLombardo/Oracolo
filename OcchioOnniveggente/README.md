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

Se ad esempio `audio.sample_rate` contiene una stringa (`"ventiquattromila"`) invece di un
numero, Pydantic segnalerà `Input should be a valid integer` e userà `24000`.

## Test

Per eseguire i test installa le dipendenze e lancia `pytest`:

```bash
pip install -r requirements.txt
pytest
```
