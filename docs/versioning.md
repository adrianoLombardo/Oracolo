# Versionamento

Per generare il changelog in modo automatico utilizza [towncrier](https://towncrier.readthedocs.io/):

```bash
pip install towncrier
```

Crea un frammento di notizia in `news/` e poi esegui:

```bash
towncrier build --version X.Y.Z
```

Il file `CHANGELOG.md` risultante va incluso nelle release.
