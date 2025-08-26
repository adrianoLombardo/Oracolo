# Deploy

## Requisiti hardware

- CPU a 4 core
- 8 GB di RAM
- GPU opzionale per modelli più pesanti

## Variabili d'ambiente

Crea un file `.env` con le chiavi necessarie:

```
OPENAI_API_KEY=...
```

## Docker

Usa il file `docker-compose.yml` di esempio nella radice del progetto:

```bash
docker-compose up --build
```

## Kubernetes

Su Kubernetes crea un deployment basato sull'immagine Docker e monta il volume con `.env`.
