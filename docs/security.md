# Sicurezza e gestione segreti

## Gestione segreti

- Utilizza variabili d'ambiente o un file `.env` (caricato automaticamente con `python-dotenv`).
- Non inserire mai le chiavi API nel controllo versione: il file `.env` è già escluso tramite `.gitignore`.
- In produzione preferisci un *secret manager* (ad es. HashiCorp Vault, AWS Secrets Manager) per distribuire le credenziali.

## Best practice

- Limita i permessi delle chiavi API al minimo indispensabile e pianifica una rotazione periodica.
- Evita di registrare nei log dati sensibili o token: imposta livelli di log appropriati e filtra le informazioni personali.
- Proteggi i file di configurazione con permessi adeguati e non condividerli su canali insicuri.

## Autenticazione metriche

L'endpoint `/metrics` espone informazioni sull'applicazione e dovrebbe essere protetto. Imposta la variabile d'ambiente `METRICS_TOKEN` per richiedere un *bearer token*:

```bash
export METRICS_TOKEN=valore-segreto
```

Le richieste dovranno includere `Authorization: Bearer valore-segreto` oppure il parametro di query `?token=valore-segreto`.
Per una sicurezza ulteriore integra un reverse proxy (Nginx, Traefik…) con autenticazione OAuth o basata su IP.
