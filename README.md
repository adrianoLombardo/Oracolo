# Oracolo

Assistente vocale basato su OpenAI con filtri di linguaggio e controlli per le luci.

## Installazione

```bash
pip install -r OcchioOnniveggente/requirements.txt
```

## Avvio

Dal repository "OcchioOnniveggente" eseguire:

```bash
python -m src.main
```

Per una semplice interfaccia grafica in stile futuristico è possibile avviare:

```bash
python -m src.ui
```

L'interfaccia offre un menu **Impostazioni** per scegliere i dispositivi audio
e modificare i parametri di illuminazione (Art-Net/sACN o WLED).

## Modalità realtime

L'interfaccia grafica offre i pulsanti **Avvia WS** e **Ferma WS** per aprire
una sessione WebSocket verso il backend realtime. Il microfono viene trasmesso
in tempo reale e nel log compaiono trascrizioni parziali e risposte TTS.

È inoltre disponibile un client da riga di comando:

```bash
python -m src.realtime_oracolo
```

Impostare la variabile d'ambiente `ORACOLO_WS_URL` con l'indirizzo del backend
realtime se diverso da `ws://localhost:8765`.

## Gestione documenti

Il menu **Documenti** consente di inserire o rimuovere file dal datastore.
Dopo ogni modifica utilizzare il pulsante **Aggiorna indice** per rigenerare
l'indice di ricerca.

I documenti vengono salvati nella cartella predefinita `DataBase`. È possibile
personalizzare il percorso modificando il parametro `docstore_path` in
`OcchioOnniveggente/settings.yaml`.

