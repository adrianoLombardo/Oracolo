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

## Gestione documenti

Il menu **Documenti** consente di inserire o rimuovere file dal datastore.
Dopo ogni modifica utilizzare il pulsante **Aggiorna indice** per rigenerare
l'indice di ricerca.

I documenti vengono salvati nella cartella predefinita `DataBase`. È possibile
personalizzare il percorso modificando il parametro `docstore_path` in
`OcchioOnniveggente/settings.yaml`.

