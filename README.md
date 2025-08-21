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

## Domande pertinenti

L'Oracolo risponde soltanto a domande sui temi ammessi:
**arte**, **scienza** e **filosofia**. Se la richiesta non rientra in questi
argomenti, il sistema risponde con un messaggio di rifiuto, ad esempio:
"Non posso parlare di questo. Chiedimi di arte, scienza o filosofia.".

Per aggiungere o rimuovere argomenti consentiti modifica la lista
`allowed_topics` in `OcchioOnniveggente/settings.yaml`. Nello stesso file puoi
personalizzare i messaggi di rifiuto modificando la sezione
`refusal_messages`.

