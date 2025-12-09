# Bot Teams IRCC

Bot Framework pour exposer le RAG via Microsoft Teams.

## Démarrage local

```bash
npm install
npm start
```

Variables attendues :

- `API_BASE_URL` pour joindre l'API backend
- `MicrosoftAppId`, `MicrosoftAppPassword` pour l'enregistrement bot Azure AD

Expose le point d'entrée `/api/messages`. Configurez le tunneling (ngrok) et le Bot Framework Emulator pour tester.
