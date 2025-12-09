# Sécurité

## Authentification

- Prévu : Azure AD / Entra ID via tokens Bearer (MSAL côté front, validation JWT côté API).
- Mode dev : `ENABLE_AUTH=False` permet de tester sans jeton.

## Réseau

- Exposition limitée à l'intranet (Ingress interne / private VNet).
- API protégée derrière un gateway ou reverse proxy avec TLS.

## Données

- Les formulaires IRCC sont publics, pas de données sensibles générées.
- Les index et modèles sont stockés sur disque/PVC avec sauvegarde périodique.

## Journaux

- Logs applicatifs à conserver (API, bot, ingestion) pour audit et diagnostic.
- Ajouter un SIEM/monitoring d'entreprise pour la production.
