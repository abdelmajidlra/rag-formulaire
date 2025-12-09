from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import get_settings

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


def get_current_token(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    settings = get_settings()
    if not settings.enable_auth:
        return None

    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = credentials.credentials
    logger.debug("Received bearer token with length %s", len(token))
    # Placeholder for Azure AD JWT validation
    return token
