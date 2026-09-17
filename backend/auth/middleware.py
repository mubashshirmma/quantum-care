"""Every /api/* route except the auth endpoints requires a valid operator session."""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from backend.auth.service import COOKIE_NAME, user_from_token

PUBLIC_PREFIXES = ("/api/auth/", "/api/health")


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        if path.startswith("/api/") and not path.startswith(PUBLIC_PREFIXES):
            user = user_from_token(request.cookies.get(COOKIE_NAME))
            if user is None:
                return JSONResponse({"detail": "authentication required"}, status_code=401)
            request.state.user = user
        return await call_next(request)
