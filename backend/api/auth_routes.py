"""Operator login/logout/session. Public endpoints (everything else under /api requires a session)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from backend.auth import service as auth
from backend.notifications.email import dev_show_otp, smtp_configured

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class PasswordChange(BaseModel):
    old_password: str
    new_password: str


def _set_cookie(resp: Response, token: str, request: Request):
    resp.set_cookie(auth.COOKIE_NAME, token, httponly=True, samesite="lax", secure=request.url.scheme == "https",
                    max_age=auth.SESSION_HOURS * 3600, path="/")


@router.get("/me")
def me(request: Request):
    user = auth.user_from_token(request.cookies.get(auth.COOKIE_NAME))
    return {"authenticated": user is not None, "user": user,
            "environment": {"default_credentials": auth.using_default_credentials(), "smtp_configured": smtp_configured(),
                            "dev_show_otp": dev_show_otp()}}


@router.post("/login")
def login(body: LoginRequest, request: Request, response: Response):
    token = auth.login(body.username, body.password, request.headers.get("user-agent"))
    if not token:
        raise HTTPException(401, "Invalid username or password.")
    _set_cookie(response, token, request)
    return {"authenticated": True, "user": auth.user_from_token(token)}


@router.post("/logout")
def logout(request: Request, response: Response):
    auth.logout(request.cookies.get(auth.COOKIE_NAME))
    response.delete_cookie(auth.COOKIE_NAME, path="/")
    return {"authenticated": False}


@router.post("/password")
def change_password(body: PasswordChange, request: Request, response: Response):
    user = auth.user_from_token(request.cookies.get(auth.COOKIE_NAME))
    if not user:
        raise HTTPException(401, "authentication required")
    try:
        ok = auth.change_password(user["username"], body.old_password, body.new_password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not ok:
        raise HTTPException(400, "Current password is incorrect.")
    response.delete_cookie(auth.COOKIE_NAME, path="/")
    return {"changed": True, "message": "Password changed. Please sign in again."}
