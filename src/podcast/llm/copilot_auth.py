# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""GitHub device-flow login and Copilot bearer-token exchange."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx
from pydantic import BaseModel, ValidationError

from podcast.config import user_config_path
from podcast.errors import ProviderError

CLIENT_ID = "01ab8ac9400c4e429b23"  # VS Code's public OAuth client id, used for Copilot
DEVICE_CODE_URL = "https://github.com/login/device/code"
ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"  # noqa: S105 — URL, not a secret
COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"  # noqa: S105


@dataclass(frozen=True)
class DeviceCode:
    """One device-flow authorization round: show `user_code` at `verification_uri`."""

    device_code: str
    user_code: str
    verification_uri: str
    interval: float


class _DeviceCodeResponse(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    interval: float = 5.0


class _AccessTokenResponse(BaseModel):
    access_token: str | None = None
    error: str | None = None


class _CopilotTokenResponse(BaseModel):
    token: str


def stored_token_path() -> Path:
    return user_config_path().parent / "github-token.json"


def load_stored_token(path: Path | None = None) -> str | None:
    """The saved GitHub OAuth token, or None when absent/corrupt."""
    token_file = path if path is not None else stored_token_path()
    if not token_file.is_file():
        return None
    try:
        data: object = json.loads(token_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    token = cast("dict[str, object]", data).get("github_token")
    return token if isinstance(token, str) and token else None


def store_token(token: str, path: Path | None = None) -> None:
    token_file = path if path is not None else stored_token_path()
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(json.dumps({"github_token": token}) + "\n", encoding="utf-8")
    token_file.chmod(0o600)


def request_device_code(client: httpx.Client) -> DeviceCode:
    try:
        response = client.post(
            DEVICE_CODE_URL,
            data={"client_id": CLIENT_ID, "scope": "read:user"},
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        parsed = _DeviceCodeResponse.model_validate_json(response.text)
    except (httpx.HTTPError, ValidationError) as exc:
        raise ProviderError(f"GitHub device-code request failed: {exc}") from exc
    return DeviceCode(
        device_code=parsed.device_code,
        user_code=parsed.user_code,
        verification_uri=parsed.verification_uri,
        interval=parsed.interval,
    )


def poll_for_access_token(
    client: httpx.Client,
    device: DeviceCode,
    *,
    timeout_seconds: float = 900.0,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Poll GitHub until the user approves the device code; returns the OAuth token."""
    interval = device.interval
    waited = 0.0
    while waited <= timeout_seconds:
        sleep(interval)
        waited += interval
        try:
            response = client.post(
                ACCESS_TOKEN_URL,
                data={
                    "client_id": CLIENT_ID,
                    "device_code": device.device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            parsed = _AccessTokenResponse.model_validate_json(response.text)
        except (httpx.HTTPError, ValidationError) as exc:
            raise ProviderError(f"GitHub token polling failed: {exc}") from exc
        if parsed.access_token:
            return parsed.access_token
        if parsed.error == "authorization_pending":
            continue
        if parsed.error == "slow_down":
            interval += 5.0
            continue
        raise ProviderError(f"GitHub device flow failed: {parsed.error or 'unknown error'}")
    raise ProviderError("GitHub device flow timed out waiting for approval")


def exchange_copilot_token(client: httpx.Client, github_token: str) -> str:
    """Trade a GitHub OAuth token for a short-lived Copilot API bearer."""
    try:
        response = client.get(
            COPILOT_TOKEN_URL,
            headers={"Authorization": f"token {github_token}", "Accept": "application/json"},
        )
        response.raise_for_status()
        parsed = _CopilotTokenResponse.model_validate_json(response.text)
    except (httpx.HTTPError, ValidationError) as exc:
        raise ProviderError(
            f"Copilot token exchange failed (is Copilot enabled for this account?): {exc}"
        ) from exc
    return parsed.token


def obtain_bearer(
    client: httpx.Client,
    *,
    announce: Callable[[str], None],
    sleep: Callable[[float], None] = time.sleep,
    token_path: Path | None = None,
) -> str:
    """End-to-end Copilot auth: stored token or interactive device flow, then exchange."""
    github_token = load_stored_token(token_path)
    if github_token is None:
        device = request_device_code(client)
        announce(
            f"To use GitHub Copilot, open {device.verification_uri} "
            f"and enter code: {device.user_code}"
        )
        github_token = poll_for_access_token(client, device, sleep=sleep)
        store_token(github_token, token_path)
    return exchange_copilot_token(client, github_token)
