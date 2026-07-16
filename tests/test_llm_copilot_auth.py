# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Tests for podcast.llm.copilot_auth (respx-mocked HTTP)."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from podcast.errors import ProviderError
from podcast.llm import copilot_auth

DEVICE_PAYLOAD = {
    "device_code": "dev-123",
    "user_code": "ABCD-1234",
    "verification_uri": "https://github.com/login/device",
    "interval": 1,
}


def _no_sleep(_seconds: float) -> None:
    return None


class TestStoredToken:
    def test_round_trip_with_restrictive_permissions(self, tmp_path: Path) -> None:
        token_file = tmp_path / "github-token.json"
        copilot_auth.store_token("gho_secret", token_file)  # pragma: allowlist secret
        assert copilot_auth.load_stored_token(token_file) == "gho_secret"
        assert (token_file.stat().st_mode & 0o777) == 0o600

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert copilot_auth.load_stored_token(tmp_path / "absent.json") is None

    def test_corrupt_file_returns_none(self, tmp_path: Path) -> None:
        broken = tmp_path / "github-token.json"
        broken.write_text("{not json", encoding="utf-8")
        assert copilot_auth.load_stored_token(broken) is None

    def test_wrong_shape_returns_none(self, tmp_path: Path) -> None:
        wrong = tmp_path / "github-token.json"
        wrong.write_text(json.dumps(["a", "list"]), encoding="utf-8")
        assert copilot_auth.load_stored_token(wrong) is None

    def test_default_path_lives_under_user_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", "/cfg")
        assert copilot_auth.stored_token_path() == Path("/cfg/podcast/github-token.json")


class TestRequestDeviceCode:
    def test_parses_device_code(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(copilot_auth.DEVICE_CODE_URL).mock(
            return_value=httpx.Response(200, json=DEVICE_PAYLOAD)
        )
        with httpx.Client() as client:
            device = copilot_auth.request_device_code(client)
        assert device.user_code == "ABCD-1234"
        assert device.interval == 1

    def test_http_failure_raises(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(copilot_auth.DEVICE_CODE_URL).mock(return_value=httpx.Response(503))
        with httpx.Client() as client, pytest.raises(ProviderError, match="device-code"):
            copilot_auth.request_device_code(client)


class TestPollForAccessToken:
    def _device(self) -> copilot_auth.DeviceCode:
        return copilot_auth.DeviceCode(
            device_code="dev-123",
            user_code="ABCD-1234",
            verification_uri="https://github.com/login/device",
            interval=1.0,
        )

    def test_pending_then_token(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(copilot_auth.ACCESS_TOKEN_URL).mock(
            side_effect=[
                httpx.Response(200, json={"error": "authorization_pending"}),
                httpx.Response(200, json={"access_token": "gho_ok"}),
            ]
        )
        with httpx.Client() as client:
            result = copilot_auth.poll_for_access_token(client, self._device(), sleep=_no_sleep)
        assert result == "gho_ok"

    def test_slow_down_increases_interval(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(copilot_auth.ACCESS_TOKEN_URL).mock(
            side_effect=[
                httpx.Response(200, json={"error": "slow_down"}),
                httpx.Response(200, json={"access_token": "gho_ok"}),
            ]
        )
        waits: list[float] = []
        with httpx.Client() as client:
            copilot_auth.poll_for_access_token(client, self._device(), sleep=waits.append)
        assert waits == [1.0, 6.0]

    def test_denied_raises(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(copilot_auth.ACCESS_TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"error": "access_denied"})
        )
        with httpx.Client() as client, pytest.raises(ProviderError, match="access_denied"):
            copilot_auth.poll_for_access_token(client, self._device(), sleep=_no_sleep)

    def test_timeout_raises(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(copilot_auth.ACCESS_TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"error": "authorization_pending"})
        )
        with httpx.Client() as client, pytest.raises(ProviderError, match="timed out"):
            copilot_auth.poll_for_access_token(
                client, self._device(), timeout_seconds=3.0, sleep=_no_sleep
            )

    def test_transport_error_raises(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(copilot_auth.ACCESS_TOKEN_URL).mock(side_effect=httpx.ConnectError("down"))
        with httpx.Client() as client, pytest.raises(ProviderError, match="polling failed"):
            copilot_auth.poll_for_access_token(client, self._device(), sleep=_no_sleep)


class TestExchangeCopilotToken:
    def test_returns_bearer(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.get(copilot_auth.COPILOT_TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"token": "cop_bearer"})
        )
        with httpx.Client() as client:
            result = copilot_auth.exchange_copilot_token(client, "gho_x")
        assert result == "cop_bearer"
        assert route.calls.last.request.headers["Authorization"] == "token gho_x"

    def test_failure_mentions_copilot_enablement(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(copilot_auth.COPILOT_TOKEN_URL).mock(return_value=httpx.Response(403))
        with httpx.Client() as client, pytest.raises(ProviderError, match="enabled"):
            copilot_auth.exchange_copilot_token(client, "gho_x")


class TestObtainBearer:
    def test_uses_stored_token_without_device_flow(
        self, tmp_path: Path, respx_mock: respx.MockRouter
    ) -> None:
        token_file = tmp_path / "github-token.json"
        copilot_auth.store_token("gho_saved", token_file)  # pragma: allowlist secret
        respx_mock.get(copilot_auth.COPILOT_TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"token": "cop_bearer"})
        )
        announcements: list[str] = []
        with httpx.Client() as client:
            bearer = copilot_auth.obtain_bearer(
                client, announce=announcements.append, token_path=token_file
            )
        assert bearer == "cop_bearer"
        assert announcements == []

    def test_runs_device_flow_and_stores_token(
        self, tmp_path: Path, respx_mock: respx.MockRouter
    ) -> None:
        token_file = tmp_path / "github-token.json"
        respx_mock.post(copilot_auth.DEVICE_CODE_URL).mock(
            return_value=httpx.Response(200, json=DEVICE_PAYLOAD)
        )
        respx_mock.post(copilot_auth.ACCESS_TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "gho_new"})
        )
        respx_mock.get(copilot_auth.COPILOT_TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"token": "cop_bearer"})
        )
        announcements: list[str] = []
        with httpx.Client() as client:
            bearer = copilot_auth.obtain_bearer(
                client,
                announce=announcements.append,
                sleep=_no_sleep,
                token_path=token_file,
            )
        assert bearer == "cop_bearer"
        assert any("ABCD-1234" in message for message in announcements)
        assert copilot_auth.load_stored_token(token_file) == "gho_new"
