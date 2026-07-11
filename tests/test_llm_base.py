"""Tests for podcast.llm.base."""

from podcast.llm import base
from podcast.llm.fake import FakeProvider


class TestMessageHelpers:
    def test_system_helper(self) -> None:
        message = base.system("be brief")
        assert message.role == "system"
        assert message.content == "be brief"

    def test_user_helper(self) -> None:
        assert base.user("hi").role == "user"

    def test_assistant_helper(self) -> None:
        assert base.assistant("hello").role == "assistant"


class TestChatProviderProtocol:
    def test_fake_provider_satisfies_protocol(self) -> None:
        assert isinstance(FakeProvider(), base.ChatProvider)

    def test_arbitrary_object_does_not(self) -> None:
        assert not isinstance(object(), base.ChatProvider)
