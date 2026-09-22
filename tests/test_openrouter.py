from unittest.mock import MagicMock, patch

import httpx
import pytest


def test_raises_error_when_no_api_key():
    from app.llm.openrouter import OpenRouterClient, OpenRouterError

    with patch("app.llm.openrouter.OPENROUTER_API_KEY", ""):
        with pytest.raises(OpenRouterError, match="OPENROUTER_API_KEY is not configured"):
            OpenRouterClient(api_key="")


def test_accepts_custom_api_key():
    from app.llm.openrouter import OpenRouterClient

    client = OpenRouterClient(api_key="test-key-123")
    assert client.api_key == "test-key-123"


@patch("app.llm.openrouter.httpx.post")
def test_chat_makes_correct_http_request(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello!"}}]
    }
    mock_post.return_value = mock_response

    from app.llm.openrouter import OpenRouterClient

    client = OpenRouterClient(api_key="test-key")
    result = client.chat(
        messages=[{"role": "user", "content": "Hi"}],
        model="openrouter/free",
    )

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert call_kwargs.kwargs["json"]["model"] == "openrouter/free"
    assert call_kwargs.kwargs["json"]["messages"] == [{"role": "user", "content": "Hi"}]
    assert result.content == "Hello!"
    assert result.tool_calls == []


@patch("app.llm.openrouter.httpx.post")
def test_chat_handles_http_error(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_post.return_value = mock_response

    from app.llm.openrouter import OpenRouterClient, OpenRouterError

    client = OpenRouterClient(api_key="test-key")
    with pytest.raises(OpenRouterError, match="status 500"):
        client.chat(messages=[{"role": "user", "content": "Hi"}], model="openrouter/free")


@patch("app.llm.openrouter.httpx.post")
def test_chat_handles_timeout(mock_post):
    mock_post.side_effect = httpx.TimeoutException("timed out")

    from app.llm.openrouter import OpenRouterClient, OpenRouterError

    client = OpenRouterClient(api_key="test-key")
    with pytest.raises(OpenRouterError, match="timed out"):
        client.chat(messages=[{"role": "user", "content": "Hi"}], model="openrouter/free")


@patch("app.llm.openrouter.httpx.post")
def test_chat_handles_http_error_exception(mock_post):
    mock_post.side_effect = httpx.HTTPError("connection failed")

    from app.llm.openrouter import OpenRouterClient, OpenRouterError

    client = OpenRouterClient(api_key="test-key")
    with pytest.raises(OpenRouterError, match="HTTP error"):
        client.chat(messages=[{"role": "user", "content": "Hi"}], model="openrouter/free")


@patch("app.llm.openrouter.httpx.post")
def test_chat_handles_missing_choices(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": []}
    mock_post.return_value = mock_response

    from app.llm.openrouter import OpenRouterClient, OpenRouterError

    client = OpenRouterClient(api_key="test-key")
    with pytest.raises(OpenRouterError, match="No choices in response"):
        client.chat(messages=[{"role": "user", "content": "Hi"}], model="openrouter/free")


@patch("app.llm.openrouter.httpx.post")
def test_chat_handles_no_choices_key(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}
    mock_post.return_value = mock_response

    from app.llm.openrouter import OpenRouterClient, OpenRouterError

    client = OpenRouterClient(api_key="test-key")
    with pytest.raises(OpenRouterError, match="No choices in response"):
        client.chat(messages=[{"role": "user", "content": "Hi"}], model="openrouter/free")
