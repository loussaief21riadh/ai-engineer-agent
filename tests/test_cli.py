from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

from app.main import main, print_help, print_status, print_header
from app.agent.orchestrator import Orchestrator
from app.config import AgentMode


@pytest.fixture
def mock_orchestrator():
    with patch("app.main.Orchestrator") as MockOrch:
        orch = MagicMock()
        orch.mode = AgentMode.READ_ONLY
        orch.core.tools = {"list_files": MagicMock(), "read_file": MagicMock()}
        MockOrch.return_value = orch
        yield orch


class TestMainFunction:
    @patch("app.main.input")
    def test_runs_without_error(self, mock_input, mock_orchestrator, capsys):
        mock_input.return_value = "/exit"
        main()
        captured = capsys.readouterr()
        assert "AI ENGINEER AGENT" in captured.out
        assert "Goodbye" in captured.out

    @patch("app.main.input")
    def test_handles_eof_error(self, mock_input, mock_orchestrator, capsys):
        mock_input.side_effect = EOFError
        main()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    @patch("app.main.input")
    def test_handles_keyboard_interrupt(self, mock_input, mock_orchestrator, capsys):
        mock_input.side_effect = KeyboardInterrupt
        main()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out


class TestHelpCommand:
    @patch("app.main.input")
    def test_help_command(self, mock_input, mock_orchestrator, capsys):
        mock_input.side_effect = ["/help", "/exit"]
        main()
        captured = capsys.readouterr()
        assert "/help" in captured.out
        assert "/exit" in captured.out
        assert "/mode" in captured.out
        assert "/clear" in captured.out
        assert "/status" in captured.out


class TestExitCommand:
    @patch("app.main.input")
    def test_exit_command(self, mock_input, mock_orchestrator, capsys):
        mock_input.side_effect = ["/exit"]
        main()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    @patch("app.main.input")
    def test_quit_command(self, mock_input, mock_orchestrator, capsys):
        mock_input.side_effect = ["quit"]
        main()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    @patch("app.main.input")
    def test_exit_word(self, mock_input, mock_orchestrator, capsys):
        mock_input.side_effect = ["exit"]
        main()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out


class TestModeCommand:
    @patch("app.main.input")
    def test_mode_with_valid_choice(self, mock_input, mock_orchestrator, capsys):
        mock_input.side_effect = ["/mode", "FULL_AUTONOMOUS", "/exit"]
        main()
        captured = capsys.readouterr()
        assert "Available modes" in captured.out
        mock_orchestrator.set_mode.assert_called_once_with(AgentMode.FULL_AUTONOMOUS)
        assert "FULL_AUTONOMOUS" in captured.out

    @patch("app.main.input")
    def test_mode_with_invalid_choice(self, mock_input, mock_orchestrator, capsys):
        mock_input.side_effect = ["/mode", "invalid_mode", "/exit"]
        main()
        captured = capsys.readouterr()
        assert "Invalid mode: invalid_mode" in captured.out
        mock_orchestrator.set_mode.assert_not_called()


class TestClearCommand:
    @patch("app.main.input")
    def test_clear_command(self, mock_input, mock_orchestrator, capsys):
        mock_input.side_effect = ["/clear", "/exit"]
        main()
        captured = capsys.readouterr()
        assert "Conversation cleared" in captured.out
        mock_orchestrator.core.history.clear.assert_called_once()


class TestStatusCommand:
    @patch("app.main.input")
    def test_status_command(self, mock_input, mock_orchestrator, capsys):
        mock_input.side_effect = ["/status", "/exit"]
        main()
        captured = capsys.readouterr()
        assert "Project:" in captured.out
        assert "Mode:" in captured.out
        assert "Tools:" in captured.out


class TestPrintFunctions:
    def test_print_help(self, capsys):
        print_help()
        captured = capsys.readouterr()
        assert "/help" in captured.out
        assert "/exit" in captured.out

    def test_print_status(self, capsys, mock_orchestrator):
        print_status(mock_orchestrator)
        captured = capsys.readouterr()
        assert "Project:" in captured.out
        assert "Mode:" in captured.out
        assert "Tools:" in captured.out

    def test_print_header(self, capsys, mock_orchestrator):
        print_header(mock_orchestrator)
        captured = capsys.readouterr()
        assert "AI ENGINEER AGENT" in captured.out
        assert "Project:" in captured.out
        assert "Mode:" in captured.out
