from __future__ import annotations

import ast
from pathlib import Path

from app.agent.validator import ValidationCheck, Validator
from app.models.schemas import TokenUsage


class TestSyntaxValidation:
    def test_valid_python_file(self, tmp_path: Path) -> None:
        filepath = tmp_path / "valid.py"
        filepath.write_text("x = 1\ny = 2\n")
        validator = Validator()
        result = validator.validate_syntax([str(filepath)])
        assert result.passed
        assert result.check == ValidationCheck.SYNTAX

    def test_invalid_python_file(self, tmp_path: Path) -> None:
        filepath = tmp_path / "invalid.py"
        filepath.write_text("def foo(:\n  pass\n")
        validator = Validator()
        result = validator.validate_syntax([str(filepath)])
        assert not result.passed
        assert "invalid.py" in result.evidence

    def test_non_python_files_skipped(self, tmp_path: Path) -> None:
        filepath = tmp_path / "data.txt"
        filepath.write_text("not python")
        validator = Validator()
        result = validator.validate_syntax([str(filepath)])
        assert result.passed
        assert "Checked 0 Python files" in result.details

    def test_mixed_files(self, tmp_path: Path) -> None:
        valid = tmp_path / "valid.py"
        valid.write_text("x = 1")
        invalid = tmp_path / "bad.py"
        invalid.write_text("def foo(:")
        validator = Validator()
        result = validator.validate_syntax([str(valid), str(invalid)])
        assert not result.passed
        assert "bad.py" in result.evidence

    def test_nonexistent_file_skipped(self) -> None:
        validator = Validator()
        result = validator.validate_syntax(["/nonexistent/file.py"])
        assert result.passed

    def test_syntax_in_validate_all(self, tmp_path: Path) -> None:
        filepath = tmp_path / "test.py"
        filepath.write_text("x = 1")
        validator = Validator()
        report = validator.validate_all(files_modified=[str(filepath)])
        syntax_results = [r for r in report.results if r.check == ValidationCheck.SYNTAX]
        assert len(syntax_results) == 1
        assert syntax_results[0].passed

    def test_syntax_error_blocks_validate_all(self, tmp_path: Path) -> None:
        filepath = tmp_path / "bad.py"
        filepath.write_text("def foo(:")
        validator = Validator()
        report = validator.validate_all(files_modified=[str(filepath)])
        assert not report.overall_passed


class TestTokenUsageCost:
    def test_default_cost(self) -> None:
        tu = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        cost = tu.estimate_cost()
        expected = 1000 * 0.000003 + 500 * 0.000015
        assert abs(cost - expected) < 1e-10

    def test_custom_pricing(self) -> None:
        tu = TokenUsage(prompt_tokens=100, completion_tokens=100, total_tokens=200)
        cost = tu.estimate_cost(prompt_price=0.01, completion_price=0.02)
        assert abs(cost - 3.0) < 1e-10

    def test_zero_tokens_zero_cost(self) -> None:
        tu = TokenUsage()
        assert tu.estimate_cost() == 0.0

    def test_cost_accumulates_through_observer(self) -> None:
        from app.agent.observer import ObserverEngine
        obs = ObserverEngine()
        t1 = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        t2 = TokenUsage(prompt_tokens=2000, completion_tokens=1000, total_tokens=3000)
        obs.emit("llm_call", tokens=t1)
        obs.emit("llm_call", tokens=t2)
        cost = obs.total_tokens.estimate_cost()
        expected = 3000 * 0.000003 + 1500 * 0.000015
        assert abs(cost - expected) < 1e-10
