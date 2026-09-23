from __future__ import annotations

from app.agent.memory import ProjectMemory


class TestMemoryQuery:
    def test_query_empty_memory(self) -> None:
        mem = ProjectMemory()
        results = mem.query("fix the bug")
        assert len(results) == 0

    def test_query_matches_files(self) -> None:
        mem = ProjectMemory()
        mem.add_important_file("src/main.py", "entry point")
        results = mem.query("main.py")
        assert len(results) >= 1
        assert any("main.py" in r["content"] for r in results)

    def test_query_matches_fixes(self) -> None:
        mem = ProjectMemory()
        mem.add_successful_fix("Fixed import error by adding missing module")
        results = mem.query("import error module")
        assert len(results) >= 1

    def test_query_matches_failures(self) -> None:
        mem = ProjectMemory()
        mem.add_failure("TypeError in parse_json when input is None")
        results = mem.query("TypeError parse_json")
        assert len(results) >= 1

    def test_query_matches_conventions(self) -> None:
        mem = ProjectMemory()
        mem.add_convention("All functions must have type hints")
        results = mem.query("type hints functions")
        assert len(results) >= 1

    def test_query_matches_architecture(self) -> None:
        mem = ProjectMemory()
        mem.add_architecture_note("Uses httpx for async HTTP calls")
        results = mem.query("httpx HTTP async")
        assert len(results) >= 1

    def test_query_matches_decisions(self) -> None:
        mem = ProjectMemory()
        mem.add_decision("Chose pydantic over dataclasses for validation")
        results = mem.query("pydantic validation")
        assert len(results) >= 1

    def test_query_matches_commands(self) -> None:
        mem = ProjectMemory()
        mem.add_known_command("pytest tests/", "run test suite")
        results = mem.query("pytest tests")
        assert len(results) >= 1

    def test_query_matches_entries(self) -> None:
        mem = ProjectMemory()
        mem.add_entry("config_path", "/etc/app/config.yaml", category="config")
        results = mem.query("config_path")
        assert len(results) >= 1
        assert results[0]["type"] == "config"

    def test_query_sorted_by_relevance(self) -> None:
        mem = ProjectMemory()
        mem.add_successful_fix("Fixed the database connection timeout issue")
        mem.add_successful_fix("Fixed a minor typo in the readme file")
        results = mem.query("database connection timeout")
        assert len(results) >= 1
        assert "database" in results[0]["content"].lower() or "timeout" in results[0]["content"].lower()

    def test_query_max_results(self) -> None:
        mem = ProjectMemory()
        for i in range(20):
            mem.add_successful_fix(f"Fix {i} for the test module")
        results = mem.query("test module fix", max_results=5)
        assert len(results) <= 5

    def test_query_no_match(self) -> None:
        mem = ProjectMemory()
        mem.add_important_file("src/main.py", "entry point")
        results = mem.query("quantum computing blockchain")
        assert len(results) == 0

    def test_to_context_string_ranked(self) -> None:
        mem = ProjectMemory()
        mem.add_important_file("app/core.py", "main logic")
        result = mem.to_context_string_ranked("core.py main logic")
        assert "PROJECT MEMORY" in result
        assert "core.py" in result

    def test_to_context_string_ranked_empty(self) -> None:
        mem = ProjectMemory()
        result = mem.to_context_string_ranked("quantum computing")
        assert "No relevant project memory found" in result

    def test_query_relevance_score_range(self) -> None:
        mem = ProjectMemory()
        mem.add_successful_fix("Fixed the database connection")
        results = mem.query("database connection")
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_query_multiple_categories(self) -> None:
        mem = ProjectMemory()
        mem.add_important_file("app/main.py", "main file")
        mem.add_successful_fix("Fixed import error in main")
        mem.add_failure("TypeError in main function")
        results = mem.query("main")
        types = {r["type"] for r in results}
        assert len(types) >= 2
