from __future__ import annotations

from app.tools.ast_security import ASTSecurityAnalyzer, ASTSecurityFinding, analyze_python_file


class TestASTSecurityAnalyzer:
    def setup_method(self) -> None:
        self.analyzer = ASTSecurityAnalyzer()

    def test_clean_code_no_findings(self) -> None:
        source = """
def add(a, b):
    return a + b

x = add(1, 2)
"""
        findings = self.analyzer.analyze(source)
        assert len(findings) == 0

    def test_import_requests(self) -> None:
        source = "import requests"
        findings = self.analyzer.analyze(source)
        assert len(findings) == 1
        assert findings[0].severity == "CRITICAL"
        assert findings[0].category == "network_import"
        assert "requests" in findings[0].description

    def test_from_requests_import(self) -> None:
        source = "from requests import get"
        findings = self.analyzer.analyze(source)
        assert len(findings) == 1
        assert findings[0].severity == "CRITICAL"
        assert findings[0].category == "network_import"

    def test_import_urllib(self) -> None:
        source = "import urllib.request"
        findings = self.analyzer.analyze(source)
        assert len(findings) == 1
        assert findings[0].severity == "CRITICAL"

    def test_import_httpx(self) -> None:
        source = "import httpx"
        findings = self.analyzer.analyze(source)
        assert len(findings) == 1
        assert findings[0].severity == "CRITICAL"

    def test_import_aiohttp(self) -> None:
        source = "from aiohttp import ClientSession"
        findings = self.analyzer.analyze(source)
        assert len(findings) == 1
        assert findings[0].severity == "CRITICAL"

    def test_eval_call(self) -> None:
        source = 'eval("1+1")'
        findings = self.analyzer.analyze(source)
        assert len(findings) == 1
        assert findings[0].severity == "CRITICAL"
        assert findings[0].category == "dangerous_function"
        assert "eval()" in findings[0].description

    def test_exec_call(self) -> None:
        source = 'exec(code_string)'
        findings = self.analyzer.analyze(source)
        assert len(findings) == 1
        assert findings[0].severity == "CRITICAL"
        assert findings[0].category == "dangerous_function"

    def test_os_system(self) -> None:
        source = 'import os\nos.system("ls")'
        findings = self.analyzer.analyze(source)
        os_findings = [f for f in findings if f.category == "os_execution"]
        assert len(os_findings) == 1
        assert os_findings[0].severity == "CRITICAL"

    def test_os_popen(self) -> None:
        source = 'import os\nos.popen("ls")'
        findings = self.analyzer.analyze(source)
        os_findings = [f for f in findings if f.category == "os_execution"]
        assert len(os_findings) == 1

    def test_subprocess_run(self) -> None:
        source = 'import subprocess\nsubprocess.run(["ls"])'
        findings = self.analyzer.analyze(source)
        sub_findings = [f for f in findings if f.category == "subprocess_execution"]
        assert len(sub_findings) == 1
        assert sub_findings[0].severity == "HIGH"

    def test_subprocess_popen(self) -> None:
        source = 'import subprocess\nsubprocess.Popen(["ls"])'
        findings = self.analyzer.analyze(source)
        sub_findings = [f for f in findings if f.category == "subprocess_execution"]
        assert len(sub_findings) == 1

    def test_subprocess_call(self) -> None:
        source = 'import subprocess\nsubprocess.call(["ls"])'
        findings = self.analyzer.analyze(source)
        sub_findings = [f for f in findings if f.category == "subprocess_execution"]
        assert len(sub_findings) == 1

    def test_dynamic_import(self) -> None:
        source = '__import__("requests")'
        findings = self.analyzer.analyze(source)
        dyn_findings = [f for f in findings if f.category == "dynamic_import"]
        assert len(dyn_findings) == 1
        assert dyn_findings[0].severity == "CRITICAL"

    def test_sensitive_file_access(self) -> None:
        source = 'open("/etc/passwd")'
        findings = self.analyzer.analyze(source)
        file_findings = [f for f in findings if f.category == "sensitive_file_access"]
        assert len(file_findings) == 1
        assert file_findings[0].severity == "CRITICAL"

    def test_sensitive_file_shadow(self) -> None:
        source = 'open("/etc/shadow")'
        findings = self.analyzer.analyze(source)
        file_findings = [f for f in findings if f.category == "sensitive_file_access"]
        assert len(file_findings) == 1

    def test_sensitive_file_ssh(self) -> None:
        source = 'open("/.ssh/id_rsa")'
        findings = self.analyzer.analyze(source)
        file_findings = [f for f in findings if f.category == "sensitive_file_access"]
        assert len(file_findings) == 1

    def test_normal_open_not_flagged(self) -> None:
        source = 'open("data.txt")'
        findings = self.analyzer.analyze(source)
        assert len(findings) == 0

    def test_multiple_findings(self) -> None:
        source = "import requests\nimport os\neval('code')\nos.system('rm -rf /')"
        findings = self.analyzer.analyze(source)
        assert len(findings) == 3
        severities = [f.severity for f in findings]
        assert all(s == "CRITICAL" for s in severities)

    def test_line_numbers_preserved(self) -> None:
        source = "x = 1\nimport requests\ny = 2"
        findings = self.analyzer.analyze(source)
        assert len(findings) == 1
        assert findings[0].line == 2

    def test_syntax_error_returns_empty(self) -> None:
        source = "def foo(:\n  pass"
        findings = self.analyzer.analyze(source)
        assert len(findings) == 0

    def test_method_call_on_module(self) -> None:
        source = "import requests\nrequests.get('http://example.com')"
        findings = self.analyzer.analyze(source)
        network_findings = [f for f in findings if f.category == "network_import"]
        assert len(network_findings) == 1

    def test_compile_call(self) -> None:
        source = 'compile(source, "<string>", "exec")'
        findings = self.analyzer.analyze(source)
        dangerous = [f for f in findings if f.category == "dangerous_function"]
        assert len(dangerous) == 1

    def test_httplib2_import(self) -> None:
        source = "import httplib2"
        findings = self.analyzer.analyze(source)
        assert len(findings) == 1
        assert findings[0].severity == "CRITICAL"

    def test_os_system_with_variable(self) -> None:
        source = "import os\nos.system(cmd)"
        findings = self.analyzer.analyze(source)
        os_findings = [f for f in findings if f.category == "os_execution"]
        assert len(os_findings) == 1

    def test_normal_function_not_flagged(self) -> None:
        source = """
import json
data = json.loads("{}")
print(data)
"""
        findings = self.analyzer.analyze(source)
        assert len(findings) == 0


class TestAnalyzePythonFile:
    def test_analyze_existing_file(self, tmp_path) -> None:
        filepath = tmp_path / "test.py"
        filepath.write_text('import requests\nrequests.get("http://example.com")')
        findings = analyze_python_file(str(filepath))
        assert len(findings) >= 1
        assert any(f.category == "network_import" for f in findings)

    def test_analyze_nonexistent_file(self) -> None:
        findings = analyze_python_file("/nonexistent/file.py")
        assert len(findings) == 0

    def test_analyze_binary_file(self, tmp_path) -> None:
        filepath = tmp_path / "binary.py"
        filepath.write_bytes(b"\x00\x01\x02\x03")
        findings = analyze_python_file(str(filepath))
        assert len(findings) == 0

    def test_analyze_clean_file(self, tmp_path) -> None:
        filepath = tmp_path / "clean.py"
        filepath.write_text("x = 1\ny = 2\nz = x + y")
        findings = analyze_python_file(str(filepath))
        assert len(findings) == 0
