"""Project Understanding Engine for V4 — analyzes repository structure."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.tools.security import is_secret_path


class LanguageInfo(BaseModel):
    name: str
    file_count: int = 0
    extensions: list[str] = Field(default_factory=list)


class ModuleInfo(BaseModel):
    path: str
    name: str
    language: str = ""
    description: str = ""
    is_test: bool = False
    is_config: bool = False
    imports: list[str] = Field(default_factory=list)
    size_bytes: int = 0


class ProjectProfile(BaseModel):
    """Structured representation of a repository."""
    root_path: str = ""
    languages: list[LanguageInfo] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    entrypoints: list[str] = Field(default_factory=list)
    source_modules: list[ModuleInfo] = Field(default_factory=list)
    test_modules: list[ModuleInfo] = Field(default_factory=list)
    config_files: list[str] = Field(default_factory=list)
    test_framework: str = ""
    dependencies: dict[str, str] = Field(default_factory=dict)
    dev_dependencies: dict[str, str] = Field(default_factory=dict)
    build_system: str = ""
    documentation_files: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_path": self.root_path,
            "languages": [l.model_dump() for l in self.languages],
            "frameworks": self.frameworks,
            "entrypoints": self.entrypoints,
            "source_modules": len(self.source_modules),
            "test_modules": len(self.test_modules),
            "config_files": self.config_files,
            "test_framework": self.test_framework,
            "dependencies": self.dependencies,
            "build_system": self.build_system,
            "risks": self.risks,
        }


class ProjectUnderstandingEngine:
    """Analyzes repository structure to build a ProjectProfile."""

    PYTHON_EXTENSIONS = {".py"}
    JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}
    CONFIG_PATTERNS = {
        "package.json", "requirements.txt", "setup.py", "setup.cfg", "pyproject.toml",
        "Cargo.toml", "go.mod", "Makefile", "Dockerfile", "docker-compose.yml",
        ".gitignore", "tsconfig.json", "webpack.config.js",
    }
    TEST_PATTERNS = {"test_", "_test", "tests", "test", "__tests__"}
    ENTRYPOINT_PATTERNS = {"main.py", "app.py", "cli.py", "index.py", "server.py", "__main__.py"}

    def __init__(self, root_path: str | Path = "") -> None:
        self.root_path = Path(root_path) if root_path else Path.cwd()

    def analyze(self) -> ProjectProfile:
        """Perform full project analysis."""
        profile = ProjectProfile(root_path=str(self.root_path))

        self._detect_languages(profile)
        self._detect_frameworks(profile)
        self._detect_dependencies(profile)
        self._scan_modules(profile)
        self._detect_entrypoints(profile)
        self._detect_config_files(profile)
        self._detect_documentation(profile)
        self._assess_risks(profile)

        return profile

    def _detect_languages(self, profile: ProjectProfile) -> None:
        ext_counts: dict[str, int] = {}
        ext_map: dict[str, list[str]] = {}

        _skip = {".git", "__pycache__", ".venv", ".opencode", "node_modules"}
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if d not in _skip]
            if ".git" in root or "__pycache__" in root or ".venv" in root or ".opencode" in root:
                continue
            for f in files:
                filepath = Path(root) / f
                if is_secret_path(filepath, self.root_path):
                    continue
                ext = Path(f).suffix.lower()
                if ext:
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
                    if ext not in ext_map:
                        ext_map[ext] = []
                    ext_map[ext].append(f)

        lang_map = {
            ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
            ".jsx": "JavaScript (React)", ".tsx": "TypeScript (React)",
            ".java": "Java", ".go": "Go", ".rs": "Rust", ".rb": "Ruby",
            ".c": "C", ".cpp": "C++", ".h": "C Header",
        }

        for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
            lang_name = lang_map.get(ext, ext)
            profile.languages.append(LanguageInfo(
                name=lang_name,
                file_count=count,
                extensions=[ext],
            ))

    def _detect_frameworks(self, profile: ProjectProfile) -> None:
        framework_indicators = {
            "django": "Django", "flask": "Flask", "fastapi": "FastAPI",
            "pytest": "pytest", "unittest": "unittest",
            "react": "React", "vue": "Vue", "angular": "Angular",
            "express": "Express", "next": "Next.js",
            "spring": "Spring", "rails": "Rails",
        }

        _skip = {".git", "__pycache__", ".venv", ".opencode", "node_modules"}
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if d not in _skip]
            if ".git" in root or "__pycache__" in root or ".opencode" in root:
                continue
            for f in files:
                f_lower = f.lower()
                for indicator, framework in framework_indicators.items():
                    if indicator in f_lower and framework not in profile.frameworks:
                        profile.frameworks.append(framework)

    def _detect_dependencies(self, profile: ProjectProfile) -> None:
        req_file = self.root_path / "requirements.txt"
        if req_file.exists() and not is_secret_path(req_file, self.root_path):
            profile.build_system = "pip"
            profile.test_framework = "pytest"
            try:
                for line in req_file.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("-"):
                        if "==" in line:
                            name, version = line.split("==", 1)
                            profile.dependencies[name.strip()] = version.strip()
                        else:
                            profile.dependencies[line] = "latest"
            except (OSError, UnicodeDecodeError):
                pass

        pkg_file = self.root_path / "package.json"
        if pkg_file.exists() and not is_secret_path(pkg_file, self.root_path):
            profile.build_system = "npm"
            try:
                import json
                data = json.loads(pkg_file.read_text())
                profile.dependencies.update(data.get("dependencies", {}))
                profile.dev_dependencies.update(data.get("devDependencies", {}))
                if "scripts" in data:
                    profile.entrypoints.extend(
                        f"npm:{name}" for name in data["scripts"]
                    )
            except (OSError, json.JSONDecodeError):
                pass

    def _scan_modules(self, profile: ProjectProfile) -> None:
        _skip = {".git", "__pycache__", ".venv", ".opencode", "node_modules"}
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if d not in _skip]
            if ".git" in root or "__pycache__" in root or ".venv" in root or ".opencode" in root:
                continue

            for f in files:
                if not f.endswith(".py"):
                    continue

                filepath = Path(root) / f
                if is_secret_path(filepath, self.root_path):
                    continue
                rel_path = filepath.relative_to(self.root_path)
                is_test = any(p in str(rel_path).lower() for p in self.TEST_PATTERNS)

                try:
                    size = filepath.stat().st_size
                except OSError:
                    size = 0

                module = ModuleInfo(
                    path=str(rel_path),
                    name=f[:-3] if f.endswith(".py") else f,
                    language="Python",
                    is_test=is_test,
                    size_bytes=size,
                )

                if is_test:
                    profile.test_modules.append(module)
                else:
                    profile.source_modules.append(module)

    def _detect_entrypoints(self, profile: ProjectProfile) -> None:
        for module in profile.source_modules:
            if any(p in module.path for p in self.ENTRYPOINT_PATTERNS):
                profile.entrypoints.append(module.path)

        main_file = self.root_path / "app" / "main.py"
        if main_file.exists() and "app/main.py" not in profile.entrypoints:
            profile.entrypoints.append("app/main.py")

    def _detect_config_files(self, profile: ProjectProfile) -> None:
        _skip = {".git", "__pycache__", ".venv", ".opencode", "node_modules"}
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if d not in _skip]
            if ".git" in root or ".opencode" in root:
                continue
            for f in files:
                if f in self.CONFIG_PATTERNS or f.endswith((".cfg", ".ini", ".toml", ".yaml", ".yml")):
                    rel_path = str(Path(root) / f)
                    try:
                        rel_path = str(Path(rel_path).relative_to(self.root_path))
                    except ValueError:
                        pass
                    profile.config_files.append(rel_path)

    def _detect_documentation(self, profile: ProjectProfile) -> None:
        doc_patterns = {"README", "CHANGELOG", "CONTRIBUTING", "LICENSE", "SECURITY"}
        _skip = {".git", "__pycache__", ".venv", ".opencode", "node_modules"}
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if d not in _skip]
            if ".git" in root or ".opencode" in root:
                continue
            for f in files:
                if any(p in f.upper() for p in doc_patterns) or f.endswith((".md", ".rst", ".txt")):
                    rel_path = str(Path(root) / f)
                    try:
                        rel_path = str(Path(rel_path).relative_to(self.root_path))
                    except ValueError:
                        pass
                    profile.documentation_files.append(rel_path)

    def _assess_risks(self, profile: ProjectProfile) -> None:
        if not profile.test_modules:
            profile.risks.append("No test files detected")

        if not profile.config_files:
            profile.risks.append("No configuration files detected")

        if not profile.documentation_files:
            profile.risks.append("No documentation files detected")

        total_modules = len(profile.source_modules) + len(profile.test_modules)
        if total_modules > 100:
            profile.risks.append(f"Large codebase ({total_modules} modules)")

        for lang in profile.languages:
            if lang.file_count > 50:
                profile.risks.append(f"Large {lang.name} codebase ({lang.file_count} files)")
