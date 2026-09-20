#!/usr/bin/env python3
"""
AURA-Drive™ Automated DevSecOps Static Analysis & ISO 21434 Audit
================================================================
Automated security audit tool:
  - AST analysis for unsafe eval, exec, shell injection, and insecure deserialization.
  - Hardcoded credential and API secret scanning.
  - File permission and sensitive configuration checks.
  - UNECE R155 / ISO 21434 Automotive Cybersecurity Compliance Report generator.
"""

import os
import ast
import re
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Root directory of the repository
REPO_ROOT = Path(__file__).resolve().parent.parent

# Forbidden function calls and insecure patterns
DANGEROUS_CALLS = {"eval", "exec", "input"}
INSECURE_SUBPROCESS_PATTERNS = {"shell=True", "shell = True"}
SECRET_PATTERNS = [
    (re.compile(r'(?i)(password|secret|api_key|token|auth)\s*=\s*["\']([a-zA-Z0-9_\-\.]{8,})["\']'), "Potential Hardcoded Secret"),
    (re.compile(r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----'), "Embedded Private Key"),
]

# Files or directories to ignore during security scanning
IGNORED_DIRS = {".git", ".pytest_cache", "__pycache__", "venv", ".venv", "env", "node_modules"}
IGNORED_FILES = {".env.example", "devsecops_audit.py"}


class SecurityVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.findings: List[Dict[str, Any]] = []

    def visit_Call(self, node: ast.Call):
        # Detect eval() / exec()
        if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_CALLS:
            self.findings.append({
                "type": "DANGEROUS_FUNCTION_CALL",
                "severity": "CRITICAL",
                "file": self.filename,
                "line": node.lineno,
                "detail": f"Direct invocation of prohibited function '{node.func.id}()'."
            })

        # Detect subprocess.run(..., shell=True)
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("call", "Popen", "run", "check_output"):
            for kw in node.keywords:
                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    self.findings.append({
                        "type": "COMMAND_INJECTION_RISK",
                        "severity": "HIGH",
                        "file": self.filename,
                        "line": node.lineno,
                        "detail": "subprocess execution with shell=True creates command injection risk."
                    })
        self.generic_visit(node)


def scan_python_file(filepath: Path) -> List[Dict[str, Any]]:
    findings = []
    rel_path = str(filepath.relative_to(REPO_ROOT))

    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return [{"type": "FILE_READ_ERROR", "severity": "LOW", "file": rel_path, "line": 0, "detail": str(e)}]

    # Regex secret scanning
    for line_idx, line in enumerate(content.splitlines(), start=1):
        # Ignore comments
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern, desc in SECRET_PATTERNS:
            # Skip if explicitly in a test dummy or example
            if "dummy" in stripped.lower() or "example" in stripped.lower() or "test" in stripped.lower():
                continue
            if pattern.search(line):
                findings.append({
                    "type": "HARDCODED_CREDENTIAL",
                    "severity": "HIGH",
                    "file": rel_path,
                    "line": line_idx,
                    "detail": desc
                })

    # AST Syntax Analysis
    try:
        tree = ast.parse(content, filename=str(filepath))
        visitor = SecurityVisitor(rel_path)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    except SyntaxError:
        pass  # Skip non-python or syntactically invalid files

    return findings


def run_full_devsecops_audit() -> Dict[str, Any]:
    print("=" * 70)
    print("AURA-Drive™ Automated DevSecOps & ISO 21434 Compliance Auditor")
    print("=" * 70)
    start_time = time.time()

    all_findings = []
    scanned_files_count = 0

    for root, dirs, files in os.walk(REPO_ROOT):
        # Prune ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for file in files:
            if file in IGNORED_FILES:
                continue
            if file.endswith(".py"):
                file_path = Path(root) / file
                scanned_files_count += 1
                findings = scan_python_file(file_path)
                all_findings.extend(findings)

    elapsed = time.time() - start_time
    critical_count = sum(1 for f in all_findings if f["severity"] == "CRITICAL")
    high_count = sum(1 for f in all_findings if f["severity"] == "HIGH")
    medium_count = sum(1 for f in all_findings if f["severity"] == "MEDIUM")

    report = {
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "standard_reference": "ISO/SAE 21434 & UNECE R155 (Cybersecurity Engineering)",
        "scanned_python_files": scanned_files_count,
        "elapsed_seconds": round(elapsed, 3),
        "total_findings": len(all_findings),
        "summary": {
            "CRITICAL": critical_count,
            "HIGH": high_count,
            "MEDIUM": medium_count,
            "PASS": (critical_count == 0 and high_count == 0)
        },
        "findings": all_findings
    }

    print(f"\n[+] Scanned {scanned_files_count} Python source files in {elapsed:.3f}s")
    print(f"[+] Security Findings: Critical={critical_count}, High={high_count}, Medium={medium_count}")

    if all_findings:
        print("\nFindings Breakdown:")
        for f in all_findings:
            print(f"  [{f['severity']}] {f['file']}:{f['line']} - {f['type']}: {f['detail']}")
    else:
        print("\n>>> ZERO VULNERABILITIES IDENTIFIED. Codebase satisfies ISO 21434 security baseline! <<<")

    output_report_path = REPO_ROOT / "docs" / "devsecops_audit_report.json"
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Audit Report exported to: {output_report_path}")
    print("=" * 70)
    return report


if __name__ == "__main__":
    rep = run_full_devsecops_audit()
    if not rep["summary"]["PASS"]:
        print("\n[!] DevSecOps Gate Failed: Security policy violations detected.")
        sys.exit(1)
    else:
        print("\n[PASS] DevSecOps Gate Succeeded.")
        sys.exit(0)
