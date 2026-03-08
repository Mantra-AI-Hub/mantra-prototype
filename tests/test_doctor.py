import json
import subprocess
import sys

from mantra import doctor


def test_doctor_checks_return_results():
    results = doctor.run_checks()
    assert results
    assert all("name" in item and "status" in item and "details" in item for item in results)


def test_doctor_runs():
    proc = subprocess.run([sys.executable, "-m", "mantra.doctor"], capture_output=True, text=True, check=False)
    assert proc.returncode in {0, 1}
    assert "MANTRA SYSTEM DIAGNOSTIC" in proc.stdout
    assert "Overall Status:" in proc.stdout


def test_doctor_json_output_valid():
    proc = subprocess.run([sys.executable, "-m", "mantra.doctor", "--json"], capture_output=True, text=True, check=False)
    assert proc.returncode in {0, 1}
    payload = json.loads(proc.stdout)
    assert "overall_status" in payload
    assert "checks" in payload
