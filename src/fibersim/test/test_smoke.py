from fibersim.main import run_main

def test_imports_and_stub_runs():
    out = run_main({"chain": [], "global": {}})
    assert out.get("status") == "ok"
