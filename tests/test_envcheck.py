from mdagent import envcheck


def test_working_platform_has_no_problem():
    assert envcheck.try_platform("Reference") is None
    notes, problems = envcheck.check_environment("Reference")
    assert problems == []
    assert "Using Reference" in notes[0]


def test_listed_platform_that_fails_stops_the_run(monkeypatch):
    error = "Error loading CUDA module: CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)"
    monkeypatch.setattr(envcheck, "available_platforms", lambda: ["Reference", "CPU", "CUDA", "OpenCL"])
    monkeypatch.setattr(envcheck, "try_platform", lambda name: error if name == "CUDA" else None)
    notes, problems = envcheck.check_environment("CUDA")
    assert len(problems) == 1
    assert "cuda-version=" in problems[0]
    assert "nvidia-smi" in problems[0]
    assert "--platform OpenCL" in problems[0]


def test_unknown_platform():
    notes, problems = envcheck.check_environment("NoSuchPlatform")
    assert len(problems) == 1 and "not available" in problems[0]
