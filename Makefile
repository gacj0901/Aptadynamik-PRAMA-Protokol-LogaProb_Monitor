.PHONY: install test verify gemini clean

install:
	python -m pip install -e .

test:
	python -m pytest

verify:
	python -m aptadynamik.verification.scenarios

gemini:
	python -m aptadynamik.pipelines.gemini

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['build', 'dist', '.pytest_cache']]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
