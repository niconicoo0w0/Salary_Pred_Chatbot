pytest -q \
  --maxfail=1 \
  --disable-warnings \
  --cov=app \
  --cov=utils \
  --cov=models/training_script \
  --cov-report=term-missing \
  --cov-report=html \
  --cov-fail-under=85