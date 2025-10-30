pytest -q \
  --maxfail=1 \
  --disable-warnings \
  --cov \
  --cov-config=.coveragerc \
  --cov-report=term-missing \
  --cov-report=html \
  --cov-fail-under=90
