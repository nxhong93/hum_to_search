from python:3.8-slim-buster

cd /model/src/

python3 config.py --test_path "./data/"
python3 predict.py
