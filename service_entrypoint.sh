#!/bin/bash

cd src

gunicorn --workers 1 --threads 2 --bind 0.0.0.0:5000 main:app