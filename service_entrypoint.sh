#!/bin/bash

cd src

flask db init

flask db migrate 

flask db upgrade

gunicorn --workers 1 --threads 2 --bind 0.0.0.0:5000 main:app