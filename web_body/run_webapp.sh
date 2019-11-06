#!/usr/bin/env bash
#Xvfb :1 -screen 0 1024x768x16 | export DISPLAY=:1.0 | python manage.py migrate 
Xvfb :1 -screen 0 1024x768x16 | export DISPLAY=:1.0 | python manage.py runserver 0.0.0.0:8000
