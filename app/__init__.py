#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 21:09:52 2020

@author: deviantpadam
"""

from flask import Flask
import os

server = Flask(__name__)
server.config['SECRET_KEY'] = os.urandom(16)
server.config['PERMANENT_SESSION_LIFETIME'] = 300
server.config['ITEMS_PER_PAGE'] = 10

from app.titanic import titanic_app

server.register_blueprint(titanic_app)


from app import main

