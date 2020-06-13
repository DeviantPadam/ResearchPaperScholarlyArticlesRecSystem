#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:37:36 2020

@author: deviantpadam
"""


from flask import Blueprint


titanic_app = Blueprint('titanic_app',__name__)

from model.titanicModel import FeatureSelector


from app.titanic import titanic