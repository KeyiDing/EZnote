#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 23:47:28 2022

@author: chris.w
"""

from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
app = Flask(__name__)
import os
app.config['VIDEO_UPLOADS'] = "./uploads"

@app.route("/", methods = ['POST', 'GET'])
def uploadVideo():
    if request.methods == "POST":
        video = request.files('files')
        filename = secure_filename(video.filename)
        return render_template("index.html", filename = filename)
    return render_template("index.html")

app.run(port = 5000)