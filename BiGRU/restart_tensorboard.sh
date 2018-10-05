#!/bin/bash
rm -rf log/train/*
tensorboard --logdir=log --port=9008
