#!/bin/bash
coverage run
coverage report > misc/coverage.txt
coverage-badge -o misc/coverage.svg -f

