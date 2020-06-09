#!/bin/bash
coverage run -m pytest
coverage report > misc/coverage.txt
coverage-badge -o misc/coverage.svg -f

