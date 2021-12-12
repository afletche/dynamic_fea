rm -r build/
rm -r *.egg-info
rm -r */**.so
find . -name "*.so" -type f -delete
find -type d -name __pycache__ -a -prune -exec rm -rf {} \;
