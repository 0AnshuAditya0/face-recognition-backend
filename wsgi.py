from app import app as application

# For Gunicorn/Waitress to discover the Flask app, it should expose 'application'
# Example runs:
#   Gunicorn (Linux/Mac): gunicorn -w 2 -k gthread -b 0.0.0.0:5000 --timeout 120 wsgi:application
#   Waitress (Windows):  python -m waitress --port=5000 --host=0.0.0.0 wsgi:application
