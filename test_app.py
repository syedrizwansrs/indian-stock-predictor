from flask import Flask
import sys

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"

if __name__ == '__main__':
    print("Starting simple Flask app...", file=sys.stderr)
    app.run(debug=True, host='0.0.0.0', port=5000)
