import os

from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hellow_world():
    Name = request.args.get('name', None)
    return "Hello {}!".format(Name)

if __name__ == "__main++":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

