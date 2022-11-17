import os

from flask import Flask, request

app = Flask(__name__)


def processVideo(url):
    return (0.23, "Get better at cricket", "Don't play cricket")


@app.route("/", methods=["GET"])
def generate_response():
    url = request.args.get('url', None)
    (score, comment1, comment2) = processVideo(url)
    return ','.join([str(score), comment1, comment2])


if __name__ == "__main++":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
