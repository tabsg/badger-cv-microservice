import os

from flask import Flask, request
from cover_drive_judge import CoverDriveJudge

app = Flask(__name__)


def processVideo(url):
    #return (0.2356, "Get better at cricket", "Don't play cricket")
    #with CoverDriveJudge(url) as judge:
    #    (averageScore, advice1, advice2) = judge.process_and_write_video()
    #    return (averageScore, advice1, advice2)

@app.route("/", methods=["GET"])
def generate_response():
    url = request.args.get('url', None)
    (score, comment1, comment2) = processVideo(url)
    return ','.join([str(int(score * 100)), comment1, comment2])


if __name__ == "__main++":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
