import flask

app = flask.Flask(__name__)

@app.route("/")
def prime():
    return flask.redirect("/Getip")

@app.route("/Getip")

def index():

    ip_address = flask.request.remote_addr

    return "IP: " + ip_address


app.run(host='0.0.0.0', port=5000)