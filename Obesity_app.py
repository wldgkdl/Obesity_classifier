
from flask import Flask, render_template, request, flash

app = Flask(__name__)
app.secret_key = "wjdghks3#"


@app.route("/spec", methods = ["POST", "GET"])
def spec():
	return render_template("question_form.html")

