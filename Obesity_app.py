
from flask import Flask, render_template, request, flash, redirect, url_for

app = Flask(__name__)
app.secret_key = "wjdghks3#"


@app.route("/spec", methods = ["POST", "GET"])
def spec():
	return render_template("question_form.html")

@app.route("/form", methods = ["POST"])
def form():
	height = request.form.get('height input')
	weight = request.form.get('weight input')
	#print(height.data)
	return render_template("index.html", height = height, weight = weight)
	#return redirect(url_for('dashboard', infos = height))

# @app.route("/dashboard/<infos>")
# def dashboard(infos):
# 	print(infos)
# 	return infos

# @app.route('/handle_data', methods=['POST'])
# def handle_data():
#     height = request.form.get('height input')

#     print(height)
#     return (height)
    # your code
    # return a response

# @app.route("/")
# def home():
# 	return render_template("index.html")


# @app.route("/login", methods = ['POST', 'GET'])
# def login():
# 	def login():
# 	    if request.method == "POST":
# 		    user = request.form["nm"]
# 		    return redirect(url_for("user", usr=user))
# 	    else:
# 		    return render_template("login.html")

# @app.route("/<usr>")
# def user(usr):
# 	return f"<h1>{usr}</h1>"

