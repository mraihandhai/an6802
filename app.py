from flask import Flask, render_template, request
import joblib, sklearn, groq, os

model = joblib.load("foodexp.pkl")

from groq import Groq

client = Groq()

app = Flask(__name__)

@app.route("/",methods=["get","post"])
def index():
    return(render_template("index.html"))

@app.route("/main",methods=["GET","POST"])
def main():
    q = request.form.get("q")
    print(q)
    return(render_template("main.html"))

@app.route("/ethics",methods=["GET","POST"])
def ethics():
    return(render_template("ethics.html"))

@app.route("/correct",methods=["GET","POST"])
def correct():
    return(render_template("correct.html"))

@app.route("/wrong",methods=["GET","POST"])
def wrong():
    return(render_template("wrong.html"))

@app.route("/econ",methods=["GET","POST"])
def econ():
    return(render_template("econ.html"))

@app.route("/chatbot",methods=["GET","POST"])
def chatbot():
    return(render_template("chatbot.html"))


@app.route("/roe",methods=["GET","POST"])
def roe():
    r=client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages = [
            {"role":"system", "content": "Please explain what is ROE in 20 words"}
        ]
    )
    return(render_template("roe.html", r=r.choices[0].message.content))

@app.route("/generalQuestion",methods=["GET","POST"])
def generalQuestion():
    return(render_template("generalQuestion.html"))

@app.route("/groqReply",methods=["GET","POST"])
def groqReply():
    q = request.form.get("q")
    r=client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages = [
            {"role":"system", "content": q}
        ]
    )
    return(render_template("groqReply.html", r=r.choices[0].message.content))

@app.route("/foodExp",methods=["GET","POST"])
def foodExp():
    q = float(request.form.get("q"))
    r = model.predict([[q]])
    return(render_template("foodExp.html", r = r[0]))

if __name__=="__main__":
    app.run()