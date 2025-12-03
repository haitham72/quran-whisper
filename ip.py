from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def home():
    name = request.args.get("name", "")
    if name:
        return f"Welcome {name}"
    return '''
        <form>
            <input name="name" placeholder="Your name">
            <button type="submit">Go</button>
        </form>
    '''

app.run(host="0.0.0.0", port=8000)

