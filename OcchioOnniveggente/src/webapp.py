from flask import Flask, render_template

app = Flask(__name__, template_folder='templates', static_folder='static')


@app.get('/docs')
def docs() -> str:
    """Render the documentation page."""
    return render_template('docs.html')


if __name__ == '__main__':
    app.run(debug=True)
