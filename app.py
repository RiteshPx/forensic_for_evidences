from flask import Flask
# from routes.home import home_bp
# from routes.userAdditionalDetails import user_bp
from routes.verifyDocuments import userVerify_bp

app = Flask(__name__)

# Register routes
# app.register_blueprint(home_bp)
@app.route("/")
def home():
    return "Welcome to the Home Page"

# app.register_blueprint(user_bp, url_prefix="/")
app.register_blueprint(userVerify_bp, url_prefix="/")

if __name__ == "__main__":
    app.run(debug=True)
