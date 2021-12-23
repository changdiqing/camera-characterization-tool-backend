from flask import Flask
from flask.helpers import url_for
from flask_cors import CORS
from .model.dut import DUT
from .routes.dut_blueprint import dut_blueprint
from .routes.dist_blueprint import dist_blueprint
from .routes.color_blueprint import color_blueprint
from .routes.fs_blueprint import fs_blueprint
from .routes.reso_blueprint import reso_blueprint
from .utils.logger import logger

UPLOAD_FOLDER = "/Users/diqingchang/cam-calibration/backend/upload/images"

app = Flask(__name__, static_folder="/upload/images/test_docs")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = "super secret key"
CORS(app)

app.register_blueprint(dut_blueprint)
app.register_blueprint(dist_blueprint)
app.register_blueprint(color_blueprint)
app.register_blueprint(reso_blueprint)
app.register_blueprint(fs_blueprint)


if __name__ == "__main__":
    logger.info("Start the app")
    # app.secret_key = os.urandom(24)
    app.run(host="localhost", port=5555, debug=True)
