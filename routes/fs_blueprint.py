import os
import sys
from flask import jsonify, request, Blueprint, send_from_directory
from flask_cors import CORS

fs_blueprint = Blueprint("fs_blueprint", __name__)
CORS(fs_blueprint)


@fs_blueprint.route("/images/<path:filename>")
def send_image(filename):
    return (
        send_from_directory("/Users/diqingchang/cam-calibration/backend/upload/images", filename, as_attachment=True),
        200,
    )


@fs_blueprint.route("/images/<path:filename>", methods=["DELETE"])
def delete_image(filename):
    pass
    # return send_from_directory('/Users/diqingchang/cam-calibration/backend/upload/images', filename, as_attachment=True), 200
