"""

    all routes and their inplementations are defined here, consider move implementations to a separate module when file
    gets larger

    JSON API Style: JSend https://github.com/omniti-labs/jsend
    example 1:
    {
        status : "success",
        data : { "post" : { "id" : 2, "title" : "Another blog post", "body" : "More content" }， “tail”： “1234”}
    }
    example 2:
    {
        "status" : "error",
        "message" : "Unable to communicate with database"
    }

"""

from backend.utils.img_handler import img_handler
from backend.model.dist_measure import DistMeasure, DistMeasureSchema
from backend.model.reso_measure import ResoMeasure, ResoMeasureSchema
from backend.model.color_measure import ColorMeasure, ColorMeasureSchema
import pymongo
from flask import jsonify, request, Blueprint
from bson import json_util, ObjectId
from ..utils.img_handler import img_handler
from ..model.dut import DUT, DUTSchema
from ..db_client.mongodb_client import devices_collection, dist_collection, color_collection, reso_collection
from flask_cors import CORS


dut_blueprint = Blueprint("dut_blueprint", __name__)
CORS(dut_blueprint)


@dut_blueprint.route("/duts")
def get_duts():
    duts = list(devices_collection.find())
    dict_duts = DUTSchema(many=True).dump(duts)
    return jsonify(dict_duts), 200


@dut_blueprint.route("/duts", methods=["POST"])
def add_dut():
    dict_dut = request.json["dut"]
    if dict_dut is None:
        return {"message": "Bad request, request must have dut as parameter"}, 400

    try:
        _id = devices_collection.insert_one(dict_dut).inserted_id
        dict_dut["_id"] = str(_id)
        return jsonify({"dut": dict_dut}), 200
    except pymongo.errors.PyMongoError as e:
        return jsonify({"message": e}), 400


@dut_blueprint.route("/duts", methods=["DELETE"])
def remove_dut():

    # Check  if there is dut_id
    dut_id = request.json["dut_id"]
    if dut_id is None:
        return {"message": "Bad request, request must have dut_id as parameter"}, 400

    # remove dist measure
    try:
        clean_dist_imgs(dut_id)
        clean_reso_imgs(dut_id)
        clean_color_imgs(dut_id)
        dist_collection.delete_one({"dut_id": dut_id})
        color_collection.delete_one({"dut_id": dut_id})
        reso_collection.delete_one({"dut_id": dut_id})
        devices_collection.delete_one({"_id": ObjectId(dut_id)})
        return jsonify({"dut_id": dut_id}), 200
    except pymongo.errors.PyMongoError as e:
        return jsonify({"message": e}), 400


def clean_dist_imgs(dut_id):
    dict_dist = dist_collection.find_one({"dut_id": dut_id})
    if dict_dist is None:
        return
    dist = DistMeasureSchema().load(DistMeasureSchema().dump(dict_dist))

    for img_url in dist.input_imgs:
        img_handler.delete(img_url)

    if not dist.dist_result is None:
        for img_url in dist.dist_result.output_imgs:
            img_handler.delete(img_url)

    dist_collection.delete_one({"dut_id": dut_id})


def clean_reso_imgs(dut_id):
    dict_reso = reso_collection.find_one({"dut_id": dut_id})
    if dict_reso is None:
        return
    reso = ResoMeasureSchema().load(ResoMeasureSchema().dump(dict_reso))

    for img_url in reso.input_imgs:
        img_handler.delete(img_url)

    if not reso.reso_result is None:
        for img_url in reso.reso_result.output_imgs:
            img_handler.delete(img_url)

    reso_collection.delete_one({"dut_id": dut_id})


def clean_color_imgs(dut_id):
    dict_color = color_collection.find_one({"dut_id": dut_id})
    if dict_color is None:
        return
    color = ColorMeasureSchema().load(ColorMeasureSchema().dump(dict_color))

    for img_url in color.input_imgs:
        img_handler.delete(img_url)

    for img_url in color.ref_imgs:
        img_handler.delete(img_url)

    if not color.color_result is None:
        for img_url in color.color_result.output_imgs:
            img_handler.delete(img_url)

    color_collection.delete_one({"dut_id": dut_id})
