import pymongo
import cv2
from ..utils.img_handler import img_handler
from ..utils.logger import logger
from ..camcalibr.reso_meas_slanted.reso_meas import reso_meas
from ..db_client.mongodb_client import reso_collection
from backend.model.reso_measure import ResoMeasure, ResoMeasureSchema, ResoResult, ResoResultSchema
from flask import jsonify, request, Blueprint
from bson import json_util
from flask_cors import CORS

reso_blueprint = Blueprint("reso_blueprint", __name__)
CORS(reso_blueprint)


@reso_blueprint.route("/reso", methods=["GET"])
def get_reso_by_id():
    # Check if key dut_id exists in the json
    params = request.args
    dut_id = params.get("dut_id")

    if dut_id is None:
        return {"message": "Bad request, request must have dut_id as parameter"}, 400

    # Find reso measurement according to dut id
    reso_measure = reso_collection.find_one({"dut_id": dut_id})

    # If found return it
    if not reso_measure is None:
        dict_reso_measure = ResoMeasureSchema().dump(reso_measure)
    else:
        reso_measure = ResoMeasure(dut_id)
        dict_reso_measure = ResoMeasureSchema().dump(reso_measure)
        # Delete the key '_id' regardless of whether it is in the dictionary, because we want to _id added by MongoDB
        dict_reso_measure.pop("_id", None)
        result = reso_collection.insert_one(dict_reso_measure).inserted_id
        dict_reso_measure["_id"] = json_util.dumps(result)

    return jsonify({"reso_meas": dict_reso_measure}), 200


@reso_blueprint.route("/reso_input_img", methods=["POST"])
def add_reso_input_img():
    """Add a new input image to reso measure
    1. find reso measure by id, if not found return error msg
    2. add image url to input_imgs
    3. return the added image url as response
    """

    # Find reso measure by id
    dut_id = request.form["dut_id"]

    logger.info("welcome to upload`")
    file = request.files["file"]
    filename = img_handler.save_file(file)

    # Update the input_imgs field in document
    try:
        reso_collection.update({"dut_id": dut_id}, {"$push": {"input_imgs": filename}})
    except pymongo.errors.PyMongoError as e:
        return jsonify({"message": e}), 400

    # TODO: respond only an imageUrl instead of the full list does not look so rubust.
    response = {"image_url": filename}
    return jsonify(response), 200


@reso_blueprint.route("/reso_input_img", methods=["DELETE"])
def remove_reso_input_img():
    """Remove input image from reso measure
    1. find reso measure by id, if not found return error msg
    2. remove image url from DB
    3. return the removed image url as response
    """

    # Extract dut_id and img_url from request
    dut_id = request.json["dut_id"]
    img_url = request.json["img_url"]

    if dut_id is None or img_url is None:
        return {"message": "Bad request, request must have dut_id and img_url as parameters"}, 400

    img_handler.delete(img_url)

    # Update the input_imgs field in document
    try:
        reso_collection.update({"dut_id": dut_id}, {"$pull": {"input_imgs": img_url}})
    except pymongo.errors.PyMongoError as e:
        return jsonify({"message": e}), 400

    # TODO: respond only an imageUrl instead of the full list does not look so rubust.
    response = {"image_url": img_url}
    return jsonify(response), 200


@reso_blueprint.route("/reso_exec", methods=["GET"])
def exec_reso():
    """Execute reso measurement
    1. find reso measurement
    2. execute reso measurement, catch err
    3. return measurement result reso_result
    """

    # Find reso measurement by id
    params = request.args
    dut_id = params.get("dut_id")

    reso_measure = reso_collection.find_one({"dut_id": dut_id})
    if reso_measure is None:
        return jsonify({"message": f"reso measurement with id {dut_id} is not found!"}), 404
    reso_measure = ResoMeasureSchema().load(ResoMeasureSchema().dump(reso_measure))

    # Load input images
    input_images = []
    for img_url in reso_measure.input_imgs:
        img = img_handler.load(img_url)
        input_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    if len(input_images) == 0:
        return jsonify({"message": "Bad request, there must be at least one input image!"}), 400

    # TODO: These parameters should be provided by use from client!

    mtf, output_imgs = reso_meas(input_images)

    # Store the ouptut images
    output_img_urls = []
    for img in output_imgs:
        img_url = img_handler.save(img)
        output_img_urls.append(img_url)

    # Form a resoResult
    reso_result = ResoResult(mtf, output_img_urls)

    # If there is already reso measurement result, remove the old output images first
    if reso_measure.reso_result != None:
        while len(reso_measure.reso_result.output_imgs) > 0:
            fname = reso_measure.reso_result.output_imgs[-1]
            img_handler.delete(fname)
            reso_measure.reso_result.output_imgs.pop()

    # Update the DB
    try:
        dict_reso_result = ResoResultSchema().dump(reso_result)
        reso_collection.update({"dut_id": dut_id}, {"$set": {"reso_result": dict_reso_result}})
    except pymongo.errors.PyMongoError as e:
        return jsonify({"message": e}), 400

    reso_result_dict = ResoResultSchema().dump(reso_result)

    return jsonify({"reso_result": reso_result_dict}), 200
