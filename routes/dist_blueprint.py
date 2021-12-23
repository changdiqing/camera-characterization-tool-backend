import pymongo
from ..utils.img_handler import img_handler
from ..utils.logger import logger
from ..camcalibr.dist_calibr.dist_calibr_opencv import dist_calibr
from ..db_client.mongodb_client import dist_collection
from backend.model.dist_measure import DistMeasure, DistMeasureSchema, DistResult, DistResultSchema
from flask import jsonify, request, Blueprint
from bson import json_util
from flask_cors import CORS

dist_blueprint = Blueprint("distortion_blueprint", __name__)
CORS(dist_blueprint)


@dist_blueprint.route("/dist", methods=["GET"])
def get_dist_by_id():
    # check if key dut_id exists in the json
    params = request.args
    dut_id = params.get("dut_id")

    if dut_id is None:
        return {"message": "Bad request, request must have dut_id as parameter"}, 400

    # find distortion measurement according to dut id
    dist_measure = dist_collection.find_one({"dut_id": dut_id})

    # if found return it
    if not dist_measure is None:
        dict_dist_measure = DistMeasureSchema().dump(dist_measure)

    else:
        dist_measure = DistMeasure(dut_id, [])
        dict_dist_measure = DistMeasureSchema().dump(dist_measure)
        # Delete the key '_id' regardless of whether it is in the dictionary, because we want to _id added by MongoDB
        dict_dist_measure.pop("_id", None)
        result = dist_collection.insert_one(dict_dist_measure).inserted_id
        dict_dist_measure["_id"] = json_util.dumps(result)

    return jsonify(dict_dist_measure), 200


@dist_blueprint.route("/dist_img", methods=["POST"])
def add_dist_img():
    """Add a new image to dist measure
    1. find dist measure by id, if not found return error msg
    2. add image url to input_imgs
    3. return the added image url as response
    """

    # Find distortion measure by id
    dut_id = request.form["dut_id"]

    logger.info("welcome to upload`")
    file = request.files["file"]
    filename = img_handler.save_file(file)

    # Update the input_imgs field in document
    try:
        dist_collection.update({"dut_id": dut_id}, {"$push": {"input_imgs": filename}})
    except pymongo.errors.PyMongoError as e:
        return jsonify({"message": e}), 400

    # TODO: respond only an imageUrl instead of the full list does not look so rubust.
    response = {"image_url": filename}
    return jsonify(response), 200


@dist_blueprint.route("/dist_img", methods=["DELETE"])
def remove_dist_img():
    """Remove image from dist measure
    1. find dist measure by id, if not found return error msg
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
        dist_collection.update({"dut_id": dut_id}, {"$pull": {"input_imgs": img_url}})
    except pymongo.errors.PyMongoError as e:
        return jsonify({"message": e}), 400

    # TODO: respond only an imageUrl instead of the full list does not look so rubust.
    response = {"image_url": img_url}
    return jsonify(response), 200


@dist_blueprint.route("/dist_exec", methods=["GET"])
def exec_dist():
    """Execute distortion measurement
    1. find distortion measurement
    2. execute distortion measurement, catch err
    3. return measurement result DistResult
    """

    # Find distortion measurement by id
    params = request.args
    dut_id = params.get("dut_id")

    dist_measure = dist_collection.find_one({"dut_id": dut_id})
    if dist_measure is None:
        return jsonify({"message": f"Distortion measurement with id {dut_id} is not found!"}), 404
    dist_measure = DistMeasureSchema().load(DistMeasureSchema().dump(dist_measure))

    # Load input images
    input_images = []
    for img_url in dist_measure.input_imgs:
        input_images.append(img_handler.load(img_url))

    if len(input_images) == 0:
        return jsonify({"message": "Bad request, there must be at least one input image!"}), 400

    mtx, dist, rvecs, tvecs, error, output_imgs = dist_calibr(input_images)

    # Store the ouptut images
    output_img_urls = []
    for img in output_imgs:
        img_url = img_handler.save(img)
        output_img_urls.append(img_url)

    # Form a DistResult
    dist_result = DistResult(mtx, dist, rvecs, tvecs, error, output_img_urls)

    # If there is already distortion measurement result, remove the old output images first
    if dist_measure.dist_result != None:
        while len(dist_measure.dist_result.output_imgs) > 0:
            fname = dist_measure.dist_result.output_imgs.pop()
            img_handler.delete(fname)

    # Update the DB
    try:
        dict_dist_result = DistResultSchema().dump(dist_result)
        dist_collection.update({"dut_id": dut_id}, {"$set": {"dist_result": dict_dist_result}})
    except pymongo.errors.PyMongoError as e:
        return jsonify({"message": e}), 400

    dist_result_dict = DistResultSchema().dump(dist_result)

    return jsonify({"dist_result": dist_result_dict}), 200
