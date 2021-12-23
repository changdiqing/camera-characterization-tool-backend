import pymongo
from ..utils.img_handler import img_handler
from ..utils.logger import logger
from ..camcalibr.color_calibr_ccm.color_calibr_ccm import color_calibration
from ..db_client.mongodb_client import color_collection
from backend.model.color_measure import ColorMeasure, ColorMeasureSchema, ColorResult, ColorResultSchema
from flask import jsonify, request, Blueprint
from bson import json_util
from flask_cors import CORS

color_blueprint = Blueprint("color_blueprint", __name__)
CORS(color_blueprint)


@color_blueprint.route("/color", methods=["GET"])
def get_color_by_id():
    # Check if key dut_id exists in the json
    params = request.args
    dut_id = params.get("dut_id")

    if dut_id is None:
        return {"message": "Bad request, request must have dut_id as parameter"}, 400

    # Find color measurement according to dut id
    color_measure = color_collection.find_one({"dut_id": dut_id})

    # If found return it
    if not color_measure is None:
        dict_color_measure = ColorMeasureSchema().dump(color_measure)
    else:
        color_measure = ColorMeasure(dut_id)
        dict_color_measure = ColorMeasureSchema().dump(color_measure)
        # Delete the key '_id' regardless of whether it is in the dictionary, because we want to _id added by MongoDB
        dict_color_measure.pop("_id", None)
        result = color_collection.insert_one(dict_color_measure).inserted_id
        dict_color_measure["_id"] = json_util.dumps(result)

    return jsonify({"color_meas": dict_color_measure}), 200


@color_blueprint.route("/color_input_img", methods=["POST"])
def add_color_input_img():
    """Add a new input image to color measure
    1. find color measure by id, if not found return error msg
    2. add image url to input_imgs
    3. return the added image url as response
    """

    # Find color measure by id
    dut_id = request.form["dut_id"]

    logger.info("welcome to upload`")
    file = request.files["file"]
    filename = img_handler.save_file(file)

    # Update the input_imgs field in document
    try:
        color_collection.update({"dut_id": dut_id}, {"$push": {"input_imgs": filename}})
    except pymongo.errors.PyMongoError as e:
        return jsonify({"message": e}), 400

    # TODO: respond only an imageUrl instead of the full list does not look so rubust.
    response = {"image_url": filename}
    return jsonify(response), 200


@color_blueprint.route("/color_ref_img", methods=["POST"])
def add_color_ref_img():
    """Add a new reference image to color measure
    1. find color measure by id, if not found return error msg
    2. add image url to input_imgs
    3. return the added image url as response
    """

    # Find color measure by id
    dut_id = request.form["dut_id"]

    logger.info("welcome to upload`")
    file = request.files["file"]
    filename = img_handler.save_file(file)

    # Update the input_imgs field in document
    try:
        color_collection.update({"dut_id": dut_id}, {"$push": {"ref_imgs": filename}})
    except pymongo.errors.PyMongoError as e:
        return jsonify({"message": e}), 400

    # TODO: responding only an imageUrl instead of the full list does not look so robust.
    response = {"image_url": filename}
    return jsonify(response), 200


@color_blueprint.route("/color_input_img", methods=["DELETE"])
def remove_color_input_img():
    """Remove input image from color measure
    1. find color measure by id, if not found return error msg
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
        color_collection.update({"dut_id": dut_id}, {"$pull": {"input_imgs": img_url}})
    except pymongo.errors.PyMongoError as e:
        return jsonify({"message": e}), 400

    # TODO: respond only an imageUrl instead of the full list does not look so rubust.
    response = {"image_url": img_url}
    return jsonify(response), 200


@color_blueprint.route("/color_ref_img", methods=["DELETE"])
def remove_color_ref_img():
    """Remove reference image from color measure
    1. find color measure by id, if not found return error msg
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
        color_collection.update({"dut_id": dut_id}, {"$pull": {"ref_imgs": img_url}})
    except pymongo.errors.PyMongoError as e:
        return jsonify({"message": e}), 400

    # TODO: respond only an imageUrl instead of the full list does not look so rubust.
    response = {"image_url": img_url}
    return jsonify(response), 200


@color_blueprint.route("/color_exec", methods=["GET"])
def exec_color():
    """Execute color measurement
    1. find color measurement
    2. execute color measurement, catch err
    3. return measurement result color_result
    """

    # Find color measurement by id
    params = request.args
    dut_id = params.get("dut_id")

    color_measure = color_collection.find_one({"dut_id": dut_id})
    if color_measure is None:
        return jsonify({"message": f"color measurement with id {dut_id} is not found!"}), 404
    color_measure = ColorMeasureSchema().load(ColorMeasureSchema().dump(color_measure))

    # Load input images
    input_images = []
    for img_url in color_measure.input_imgs:
        input_images.append(img_handler.load(img_url))

    # Load reference images
    ref_images = []
    for img_url in color_measure.ref_imgs:
        ref_images.append(img_handler.load(img_url))

    if len(input_images) == 0 or len(ref_images) == 0:
        return jsonify({"message": "Bad request, there must be at least one input image!"}), 400

    # TODO: These parameters should be provided by use from client!

    src_color_space = "sRGB"
    src_is_linear = False
    ref_color_space = "sRGB"
    ref_is_linear = False
    ccm, error, output_imgs = color_calibration(
        input_images, src_color_space, src_is_linear, ref_images, ref_color_space, ref_is_linear, verbose=False
    )

    # Store the ouptut images
    output_img_urls = []
    for img in output_imgs:
        img_url = img_handler.save(img)
        output_img_urls.append(img_url)

    # Form a colorResult
    color_result = ColorResult(ccm.tolist(), error, output_img_urls)

    # If there is already color measurement result, remove the old output images first
    if color_measure.color_result != None:
        while len(color_measure.color_result.output_imgs) > 0:
            fname = color_measure.color_result.output_imgs[-1]
            img_handler.delete(fname)
            color_measure.color_result.output_imgs.pop()

    # Update the DB
    try:
        dict_color_result = ColorResultSchema().dump(color_result)
        color_collection.update({"dut_id": dut_id}, {"$set": {"color_result": dict_color_result}})
    except pymongo.errors.PyMongoError as e:
        return jsonify({"message": e}), 400

    color_result_dict = ColorResultSchema().dump(color_result)

    return jsonify({"color_result": color_result_dict}), 200
