# get one distortion measure by dut_id
curl http://localhost:5000/dist?dut_id=12345

# add new distortion measure
curl -X POST -H "Content-Type: application/json" -d '{
    "_id": "aeaf6f78-15f7-11ec-9408-a45e60c16ea9",
    "img_url": "/Users/diqingchang/cam-calibration/img/left01.jpg"
}' http://localhost:5000/dist_img

# check if img_url is added
curl http://localhost:5000/dist?dut_id=12345