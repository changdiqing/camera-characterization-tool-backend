""" Model for Distortion Measurement
"""
import datetime as dt
from marshmallow import Schema, post_load, fields
from .base_schema import BaseSchema


class DistResult:
    def __init__(self, mtx, dist, rvecs, tvecs, error, output_imgs):
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.error = error
        self.output_imgs = output_imgs

    def __repr__(self):
        return "<Distortion Measurement Result>".format(self=self)


class DistResultSchema(BaseSchema):
    mtx = fields.List(fields.List(fields.Float()))
    dist = fields.List(fields.List(fields.Float()))
    rvecs = fields.List(fields.List(fields.List(fields.Float())))
    tvecs = fields.List(fields.List(fields.List(fields.Float())))
    error = fields.Float()
    output_imgs = fields.List(fields.String())

    @post_load
    def make_dist_result(self, data, **kwargs):
        return DistResult(**data)


class DistMeasure:
    def __init__(
        self,
        dut_id,
        input_imgs,
        dist_result=None,
        _id=None,
        created_at=None,
    ):
        self._id = _id
        self.dut_id = dut_id
        self.input_imgs = input_imgs
        self.created_at = dt.datetime.now() if created_at == None else created_at
        self.dist_result = dist_result

    def __repr__(self):
        return "<Distortion Measurement(dut_id={self.dut_id!r})>".format(self=self)


class DistMeasureSchema(Schema):
    _id = fields.String()
    dut_id = fields.String()
    input_imgs = fields.List(fields.String())
    created_at = fields.String()
    dist_result = fields.Nested(DistResultSchema, missing=None)

    @post_load
    def make_dist_measure(self, data, **kwargs):
        return DistMeasure(**data)
