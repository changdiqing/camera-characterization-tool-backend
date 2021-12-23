""" Model for Reso Measurement
"""
from .base_schema import BaseSchema
import datetime as dt
from marshmallow import post_load, fields


class ResoResult:
    def __init__(self, mtf, output_imgs):
        self.mtf = mtf
        self.output_imgs = output_imgs

    def __repr__(self):
        return "<Reso Measurement Result>".format(self=self)


class ResoResultSchema(BaseSchema):
    mtf = fields.List(fields.List(fields.Float()))
    output_imgs = fields.List(fields.String())

    @post_load
    def make_reso_result(self, data, **kwargs):
        return ResoResult(**data)


class ResoMeasure:
    def __init__(self, dut_id, input_imgs=[], reso_result=None, _id=None, created_at=None):
        self.id = _id
        self.dut_id = dut_id
        self.input_imgs = input_imgs
        self.reso_result = reso_result
        self.created_at = dt.datetime.now() if created_at is None else created_at

    def __repr__(self):
        return "<Reso Measurement(dut_id={self.dut_id!r})>".format(self=self)


class ResoMeasureSchema(BaseSchema):
    _id = fields.String()
    dut_id = fields.String()
    input_imgs = fields.List(fields.String())
    reso_result = fields.Nested(ResoResultSchema, missing=None)
    created_at = fields.String()

    @post_load
    def make_reso_measure(self, data, **kwargs):
        return ResoMeasure(**data)
