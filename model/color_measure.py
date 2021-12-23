""" Model for Color Measurement
"""
from .base_schema import BaseSchema
import datetime as dt
from marshmallow import Schema, post_load, fields


class ColorResult:
    def __init__(self, ccm, error, output_imgs):
        self.ccm = ccm
        self.error = error
        self.output_imgs = output_imgs

    def __repr__(self):
        return "<Color Measurement Result>".format(self=self)


class ColorResultSchema(BaseSchema):
    ccm = fields.List(fields.List(fields.Float()))
    error = fields.Float()
    output_imgs = fields.List(fields.String())

    @post_load
    def make_color_result(self, data, **kwargs):
        return ColorResult(**data)


class ColorMeasure:
    def __init__(self, dut_id, input_imgs=[], ref_imgs=[], color_result=None, _id=None, created_at=None):
        self.id = _id
        self.dut_id = dut_id
        self.input_imgs = input_imgs
        self.ref_imgs = ref_imgs
        self.color_result = color_result
        self.created_at = dt.datetime.now() if created_at is None else created_at

    def __repr__(self):
        return "<Color Measurement(dut_id={self.dut_id!r})>".format(self=self)


class ColorMeasureSchema(BaseSchema):
    _id = fields.String()
    dut_id = fields.String()
    input_imgs = fields.List(fields.String())
    ref_imgs = fields.List(fields.String())
    color_result = fields.Nested(ColorResultSchema, missing=None)
    created_at = fields.String()

    @post_load
    def make_color_measure(self, data, **kwargs):
        return ColorMeasure(**data)
