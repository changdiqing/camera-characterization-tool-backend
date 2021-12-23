""" Model for DUT (device  under test)
"""
import datetime as dt
from marshmallow import Schema, post_load, fields


class DUT:
    def __init__(self, _id, name, description):
        self._id = _id
        self.name = name
        self.description = description
        self.created_at = dt.datetime.now()

    def __repr__(self):
        return "<Device Under Test(name={self.name!r})>".format(self=self)


class DUTSchema(Schema):
    _id = fields.String()
    name = fields.String()
    description = fields.String()
    created_at = fields.Date()

    @post_load
    def make_dut(self, data, **kwargs):
        return DUT(**data)
