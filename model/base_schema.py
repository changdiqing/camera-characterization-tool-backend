from marshmallow import Schema


class BaseSchema(Schema):
    class Meta:
        dateformat = "%Y-%m-%dT%H:%M:%S+03:00"
