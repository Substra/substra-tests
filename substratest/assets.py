import enum


class AssetType(enum.Enum):
    function = enum.auto()
    compute_plan = enum.auto()
    data_sample = enum.auto()
    dataset = enum.auto()
    organization = enum.auto()
    task = enum.auto()

    @classmethod
    def all(cls):
        return [e for e in cls]

    @classmethod
    def can_be_get(cls):
        gettable = cls.all()
        gettable.remove(cls.data_sample)
        gettable.remove(cls.organization)
        return gettable

    @classmethod
    def can_be_listed(cls):
        return cls.all()
