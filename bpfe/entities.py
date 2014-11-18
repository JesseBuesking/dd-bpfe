

# noinspection PyUnresolvedReferences
class Data(object):
    __slots__ = (
        'object_description', 'program_description', 'subfund_description',
        'job_title_description', 'facility_or_department',
        'sub_object_description', 'location_description', 'fte',
        'function_description', 'position_extra', 'text_4', 'total', 'text_2',
        'text_3', 'fund_description', 'text_1'
    )

    def __str__(self):
        val = '{}:\n'.format(self.__class__.__name__)
        for key in self.__slots__:
            val += '  {:>25}: "{}"\n'.format(key, getattr(self, key))
        return val


# noinspection PyUnresolvedReferences
class Label(object):
    __slots__ = (
        'function', 'use', 'sharing', 'reporting', 'student_type',
        'position_type', 'object_type', 'pre_k', 'operating_status'
    )

    def __str__(self):
        val = '{}:\n'.format(self.__class__.__name__)
        for key in self.__slots__:
            val += '  {:>25}: "{}"\n'.format(key, getattr(self, key))
        return val
