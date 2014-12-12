

# noinspection PyUnresolvedReferences
from bpfe.config import LABELS, LABEL_MAPPING, FLAT_LABELS, KLASS_LABEL_INFO


class Data(object):
    __slots__ = (
        'id', 'object_description', 'program_description',
        'subfund_description', 'job_title_description',
        'facility_or_department', 'sub_object_description',
        'location_description', 'fte', 'function_description', 'position_extra',
        'text_4', 'total', 'text_2', 'text_3', 'fund_description', 'text_1'
    )

    attributes = (
        'object_description', 'program_description',
        'subfund_description', 'job_title_description',
        'facility_or_department', 'sub_object_description',
        'location_description', 'fte', 'function_description', 'position_extra',
        'text_4', 'total', 'text_2', 'text_3', 'fund_description', 'text_1'
    )

    attribute_types = (
        str, str, str, str, str, str, str, float, str, str, str, float, str,
        str, str, str
    )

    def __str__(self):
        val = '{}:\n'.format(self.__class__.__name__)
        for key in self.__slots__:
            val += '  {:>25}: "{}"\n'.format(key, getattr(self, key))
        return val


class Label(object):
    __slots__ = (
        'id', 'function', 'use', 'sharing', 'reporting', 'student_type',
        'position_type', 'object_type', 'pre_k', 'operating_status'
    )

    def __str__(self):
        val = '{}:\n'.format(self.__class__.__name__)
        for key in self.__slots__:
            val += '  {:>25}: "{}"\n'.format(key, getattr(self, key))
        return val

    def to_vec(self, klass_num):
        ret = None

        idx = 0
        for key, val in FLAT_LABELS:
            if KLASS_LABEL_INFO[key][0] != klass_num:
                continue
            if ret is None:
                ret = [0] * KLASS_LABEL_INFO[key][1]
            # noinspection PyTypeChecker
            attr = LABEL_MAPPING[key]
            value = getattr(self, attr)
            if value == val:
                ret[idx] = 1
                break
            idx += 1

        return ret

    def to_klass_num(self, klass_num):
        ret_idx = 0
        for _, (key, val) in enumerate(FLAT_LABELS):
            if KLASS_LABEL_INFO[key][0] != klass_num:
                continue
            # noinspection PyTypeChecker
            attr = LABEL_MAPPING[key]
            value = getattr(self, attr)
            if value == val:
                return ret_idx
            ret_idx += 1
