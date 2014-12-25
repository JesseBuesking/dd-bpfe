

# noinspection PyUnresolvedReferences
import re
import sys
from bpfe.config import LABELS, LABEL_MAPPING, FLAT_LABELS, KLASS_LABEL_INFO,\
    GRADE_ORDER


class Data(object):
    __slots__ = (
        'id', 'object_description', 'program_description',
        'subfund_description', 'job_title_description',
        'facility_or_department', 'sub_object_description',
        'location_description', 'fte', 'function_description', 'position_extra',
        'text_4', 'total', 'text_2', 'text_3', 'fund_description', 'text_1',
        'cleaned'
    )

    float_attributes = (
        'fte', 'total'
    )

    text_attributes = (
        'object_description', 'program_description',
        'subfund_description', 'job_title_description',
        'facility_or_department', 'sub_object_description',
        'location_description', 'function_description', 'position_extra',
        'text_4', 'text_2', 'text_3', 'fund_description', 'text_1'
    )

    attributes = tuple(float_attributes + text_attributes)

    attribute_types = (
        str, str, str, str, str, str, str, float, str, str, str, float, str,
        str, str, str
    )

    def __init__(self):
        self.cleaned = dict()

    @property
    def grades(self):
        s, e = [], []
        for attr in Data.text_attributes:
            value = self.cleaned[attr + '-mapped']
            for start, end in re.findall('GRADE=(k|\d+)\|(k|\d+)', value):
                s.append(start)
                e.append(end)

        fin = [False] * len(GRADE_ORDER)
        if len(s) == 0:
            return fin

        s_min = min(s, key=GRADE_ORDER.get)
        e_max = max(e, key=GRADE_ORDER.get)

        for i in range(GRADE_ORDER.get(s_min), GRADE_ORDER.get(e_max) + 1):
            fin[i] = True

        return fin

    @property
    def any_grades(self):
        return any(self.grades)

    @property
    def is_prek(self):
        return self.grades[0] and not any(self.grades[1:])

    @property
    def is_k(self):
        return self.grades[1]

    @property
    def is_elementary(self):
        return any(self.grades[2:8])

    @property
    def is_middle(self):
        return any(self.grades[8:10])

    @property
    def is_high(self):
        return any(self.grades[10:])

    @property
    def title(self):
        ttl = None
        for attr in Data.text_attributes:
            value = self.cleaned[attr + '-mapped']
            for t in re.findall('TITLE=(\d+)', value):
                ttl = int(t)

        title = [False] * 11
        if ttl is not None:
            title[ttl] = True

        return title

    @property
    def is_title(self):
        return any(self.title)

    def __str__(self):
        val = '{}:\n'.format(self.__class__.__name__)
        for key in self.__slots__:
            val += '  {:>25}: "{}"\n'.format(key, getattr(self, key))
        return val

    def __eq__(self, other):
        same = []
        for attr in Data.text_attributes:
            same.append(getattr(self, attr) == getattr(other, attr))
        return all(same)

    def __hash__(self):
        s = []
        for attr in Data.text_attributes:
            s.append((attr, getattr(self, attr)))

        return hash(frozenset(s))


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

    def __eq__(self, other):
        same = []
        for attr in self.__slots__:
            if attr == 'id':
                continue
            same.append(getattr(self, attr) == getattr(other, attr))
        return all(same)

    def __hash__(self):
        s = []
        for attr in self.__slots__:
            if attr == 'id':
                continue
            s.append((attr, getattr(self, attr)))

        return hash(frozenset(s))

