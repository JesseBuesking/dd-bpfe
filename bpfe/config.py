

import numpy


INPUT_MAPPING = {
    'Object_Description': 'object_description',
    'Program_Description': 'program_description',
    'SubFund_Description': 'subfund_description',
    'Job_Title_Description': 'job_title_description',
    'Facility_or_Department': 'facility_or_department',
    'Sub_Object_Description': 'sub_object_description',
    'Location_Description': 'location_description',
    'FTE': 'fte',
    'Function_Description': 'function_description',
    'Position_Extra': 'position_extra',
    'Text_4': 'text_4',
    'Total': 'total',
    'Text_2': 'text_2',
    'Text_3': 'text_3',
    'Fund_Description': 'fund_description',
    'Text_1': 'text_1'
}
REVERSE_INPUT_MAPPING = {}
INPUT_CODES = {}
i = 0
for key, value in INPUT_MAPPING.iteritems():
    INPUT_CODES[value] = str(i)
    i += 1
    REVERSE_INPUT_MAPPING[value] = key

LABEL_MAPPING = {
    'Function': 'function',
    'Use': 'use',
    'Sharing': 'sharing',
    'Reporting': 'reporting',
    'Student_Type': 'student_type',
    'Position_Type': 'position_type',
    'Object_Type': 'object_type',
    'Pre_K': 'pre_k',
    'Operating_Status': 'operating_status'
}
REVERSE_LABEL_MAPPING = {}
for key, value in LABEL_MAPPING.iteritems():
    REVERSE_LABEL_MAPPING[value] = key

LABELS = {
    'Function': [
        'Aides Compensation',
        'Career & Academic Counseling',
        'Communications',
        'Curriculum Development',
        'Data Processing & Information Services',
        'Development & Fundraising',
        'Enrichment',
        'Extended Time & Tutoring',
        'Facilities & Maintenance',
        'Facilities Planning',
        'Finance, Budget, Purchasing & Distribution',
        'Food Services',
        'Governance',
        'Human Resources',
        'Instructional Materials & Supplies',
        'Insurance',
        'Legal',
        'Library & Media',
        'NO_LABEL',
        'Other Compensation',
        'Other Non-Compensation',
        'Parent & Community Relations',
        'Physical Health & Services',
        'Professional Development',
        'Recruitment',
        'Research & Accountability',
        'School Administration',
        'School Supervision',
        'Security & Safety',
        'Social & Emotional',
        'Special Population Program Management & Support',
        'Student Assignment',
        'Student Transportation',
        'Substitute Compensation',
        'Teacher Compensation',
        'Untracked Budget Set-Aside',
        'Utilities'
    ],
    'Object_Type': [
        'Base Salary/Compensation',
        'Benefits',
        'Contracted Services',
        'Equipment & Equipment Lease',
        'NO_LABEL',
        'Other Compensation/Stipend',
        'Other Non-Compensation',
        'Rent/Utilities',
        'Substitute Compensation',
        'Supplies/Materials',
        'Travel & Conferences'
    ],
    'Operating_Status': [
        'Non-Operating',
        'Operating, Not PreK-12',
        'PreK-12 Operating'
    ],
    'Position_Type': [
        '(Exec) Director',
        'Area Officers',
        'Club Advisor/Coach',
        'Coordinator/Manager',
        'Custodian',
        'Guidance Counselor',
        'Instructional Coach',
        'Librarian',
        'NO_LABEL',
        'Non-Position',
        'Nurse',
        'Nurse Aide',
        'Occupational Therapist',
        'Other',
        'Physical Therapist',
        'Principal',
        'Psychologist',
        'School Monitor/Security',
        'Sec/Clerk/Other Admin',
        'Social Worker',
        'Speech Therapist',
        'Substitute',
        'TA',
        'Teacher',
        'Vice Principal'
    ],
    'Pre_K': [
        'NO_LABEL',
        'Non PreK',
        'PreK'
    ],
    'Reporting': [
        'NO_LABEL',
        'Non-School',
        'School'
    ],
    'Sharing': [
        'Leadership & Management',
        'NO_LABEL',
        'School Reported',
        'School on Central Budgets',
        'Shared Services'
    ],
    'Student_Type': [
        'Alternative',
        'At Risk',
        'ELL',
        'Gifted',
        'NO_LABEL',
        'Poverty',
        'PreK',
        'Special Education',
        'Unspecified'
    ],
    'Use': [
        'Business Services',
        'ISPD',
        'Instruction',
        'Leadership',
        'NO_LABEL',
        'O&M',
        'Pupil Services & Enrichment',
        'Untracked Budget Set-Aside'
    ]
}

FLAT_LABELS = []
tups = [(k, v) for k, v in LABELS.iteritems()]
tups.sort(key=lambda x: x[0])
for tup in tups:
    for value in tup[1]:
        FLAT_LABELS.append((tup[0], value))

KLASS_LABEL_INFO = dict()
_klass_idx = 0
for tup in FLAT_LABELS:
    if tup[0] not in KLASS_LABEL_INFO:
        KLASS_LABEL_INFO[tup[0]] = (_klass_idx, 0)
        _klass_idx += 1
    info = KLASS_LABEL_INFO[tup[0]]
    KLASS_LABEL_INFO[tup[0]] = (info[0], info[1] + 1)


class HiddenLayerSettings(object):

    def __init__(self, num_nodes, epochs, learning_rate):
        self.num_nodes = num_nodes
        self.epochs = epochs
        self.learning_rate = learning_rate

    @property
    def num_nodes(self):
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, value):
        assert isinstance(value, int)
        self._num_nodes = value

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        assert isinstance(value, int)
        self._epochs = value

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        assert isinstance(value, float)
        self._learning_rate = value

    def filename(self):
        return '{}-{}'.format(
            self.num_nodes,
            self.learning_rate
        )

    def __str__(self):
        return '{}-{}-{}'.format(
            self.num_nodes,
            self.epochs,
            self.learning_rate
        )


class ClassSettings(object):

    @property
    def best_validation_loss(self):
        return self._best_validation_loss

    @best_validation_loss.setter
    def best_validation_loss(self, value):
        assert isinstance(value, float)
        self._best_validation_loss = value

    @property
    def best_test_loss(self):
        return self._best_test_loss

    @best_test_loss.setter
    def best_test_loss(self, value):
        assert isinstance(value, float)
        self._best_test_loss = value

    @property
    def best_iteration(self):
        return self._best_iteration

    @best_iteration.setter
    def best_iteration(self, value):
        assert isinstance(value, int)
        self._best_iteration = value

    @property
    def improvement_threshold(self):
        return 0.995

    @property
    def patience(self):
        return self._patience

    @patience.setter
    def patience(self, patience):
        assert isinstance(patience, float)
        if isinstance(self.patience, float):
            self._patience = max(self.patience, patience)
        else:
            self._patience = patience

    @property
    def patience_increase(self):
        return 2.0

    @property
    def minimum_improvement(self):
        return self.best_validation_loss * self.improvement_threshold


class FinetuningSettings(object):

    def __init__(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self._klass_info = dict()

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        assert isinstance(value, int)
        self._epochs = value

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        assert isinstance(value, float)
        self._learning_rate = value

    def filename(self):
        return '{}'.format(
            self.learning_rate
        )

    def __str__(self):
        return '{}-{}'.format(
            self.epochs,
            self.learning_rate
        )

    def __eq__(self, other):
        return \
            self.epochs == other.epochs and \
            self.learning_rate == other.learning_rate

    def __getitem__(self, item):
        if item not in self._klass_info:
            cs = ClassSettings()
            cs._best_iteration = 0
            cs._best_test_loss = numpy.inf
            cs._best_validation_loss = numpy.inf
            cs._patience = None
            self._klass_info[item] = cs

        return self._klass_info[item]


class ChunkSettings(object):

    def __init__(self, train, validate, test, submission=None):
        self.train = train
        self.validate = validate
        self.test = test
        if submission is None:
            self.submission = train
        else:
            self.submission = submission

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        assert isinstance(value, int)
        self._train = value

    @property
    def validate(self):
        return self._validate

    @validate.setter
    def validate(self, value):
        assert isinstance(value, int)
        self._validate = value

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, value):
        assert isinstance(value, int)
        self._test = value

    @property
    def submission(self):
        return self._submission

    @submission.setter
    def submission(self, value):
        assert isinstance(value, int)
        self._submission = value

    def __eq__(self, other):
        return \
            self.train == other.train and \
            self.validate == other.validate and \
            self.test == other.test and \
            self.submission == other.submission

    def __str__(self):
        return '{}-{}-{}-{}'.format(
            self.train,
            self.validate,
            self.test,
            self.submission
        )


class Settings(object):

    def __init__(self):
        self._patience = None
        self._train_size = None
        self._batch_size = None

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, value):
        assert isinstance(value, float)
        self._version = value

    @property
    def theano_rng(self):
        return self._theano_rng

    @theano_rng.setter
    def theano_rng(self, value):
        from theano.tensor.shared_randomstreams import RandomStreams
        assert isinstance(value, RandomStreams)
        self._theano_rng = value

    @property
    def numpy_rng(self):
        return self._numpy_rng

    @numpy_rng.setter
    def numpy_rng(self, value):
        # noinspection PyUnresolvedReferences
        assert isinstance(value, numpy.random.RandomState)
        self._numpy_rng = value

    @property
    def train_size(self):
        return self._train_size

    @train_size.setter
    def train_size(self, value):
        assert isinstance(value, int)
        self._train_size = value
        if isinstance(self.batch_size, int):
            for value in self.finetuning._klass_info.values():
                value.patience = 4. * (self.train_size / self.batch_size)

    @property
    def num_cols(self):
        return self._num_cols

    @num_cols.setter
    def num_cols(self, value):
        assert isinstance(value, int)
        self._num_cols = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        assert isinstance(value, int)
        self._batch_size = value
        if isinstance(self.train_size, int):
            for value in self.finetuning._klass_info.values():
                value.patience = 4. * (self.train_size / self.batch_size)

    @property
    def hidden_layers(self):
        return self._hidden_layers

    @hidden_layers.setter
    def hidden_layers(self, value):
        assert isinstance(value, list)
        assert isinstance(value[0], HiddenLayerSettings)
        self._hidden_layers = value

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        assert isinstance(value, int)
        self._k = value

    @property
    def finetuning(self):
        return self._finetuning

    @finetuning.setter
    def finetuning(self, value):
        assert isinstance(value, FinetuningSettings)
        self._finetuning = value

    @property
    def chunks(self):
        return self._chunks

    @chunks.setter
    def chunks(self, value):
        assert isinstance(value, ChunkSettings)
        self._chunks = value

    @property
    def train_batches(self):
        return self.train_size / self.batch_size

    @property
    def validation_frequency(self):
        return self.train_batches

    def pretrain_string(self, layer_num):
        pre_hl = '-'.join([
            str(i.epochs) for i in self.hidden_layers[:layer_num]
        ])
        if pre_hl != '':
            return '{}-{}-{}-{}-{}'.format(
                self.batch_size,
                self.k,
                self.chunks,
                pre_hl,
                self.hidden_layers[layer_num].filename()
            )
        else:
            return '{}-{}-{}-{}'.format(
                self.batch_size,
                self.k,
                self.chunks,
                self.hidden_layers[layer_num].filename()
            )

    def finetune_string(self):
        hl = '-'.join([str(i.epochs) for i in self.hidden_layers])
        return '{}-{}-{}-{}-{}'.format(
            self.batch_size,
            self.k,
            self.chunks,
            self.finetuning.filename(),
            hl
        )

    def pretrain_fname(self, layer_num):
        return '{}-{}-{}'.format(
            self.version,
            self.pretrain_string(layer_num),
            layer_num
        )

    def finetuning_fname(self):
        return '{}-{}'.format(
            self.version,
            self.finetune_string()
        )

    def __eq__(self, other):
        return \
            self.version == other.version and \
            self.batch_size == other.batch_size and \
            self.hidden_layers == other.hidden_layers and \
            self.k == other.k and \
            self.finetuning == other.finetuning and \
            self.chunks == other.chunks


class StatsSettings(object):

    fname = 'data/settings_stats.pkl'
