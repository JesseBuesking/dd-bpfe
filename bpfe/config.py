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
