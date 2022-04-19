import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

TitleStyle = ["objective", "subjective", "unknown"]
TitleStance = ["agree", "disagree", "unrelated"]


def load_data(file):
    df = pd.read_json(file, 'index')
    df = df.rename(columns={"VALUE_ACUERDO": "label", "TITLE": 'text_a', 'TEXT': 'text_b'})
    labels = df['label']
    text_a = df['text_a'].replace(r'\n','', regex=True)

    text_b = df['text_b']
    reduce = [ "TitleStyleObjective",
                "TitleStyleSubjective",
                "TitleStyleUnknown",
                "TitleTitle-StanceAgree",
                "TitleTitle-StanceDisagree",
                "TitleTitle-StanceUnrelated"]
    features = ["Title",
                "Subtitle",
                "Lead",
                "Body",
                "Conclusion",
                "What",
                "WhatReliabilityReliable",
                "WhatReliabilityUnreliable",
                "WhatLack-Of-InformationYes",
                "WhatLack-Of-InformationNo",
                "WhatMain-Event",
                "Who",
                "WhoReliabilityReliable",
                "WhoReliabilityUnreliable",
                "WhoLack-Of-InformationYes",
                "WhoLack-Of-InformationNo",
                "When",
                "WhenReliabilityReliable",
                "WhenReliabilityUnreliable",
                "WhenLack-Of-InformationYes",
                "WhenLack-Of-InformationNo",
                "Where",
                "WhereReliabilityReliable",
                "WhereReliabilityUnreliable",
                "WhereLack-Of-InformationYes",
                "WhereLack-Of-InformationNo",
                "Why",
                "WhyReliabilityReliable",
                "WhyReliabilityUnreliable",
                "WhyLack-Of-InformationYes",
                "WhyLack-Of-InformationNo",
                "How",
                "HowReliabilityReliable",
                "HowReliabilityUnreliable",
                "HowLack-Of-InformationYes",
                "HowLack-Of-InformationNo",
                "Quote",
                "QuoteAuthor-StanceAgree",
                "QuoteAuthor-StanceDisagree",
                "QuoteAuthor-StanceUnknown",
                "WhoRoleSubject",
                "WhoRoleTarget",
                "WhoRoleBoth",
                "Key-Expression",
                "Orthotypography",
                "Figure",
                ]

    df = pd.json_normalize(df['DATA'])
    df_1 = df[reduce]
    conditions_stance = [
        (df_1['TitleTitle-StanceAgree'] == 1),
        (df_1['TitleTitle-StanceDisagree'] == 1),
        (df_1['TitleTitle-StanceUnrelated'] == 1)
    ]
    conditions_style = [
        (df_1['TitleStyleObjective'] == 1),
        (df_1['TitleStyleSubjective'] == 1),
        (df_1['TitleStyleUnknown'] == 1)
    ]

    # create a new column and use np.select to assign values to it using our lists as arguments
    df_1['stance'] = np.select(conditions_stance, TitleStance)
    df_1['style'] = np.select(conditions_style, TitleStyle)
    df_1 = df_1.drop(columns=reduce, axis=1)

    # encode columns stance and style
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df_1['stance'].values)
    df_1['stance'] = integer_encoded
    integer_encoded = label_encoder.fit_transform(df_1['style'].values)
    df_1['style'] = integer_encoded

    #other features
    df_2 = df[features]
    features = pd.concat([df_1, df_2], axis=1).values.tolist()

    list_of_tuples = list(zip(list(text_a), list(text_b), list(labels), features))
    df = pd.DataFrame(list_of_tuples, columns=['text_a', 'text_b', 'labels', 'features'])
    return df