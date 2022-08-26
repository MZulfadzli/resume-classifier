from spacy.lang.en import English
from spacy.gold import biluo_tags_from_offsets
import pandas as pd


def bilou_tags(data):
    docs = []
    annots = []
    nlp = English()
    for text, annotations in data:
        offsets = annotations["entities"]
        doc = nlp(text)
        tags = biluo_tags_from_offsets(doc, offsets)
        for i in range(len(tags)):
            if tags[i].startswith("U"):
                tags[i] = "B" + tags[i][1:]
            elif tags[i].startswith("L"):
                tags[i] = "I" + tags[i][1:]
            if not (doc[i].text.isalnum() or len(doc[i].text) > 1):
                tags[i] = "O"
        docs.append([token.text for token in doc])
        annots.append(tags)

    df_data = pd.DataFrame({'docs': docs, 'annots': annots})

    return df_data


def remove_mislabel(df_data):
    for i in range(len(df_data)):
        if "-" in df_data.loc[i, "annots"]:
            df_data.drop(i, axis="index", inplace=True)

    return df_data
