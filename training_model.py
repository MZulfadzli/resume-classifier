from dataing import *
from tagging import bilou_tags, remove_mislabel
from modeling import *
from sklearn.model_selection import train_test_split

model_conf = {
    "l1": 1.0,
    "l2": 1e-2,
    "max_iter": 30
}


def main():
    data = trim_entity_spans(convert_data_to_spacy("data/Entity-Recognition-in-Resumes.json"))
    df_data = remove_mislabel(bilou_tags(data))
    df_data.reset_index(inplace=True)
    sentences = sentence_getter(df_data)
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_ner(X_train, y_train, model_conf["l1"], model_conf["l2"], model_conf["max_iter"])
    y_pred = prediction(X_test)
    report, accuracy = ner_report(y_test, y_pred)
    print("The accuracy of the NER model is", accuracy)


if __name__ == "__main__":
    main()
