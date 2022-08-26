import pycrfsuite
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from itertools import chain
import warnings
warnings.filterwarnings('ignore')


def model_ner(X_train, y_train, l1, l2, max_iter):
    trainer = pycrfsuite.Trainer(verbose = True)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({
        'c1': l1,
        'c2': l2,
        'max_iterations': max_iter,
        'feature.possible_transitions': True
    })

    trainer.train('ner-classifier.crfsuite')


def prediction(X_test):
    tagger = pycrfsuite.Tagger()
    tagger.open('./ner-classifier.crfsuite')

    return [tagger.tag(xseq) for xseq in X_test]


def ner_report(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_)
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset
    ), accuracy_score(y_true_combined, y_pred_combined)
