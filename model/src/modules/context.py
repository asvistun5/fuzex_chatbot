from strings import normalize


def sentiment(text):
    text_norm = normalize(text)
    words = set(text_norm.split())

    positive = {"хорошо", "отлично", "прекрасно", "замечательно", "круто", "супер"}
    negative = {"плохо", "ужасно", "отвратительно", "хреново", "хуёво"}

    pos_score = len(words & positive)
    neg_score = len(words & negative)

    letters = [c for c in text if c.isalpha()]

    if letters:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    else:
        caps_ratio = 0

    if caps_ratio > 0.5:
        neg_score += 1

    if pos_score > 0 and pos_score >= neg_score:
        return "positive"
    elif neg_score > 0:
        return "negative"
    else:
        return "neutral"