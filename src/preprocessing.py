import pandas as pd


# Convert all reviews to lower case (optional according to study)
def to_lower(data: pd.Series):
    return data.str.lower()


def remove_accented_characters(data: pd.Series):
    import unicodedata

    """Removes accented characters from the Series

    Args:
        data (pd.Series): Series of string

    Returns:
        _type_: pd.Series
    """
    import unicodedata

    return data.apply(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("utf-8", "ignore"))


def remove_html_encodings(data: pd.Series):
    return data.str.replace(r"&#\d+;", " ", regex=True)


def remove_html_tags(data: pd.Series):
    return data.str.replace(r"<[a-zA-Z]+\s?/?>", " ", regex=True)


def remove_url(data: pd.Series):
    return data.str.replace(r"https?://([\w\-\._]+){2,}/[\w\-\.\-/=\+_\?]+", " ", regex=True)


def remove_html_and_url(data: pd.Series):
    """Function to remove
             1. HTML encodings
             2. HTML tags (both closed and open)
             3. URLs

    Args:
        data (pd.Series): A Pandas series of type string

    Returns:
        _type_: pd.Series
    """
    # Remove HTML encodings
    data.str.replace(r"&#\d+;", " ", regex=True)

    # Remove HTML tags (both open and closed)
    data.str.replace(r"<[a-zA-Z]+\s?/?>", " ", regex=True)

    # Remove URLs
    data.str.replace(r"https?://([\w\-\._]+){2,}/[\w\-\.\-/=\+_\?]+", " ", regex=True)

    return data


# Remove non-alphabetical characters
def remove_non_alpha_characters(data: pd.Series):
    return data.str.replace(r"_+|\\|[^a-zA-Z0-9\s]", " ", regex=True)


# Remove extra spaces
def remove_extra_spaces(data: pd.Series):
    return data.str.replace(r"^\s*|\s\s*", " ", regex=True)


# Expanding contractions
def fix_contractions(data: pd.Series):
    import contractions

    def contraction_fixer(txt: str):
        return " ".join([contractions.fix(word) for word in txt.split()])

    return data.apply(contraction_fixer)


# remove "-lrb-"
def remove_special_words(data: pd.Series):
    return data.str.replace(r"\-[^a-zA-Z]{3}\-", " ", regex=True)
