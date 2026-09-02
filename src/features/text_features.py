import re
import pandas as pd

def extract_text_statistics(df: pd.DataFrame, text_column: str = "statement") -> pd.DataFrame:
    df_out = df.copy()
    series = df_out[text_column].astype(str)
    df_out["statement_char_len"] = series.str.len()
    df_out["statement_word_count"] = series.str.split().apply(len)
    word_count_safe = df_out["statement_word_count"].replace(0, 1)
    df_out["avg_word_length"] = (df_out["statement_char_len"] / word_count_safe).round(2)
    return df_out

def clean_text_simple(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()
