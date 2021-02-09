import os


def createSubmissionFile(df, name, submission_folder="../submissions/"):
    assert "item_cnt_month" in df.columns
    assert "ID" in df.columns
    df["item_cnt_month"] = df["item_cnt_month"].clip(upper=20, lower=0)
    df[["ID", "item_cnt_month"]].to_csv(os.path.join(os.getcwd(), submission_folder, name), index=False)
    return None
