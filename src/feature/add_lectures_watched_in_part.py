import pandas as pd
from loguru import logger

if __name__ == '__main__':
    logger.add("logs/logs_add_lectures_watched_in_part.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        train_df = pd.read_csv("data/interim/train_cropped.csv")
        lectures = pd.read_csv("data/raw/lectures.csv")

        train_df = pd.merge(
            train_df,
            lectures[['lecture_id', 'part', 'type_of']],
            how='left',
            left_on='content_id',
            right_on='lecture_id'
        )

        train_df.drop(columns=['lecture_id'], inplace=True)
        print(train_df.head())

        lectures_watched_in_part = train_df[train_df['content_type_id'] == 1].groupby(['user_id', 'part'])['row_id']\
            .nunique()
        lectures_watched_in_part = pd.DataFrame(lectures_watched_in_part)
        lectures_watched_in_part = lectures_watched_in_part.reset_index()
        lectures_watched_in_part = lectures_watched_in_part.rename(columns={'row_id': 'lectures_watched_in_part'})
        print(lectures_watched_in_part.head())
        lectures_watched_in_part.to_csv("data/processed/lectures_watched_in_part.csv")

        train_with_bundle_id_part_df = pd.read_csv("data/interim/train_with_bundle_id_part.csv")
        train_with_lecture_count = train_with_bundle_id_part_df.merge(lectures_watched_in_part,
                                                                      on=['user_id', 'part'],
                                                                      how='left')

        train_with_lecture_count['lectures_watched_in_part'] = train_with_lecture_count['lectures_watched_in_part']\
            .fillna(0)
        print(train_with_lecture_count.head())
        train_with_lecture_count.to_csv("data/interim/train_with_lecture_count.csv")
