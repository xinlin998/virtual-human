from src.config import load_config
from src.preprocessed import *
from virtual_human.sticker_pipeline import (
    QwenStickerAnalyzer,
    process_stickers,
    save_sticker_metadata,
)
import pandas as pd

def main():
    CONFIGS = load_config("configs/pipeline.yaml")
    INPUT_CSV = CONFIGS["data"]['input_csv']
    ALLOWED_TYPES = CONFIGS['filter']['allowed_types']
    PROCESSED_DIR = CONFIGS['data']['processed_dir']
    STICKERED_DIR = CONFIGS['data']['stickered_dir']
    MODEL_NAME = CONFIGS['sticker']['model_name']
    MODEL_DIR = CONFIGS['data']['model_dir']
    OUTPUT_DIR = CONFIGS['data']['output_dir']
    df = pd.read_csv(INPUT_CSV)
    df = filter_message_types(df,type_col='type_name',allowed_types=ALLOWED_TYPES)
    df = drop_empty_messages(df,content_col='msg')
    df = process_chat_csv(df)
    df = add_message_id(df)
    df.to_csv(PROCESSED_DIR)
    df = pd.read_csv(PROCESSED_DIR)
    analyzer = QwenStickerAnalyzer(model_name="Qwen/Qwen2.5-VL-3B-Instruct",model_dir=MODEL_DIR)
    df, sticker_metadata = process_stickers(
        df=df,
        analyzer=analyzer,
        message_type_col=(
            "message_type"
        ),
        sticker_path_col=(
            "sticker_path"
        ),
        phash_threshold=5,
    )
    df.to_csv(STICKERED_DIR)
    save_sticker_metadata(sticker_metadata,OUTPUT_DIR)




if __name__ == "__main__":
    main()