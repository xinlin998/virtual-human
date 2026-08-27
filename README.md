# virtual-human

当前阶段实现私人聊天数据预处理与表情包处理流水线：

1. 读取并校验聊天 CSV；
2. 标准化消息类型与时间字段；
3. 过滤无效消息并生成 `message_id`；
4. 使用 pHash 对表情包去重；
5. 使用 `Qwen/Qwen2.5-VL-3B-Instruct` 批量分析唯一表情包；
6. 回填 `sticker_id`、`sticker_caption` 和 `normalized_text`；
7. 保存处理后的 CSV 与 `sticker_metadata.jsonl`。

## 初始化

```powershell
uv lock
uv sync
```

## 运行

将私人聊天 CSV 放到 `data/raw/chat.csv`，表情包文件放到 `data/stickers/`，然后执行：

```powershell
uv run python main.py
```

## 静态检查

```powershell
uv run ruff check .
uv run python -m compileall main.py src
```

`data/raw/`、`data/processed/`、`data/stickers/` 和 `data/outputs/` 均被 `.gitignore` 忽略，不应提交私人聊天数据或表情包文件。
