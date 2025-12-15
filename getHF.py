from huggingface_hub import HfApi
import csv
import os
from datetime import datetime

def fetch_all_models(output_file="huggingface_models.csv"):
    api = HfApi()
    models_iter= api.list_models()#不指定 1imit,返回迭代器

    #打开 CSV 文件写入
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "author", "downloads", "likes", "pipeline_tag", "trending_score","tags"])

        count = 0
        for model in models_iter:
            writer.writerow([
                model.id,
                model.author,
                model.downloads,
                model.likes,
                model.pipeline_tag,
                model.trending_score,
                model.tags
            ])
            count += 1
            if count % 1000 == 0:
                print(f"已处理{count}个模型 ... ")

    print(f"完成！共保存 {count}个模型到 {output_file}")

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name="huggingface_models"
    ext=".csv"
    final_filename = f"{name}_{timestamp}{ext}"
    fetch_all_models(final_filename)