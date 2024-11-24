import os
import datasets
import glob

# 数据集信息
_DESCRIPTION = "NuminaMath CoT dataset."
_CITATION = """\
@misc{numina_math_datasets,
  author = {Jia LI and Edward Beeching and Lewis Tunstall and Ben Lipkin and Roman Soletskyi and Shengyi Costa Huang and Kashif Rasul and Longhui Yu and Albert Jiang and Ziju Shen and Zihan Qin and Bin Dong and Li Zhou and Yann Fleureau and Guillaume Lample and Stanislas Polu},
  title = {NuminaMath},
  year = {2024},
  publisher = {Numina},
  journal = {Hugging Face repository},
  howpublished = {\url{[https://huggingface.co/AI-MO/NuminaMath-CoT](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf)}}
}
"""
#_HOMEPAGE = "https://huggingface.co/datasets/numinamath-cot"
#_LICENSE = "cc-by-nc-4.0"

# 数据集的本地路径
current_dir = os.path.dirname(os.path.dirname(os.getcwd()))  # 获取上一级目录
_DATA_DIR = os.path.join(current_dir, "NuminaMath-CoT", "data")
class NuminaMathCoT(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "source": datasets.Value("string"),
                "problem": datasets.Value("string"),
                "solution": datasets.Value("string"),
                "messages": datasets.Sequence({
                    "content": datasets.Value("string"),
                    "role": datasets.Value("string"),
                }),
            }),
            #homepage=_HOMEPAGE,
            #license=_LICENSE,
            citation=_CITATION,
            task_categories=["text-generation"],
            language=["en"],
            tags=["aimo", "math"],
            pretty_name="NuminaMath CoT",
            splits=[
                datasets.Split.TRAIN,
                datasets.Split.TEST,
            ],
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_DATA_DIR)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepaths": os.path.join(data_dir, "train-*"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": os.path.join(data_dir, "test-*"),
                },
            ),
        ]

    def _generate_examples(self, filepaths):
    # 这里我们使用 pyarrow 来读取 Parquet 文件
        import pyarrow.parquet as pq

        for filepath in glob.glob(filepaths):
            table = pq.read_table(filepath)
            df = table.to_pandas()

        # 过滤 source 为 Olympiads 的数据
            df = df[df['source'] == 'Olympiads']

            for idx, row in df.iterrows():
                yield idx, {
                    "source": row["source"],
                    "problem": row["problem"],
                    "solution": row["solution"],
                    "messages": row["messages"],
                }

# 使用脚本
if __name__ == "__main__":
    dataset = NuminaMathCoT()
    dataset.download_and_prepare()