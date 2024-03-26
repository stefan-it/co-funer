import flair

from flair.datasets.sequence_labeling import ColumnCorpus
from flair.file_utils import cached_path

from pathlib import Path
from typing import Optional, Union

# Taken from my example notebook:
# https://huggingface.co/datasets/stefan-it/co-funer/blob/main/FlairDatasetExample.ipynb
class NER_CO_FUNER(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)
        dataset_name = self.__class__.__name__.lower()
        data_folder = base_path / dataset_name
        data_path = flair.cache_root / "datasets" / dataset_name

        columns = {0: "text", 2: "ner"}

        hf_download_path = "https://huggingface.co/datasets/stefan-it/co-funer/resolve/main"

        for split in ["train", "dev", "test"]:
            cached_path(f"{hf_download_path}/{split}.tsv", data_path)

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            comment_symbol=None,
            **corpusargs,
        )
