import polars as pl
from pathlib import Path
from typing import List, Tuple, Union, Iterable


class DatasetMetadataLoader:
    """
    A utility class to scan directories and build a metadata DataFrame
    mapping file paths to their corresponding category labels.
    """

    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize the loader with a base directory.

        Args:
                base_path (Union[str, Path]): The root directory containing category subfolders.
        """
        self.base_path = Path(base_path)

    def get_files_by_category(self, category: str) -> List[Tuple[str, str]]:
        """
        Retrieves all files within a specific category (subdirectory).

        Args:
                category (str): The name of the subdirectory acting as the label.

        Returns:
                List[Tuple[str, str]]: A list of tuples formatted as (absolute_file_path, label).
        """
        target_dir = self.base_path / category

        return [
            (str(p.resolve()), category) for p in target_dir.iterdir() if p.is_file()
        ]

    def collect_all_metadata(self, categories: Iterable[str]) -> List[Tuple[str, str]]:
        """
        Aggregates file data from multiple categories.

        Args:
                categories (Iterable[str]): A list or sequence of category names to scan.

        Returns:
                List[Tuple[str, str]]: Combined list of (path, label) tuples.
        """
        all_data = []

        for category in categories:
            all_data.extend(self.get_files_by_category(category))

        return all_data

    def to_dataframe(self, categories: Iterable[str]) -> pl.DataFrame:
        """
        Builds a Polars DataFrame containing file paths and labels.

        Args:
                categories (Iterable[str]): The categories to include in the DataFrame.

        Returns:
                pl.DataFrame: A DataFrame with columns ["file_path", "label"].
        """
        all_data = self.collect_all_metadata(categories)

        return pl.DataFrame(all_data, schema=["File Path", "Label"], orient="row")
