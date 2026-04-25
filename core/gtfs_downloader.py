"""
Module for downloading and extracting Warsaw GTFS data
"""
import requests
import zipfile
from pathlib import Path
from datetime import datetime
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GTFS_URL, RAW_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GTFSDownloader:
    """Download and extract GTFS data"""

    def __init__(self, url: str = GTFS_URL, output_dir: Path = RAW_DIR):
        self.url = url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> Path:
        """
        Download GTFS zip file

        Returns:
            Path to downloaded zip file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = self.output_dir / f"warsaw_gtfs_{timestamp}.zip"

        logger.info(f"Downloading GTFS data from {self.url}")

        try:
            response = requests.get(self.url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(zip_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        downloaded += len(chunk)
                        f.write(chunk)
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownload progress: {progress:.1f}%", end='')

            print()  # New line after progress
            logger.info(f"Downloaded successfully to {zip_path}")
            return zip_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download GTFS data: {e}")
            raise

    def extract(self, zip_path: Path) -> Path:
        """
        Extract GTFS zip file

        Args:
            zip_path: Path to zip file

        Returns:
            Path to extracted directory
        """
        extract_dir = zip_path.parent / zip_path.stem
        extract_dir.mkdir(exist_ok=True)

        logger.info(f"Extracting to {extract_dir}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            logger.info(f"Extracted {len(list(extract_dir.glob('*.txt')))} files")
            return extract_dir

        except zipfile.BadZipFile as e:
            logger.error(f"Failed to extract zip file: {e}")
            raise

    def download_and_extract(self) -> Path:
        """
        Download and extract GTFS data in one step

        Returns:
            Path to extracted directory
        """
        zip_path = self.download()
        extract_dir = self.extract(zip_path)

        # Clean up zip file after extraction
        zip_path.unlink()
        logger.info(f"Cleaned up temporary zip file")

        return extract_dir

    def get_latest_data_dir(self) -> Path:
        """
        Get the most recently extracted GTFS directory (ZTM only, not combined).

        Returns:
            Path to latest GTFS data directory
        """
        gtfs_dirs = sorted(
            [d for d in self.output_dir.iterdir()
             if d.is_dir() and d.name.startswith('warsaw_gtfs') and '_with_km' not in d.name],
            key=lambda x: x.name,
            reverse=True
        )

        if not gtfs_dirs:
            logger.info("No existing GTFS data found, downloading fresh data")
            return self.download_and_extract()

        latest = gtfs_dirs[0]
        logger.info(f"Using existing GTFS data from {latest}")
        return latest


if __name__ == "__main__":
    downloader = GTFSDownloader()
    data_dir = downloader.download_and_extract()
    print(f"\nGTFS data ready at: {data_dir}")
